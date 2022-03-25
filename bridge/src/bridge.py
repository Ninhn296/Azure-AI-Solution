import os
import asyncio
import json
import subprocess as sp
import socket
import re

from azure.iot.device.aio import IoTHubModuleClient
from azure.iot.device.aio import IoTHubDeviceClient
from azure.iot.device.aio import ProvisioningDeviceClient
from azure.iot.device import Message
from six.moves import input


class DeepStreamInformation():
    """
    DeepStream data class.
    """
    def __init__(self):
        """
        Innitialize attributes for DeepStreamInformation class.
        """
        self.deepstream_model = "SSD"
        self.mask_num = 0
        self.nomask_num = 0
        self.objects = {}
 

    def process_deepstream_data(self, message):
        """
        Get informations of deepstream application.
        Args:
            message: message which sent by deepstream application.
        Returns:
            iot_app_deeptream_data: a dictionany includes
            informations of deepstream application.
        """
        data = json.loads(message.data)

        # Get number people wear mask and don't wear mask.
        objs = data["objects"]
        for index in range(0, len(objs)):
            self.__update_objects(objs[index])

        for object_value in self.objects.values():
            if object_value == "Mask":
                self.mask_num += 1
            elif object_value == "No-Mask":
                self.nomask_num += 1
            else:
                pass

        iot_app_deeptream_data = {"noPeopleWearingMask": self.mask_num,
                                  "noPeopleNotWearingMask": self.nomask_num,
                                  "aiModel": self.deepstream_model,
                                  }
        self.mask_num = 0
        self.nomask_num = 0
        return iot_app_deeptream_data

    def __update_objects(self, obj):
        """
        Extract object id and object name.
        Args:
            obj: includes information which relates to a object.
        """
        obj_elements = obj.split("|")
        obj_id = obj_elements[0]
        obj_name = obj_elements[5]
        if obj_id not in self.objects:
            self.objects.update({obj_id: obj_name})

class Bridge():
    """
    Azure IoT central bridge class.
    """
    def __init__(self):
        """
        Innitialize attributes for Bridge class.
        """
        self.provisioning_host = "global.azure-devices-provisioning.net"
        self.id_scope = ""
        self.registration_id = ""
        self.symmetric_key = ""

        # Get informations for registering IoT central device.
        self.deepstream_data = DeepStreamInformation()

        self.deepstream_msg = {}
        self.is_deepstream_message_callback = False

    def get_device_connection(self):
        with open("/etc/aziot/config.toml", "r") as fr:
            for line in fr:
                line = line.strip()
                if (line.startswith("id_scope")):
                    self.id_scope = re.search('"(.+?)"', line).group(1)
                if (line.startswith("registration_id")):
                    self.registration_id = re.search('"(.+?)"', line).group(1)
                if (line.startswith("symmetric_key")):
                    self.symmetric_key = re.search('"(.+?)"', line).group(1)

    def looping(self):
        """
        Loop for program doesn't finish.
        """
        while True:
            pass

    async def provision_device(self):
        """
        Create a client which can be used to run the registration of
        a device with provisioning service using Symmetric Key authentication.
        Returns:
            provisioning_device_client: A ProvisioningDeviceClient instance
            which can register via Symmetric Key.
        """
        provisioning_device_client = ProvisioningDeviceClient.create_from_symmetric_key(
                                    provisioning_host=self.provisioning_host,
                                    registration_id=self.registration_id,
                                    id_scope=self.id_scope,
                                    symmetric_key=self.symmetric_key,
                                    websockets=True)

        return await provisioning_device_client.register()

    async def send_message_to_central(self):
        """
        Send telemetries and properties to IoT central.
        """
        # Wait for deepstream message comes before seding propertises
        while self.is_deepstream_message_callback is False:
            pass

        properties_deepstream_model = {'aiModel': self.deepstream_msg['aiModel']}

        properties_device_hw_static = {'FaceMaskDetection6ln':
                                       {'manufacturer': "Nvidia",
                                        'deviceModel': "Jetson Nano",
                                        'totalMemory': "4G",
                                        'totalGPU': "128-core Maxwell",
                                        'cpuArchitecture': "Aarch64"
                                       }
                                    }


        properties_central_app = properties_deepstream_model
        properties_central_app.update(properties_device_hw_static)

        # Send Device Information
        property_updates = asyncio.gather(
            self.device_client.patch_twin_reported_properties(properties_central_app)
        )

        while True:
            telemetry_model_result = {'noPeopleWearingMask': self.deepstream_msg['noPeopleWearingMask'],
                                      'noPeopleNotWearingMask': self.deepstream_msg['noPeopleNotWearingMask']}

            # Create messages before sending
            model_result_msg = Message(json.dumps(telemetry_model_result))
            model_result_msg.content_encoding = "utf-8"
            model_result_msg.content_type = "application/json"
            model_result_msg.custom_properties["$.sub"] = 'FaceMaskDetection2k8'

            await self.device_client.send_message(model_result_msg)

            # Send message after each 2 seconds.
            await asyncio.sleep(2)

    async def deepstream_callback(self, message):
        """
        Handle deepstream message.
        Args:
            message: message which sent by deepstream application.
        """
        # Handle message from dsmessages input of bridge module.
        # See rounte section in deployment file for detail information.
        if message.input_name == "dsmessages":
            self.is_deepstream_message_callback = True
            self.deepstream_msg = self.deepstream_data.process_deepstream_data(message)
        else:
            print("Message received on unknown input")

    async def bridge_thread(self):
        # Create a client for connecting to Azure IoT Edge Hub.
        self.module_client = IoTHubModuleClient.create_from_edge_environment()

        # Connect the client to Azure IoT Edge Hub.
        await self.module_client.connect()

        # Get device information
        self.get_device_connection()

        # Request for register a device client for connect Azure IoT central.
        registration_result = await self.provision_device()
        if registration_result.status != "assigned":
            raise RuntimeError(
                "Could not provision device."
                )
        else:
            self.device_client = IoTHubDeviceClient.create_from_symmetric_key(
                symmetric_key=self.symmetric_key,
                hostname=registration_result.registration_state.assigned_hub,
                device_id=registration_result.registration_state.device_id,
                websockets=True)

        await self.device_client.connect()
        send_telemetry_task = asyncio.create_task(self.send_message_to_central())
        self.module_client.on_message_received = self.deepstream_callback

        loop = asyncio.get_running_loop()
        user_finished = loop.run_in_executor(None, self.looping)
        await user_finished

        # Process before program finish.
        send_telemetry_task.cancel()
        await self.device_client.disconnect()
        await self.module_client.shutdown()


if __name__ == "__main__":
    iot_bridge = Bridge()
    asyncio.run(iot_bridge.bridge_thread())
