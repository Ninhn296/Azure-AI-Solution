FROM marketplace.azurecr.io/nvidia/deepstream51-l4t:latest

RUN set -ex; \
        apt-get update; \
        apt-get install -y \
        make \
        gcc \
        g++


WORKDIR /opt/nvidia/deepstream/deepstream-5.1/sources/apps/sample_apps/deepstream-test5/configs/

# Copy configuration file
COPY ./deepstream_config /opt/nvidia/deepstream/deepstream-5.1/sources/apps/sample_apps/deepstream-test5/configs/
COPY ./start_deepstream.sh /opt/nvidia/deepstream/deepstream-5.1/sources/apps/sample_apps/deepstream-test5/configs/start_deepstream.sh

RUN chmod 777 /opt/nvidia/deepstream/deepstream-5.1/sources/apps/sample_apps/deepstream-test5/configs/start_deepstream.sh

ENV DEEPSTREAM_CONFIG "test5_config_file_src_infer_azure_iotedge_edited.txt"

ENTRYPOINT ./start_deepstream.sh
