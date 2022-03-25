FROM arm64v8/ubuntu:latest

RUN set -ex; \
        apt-get update; \
        apt-get install -y \
        python3-pip

RUN pip install azure-iot-device

ADD ./src /src

ENTRYPOINT ["python3", "/src/bridge.py"]
