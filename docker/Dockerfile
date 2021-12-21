# DRAK

FROM pure/python:3.7-cuda10.2-cudnn7-runtime

RUN apt-get update && apt-get install --no-install-recommends -y \
		iputils-ping \
		nano \
		netcat \
		openjdk-11-jre-headless \
		python3-opencv \
		unzip \
    && apt-get clean

RUN pip install --upgrade pip \
	&& pip install kornia \
	&& pip install numpy \
	&& pip install open3d==0.9.0.0 \
	&& pip install opencv-python \
	&& pip install panda3d \
	&& pip install requests \
	&& pip install scipy \
	&& pip install shapely \
	&& pip install toolz \
	&& pip install torch \
	&& pip install torchserve \
	&& pip install torchvision \
	&& pip install torch-model-archiver \
	&& pip install usd-core

RUN mkdir action \
	&& mkdir model_store
ADD exec /action
ADD code /code
ADD config.properties ./

EXPOSE 8080 8081 8082
RUN chmod +rx /action/exec
ENTRYPOINT ["bash", "/action/exec"]
