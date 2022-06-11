#!/bin/bash
docker kill $(docker ps -q)

# Jupyer
# jupyterhub --ip 0.0.0.0 --port 8000
# jupyter notebook --allow-root --ip 0.0.0.0 --port 8000


# Run Yolo Container
YOLO_HOME="$PWD"
docker run \
	-p 8000:8000 \
	-v $YOLO_HOME:/code  \
	-v $HOME/Desktop/data:/data \
	-v $YOLO_HOME/yolo_package:/yolo \
	-it ptyolo bash
