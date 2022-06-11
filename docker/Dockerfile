ARG IMAGE=pytorch/pytorch
ARG VERSION=latest
FROM $IMAGE:$VERSION

# Python 3 already good!

RUN apt upgrade \
	&& apt update -y \
	&& apt install -y \
		libglib2.0-0 \
		libgl1-mesa-glx

RUN pip install opencv-python

# Install other python deps
RUN pip install \
	pandas \
	matplotlib \
	seaborn

# Copy Code into container
# TODO

WORKDIR /code
