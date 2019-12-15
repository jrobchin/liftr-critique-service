FROM python:3.7

ENV DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        swig \
        ffmpeg \
        libavcodec-extra \
        vim && \
    rm -r /var/lib/apt/lists/*

RUN pip3 install pylint

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

# Install pafprocess
COPY lib/pafprocess /opt/pafprocess
RUN cd /opt/pafprocess && \
    swig -python -c++ pafprocess.i && python setup.py build_ext --inplace

# Add to python path
ENV PYTHONPATH=/opt:/workspaces/liftr-critique-service:$PYTHONPATH

RUN useradd -m critique
USER critique

ENV HOME /home/critique

ENV DEBIAN_FRONTEND=dialog