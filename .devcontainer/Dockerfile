FROM nvidia/cuda:9.0-base

ENV DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        software-properties-common \
        libreadline-gplv2-dev \
        libncursesw5-dev \ 
        libssl-dev \
        libsqlite3-dev \
        tk-dev \
        libgdbm-dev \
        libc6-dev \
        libbz2-dev \
        libffi-dev \
        zlib1g-dev \
        cmake \
        git \
        tar \
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
        vim

RUN wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
RUN tar -xf Python-3.7.5.tgz
RUN cd Python-3.7.5 && \
    ./configure && \
    make -j 8 && \
    make altinstall

RUN python3.7 -m pip install pylint cython numpy rope

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN python3.7 -m pip install -r requirements.txt

# Add to python path
ENV PYTHONPATH=/opt:/workspaces/liftr-critique-service:$PYTHONPATH

RUN useradd -m critique
USER critique

ENV HOME /home/critique
RUN echo 'alias python=python3.7' >> ~/.bashrc

ENV DEBIAN_FRONTEND=dialog