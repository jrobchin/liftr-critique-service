FROM python:3.7

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

# Install numpy
RUN pip install numpy

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

# Install pafprocess
COPY lib/pafprocess /opt/pafprocess
RUN cd /opt/pafprocess && \
    swig -python -c++ pafprocess.i && python setup.py build_ext --inplace

ENV PYTHONPATH="/opt/:{$PYTHONPATH}"

RUN mkdir /src

RUN useradd -m critique && chown -R critique /src
RUN chown -R critique /home/critique
USER critique

# Copy source code
# COPY ./src /src
# WORKDIR /src

CMD [ "bash" ]