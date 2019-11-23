sudo apt-get update

sudo apt-get install -y \
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
    vim

pip install -r ../requirements.txt

cp lib/pafprocess /opt/pafprocess
cd /opt/pafprocess
swig -python -c++ pafprocess.i && python setup.py build_ext --inplace

export PYTHONPATH="/opt/:{$PYTHONPATH}"