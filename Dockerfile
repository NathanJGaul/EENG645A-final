FROM tensorflow/tensorflow:2.3.0-gpu as develop

# Install system libraries for python packages
RUN apt-get update &&  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        graphviz \
        python3-tk \
        # required for cv2
        libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install other python library requirements
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt