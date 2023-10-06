# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:22.11-py3
RUN rm -rf /opt/pytorch  # remove 1.2GB dir

# Install linux packages
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y libgl1-mesa-glx ffmpeg libsm6 libxext6 python3-tk

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip wheel
RUN pip install --no-cache -r requirements.txt

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
# COPY . /usr/src/app

# Set environment variables
ENV HOME=/usr/src/app
