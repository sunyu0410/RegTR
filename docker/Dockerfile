FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /data
COPY ./wheels ./

# Upgrade and install essentials
RUN apt update \
    && apt upgrade -y \
    && apt install -y libopenblas-dev nano unzip libxrender-dev 

# There's a step downstream that needs to uninstalled blinker
# However, it will fail unless it's reinstalled from source (not from .whl)
RUN wget https://files.pythonhosted.org/packages/1b/51/e2a9f3b757eb802f61dc1f2b09c8c99f6eb01cf06416c0671253536517b6/blinker-1.4.tar.gz && tar -xvzf blinker-1.4.tar.gz
WORKDIR /data/blinker-1.4
RUN python setup.py install
WORKDIR /data

# Install the pytorch3d, compiled MinkowskiEngine and other packages
RUN pip install pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl \
    && pip install MinkowskiEngine-0.5.4-cp310-cp310-linux_x86_64.whl \
    && pip install coloredlogs easydict h5py GitPython nibabel numpy scipy open3d tensorboard vtk tensorboard

CMD /bin/bash
