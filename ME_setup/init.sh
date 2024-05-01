apt update
apt -y upgrade
apt install libopenblas-dev nano unzip libxrender-dev

pip install wheels/MinkowskiEngine-0.5.4-cp310-cp310-linux_x86_64.whl
python blinker-1.4/setup.py install # pip install wheels/blinker-1.4-py3-none-any.whl
pip install wheels/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl

pip install coloredlogs easydict h5py GitPython nibabel numpy scipy open3d tensorboard vtk tensorboard
