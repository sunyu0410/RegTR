apt update
apt -y upgrade
apt install -y libopenblas-dev nano unzip libxrender-dev

pip install wheels/MinkowskiEngine-0.5.4-cp310-cp310-linux_x86_64.whl

cd blinker-1.4
python setup.py install # pip install wheels/blinker-1.4-py3-none-any.whl
cd ..

pip install wheels/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl
pip install coloredlogs easydict h5py GitPython nibabel numpy scipy open3d tensorboard vtk tensorboard

git config --global user.email "sunyu0410@gmail.com"
git config --global user.name "Yu Sun"

