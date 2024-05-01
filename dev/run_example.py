import sys
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict
from matplotlib.pyplot import cm as colormap

src_dir = Path('/workspace/RegTR/src')
sys.path.append('/workspace/RegTR/src')

import cvhelpers.visualization as cvv
import cvhelpers.colors as colors
from cvhelpers.torch_helpers import to_numpy
from models.regtr import RegTR
from utils.misc import load_config
from utils.se3_numpy import se3_transform

def load_point_cloud(fname):
    if fname.endswith('.pth'):
        data = torch.load(fname)
    elif fname.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(fname)
        data = np.asarray(pcd.points)
    elif fname.endswith('.bin'):
        data = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
    else:
        raise AssertionError('Cannot recognize point cloud format')
    return data[:, :3]


ckpt_path = src_dir / '../logs/modelnet/240501_040054_regtr_regressCoor/ckpt/model-30672.pth'
src_path = src_dir / '../data/modelnet_demo_data/modelnet_test_2_0.ply'
tgt_path = src_dir / '../data/modelnet_demo_data/modelnet_test_2_1.ply'

cfg = EasyDict(load_config(Path(ckpt_path).parents[1] / 'config.yaml'))
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


cfg = EasyDict(load_config(Path(ckpt_path).parents[1] / 'config.yaml'))
model = RegTR(cfg).to(device)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state['state_dict'])

src_xyz = load_point_cloud(src_path.as_posix())
tgt_xyz = load_point_cloud(tgt_path.as_posix())

data_batch = dict(
    src_xyz = [torch.from_numpy(src_xyz).float().to(device)],
    tgt_xyz = [torch.from_numpy(tgt_xyz).float().to(device)],
)

outputs = model(data_batch)

b = 0
pose = to_numpy(outputs['pose'][-1, b])
src_kp = to_numpy(outputs['src_kp'][b])
src2tgt = to_numpy(outputs['src_kp_warped'][b][-1])
overlap_score = to_numpy(torch.sigmoid(outputs['src_overlap'][b][-1]))

args = [src_xyz, tgt_xyz, src_kp, src2tgt, overlap_score, pose]

with open('args.pkl', 'wb') as f:
    pickle.dump(args, f)
    print('Pickle saved')


