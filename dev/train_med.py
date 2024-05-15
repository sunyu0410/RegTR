import sys
sys.path.append('../src')

import os
from data_loaders import modelnet
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision.transforms import Compose
from easydict import EasyDict
import argparse

from cvhelpers.misc import prepare_logger
from data_loaders.collate_functions import collate_pair
from models import get_model
from trainer import Trainer
from utils.misc import load_config

parser = argparse.ArgumentParser()
# General
parser.add_argument('--config', type=str, help='Path to the config file.')
# Logging
parser.add_argument('--logdir', type=str, default='../logs',
                    help='Directory to store logs, summaries, checkpoints.')
parser.add_argument('--dev', action='store_true',
                    help='If true, will ignore logdir and log to ../logdev instead')
parser.add_argument('--name', type=str,
                    help='Experiment name (used to name output directory')
parser.add_argument('--summary_every', type=int, default=500,
                    help='Interval to save tensorboard summaries')
parser.add_argument('--validate_every', type=int, default=-1,
                    help='Validation interval. Default: every epoch')
parser.add_argument('--debug', action='store_true',
                    help='If set, will enable autograd anomaly detection')
# Misc
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of worker threads for dataloader')
# Training and model options
parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
parser.add_argument('--nb_sanity_val_steps', type=int, default=2,
                    help='Number of validation sanity steps to run before training.')

opt = parser.parse_args()
# Override config if --resume is passed
if opt.config is None:
    if opt.resume is None or not os.path.exists(opt.resume):
        print('--config needs to be supplied unless resuming from checkpoint')
        exit(-1)
    else:
        resume_folder = opt.resume if os.path.isdir(opt.resume) else os.path.dirname(opt.resume)
        opt.config = os.path.normpath(os.path.join(resume_folder, '../config.yaml'))
        if os.path.exists(opt.config):
            print(f'Using config file from checkpoint directory: {opt.config}')
        else:
            print('Config not found in resume directory')
            exit(-2)

cfg = EasyDict(load_config(opt.config))


# Hack: Stores different datasets to its own subdirectory
opt.logdir = os.path.join(opt.logdir, cfg.dataset)

if opt.name is None and len(cfg.get('expt_name', '')) > 0:
    opt.name = cfg.expt_name
logger, opt.log_path = prepare_logger(opt)

# Save config to log
config_out_fname = os.path.join(opt.log_path, 'config.yaml')
with open(opt.config, 'r') as in_fid, open(config_out_fname, 'w') as out_fid:
    out_fid.write(f'# Original file name: {opt.config}\n')
    out_fid.write(in_fid.read())

class Med3DDataset(Dataset):
    def __init__(self, filelist, transform=None):
        super().__init__()
        self.files = filelist
        self.transform = transform

    @staticmethod
    def norm(points):
        _min = points.min()
        _max = points.max()
        return 2 * (points-_min) / (_max-_min) - 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        points = np.load(self.files[idx])
        # Normalise to [-1, 1]
        points = self.norm(points)
        # Make it 6 channel (the same as RegTR)
        # Can modify the network to take 3 channels in the next step
        sample = dict(
            points=np.concatenate([points,points], 1),
            idx = np.array([idx])
        )
        if self.transform:
            sample = self.transform(sample)

        corr_xyz = np.concatenate([
            sample['points_src'][sample['correspondences'][0], :3],
            sample['points_ref'][sample['correspondences'][1], :3]], axis=1)

        # Transform to my format
        sample_out = {
            'src_xyz': torch.from_numpy(sample['points_src'][:, :3]).float(),
            'tgt_xyz': torch.from_numpy(sample['points_ref'][:, :3]).float(),
            'tgt_raw': torch.from_numpy(sample['points_raw'][:, :3]).float(),
            'src_overlap': torch.from_numpy(sample['src_overlap']),
            'tgt_overlap': torch.from_numpy(sample['ref_overlap']),
            'correspondences': torch.from_numpy(sample['correspondences']),
            'pose': torch.from_numpy(sample['transform_gt']).float(),
            'idx': torch.from_numpy(sample['idx']),
            'corr_xyz': torch.from_numpy(corr_xyz).float(),
        }

        return sample_out

if __name__ == "__main__":

    train_transform, val_transform = modelnet.get_transforms(
        'crop', 45.0, 0.5, 1024, [0.7, 0.7]
    )

    data_dir = Path('data/regtr_liver_npy')
    filelist = list(data_dir.iterdir())

    files_train = filelist[:800]
    files_val = filelist[800:]

    data_train = Med3DDataset(files_train, Compose(train_transform))
    data_val = Med3DDataset(files_val, Compose(val_transform))

    def get_dataloader(dataset, cfg, phase, num_workers=0):
        batch_size = cfg[f'{phase}_batch_size']
        shuffle = phase == 'train'

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_pair,
        )

        return data_loader
    
    train_loader = get_dataloader(data_train, cfg, phase='train', num_workers=opt.num_workers)
    val_loader = get_dataloader(data_val, cfg, phase='val', num_workers=opt.num_workers)

    Model = get_model(cfg.model)
    model = Model(cfg)

    trainer = Trainer(opt, niter=cfg.niter, grad_clip=cfg.grad_clip)
    trainer.fit(model, train_loader, val_loader)
    # trainer.fit(model, train_loader)
