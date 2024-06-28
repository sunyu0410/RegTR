import logging
from typing import Tuple

import torch.nn
from torch.utils.tensorboard import SummaryWriter

import os
from abc import ABC

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cvhelpers.torch_helpers import to_numpy
from models.scheduler.warmup import WarmUpScheduler
from benchmark.benchmark_predator import benchmark as benchmark_predator
import benchmark.benchmark_modelnet as benchmark_modelnet
from utils.misc import StatsMeter, metrics_to_string
from utils.se3_torch import se3_compare

import math

import torch.nn as nn

from models.backbone_kpconv.kpconv import KPFEncoder, PreprocessorGPU, compute_overlaps
from models.losses.corr_loss import CorrCriterion
from models.losses.feature_loss import InfoNCELossFull, CircleLossFull
from models.transformer.position_embedding import PositionEmbeddingCoordsSine, \
    PositionEmbeddingLearned
from models.transformer.transformers import \
    TransformerCrossEncoderLayer, TransformerCrossEncoder
from utils.se3_torch import compute_rigid_transform, se3_transform_list, se3_inv
from utils.seq_manipulation import split_src_tgt, pad_sequence, unpad_sequences
from utils.viz import visualize_registration
_TIMEIT = False


class RegTR(torch.nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.device = None
        self.logger = logging.getLogger(self.__class__.__name__)

        self.optimizer = None
        self.scheduler = None
        self.optimizer_handled_by_trainer = True
        self._trainer = None
        self.logger.info(f'Instantiating model {self.__class__.__name__}')

        self.loss_stats_meter = StatsMeter()  # For accumulating losses
        self.reg_success_thresh_rot = cfg.reg_success_thresh_rot
        self.reg_success_thresh_trans = cfg.reg_success_thresh_trans

        #######################
        # Preprocessor
        #######################
        self.preprocessor = PreprocessorGPU(cfg)

        #######################
        # KPConv Encoder/decoder
        #######################
        self.kpf_encoder = KPFEncoder(cfg, cfg.d_embed)
        # Bottleneck layer to shrink KPConv features to a smaller dimension for running attention
        self.feat_proj = nn.Linear(self.kpf_encoder.encoder_skip_dims[-1], cfg.d_embed, bias=True)

        #######################
        # Embeddings
        #######################
        if cfg.get('pos_emb_type', 'sine') == 'sine':
            self.pos_embed = PositionEmbeddingCoordsSine(3, cfg.d_embed,
                                                         scale=cfg.get('pos_emb_scaling', 1.0))
        elif cfg['pos_emb_type'] == 'learned':
            self.pos_embed = PositionEmbeddingLearned(3, cfg.d_embed)
        else:
            raise NotImplementedError

        #######################
        # Attention propagation
        #######################
        encoder_layer = TransformerCrossEncoderLayer(
            cfg.d_embed, cfg.nhead, cfg.d_feedforward, cfg.dropout,
            activation=cfg.transformer_act,
            normalize_before=cfg.pre_norm,
            sa_val_has_pos_emb=cfg.sa_val_has_pos_emb,
            ca_val_has_pos_emb=cfg.ca_val_has_pos_emb,
            attention_type=cfg.attention_type,
        )
        encoder_norm = nn.LayerNorm(cfg.d_embed) if cfg.pre_norm else None
        self.transformer_encoder = TransformerCrossEncoder(
            encoder_layer, cfg.num_encoder_layers, encoder_norm,
            return_intermediate=True)

        #######################
        # Output layers
        #######################
        if cfg.get('direct_regress_coor', False):
            self.correspondence_decoder = CorrespondenceRegressor(cfg.d_embed)
        else:
            self.correspondence_decoder = CorrespondenceDecoder(cfg.d_embed,
                                                                cfg.corr_decoder_has_pos_emb,
                                                                self.pos_embed)

        #######################
        # Losses
        #######################
        self.overlap_criterion = nn.BCEWithLogitsLoss()
        if self.cfg.feature_loss_type == 'infonce':
            self.feature_criterion = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
        elif self.cfg.feature_loss_type == 'circle':
            self.feature_criterion = CircleLossFull(dist_type='euclidean', r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = self.feature_criterion
        else:
            raise NotImplementedError

        self.corr_criterion = CorrCriterion(metric='mae')

        self.weight_dict = {}
        for k in ['overlap', 'feature', 'corr']:
            for i in cfg.get(f'{k}_loss_on', [cfg.num_encoder_layers - 1]):
                self.weight_dict[f'{k}_{i}'] = cfg.get(f'wt_{k}')
        self.weight_dict['feature_un'] = cfg.wt_feature_un

        self.logger.info('Loss weighting: {}'.format(self.weight_dict))
        self.logger.info(
            f'Config: d_embed:{cfg.d_embed}, nheads:{cfg.nhead}, pre_norm:{cfg.pre_norm}, '
            f'use_pos_emb:{cfg.transformer_encoder_has_pos_emb}, '
            f'sa_val_has_pos_emb:{cfg.sa_val_has_pos_emb}, '
            f'ca_val_has_pos_emb:{cfg.ca_val_has_pos_emb}'
        )

    def set_trainer(self, trainer):
        self._trainer = trainer

    def get_trainer(self):
        """Returns the trainer instance"""
        return self._trainer

    def train_epoch_start(self):
        pass

    def training_step(self, batch, batch_idx):
        """Training step.

        Returns:
            losses(Dict): Which should be a python dictionary and should have at
              least one term 'total' for the total loss
        """
        pred = self.forward(batch)
        losses = self.compute_loss(pred, batch)

        # Stores the losses for summary writing
        for k in losses:
            self.loss_stats_meter[k].update(losses[k])

        # visualize_registration(batch, pred)
        return pred, losses

    def train_epoch_end(self):
        pass

    def train_summary_fn(self, writer: SummaryWriter, step: int,
                         data_batch, train_output, train_losses):

        losses_dict = {k: self.loss_stats_meter[k].avg for k in self.loss_stats_meter}
        self._generic_summary_function(writer, step, model=self, losses=losses_dict)
        self.loss_stats_meter.clear()

    def validation_epoch_start(self):
        pass

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        losses = self.compute_loss(pred, batch)
        metrics = self._compute_metrics(pred, batch)

        # visualize_registration(batch, pred, metrics=metrics, iter_idx=5, b=2)

        val_outputs = (losses, metrics)

        return val_outputs

    def validation_epoch_end(self, validation_step_outputs) -> Tuple[float, dict]:
        pass

    def validation_summary_fn(self, writer: SummaryWriter, step: int, val_outputs):
        """Logs data during validation. This function will be called after every
        validation run.
        The default implementation saves out the scalars from losses and metrics.

        Args:
            writer: validation writer
            step: The current step number
            val_outputs: Whatever that is returned from validation_epoch_end()

        """
        super().validation_summary_fn(writer, step, val_outputs)

        # Save histogram summaries
        metrics = val_outputs['metrics']
        for k in metrics:
            if k.endswith('hist'):
                writer.add_histogram(f'metrics/{k}', metrics[k], step)

    def validation_epoch_end(self, validation_step_outputs):

        losses = [v[0] for v in validation_step_outputs]
        metrics = [v[1] for v in validation_step_outputs]

        loss_keys = set(losses[0].keys())
        losses_stacked = {k: torch.stack([l[k] for l in losses]) for k in loss_keys}

        # Computes the mean over all metrics
        avg_losses = {k: torch.mean(losses_stacked[k]) for k in loss_keys}
        avg_metrics = self._aggregate_metrics(metrics)
        return avg_metrics['reg_success_final'].item(), {'losses': avg_losses, 'metrics': avg_metrics}

    def test_epoch_start(self):
        if self.cfg.dataset == 'modelnet':
            self.modelnet_metrics = []
            self.modelnet_poses = []

    def test_step(self, batch, batch_idx):

        pred = self.forward(batch)
        losses = self.compute_loss(pred, batch)
        metrics = self._compute_metrics(pred, batch)

        # Dataset specific handling
        if self.cfg.dataset == '3dmatch':
            self._save_3DMatch_log(batch, pred)

        elif self.cfg.dataset == 'modelnet':
            modelnet_data = {
                'points_src': torch.stack(batch['src_xyz']),
                'points_ref': torch.stack(batch['tgt_xyz']),
                'points_raw': torch.stack(batch['tgt_raw']),
                'transform_gt': batch['pose'],
            }
            self.modelnet_metrics.append(
                benchmark_modelnet.compute_metrics(modelnet_data, pred['pose'][-1])
            )
            self.modelnet_poses.append(
                pred['pose'][-1]
            )

        else:
            raise NotImplementedError

        test_outputs = (losses, metrics)
        return test_outputs


    def test_epoch_end(self, test_step_outputs):

        losses = [v[0] for v in test_step_outputs]
        metrics = [v[1] for v in test_step_outputs]

        loss_keys = losses[0].keys()
        losses = {k: torch.stack([l[k] for l in losses]) for k in loss_keys}

        # Computes the mean over all metrics
        avg_losses = {k: torch.mean(losses[k]) for k in loss_keys}
        avg_metrics = self._aggregate_metrics(metrics)

        log_str = 'Test ended:\n'
        log_str += metrics_to_string(avg_losses, '[Losses]') + '\n'
        log_str += metrics_to_string(avg_metrics, '[Metrics]') + '\n'
        self.logger.info(log_str)

        if self.cfg.dataset == '3dmatch':
            # Evaluate 3DMatch registration recall
            results_str, mean_precision = benchmark_predator(
                os.path.join(self._log_path, self.cfg.benchmark),
                os.path.join('datasets', '3dmatch', 'benchmarks', self.cfg.benchmark))
            self.logger.info('\n' + results_str)
            return mean_precision

        elif self.cfg.dataset == 'modelnet':
            metric_keys = self.modelnet_metrics[0].keys()
            metrics_cat = {k: np.concatenate([m[k] for m in self.modelnet_metrics])
                           for k in metric_keys}
            summary_metrics = benchmark_modelnet.summarize_metrics(metrics_cat)
            benchmark_modelnet.print_metrics(self.logger, summary_metrics)

            # Also save out the predicted poses, which can be evaluated using
            # RPMNet's eval.py
            poses_to_save = to_numpy(torch.stack(self.modelnet_poses, dim=0))
            np.save(os.path.join(self._log_path, 'pred_transforms.npy'), poses_to_save)

    def configure_optimizers(self):
        """Sets and returns the optimizers. Default implementation does nothing.
        """
        scheduler_type = self.cfg.get('scheduler', None)
        if scheduler_type is None or scheduler_type in ['none', 'step']:
            base_lr = self.cfg.base_lr
        elif scheduler_type == 'warmup':
            base_lr = 0.0  # start from 0
        else:
            raise NotImplementedError

        # Create optimizer
        if self.cfg.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=base_lr,
                                               weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=base_lr,
                                              weight_decay=self.cfg.weight_decay)
        else:
            raise NotImplementedError

        # Create scheduler
        if scheduler_type == 'warmup':
            # Warmup, then smooth exponential decay
            self.scheduler = WarmUpScheduler(self.optimizer, self.cfg.scheduler_param, self.cfg.base_lr)
        elif scheduler_type == 'step':
            # Step decay
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.cfg.scheduler_param[0],
                                                             self.cfg.scheduler_param[1])
        elif scheduler_type == 'none' or scheduler_type is None:
            # No decay
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 50, 1.0)
        else:
            raise AssertionError('Invalid scheduler')

        self.logger.info(f'Using optimizer {self.optimizer} with scheduler {self.scheduler}')

    def to(self, *args, **kwargs):
        """Sends the model to the specified device. Also sets self.device
        so that it can be accessed by code within the model.
        """
        super().to(*args, **kwargs)

        # Keep track of device in an easily accessible place
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = args[0]
        return self

    def _generic_summary_function(self, writer: SummaryWriter, step: int, **kwargs):
        # Generic summary function that saves losses and metrics as tensorboard
        # summary scalars.
        losses = kwargs.get('losses', None)
        if losses is not None:
            for k in losses:
                if isinstance(losses[k], torch.Tensor) and losses[k].ndim > 0:
                    continue
                writer.add_scalar('losses/{}'.format(k), losses[k], step)

        metrics = kwargs.get('metrics', None)
        if metrics is not None:
            for k in metrics:
                if isinstance(metrics[k], torch.Tensor) and metrics[k].ndim > 0:
                    continue
                writer.add_scalar('metrics/{}'.format(k), metrics[k], step)

        if self.scheduler is not None:
            writer.add_scalar('lr', self.scheduler.get_last_lr()[0], step)

    def _compute_metrics(self, pred, batch):
        metrics = {}
        with torch.no_grad():

            pose_keys = [k for k in pred.keys() if k.startswith('pose')]
            for k in pose_keys:
                suffix = k[4:]
                pose_err = se3_compare(pred[k], batch['pose'][None, :])
                metrics[f'rot_err_deg{suffix}'] = pose_err['rot_deg']
                metrics[f'trans_err{suffix}'] = pose_err['trans']

        return metrics
    
    def _aggregate_metrics(self, metrics):

        if len(metrics[0]) == 0:
            return {}

        batch_dim = 1  # dim=1 is batch dimension (0 is decoder layer)
        metrics_keys = set(metrics[0].keys())
        metrics_cat = {k: torch.cat([m[k] for m in metrics], dim=batch_dim) for k in metrics_keys}
        num_instances = next(iter(metrics_cat.values())).shape[batch_dim]
        self.logger.info(f'Aggregating metrics, total number of instances: {num_instances}')
        assert all([metrics_cat[k].shape[batch_dim] == num_instances for k in metrics_keys]), \
            'Dimensionality incorrect, check whether batch dimension is consistent'

        rot_err_keys = [k for k in metrics_cat.keys() if k.startswith('rot_err_deg')]
        if len(rot_err_keys) > 0:
            num_pred = metrics_cat[rot_err_keys[0]].shape[0]

        avg_metrics = {}
        for p in range(num_pred):
            suffix = f'{p}' if p < num_pred - 1 else 'final'

            for rk in rot_err_keys:
                pose_type_suffix = rk[11:]

                avg_metrics[f'rot_err_deg{pose_type_suffix}_{suffix}'] = torch.mean(metrics_cat[rk][p])
                avg_metrics[f'rot_err{pose_type_suffix}_{suffix}_hist'] = metrics_cat[rk][p]

                tk = 'trans_err' + pose_type_suffix
                avg_metrics[f'{tk}_{suffix}'] = torch.mean(metrics_cat[tk][p])
                avg_metrics[f'{tk}_{suffix}_hist'] = metrics_cat[tk][p]

                reg_success = torch.logical_and(metrics_cat[rk][p, :] < self.reg_success_thresh_rot,
                                                metrics_cat[tk][p, :] < self.reg_success_thresh_trans)
                avg_metrics[f'reg_success{pose_type_suffix}_{suffix}'] = reg_success.float().mean()

            if 'corr_err' in metrics_cat:
                avg_metrics[f'corr_err_{suffix}_hist'] = metrics_cat['corr_err'][p].flatten()
                avg_metrics[f'corr_err_{suffix}'] = torch.mean(metrics_cat['corr_err'][p])

        return avg_metrics

    @property
    def _log_path(self):
        return self.get_trainer().log_path

    """
    Dataset specific functions
    """
    def _save_3DMatch_log(self, batch, pred):
        B = len(batch['src_xyz'])

        for b in range(B):
            scene = batch['src_path'][b].split(os.path.sep)[1]
            src_idx = int(os.path.basename(batch['src_path'][b]).split('_')[-1].replace('.pth', ''))
            tgt_idx = int(os.path.basename(batch['tgt_path'][b]).split('_')[-1].replace('.pth', ''))

            pred_pose_np = to_numpy(pred['pose'][-1][b]) if pred['pose'].ndim == 4 else \
                to_numpy(pred['pose'][b])
            if pred_pose_np.shape[0] == 3:
                pred_pose_np = np.concatenate([pred_pose_np, [[0., 0., 0., 1.]]], axis=0)

            scene_folder = os.path.join(self._log_path, self.cfg.benchmark, scene)
            os.makedirs(scene_folder, exist_ok=True)
            est_log_path = os.path.join(scene_folder, 'est.log')
            with open(est_log_path, 'a') as fid:
                # We don't know the number of frames, so just put -1
                # This will be ignored by the benchmark function in any case
                fid.write('{}\t{}\t{}\n'.format(tgt_idx, src_idx, -1))
                for i in range(4):
                    fid.write('\t'.join(map('{0:.12f}'.format, pred_pose_np[i])) + '\n')

    def forward(self, batch):
        B = len(batch['src_xyz'])
        outputs = {}

        if _TIMEIT:
            t_start_all_cuda, t_end_all_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_pp_cuda, t_end_pp_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_all_cuda.record()
            t_start_pp_cuda.record()

        # Preprocess
        kpconv_meta = self.preprocessor(batch['src_xyz'] + batch['tgt_xyz'])
        batch['kpconv_meta'] = kpconv_meta
        slens = [s.tolist() for s in kpconv_meta['stack_lengths']]
        slens_c = slens[-1]
        src_slens_c, tgt_slens_c = slens_c[:B], slens_c[B:]
        feats0 = torch.ones_like(kpconv_meta['points'][0][:, 0:1])

        if _TIMEIT:
            t_end_pp_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_pp_cuda = t_start_pp_cuda.elapsed_time(t_end_pp_cuda) / 1000
            t_start_enc_cuda, t_end_enc_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_enc_cuda.record()

        ####################
        # REGTR Encoder
        ####################
        # KPConv encoder (downsampling) to obtain unconditioned features
        feats_un, skip_x = self.kpf_encoder(feats0, kpconv_meta)
        if _TIMEIT:
            t_end_enc_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_enc_cuda = t_start_enc_cuda.elapsed_time(t_end_enc_cuda) / 1000
            t_start_att_cuda, t_end_att_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_att_cuda.record()

        both_feats_un = self.feat_proj(feats_un)
        src_feats_un, tgt_feats_un = split_src_tgt(both_feats_un, slens_c)

        # Position embedding for downsampled points
        src_xyz_c, tgt_xyz_c = split_src_tgt(kpconv_meta['points'][-1], slens_c)
        src_pe, tgt_pe = split_src_tgt(self.pos_embed(kpconv_meta['points'][-1]), slens_c)
        src_pe_padded, _, _ = pad_sequence(src_pe)
        tgt_pe_padded, _, _ = pad_sequence(tgt_pe)

        # Performs padding, then apply attention (REGTR "encoder" stage) to condition on the other
        # point cloud
        src_feats_padded, src_key_padding_mask, _ = pad_sequence(src_feats_un,
                                                                 require_padding_mask=True)
        tgt_feats_padded, tgt_key_padding_mask, _ = pad_sequence(tgt_feats_un,
                                                                 require_padding_mask=True)
        src_feats_cond, tgt_feats_cond = self.transformer_encoder(
            src_feats_padded, tgt_feats_padded,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_pos=src_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
            tgt_pos=tgt_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
        )

        src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list = \
            self.correspondence_decoder(src_feats_cond, tgt_feats_cond, src_xyz_c, tgt_xyz_c)

        src_feats_list = unpad_sequences(src_feats_cond, src_slens_c)
        tgt_feats_list = unpad_sequences(tgt_feats_cond, tgt_slens_c)
        num_pred = src_feats_cond.shape[0]

        ## TIMING CODE
        if _TIMEIT:
            t_end_att_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_att_cuda = t_start_att_cuda.elapsed_time(t_end_att_cuda) / 1000
            t_start_pose_cuda, t_end_pose_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_pose_cuda.record()

        # Stacks correspondences in both directions and computes the pose
        corr_all, overlap_prob = [], []
        for b in range(B):
            corr_all.append(torch.cat([
                torch.cat([src_xyz_c[b].expand(num_pred, -1, -1), src_corr_list[b]], dim=2),
                torch.cat([tgt_corr_list[b], tgt_xyz_c[b].expand(num_pred, -1, -1)], dim=2)
            ], dim=1))
            overlap_prob.append(torch.cat([
                torch.sigmoid(src_overlap_list[b][:, :, 0]),
                torch.sigmoid(tgt_overlap_list[b][:, :, 0]),
            ], dim=1))

            # # Thresholds the overlap probability. Enable this for inference to get a slight boost
            # # in performance. However, we do not use this in the paper.
            # overlap_prob = [nn.functional.threshold(overlap_prob[b], 0.5, 0.0) for b in range(B)]

        pred_pose_weighted = torch.stack([
            compute_rigid_transform(corr_all[b][..., :3], corr_all[b][..., 3:],
                                    overlap_prob[b])
            for b in range(B)], dim=1)

        ## TIMING CODE
        if _TIMEIT:
            t_end_pose_cuda.record()
            t_end_all_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_pose_cuda = t_start_pose_cuda.elapsed_time(t_end_pose_cuda) / 1000
            t_elapsed_all_cuda = t_start_all_cuda.elapsed_time(t_end_all_cuda) / 1000
            with open('timings.txt', 'a') as fid:
                fid.write('{:10f}\t{:10f}\t{:10f}\t{:10f}\t{:10f}\n'.format(
                    t_elapsed_pp_cuda, t_elapsed_enc_cuda, t_elapsed_att_cuda,
                    t_elapsed_pose_cuda, t_elapsed_all_cuda
                ))

        outputs = {
            # Predictions
            'src_feat_un': src_feats_un,
            'tgt_feat_un': tgt_feats_un,
            'src_feat': src_feats_list,  # List(B) of (N_pred, N_src, D)
            'tgt_feat': tgt_feats_list,  # List(B) of (N_pred, N_tgt, D)

            'src_kp': src_xyz_c,
            'src_kp_warped': src_corr_list,
            'tgt_kp': tgt_xyz_c,
            'tgt_kp_warped': tgt_corr_list,

            'src_overlap': src_overlap_list,
            'tgt_overlap': tgt_overlap_list,

            'pose': pred_pose_weighted,
        }
        return outputs
    
    def compute_loss(self, pred, batch):
        losses = {}
        kpconv_meta = batch['kpconv_meta']
        pose_gt = batch['pose']
        p = len(kpconv_meta['stack_lengths']) - 1  # coarsest level

        # Compute groundtruth overlaps first
        batch['overlap_pyr'] = compute_overlaps(batch)
        src_overlap_p, tgt_overlap_p = \
            split_src_tgt(batch['overlap_pyr'][f'pyr_{p}'], kpconv_meta['stack_lengths'][p])

        # Overlap prediction loss
        all_overlap_pred = torch.cat(pred['src_overlap'] + pred['tgt_overlap'], dim=-2)
        all_overlap_gt = batch['overlap_pyr'][f'pyr_{p}']
        for i in self.cfg.overlap_loss_on:
            losses[f'overlap_{i}'] = self.overlap_criterion(all_overlap_pred[i, :, 0], all_overlap_gt)

        # Feature criterion
        for i in self.cfg.feature_loss_on:
            losses[f'feature_{i}'] = self.feature_criterion(
                [s[i] for s in pred['src_feat']],
                [t[i] for t in pred['tgt_feat']],
                se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
            )
        losses['feature_un'] = self.feature_criterion_un(
            pred['src_feat_un'],
            pred['tgt_feat_un'],
            se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
        )

        # Loss on the 6D correspondences
        for i in self.cfg.corr_loss_on:
            src_corr_loss = self.corr_criterion(
                pred['src_kp'],
                [w[i] for w in pred['src_kp_warped']],
                batch['pose'],
                overlap_weights=src_overlap_p
            )
            tgt_corr_loss = self.corr_criterion(
                pred['tgt_kp'],
                [w[i] for w in pred['tgt_kp_warped']],
                torch.stack([se3_inv(p) for p in batch['pose']]),
                overlap_weights=tgt_overlap_p
            )
            losses[f'corr_{i}'] = src_corr_loss + tgt_corr_loss

        debug = False  # Set this to true to look at the registration result
        if debug:
            b = 0
            o = -1  # Visualize output of final transformer layer
            visualize_registration(batch['src_xyz'][b], batch['tgt_xyz'][b],
                                   torch.cat([pred['src_kp'][b], pred['src_kp_warped'][b][o]], dim=1),
                                   correspondence_conf=torch.sigmoid(pred['src_overlap'][b][o])[:, 0],
                                   pose_gt=pose_gt[b], pose_pred=pred['pose'][o, b])

        losses['total'] = torch.sum(
            torch.stack([(losses[k] * self.weight_dict[k]) for k in losses]))
        return losses
    
class CorrespondenceDecoder(nn.Module):
    def __init__(self, d_embed, use_pos_emb, pos_embed=None, num_neighbors=0):
        super().__init__()

        assert use_pos_emb is False or pos_embed is not None, \
            'Position encoder must be supplied if use_pos_emb is True'

        self.use_pos_emb = use_pos_emb
        self.pos_embed = pos_embed
        self.q_norm = nn.LayerNorm(d_embed)

        self.q_proj = nn.Linear(d_embed, d_embed)
        self.k_proj = nn.Linear(d_embed, d_embed)
        self.conf_logits_decoder = nn.Linear(d_embed, 1)
        self.num_neighbors = num_neighbors

        # nn.init.xavier_uniform_(self.q_proj.weight)
        # nn.init.xavier_uniform_(self.k_proj.weight)

    def simple_attention(self, query, key, value, key_padding_mask=None):
        """Simplified single-head attention that does not project the value:
        Linearly projects only the query and key, compute softmax dot product
        attention, then returns the weighted sum of the values

        Args:
            query: ([N_pred,] Q, B, D)
            key: ([N_pred,] S, B, D)
            value: (S, B, E), i.e. dimensionality can be different
            key_padding_mask: (B, S)

        Returns:
            Weighted values (B, Q, E)
        """

        q = self.q_proj(query) / math.sqrt(query.shape[-1])
        k = self.k_proj(key)

        attn = torch.einsum('...qbd,...sbd->...bqs', q, k)  # (B, N_query, N_src)

        if key_padding_mask is not None:
            attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(key_padding_mask, float('-inf'))
            attn = attn + attn_mask[:, None, :]  # ([N_pred,] B, Q, S)

        if self.num_neighbors > 0:
            neighbor_mask = torch.full_like(attn, fill_value=float('-inf'))
            haha = torch.topk(attn, k=self.num_neighbors, dim=-1).indices
            neighbor_mask[:, :, haha] = 0
            attn = attn + neighbor_mask

        attn = torch.softmax(attn, dim=-1)

        attn_out = torch.einsum('...bqs,...sbd->...qbd', attn, value)

        return attn_out

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3)
            tgt_xyz: List of ([N_pred,] N_tgt, 3)

        Returns:

        """

        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)
        assert src_xyz_padded.shape[:-1] == src_feats_padded.shape[-3:-1] and \
               tgt_xyz_padded.shape[:-1] == tgt_feats_padded.shape[-3:-1]

        if self.use_pos_emb:
            both_xyz_packed = torch.cat(src_xyz + tgt_xyz)
            slens = list(map(len, src_xyz)) + list(map(len, tgt_xyz))
            src_pe, tgt_pe = split_src_tgt(self.pos_embed(both_xyz_packed), slens)
            src_pe_padded, _, _ = pad_sequence(src_pe)
            tgt_pe_padded, _, _ = pad_sequence(tgt_pe)

        # Decode the coordinates
        src_feats2 = src_feats_padded + src_pe_padded if self.use_pos_emb else src_feats_padded
        tgt_feats2 = tgt_feats_padded + tgt_pe_padded if self.use_pos_emb else tgt_feats_padded
        src_corr = self.simple_attention(src_feats2, tgt_feats2, pad_sequence(tgt_xyz)[0],
                                         tgt_key_padding_mask)
        tgt_corr = self.simple_attention(tgt_feats2, src_feats2, pad_sequence(src_xyz)[0],
                                         src_key_padding_mask)

        src_overlap = self.conf_logits_decoder(src_feats_padded)
        tgt_overlap = self.conf_logits_decoder(tgt_feats_padded)

        src_corr_list = unpad_sequences(src_corr, src_lens)
        tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)
        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)

        return src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list


class CorrespondenceRegressor(nn.Module):

    def __init__(self, d_embed):
        super().__init__()

        self.coor_mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 3)
        )
        self.conf_logits_decoder = nn.Linear(d_embed, 1)

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3). Ignored
            tgt_xyz: List of ([N_pred,] N_tgt, 3). Ignored

        Returns:

        """

        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)

        # Decode the coordinates
        src_corr = self.coor_mlp(src_feats_padded)
        tgt_corr = self.coor_mlp(tgt_feats_padded)

        src_overlap = self.conf_logits_decoder(src_feats_padded)
        tgt_overlap = self.conf_logits_decoder(tgt_feats_padded)

        src_corr_list = unpad_sequences(src_corr, src_lens)
        tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)
        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)

        return src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list