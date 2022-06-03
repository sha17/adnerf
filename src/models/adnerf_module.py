import os
import numpy as np
import imageio
import cv2
from typing import Any, List
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid
from torchmetrics import MaxMetric
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from pytorch_lightning import LightningModule
import pytorch_lightning.utilities.distributed as dist

from src.models.components.face_nerf import FaceNeRF
from src.models.components.pos_encoder import PositionEncoder
from src.models.components.audio_net import AudioNet
from src.models.components.audio_attn_net import AudioAttnNet

from src.utils.utils import (raw2outputs, sample_pdf,
                             update_n_append_dict, concat_all_items_in_dict,
                             sort_preds, log_video, log_image)


class AdNeRFLitModule(LightningModule):

    def __init__(
        self,
        pos_encoder: DictConfig,
        pos_encoder_view_dirs: DictConfig, 
        nerf_coarse: DictConfig,
        nerf_fine: DictConfig,
        audio_net: DictConfig,
        audio_attn_net: DictConfig,
        optim_hparams: DictConfig,
        render_hparams: DictConfig
    ):
        """
        Args:
            pos_encoder:
            pos_encoder_view_dirs:
            nerf_coarse:
            nerf_fine:
            audio_net:
            audio_attn_net:
            optim_hparams:
            render_hparams:
        """

        super().__init__()

        self.save_hyperparameters(logger=False)

        # Create NeRF
        pos_encoder = PositionEncoder(**pos_encoder)
        pos_encoder_view_dirs = PositionEncoder(**pos_encoder_view_dirs)

        audio_net = AudioNet(**audio_net)
        audio_attn_net = AudioAttnNet(**audio_attn_net)
        
        nerf_coarse['input_ch'] = pos_encoder.get_output_dim()
        nerf_coarse['input_ch_views'] = pos_encoder_view_dirs.get_output_dim()
        nerf_coarse['dim_aud'] = audio_net.dim_aud
        nerf_coarse['output_ch'] = 5 if render_hparams.n_samples_fine > 0 else 4
        nerf_coarse = FaceNeRF(**nerf_coarse)

        nerf_fine['input_ch'] = pos_encoder.get_output_dim()
        nerf_fine['input_ch_views'] = pos_encoder_view_dirs.get_output_dim()
        nerf_fine['dim_aud'] = audio_net.dim_aud
        nerf_fine['output_ch'] = 5 if render_hparams.n_samples_fine > 0 else 4
        nerf_fine = FaceNeRF(**nerf_fine)

        self.pos_encoder = pos_encoder
        self.pos_encoder_view_dirs = pos_encoder_view_dirs
        self.nerf_coarse = nerf_coarse
        self.nerf_fine = nerf_fine
        self.audio_net = audio_net
        self.audio_attn_net = audio_attn_net
        self.render_hparams = render_hparams
        self.optim_hparams = optim_hparams
        self.criterion = nn.MSELoss()
        self.train_psnr = PeakSignalNoiseRatio()
        self.val_psnr = PeakSignalNoiseRatio()
        self.test_psnr = PeakSignalNoiseRatio()
        self.val_psnr_best = MaxMetric()

    def forward(self, 
                rays_o, 
                rays_d, 
                z_vals, 
                view_dirs, 
                auds, 
                bg_imgs, 
                nerf, 
                mode, 
                output_raw_noise_std=0, 
                chunk_sz:int=1024*64):
        """
        Args:
            rays_o: 
            rays_d:
            z_vals: 
            view_dirs:
            auds: 
            bg_imgs: 
            nerf: 
            mode: 
            output_raw_noise_std:, 
            chunk_sz:
        Returns:
            outputs:
        """
        # [B, num_rays, num_samples]
        pnts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  
        B, R, P = pnts.shape[:-1]

        view_dirs = view_dirs.unsqueeze(2).expand(-1, R, P, -1)  # [B, num_rays, num_samples, 3]
        auds = auds.unsqueeze(2).expand(-1, R, P, -1)  # [B, num_rays, num_samples, 64]
        
        pnts = self.pos_encoder(torch.flatten(pnts, end_dim=2)) # [BxRxP, 3] 
        auds = torch.flatten(auds, end_dim=2)                   # [BxRxP, aud_dim]
        view_dirs = self.pos_encoder_view_dirs(torch.flatten(view_dirs, end_dim=2)) # [BxRxP, 3] 
        inputs = torch.cat([pnts, auds, view_dirs], -1).float()

        raw = []
        for chunk_idx in range(0, inputs.shape[0], chunk_sz):
            chunk = inputs[chunk_idx:chunk_idx+chunk_sz]
            raw.append(nerf(chunk))
        raw = torch.cat(raw).view(B, R, P, -1)
        
        rgb_map, disp_map, acc_map, weights, depth_map =\
            raw2outputs(   
                raw, z_vals, rays_d, bg_imgs,
                output_raw_noise_std,
                self.render_hparams.white_background)

        keys = ['raw', 'rgb_map', 'disp_map', 'acc_map', 'weight', 'depth_map']
        outputs = dict(zip(
            [k+f'_{mode}' for k in keys],
            [raw, rgb_map, disp_map, acc_map, weights, depth_map]))
        return outputs
        # return raw, rgb_map, disp_map, acc_map, weights, depth_map

    def step(self,
             batch,
             batch_idx, 
             chunk_sz=1024, 
             jitter=False, 
             output_raw_noise_std=0, 
             output_to_cpu=False,
             audio_smoothing=False):
        """
        Args:
            batch:
            batch_idx:
            chunk_sz:
            jitter:
            output_raw_noise_std:
            output_to_cpu:
            audio_smoothing:
        """
        B, R = batch[0].shape[0:2]
        rays_o, rays_d, view_dirs, bg_imgs, auds = batch

        if audio_smoothing:
            # [B, W, 16, 29] -> [BxW, 16, 29] 
            auds = self.audio_net(torch.flatten(auds,end_dim=1)) 
            # [B, W, 64] -> [B, W, 64] 
            auds = self.audio_attn_net(auds.view(B, self.render_hparams.audio_smoothing_size, -1))
        else:
            auds = self.audio_net(auds) # [B, 64]

        outputs = {
            # 'raw_coarse':[],
            # 'rgb_map_coarse':[], 
            # 'disp_map_coarse':[],
            # 'acc_map_coarse':[],
            # 'weights_coarse':[],
            # 'depth_map_coarse':[],
            # 'raw_fine':[],
            # 'rgb_map_fine':[], 
            # 'disp_map_fine':[],
            # 'acc_map_fine':[],
            # 'weights_fine':[],
            # 'depth_map_fine':[],
        }
        for i in range(0, R, chunk_sz):
             # [B, R, ch] is too large, so run in chunk [B, chunk, ch]
            rays_o_chunk = rays_o[:, i:i+chunk_sz]
            rays_d_chunk = rays_d[:, i:i+chunk_sz]
            view_dirs_chunk = view_dirs[:, i:i+chunk_sz]
            bg_imgs_chunk = bg_imgs[:, i:i+chunk_sz]
            auds_chunk = auds.unsqueeze(1).expand(-1, rays_o_chunk.shape[1], -1) #[B, 64] > [B, chunk, 64]

            # process coarse points by chunk as well
            n_samples_coarse = self.render_hparams.n_samples_coarse
            near_bound_chunk = self.render_hparams.near_bound * torch.ones_like(rays_d_chunk[..., :1]) # [B, chunk]
            far_bound_chunk = self.render_hparams.far_bound * torch.ones_like(rays_d_chunk[..., :1])
            t_vals = torch.linspace(0., 1., steps=n_samples_coarse).to(self.device) # 
            z_vals = near_bound_chunk * (1.-t_vals) + far_bound_chunk * (t_vals)

            if jitter:
                # get intervals between samples
                mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).to(self.device) #
                t_rand[..., -1] = 1.0
                z_vals = lower + (upper - lower) * t_rand

            outputs_chunk = self.forward( 
                rays_o_chunk, rays_d_chunk, z_vals,
                view_dirs_chunk, auds_chunk, bg_imgs_chunk,
                self.nerf_coarse, mode="coarse", chunk_sz=1024*64,
                output_raw_noise_std=output_raw_noise_std)
            update_n_append_dict(outputs_chunk, outputs, output_to_cpu)
            
            # process fine points by chunk as well
            n_samples_fine = self.render_hparams.n_samples_fine
            if n_samples_fine > 0:
                weights = outputs_chunk['weight_coarse']
                z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1],
                                       n_samples_fine,
                                       det=(jitter == 0)) # TODO Check
                z_samples = z_samples.detach()
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
                
                outputs_chunk = self.forward(
                    rays_o_chunk, rays_d_chunk, z_vals,
                    view_dirs_chunk, auds_chunk, bg_imgs_chunk,
                    self.nerf_fine, mode="fine", chunk_sz=1024*128,
                    output_raw_noise_std=output_raw_noise_std)
                outputs_chunk['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)
                outputs_chunk['last_weight'] = weights[..., -1]
                update_n_append_dict(outputs_chunk, outputs, output_to_cpu)

        # gather all the results and reshape
        outputs = concat_all_items_in_dict(outputs)

        # TODO 이거좀 정리
        keys_interested = ['rgb_map_coarse', 'rgb_map_fine']#, 'disp_map_fine', 'acc_map_fine'] #TODO key이름
        ret_list = [outputs[k] for k in keys_interested]
        #ret_dict = [outputs[k] for k in outputs if k not in keys_interested]
        return outputs['rgb_map_coarse'], outputs['rgb_map_fine'] #ret_list[0], ret_list[1]

    def on_train_start(self):
        self.val_psnr_best.reset()

    def on_train_epoch_start(self):
        if self.current_epoch == self.render_hparams.audio_smoothing_start_epoch:
            cfg = {'name': "audio_attn_net",
                   'params': self.audio_attn_net.parameters(),
                   'lr': self.optim_hparams.base_lr,
                   'betas': (0.9,0.999)}
            self.trainer.optimizers[0].add_param_group(cfg)
            self.trainer.datamodule.train_ds.audio_smoothing = True
            self.print(f'audio smoothing started at {self.global_step}')

    def training_step(self, batch: Any, batch_idx: int):
        preds_coarse, preds_fine = self.step(batch[:-1], batch_idx, chunk_sz=2048,
                            jitter=True,
                            output_raw_noise_std=self.render_hparams.output_raw_noise_std,
                            audio_smoothing=self.current_epoch>=self.render_hparams.audio_smoothing_start_epoch)
        targets = batch[-1]
        loss = self.criterion(preds_coarse, targets) + self.criterion(preds_fine, targets)
        psnr = self.train_psnr(preds_fine, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds_fine, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        _, preds = self.step(batch[:-1], batch_idx, chunk_sz=1024, output_to_cpu=True, audio_smoothing=True)
        preds = preds.view(-1, *self.trainer.datamodule.val_ds.img_size, 3)
        targets = batch[-1].cpu().view(-1, *self.trainer.datamodule.val_ds.img_size, 3)
        loss = self.criterion(preds, targets)
        psnr = self.val_psnr(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        """
        All the outputs after one epoch will be given to generate a video.
        """
        psnr = self.val_psnr.to(self.device).compute()
        self.val_psnr_best.update(psnr)
        self.log("val/psnr_best", self.val_psnr_best.compute(), on_epoch=True, prog_bar=True)
        preds_gpu = torch.cat([output["preds"] for output in outputs], 0)
        preds_all = dist.gather_all_tensors(preds_gpu)
        if self.trainer.is_global_zero:
            dataset_size = len(self.trainer.datamodule.val_ds)
            img_size = self.trainer.datamodule.val_ds.img_size
            world_size = self.trainer.world_size
            preds_all = sort_preds(preds_all, dataset_size, img_size, world_size)
            log_image(self.logger, "val_imgs", self.trainer.global_step, preds_all)

    def test_step(self, batch: Any, batch_idx: int):
        """
        Batches will be assigned to each GPU.
        Every GPU goes through the same test step.
        """
        _, preds = self.step(batch, batch_idx, chunk_sz=1024, output_to_cpu=True, audio_smoothing=True)
        preds = preds.view(-1, *self.trainer.datamodule.test_ds.img_size, 3)
        return {"preds": preds}

    def test_epoch_end(self, outputs):
        """
        Predictions from the GPUs after one epoch will be given to generate a video.
        """
        preds_gpu = torch.cat([output["preds"] for output in outputs], 0)
        preds_all = dist.gather_all_tensors(preds_gpu)
        if self.trainer.is_global_zero:
            dataset_size = len(self.trainer.datamodule.test_ds)
            img_size = self.trainer.datamodule.test_ds.img_size
            world_size = self.trainer.world_size
            preds_all = sort_preds(preds_all, dataset_size, img_size, world_size)
            log_video(self.logger, "test_video", preds_all, fps=self.render_hparams.fps)

    def on_epoch_end(self):
        """
        """
        # reset metrics at the end of every epoch
        self.train_psnr.reset()
        self.test_psnr.reset()
        self.val_psnr.reset()

    def configure_optimizers(self):
        """
        """
        cfgs = [
            {'name': "nerf_coarse",
             'params': self.nerf_coarse.parameters(),
             'lr': self.optim_hparams.base_lr,
             'betas':(0.9,0.999)},
            {'name': "audio_net",
             'params': self.audio_net.parameters(),
             'lr': self.optim_hparams.base_lr,
             'betas':(0.9,0.999)},
        ]
        if self.render_hparams.n_samples_fine > 0:
            cfgs.append({'name': "nerf_fine",
                         'params': self.nerf_fine.parameters(),
                         'lr': self.optim_hparams.base_lr,
                         'betas':(0.9,0.999)})
        optimizer = torch.optim.Adam(cfgs)
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                    optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        optimizer.step(closure=optimizer_closure)
        new_lr = self.optim_hparams.base_lr *\
                 self.optim_hparams.decay_rate **\
                 (self.trainer.global_step / self.optim_hparams.decay_steps)
        for i, pg in enumerate(optimizer.param_groups):
            pg['name'] = new_lr if pg['name']!="audio_attn_net" else new_lr*5