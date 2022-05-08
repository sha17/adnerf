from typing import Any, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid

from pytorch_lightning import LightningModule

from torchmetrics import MaxMetric
from torchmetrics.image.psnr import PeakSignalNoiseRatio

from src.models.components.face_nerf import FaceNeRF
from src.models.components.pos_encoder import PositionEncoder
from src.models.components.audio_net import AudioNet
from src.models.components.audio_attn_net import AudioAttnNet

from src.utils.utils import raw2outputs, sample_pdf, update_n_append_dict, concat_all_items_in_dict

from omegaconf import DictConfig

class AdNeRFLitModule(LightningModule):

    def __init__(
        self,
        lr,
        weight_decay,
        pos_encoder: DictConfig,
        pos_encoder_view_dirs: DictConfig, 
        nerf_coarse: DictConfig,
        nerf_fine: DictConfig,
        audio_net: DictConfig,
        audio_attn_net: DictConfig,
        options: DictConfig
    ):  # TODO: 이거 config 어떻게 처리할지 확인

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
        nerf_coarse['output_ch'] = 5 if options.n_samples_per_ray_fine > 0 else 4
        nerf_coarse = FaceNeRF(**nerf_coarse)

        nerf_fine['input_ch'] = pos_encoder.get_output_dim()
        nerf_fine['input_ch_views'] = pos_encoder_view_dirs.get_output_dim()
        nerf_fine['dim_aud'] = audio_net.dim_aud
        nerf_fine['output_ch'] = 5 if options.n_samples_per_ray_fine > 0 else 4
        nerf_fine = FaceNeRF(**nerf_fine)

        self.pos_encoder = pos_encoder
        self.pos_encoder_view_dirs = pos_encoder_view_dirs
        self.nerf_coarse = nerf_coarse
        self.nerf_fine = nerf_fine
        self.audio_net = audio_net
        self.audio_attn_net = audio_attn_net

        self.options = options

        self.criterion = nn.MSELoss()

        self.train_psnr = PeakSignalNoiseRatio()
        self.val_psnr = PeakSignalNoiseRatio()
        self.test_psnr = PeakSignalNoiseRatio()

        self.val_psnr_best = MaxMetric()

    def forward(self, rays_o, rays_d, z_vals, view_dirs, auds, bg_imgs, nerf, mode, chunk_sz:int=1024*64):
        """
        """
        pnts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [B, num_rays, num_samples]
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
                self.options.output_raw_noise_std,
                self.options.white_background)

        keys = ['raw', 'rgb_map', 'disp_map', 'acc_map', 'weight', 'depth_map']
        outputs = dict(zip(
            [k+f'_{mode}' for k in keys],
            [raw, rgb_map, disp_map, acc_map, weights, depth_map]))
        return outputs

    def step(self, batch, batch_idx, chunk_sz=1024, output_to_cpu=False): # chunk size 설정
        """
        """
        B, R = batch[0].shape[0:2]
        rays_o, rays_d, view_dirs, bg_imgs, imgs, auds = batch

        temp = False
        if temp: #self.global_step >= self.options.audio_smoothing_start_step:
            auds = self.audio_net(torch.flatten(auds, end_dim=1)) # 16 64
            auds = self.audio_attn_net(auds.view(B, self.options.smoothing_size, -1))
        else:
            auds = self.audio_net(auds) # [B, 64]

        outputs = {}
        for i in range(0, R, chunk_sz):
             # [B, R, ch] is too large, so run in chunk [B, chunk, ch]
            rays_o_chunk = rays_o[:, i:i+chunk_sz]
            rays_d_chunk = rays_d[:, i:i+chunk_sz]
            view_dirs_chunk = view_dirs[:, i:i+chunk_sz]
            bg_imgs_chunk = bg_imgs[:, i:i+chunk_sz]
            auds_chunk = auds.unsqueeze(1).expand(-1, rays_o_chunk.shape[1], -1) #[B, 64] > [B, chunk, 64]

            # PROCESS COARSE POINTS
            n_samples_coarse = self.options.n_samples_per_ray_coarse
            rays_n_chunk = self.options.neareast_distance * torch.ones_like(rays_d_chunk[..., :1]) # [B, chunk]
            rays_f_chunk = self.options.farthest_distance * torch.ones_like(rays_d_chunk[..., :1])
            t_vals = torch.linspace(0., 1., steps=n_samples_coarse).to(self.device) # 
            if not self.options.sampling_linearly_in_disparity:
                z_vals = rays_n_chunk * (1.-t_vals) + rays_f_chunk * (t_vals)
            else:
                z_vals = 1./(1./rays_n_chunk * (1.-t_vals) + 1./rays_f_chunk * (t_vals))

            #z_vals = z_vals.expand([B*R, n_samples_coarse])
            if self.options.use_jitter:
                # get intervals between samples
                mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).to(self.device) #
                t_rand[..., -1] = 1.0
                z_vals = lower + (upper - lower) * t_rand

            # run network by another chunk
            outputs_chunk = self.forward( 
                rays_o_chunk, rays_d_chunk, z_vals,
                view_dirs_chunk, auds_chunk, bg_imgs_chunk,
                self.nerf_coarse, mode="coarse", chunk_sz=1024*64)
            update_n_append_dict(outputs_chunk, outputs, output_to_cpu)

            # PROCESS FINE POINTS
            n_samples_fine = self.options.n_samples_per_ray_fine
            if n_samples_fine > 0:
                weights = outputs_chunk['weight_coarse']
                z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples_fine, det=(self.options.use_jitter == 0.))
                z_samples = z_samples.detach()
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
                
                # run network by another chunk
                outputs_chunk = self.forward(
                    rays_o_chunk, rays_d_chunk, z_vals,
                    view_dirs_chunk, auds_chunk, bg_imgs_chunk,
                    self.nerf_fine, mode="fine", chunk_sz=1024*64)
                outputs_chunk['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)
                outputs_chunk['last_weight'] = weights[..., -1]
                update_n_append_dict(outputs_chunk, outputs, output_to_cpu)

        # gather all the results and reshape
        outputs = concat_all_items_in_dict(outputs)

        # TODO 이거좀 정리
        keys_interested = ['rgb_map_fine', 'disp_map_fine', 'acc_map_fine'] #TODO key이름
        ret_list = [outputs[k] for k in keys_interested]
        ret_dict = [outputs[k] for k in outputs if k not in keys_interested]

        if output_to_cpu:
            imgs = imgs.cpu()
        loss = self.criterion(ret_list[0], imgs) # 
        return loss, ret_list[0], imgs

    def on_train_start(self):
        self.val_psnr_best.reset()

    def training_step(self, batch: Any, batch_idx: int):
        batch_size = batch[0].shape[0]
        loss, preds, targets = self.step(batch, batch_idx, chunk_sz=1024)
        psnr = self.train_psnr(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def on_validation_epoch_start(self):
        self.hwfcxy = self.trainer.val_dataloaders[0].dataset.hwfcxy
        self.img_size = [int(self.hwfcxy[0]), int(self.hwfcxy[1])]

    def validation_step(self, batch: Any, batch_idx: int):
        batch_size = batch[0].shape[0]
        loss, preds, targets = self.step(batch, batch_idx, chunk_sz=1024, output_to_cpu=True)
        preds = preds.view(batch_size, *self.img_size, 3)
        targets = targets.view(batch_size, *self.img_size, 3)
        psnr = self.val_psnr(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.logger.experiment.add_image('val_pred', make_grid(preds.permute(0,3,1,2)), self.global_step)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        psnr = self.val_psnr.to(self.device).compute()
        self.val_psnr_best.update(psnr)
        self.log("val/psnr_best", self.val_psnr_best.compute(), on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        self.hwfcxy = self.trainer.test_dataloaders[0].dataset.hwfcxy
        self.img_size = [int(self.hwfcxy[0]), int(self.hwfcxy[1])]

    def test_step(self, batch: Any, batch_idx: int):
        batch_size = batch[0].shape[0]
        loss, preds, targets = self.step(batch, batch_idx, chunk_sz=1024, output_to_cpu=True)
        preds = preds.view(batch_size, *self.img_size, 3)
        targets = targets.view(batch_size, *self.img_size, 3)
        psnr = self.test_psnr(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.logger.experiment.add_image('test_pred', make_grid(preds.permute(0,3,1,2)), self.global_step)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_psnr.reset()
        self.test_psnr.reset()
        self.val_psnr.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # cfgs = [
        #     {'params': self.nerf_coarse.parameters(), 'lr': self.hparams.nerf_coarse.loss*},          # for faceNeRF
        #     {'params': self.audio_net.parameters(), 'lr': self.hparams.audio_net},     # for AudioNet
        #     {'params': self.audio_attn_net.parameters(), 'lr': self.hparams.audio_attn_net}, # for AudioAttnNet
        # ]
        # if self.options.n_samples_fine > 0:
        #     cfgs.append({'params': self.nerf_fine.parameters(), 'lr': self.hparams.nerf_fine})

        optimizer = torch.optim.Adam([ #'weight_decay':self.hparams.weight_decay
            {'params': self.nerf_coarse.parameters(), 'lr':self.hparams.lr, 'weight_decay':self.hparams.weight_decay},          # for faceNeRF
            {'params': self.nerf_fine.parameters(), 'lr':self.hparams.lr, 'weight_decay':self.hparams.weight_decay},
            {'params': self.audio_net.parameters(), 'lr':self.hparams.lr, 'weight_decay':self.hparams.weight_decay},     # for AudioNet
            {'params': self.audio_attn_net.parameters(), 'lr':self.hparams.lr, 'weight_decay':self.hparams.weight_decay}, # for AudioAttnNet
        ])
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
        return [optimizer]#, [scheduler]
