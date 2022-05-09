import os
import numpy as np
import imageio
from typing import Any, List
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchmetrics import MaxMetric
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from pytorch_lightning import LightningModule

from src.models.components.face_nerf import FaceNeRF
from src.models.components.pos_encoder import PositionEncoder
from src.models.components.audio_net import AudioNet
from src.models.components.audio_attn_net import AudioAttnNet

from src.utils.utils import raw2outputs, sample_pdf, update_n_append_dict, concat_all_items_in_dict


class AdNeRFLitModule(LightningModule):

    def __init__(
        self,
        lr,
        #weight_decay,
        pos_encoder: DictConfig,
        pos_encoder_view_dirs: DictConfig, 
        nerf_coarse: DictConfig,
        nerf_fine: DictConfig,
        audio_net: DictConfig,
        audio_attn_net: DictConfig,
        options: DictConfig
    ):
        """
        Args:
            lr:
            pos_encoder:
            pos_encoder_view_dirs:
            nerf_coarse:
            nerf_fine:
            audio_net:
            audio_attn_net:
            options:
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
                self.options.white_background)

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
             audio_smoothing=False): # chunk size 설정
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
        rays_o, rays_d, view_dirs, bg_imgs, imgs, auds = batch
        if audio_smoothing:
            # [B, W, 16, 29] -> [BxW, 16, 29] 
            auds = self.audio_net(torch.flatten(auds,end_dim=1)) 
            # [B, W, 64] -> [B, W, 64] 
            auds = self.audio_attn_net(auds.view(B, self.options.audio_smoothing_size, -1))
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
            if jitter:
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
                self.nerf_coarse, mode="coarse", chunk_sz=1024*64,
                output_raw_noise_std=output_raw_noise_std)
            #update_n_append_dict(outputs_chunk, outputs, output_to_cpu) # 이거 굳이할 필요가?
            
            # PROCESS FINE POINTS
            n_samples_fine = self.options.n_samples_per_ray_fine
            if n_samples_fine > 0:
                weights = outputs_chunk['weight_coarse']
                z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1],
                                       n_samples_fine,
                                       det=(jitter == 0)) # TODO Check
                z_samples = z_samples.detach()
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
                
                # run network by another chunk
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
        keys_interested = ['rgb_map_fine', 'disp_map_fine', 'acc_map_fine'] #TODO key이름
        ret_list = [outputs[k] for k in keys_interested]
        ret_dict = [outputs[k] for k in outputs if k not in keys_interested]

        if output_to_cpu:
            imgs = imgs.cpu()
        loss = self.criterion(ret_list[0], imgs) # 
        return loss, ret_list[0], imgs

    def on_train_start(self):
        """
        """
        self.val_psnr_best.reset()

    def on_train_epoch_start(self):
        if self.current_epoch == self.options.audio_smoothing_start_epoch:
            cfg = {'params': self.audio_attn_net.parameters(),
                   'lr': self.hparams.lr,
                   'betas': (0.9,0.999)}
            self.trainer.optimizers[0].add_param_group(cfg)
            self.trainer.datamodule.train_ds.audio_smoothing = True
            self.print(f'audio smoothing started at {self.global_step}')

    def training_step(self, batch: Any, batch_idx: int):
        """
        """
        loss, preds, targets = self.step(batch, batch_idx, chunk_sz=2048,
                                        jitter=True,
                                        output_raw_noise_std=self.options.output_raw_noise_std,
                                        audio_smoothing=True \
                                            if self.current_epoch >= self.options.audio_smoothing_start_epoch
                                            else False)
        psnr = self.train_psnr(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        """
        """
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        """
        """
        loss, preds, targets = self.step(batch, batch_idx, chunk_sz=1024, output_to_cpu=True)
        preds, targets = self.reshape_outputs("val", preds, targets)
        psnr = self.val_psnr(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        # self.log_image("val", preds, targets)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        """
        """
        psnr = self.val_psnr.to(self.device).compute()
        self.val_psnr_best.update(psnr)
        self.log("val/psnr_best", self.val_psnr_best.compute(), on_epoch=True, prog_bar=True)
        preds = torch.cat([output['preds'] for output in outputs],0)
        self.log_video(preds)

    def test_step(self, batch: Any, batch_idx: int):
        """
        """
        # TODO: 수정
        loss, preds, targets = self.step(batch, batch_idx, chunk_sz=1024, output_to_cpu=True)
        preds, targets = self.reshape_outputs("test", preds, targets)
        psnr = self.val_psnr(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log_image("test", preds, targets)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        """
        """
        self.log_video(outputs)

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
            {'params': self.nerf_coarse.parameters(), 'lr': self.hparams.lr, 'betas':(0.9,0.999)},
            {'params': self.audio_net.parameters(), 'lr': self.hparams.lr, 'betas':(0.9,0.999)},
        ]
        if self.options.n_samples_per_ray_fine > 0:
            cfgs.append({'params': self.nerf_fine.parameters(), 'lr': self.hparams.lr, 'betas':(0.9,0.999)})
        optimizer = torch.optim.Adam(cfgs)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
        return [optimizer], [scheduler]

    def reshape_outputs(self, split, preds, targets):
        """
        Args:
            split:
            preds:
            targets:
        Returns:
            preds:
            targets:
        """
        batch_size = preds.shape[0]
        if split=="val":
            hwfcxy = self.trainer.datamodule.val_ds.hwfcxy
        elif split=="test":
            hwfcxy = self.trainer.datamodule.test_ds.hwfcxy
        img_size = [int(hwfcxy[0]), int(hwfcxy[1])]
        preds = preds.view(batch_size, *img_size, 3)
        targets = targets.view(batch_size, *img_size, 3)
        return preds, targets

    def log_image(self, split, preds, targets):
        """
        Args:
            split:
            presd:
            targets:
            batch_size:
        """
        self.logger.experiment.add_image(f'{split}_pred',
                                         make_grid(preds.permute(0,3,1,2)),
                                         self.global_step)

    def log_video(self, imgs, fps=30, quality=8):
        """
        Args:
            imgs:
            fps:
            quality:
        """
        def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)
        video_root = os.path.join(self.logger.log_dir, '../../videos')
        if not os.path.exists(video_root):
            os.makedirs(video_root)
        video_path = os.path.join(video_root, f'{self.global_step}.mp4')
        imgs = np.array(list(map(lambda x: to8b(x.numpy()), imgs)))
        imageio.mimwrite(video_path, imgs, fps=fps, quality=quality)
        print(f'Video saved at {video_path}.')