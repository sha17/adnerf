from typing import Optional, Tuple, Any
from omegaconf import DictConfig

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule

import os
import cv2
import numpy as np

from src.utils.utils import load_audface_data, get_rays_np


class RayDataset(Dataset):
    def __init__(
        self,
        rays,
        auds, 
        bg_img,
        hwfcxy,
        render_hparams,
        split, 
        sample_rects=None, 
        mouth_rects=None, 
        img_paths=None,  
    ):
        rays_o, rays_d = rays.transpose(3, 0, 1, 2, 4)

        self.rays_o = rays_o 
        self.rays_d = rays_d 
        self.auds = auds
        self.bg_img = bg_img 
        self.hwfcxy = hwfcxy
        self.render_hparams = render_hparams
        self.split = split
        self.sample_rects = sample_rects
        self.mouth_rects = mouth_rects
        self.img_paths = img_paths

        H, W, focal, cx, cy = hwfcxy
        coords = torch.stack(torch.meshgrid(torch.linspace(0, int(H)-1, int(H)),
                                            torch.linspace(0, int(W)-1, int(W))), -1)  
        self.coords = torch.reshape(coords, [-1,2]) # (HxW, 2)
        self.audio_smoothing = False

        print(f'[{self.split}] RayDataset created\n'
              f'\t rays_o:{self.rays_o.shape}\n'
              f'\t rays_d:{self.rays_d.shape}\n'
              f'\t audios:{self.auds.shape}\n'
              f'\t bg_img:{self.bg_img.shape}\n'
              f'\t hwfcxy:{self.hwfcxy}\n')
        if self.split=="train":
            print(f'\t sample_rects:{self.sample_rects.shape}\n'
                  f'\t mouth_rects:{self.mouth_rects.shape}\n'
                  f'\t img_paths:{self.img_paths.shape}\n')

    def __len__(self):
        return len(self.rays_o)

    def __getitem__(self, index):
        """
        point sampling per ray  # TODO : 함수로 빼기
        """
        if self.split=="train":
            # train일 때는 얼굴 부분에서 ray들을 sampling 해온다.
            coords = self.coords
            sample_rect = self.sample_rects[index]
            mouth_rect = self.mouth_rects[index]
            if self.render_hparams.sample_rate_in_bbox > 0:
                rect_inds = (coords[:, 0] >= sample_rect[0]) &\
                            (coords[:, 0] <= sample_rect[0] + sample_rect[2]) &\
                            (coords[:, 1] >= sample_rect[1]) &\
                            (coords[:, 1] <= sample_rect[1] + sample_rect[3])
                coords_rect = coords[rect_inds]
                coords_norect = coords[~rect_inds]
                rect_num = int(self.render_hparams.n_rays_per_batch * self.render_hparams.sample_rate_in_bbox)
                norect_num = self.render_hparams.n_rays_per_batch - rect_num
                select_inds_rect = np.random.choice(
                    coords_rect.shape[0], size=[rect_num], replace=False) # (N_rand,)
                select_coords_rect = coords_rect[select_inds_rect].long() # (N_rand, 2)
                select_inds_norect = np.random.choice(
                    coords_norect.shape[0], size=[norect_num], replace=False)  # (N_rand,)
                select_coords_norect = coords_norect[select_inds_norect].long() # (N_rand, 2)
                select_coords = np.concatenate((select_coords_rect, select_coords_norect), 0)
            else:
                select_idxs = np.random.choice(coords.shape[0], size=[self.render_hparams.n_rays_per_batch], replace=False)
                select_coords = coords[select_idxs].long()
            rays_o = self.rays_o[index][select_coords[:, 0], select_coords[:, 1]]  # (n_rays_per_batch, 3)
            rays_d = self.rays_d[index][select_coords[:, 0], select_coords[:, 1]]  # (n_rays_per_batch, 3)
            bg_img = self.bg_img[select_coords[:, 0], select_coords[:, 1]]
        else:
            H, W, focal, cx, cy = self.hwfcxy
            rays_o = np.reshape(self.rays_o[index], [int(H*W), 3])  # (n_rays_per_batch, 3)
            rays_d = np.reshape(self.rays_d[index], [int(H*W), 3])  # (n_rays_per_batch, 3)
            bg_img = np.reshape(self.bg_img, [int(H*W), 3])

        if self.split!="test":
            img = cv2.imread(self.img_paths[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.split=="train":
                img = img[select_coords[:, 0], select_coords[:, 1]]
            elif self.split=="valid":
                img = np.reshape(img, [int(H*W), 3])
            img = np.reshape(img,[-1,3])/255.0 # norm flag 설정

        view_dirs = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)      # (n_rays_per_batch, 3)
        bg_img = np.reshape(bg_img, [-1,3])/255.0
        if self.split=="train" and self.audio_smoothing:
            smoothing_window_half = int(self.render_hparams.audio_smoothing_size/2)
            left_i = index - smoothing_window_half
            right_i = index + smoothing_window_half
            pad_left, pad_right = 0, 0
            if left_i < 0:
                pad_left = -left_i
                left_i = 0
            if right_i > self.__len__():
                pad_right = right_i-self.__len__()
                right_i = self.__len__()
            auds_win = self.auds[left_i:right_i]
            if pad_left > 0:
                auds_win = np.concatenate((np.zeros_like(auds_win)[:pad_left], auds_win), 0)
            if pad_right > 0:
                auds_win = np.concatenate((auds_win, np.zeros_like(auds_win)[:pad_right]), 0)
            aud = auds_win # [8, 29] TODO: smoothed with padding 출력해서 잘 되었는지 확인해보기
        else:
            aud = self.auds[index]

        if self.split != "test":
            return rays_o, rays_d, view_dirs, bg_img, aud, img
        else:
            return rays_o, rays_d, view_dirs, bg_img, aud


class RayDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        aud_file: str,
        ray_file: str,
        test_file: str,
        test_ray_file: str,
        render_hparams: DictConfig,
        num_workers: int,
        pin_memory: bool,
        train_batch_size: int,
        valid_batch_size: int,
        test_batch_size: int,
        val_frame_sample_rate:int, 
        test_frame_sample_rate: int,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.ds_train: Optional[Dataset] = None
        self.ds_valid: Optional[Dataset] = None
        self.ds_test: Optional[Dataset] = None

    @property
    def train_ds(self):
        return self.ds_train

    @property
    def val_ds(self):
        return self.ds_valid

    @property
    def test_ds(self):
        return self.ds_test

    def prepare_data(self):
        """
        Download data if needed.
        """
        pass

    def get_rays(self, poses, hwfcxy, base_dir, ray_file):
        ray_file = os.path.join(base_dir, ray_file)
        if os.path.exists(ray_file):
            print(f'loading ray from {ray_file} ...')
            rays = np.load(ray_file)
            print(f'ray loaded from {ray_file} !')
        else:
            rays = []
            for idx, p in enumerate(poses[:, :3, :4]):
                if idx%100==0:
                    print(f'{idx}th get_rays_np...')
                rays.append(get_rays_np(*hwfcxy, p))
            rays = np.stack(rays, 0)
            rays = np.transpose(rays, (0, 2, 3, 1, 4)).astype(np.float32) # [N, H, W, ro+rd, 3]
            np.save(ray_file, rays)
            print(f'ray saved at {ray_file}')
        return rays

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already
        if (stage=="fit" or stage is None) and not self.ds_train and not self.ds_valid:
            img_paths, poses, auds, bg_img, hwfcxy, sample_rects, mouth_rects, ray_idxs_splits =\
                load_audface_data(self.hparams.data_dir,
                                  self.hparams.val_frame_sample_rate, 
                                  aud_file=self.hparams.aud_file) 
            rays = self.get_rays(poses, hwfcxy, self.hparams.data_dir, self.hparams.ray_file)

            self.ds_train = RayDataset(rays[ray_idxs_splits[0]], 
                                        auds[ray_idxs_splits[0]], 
                                        bg_img, 
                                        np.array(hwfcxy),
                                        self.hparams.render_hparams,
                                        split="train", 
                                        sample_rects=sample_rects[ray_idxs_splits[0]], 
                                        mouth_rects=mouth_rects[ray_idxs_splits[0]], 
                                        img_paths=img_paths[ray_idxs_splits[0]])
            self.ds_valid = RayDataset(rays[ray_idxs_splits[1]], 
                                        auds[ray_idxs_splits[1]], 
                                        bg_img, 
                                        np.array(hwfcxy),
                                        self.hparams.render_hparams,
                                        split="val",
                                        img_paths=img_paths[ray_idxs_splits[1]])

        if stage=="test" and not self.ds_test and self.hparams.test_file:
            poses, auds, bg_img, hwfcxy =\
                load_audface_data(self.hparams.data_dir,
                                  self.hparams.test_frame_sample_rate,
                                  test_file=self.hparams.test_file,
                                  aud_file=self.hparams.aud_file)       
            rays = self.get_rays(poses, hwfcxy, self.hparams.data_dir, self.hparams.test_ray_file)              
            self.ds_test = RayDataset(rays, auds, bg_img, np.array(hwfcxy), self.hparams.render_hparams, split="test")      
   
    def train_dataloader(self):
        return DataLoader(
            dataset=self.ds_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.ds_valid,
            batch_size=self.hparams.valid_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.ds_test,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )