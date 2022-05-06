from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

import numpy as np
import cv2 # imageio 대신 opency 사용

from src.utils.utils import load_audface_data, get_rays_np

from omegaconf import DictConfig


class RayDataset(Dataset):
    def __init__(
        self,
        rays,
        auds, 
        bg_img,
        hwfcxy, 
        sample_rects, 
        mouth_rects, 
        img_paths,  
        idxs_split, 
        options,
    ):
        """
        해결해야될 옵션들
            near
            far
            use_viewdirs
            ndc
            c2w
        """ 
        rays = rays[idxs_split[1]].astype(np.float32)
        rays_o, rays_d = rays.transpose(3, 0, 1, 2, 4)
        H, W, focal, cx, cy = hwfcxy
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H),
                                            torch.linspace(0, W-1, W)), -1)  
        self.coords = torch.reshape(coords, [-1,2]) # (HxW, 2)

        self.rays_o = rays_o 
        self.rays_d = rays_d 
        self.auds = auds[idxs_split[1]]

        self.sample_rects = sample_rects[idxs_split[1]]
        self.mouth_rects = mouth_rects[idxs_split[1]]
        self.img_paths = img_paths[idxs_split[1]]

        self.bg_img = bg_img 
        self.hwfcxy = np.array(hwfcxy)

        self.options = options

        self.split = idxs_split[0]
        self.audio_smoothing = False

        print(f'[{idxs_split[0]}] RayDataset created\n'
              f'\t rays_o:{self.rays_o.shape}\n'
              f'\t rays_d:{self.rays_d.shape}\n'
              f'\t audios:{self.auds.shape}\n'
              f'\t sample_rects:{self.sample_rects.shape}\n'
              f'\t mouth_rects:{self.mouth_rects.shape}\n'
              f'\t bg_img:{self.bg_img.shape}\n'
              f'\t hwfcxy:{self.hwfcxy}\n'
              f'\t img_paths:{self.img_paths.shape}\n'  
              )

    def __len__(self):
        """
        """
        return len(self.img_paths)

    def __getitem__(self, index):
        """
        point sampling per ray  # TODO : augmentation 부분들 함수로 빼기
        """
        sample_rect = self.sample_rects[index]
        mouth_rect = self.mouth_rects[index]

        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.split=="train":
            coords = self.coords
            if self.options.sample_rate_in_bbox > 0:
                rect_inds = (coords[:, 0] >= sample_rect[0]) &\
                            (coords[:, 0] <= sample_rect[0] + sample_rect[2]) &\
                            (coords[:, 1] >= sample_rect[1]) &\
                            (coords[:, 1] <= sample_rect[1] + sample_rect[3])
                coords_rect = coords[rect_inds]
                coords_norect = coords[~rect_inds]
                rect_num = int(self.options.n_rays_per_batch * self.options.sample_rate_in_bbox)
                norect_num = self.options.n_rays_per_batch - rect_num
                select_inds_rect = np.random.choice(
                    coords_rect.shape[0], size=[rect_num], replace=False)  # (N_rand,)
                # (N_rand, 2)
                select_coords_rect = coords_rect[select_inds_rect].long()
                select_inds_norect = np.random.choice(
                    coords_norect.shape[0], size=[norect_num], replace=False)  # (N_rand,)
                # (N_rand, 2)
                select_coords_norect = coords_norect[select_inds_norect].long()
                select_coords = np.concatenate((select_coords_rect, select_coords_norect), 0)
            else:
                select_idxs = np.random.choice(coords.shape[0], size=[self.options.n_rays_per_batch], replace=False)
                select_coords = coords[select_idxs].long()
            rays_o = self.rays_o[index][select_coords[:, 0], select_coords[:, 1]]  # (n_rays_per_batch, 3)
            rays_d = self.rays_d[index][select_coords[:, 0], select_coords[:, 1]]  # (n_rays_per_batch, 3)
            img = img[select_coords[:, 0], select_coords[:, 1]] 
            bg_img = self.bg_img[select_coords[:, 0], select_coords[:, 1]]
        else:
            H, W, focal, cx, cy = self.hwfcxy
            rays_o = np.reshape(self.rays_o[index], [int(H*W), 3])  # (n_rays_per_batch, 3)
            rays_d = np.reshape(self.rays_d[index], [int(H*W), 3])  # (n_rays_per_batch, 3)
            img = np.reshape(img, [int(H*W), 3])
            bg_img = np.reshape(self.bg_img, [int(H*W), 3])

        view_dirs = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)      # (n_rays_per_batch, 3)
        img = np.reshape(img,[-1,3])/255.0 # norm flag 설정
        bg_img = np.reshape(bg_img, [-1,3])/255.0

        if self.split=="train" and self.audio_smoothing:
            smoothing_window_half = int(self.options.smoothing_size/2)
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
            aud = auds_win # [8, 29] smoothed with padding 출력해서 잘 되었는지 확인해보기
        else:
            aud = self.auds[index]
        return rays_o, rays_d, view_dirs, bg_img, img, aud


class RayDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_batch_size: int,
        valid_batch_size: int,
        test_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        # use_test_data: bool,
        # test_data_dir: str,
        test_data_sample_rate: int,
        options: DictConfig,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.ds_train: Optional[Dataset] = None
        self.ds_valid: Optional[Dataset] = None
        self.ds_test: Optional[Dataset] = None

    def prepare_data(self):
        """
        Download data if needed.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already
        if not self.ds_train and not self.ds_valid: # and not self.ds_test:
            # if self.hparams.use_test_data:
            #     #poses, audios, bg_image, hwfcxy =\
            #     #    load_audface_data(self.hparams.data_dir, self.hparams.test_skip, self.hparams.audio_file)
            #     imgs = np.zeros(1)
            # else:
            frame_ids, img_paths, poses, auds, bg_img, hwfcxy, sample_rects, mouth_rects, ray_idxs_splits =\
                load_audface_data(self.hparams.data_dir, self.hparams.test_data_sample_rate)
            ray_idxs_train = ("train", ray_idxs_splits[0])
            ray_idxs_valid = ("valid", ray_idxs_splits[1])     

            impl_bound = 200
            ray_idxs_train = ("train", np.arange(1, 199))
            ray_idxs_valid = ("valid", np.arange(199, impl_bound)) # impl_bound
            rays = []
            for idx, p in enumerate(poses[:impl_bound, :3, :4]):
                if idx%100==0:
                    print(f'{idx}th get_rays_np...')
                rays.append(get_rays_np(*hwfcxy, p))
            rays = np.stack(rays, 0)
            rays = np.transpose(rays, (0, 2, 3, 1, 4)) # [N, H, W, ro+rd, 3]

            sample_rects = sample_rects[:impl_bound]
            mouth_rects = mouth_rects[:impl_bound]

            self.ds_train = RayDataset(rays, auds, bg_img, hwfcxy, sample_rects, mouth_rects, img_paths, ray_idxs_train, self.hparams.options)
            self.ds_valid = RayDataset(rays, auds, bg_img, hwfcxy, sample_rects, mouth_rects, img_paths, ray_idxs_valid, self.hparams.options)
            self.ds_test = RayDataset(rays, auds, bg_img, hwfcxy, sample_rects, mouth_rects, img_paths, ray_idxs_valid, self.hparams.options)            

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