<div align="center">

# AD-NeRF

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2103.11078-B31B1B.svg)](https://arxiv.org/abs/2103.11078)
[![Conference](http://img.shields.io/badge/ICCV-2021-4b44ce.svg)](https://openaccess.thecvf.com/ICCV2021)

</div>

## Description
*WIP*
This repo contains AD-NeRF reimplemented and ported over Lightning-Hydra-Template.
The reimplemented version supports Multi-GPU training and evaluation with DDP + GLOO backend,
but the most of details and structures follow the original repo.
We thank the authors for the great work, and hope this help others interested in the audio-driven talking head generation but dragged down with slow training and complicated code.

@inproceedings{guo2021adnerf,
  title={AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis},
  author={Yudong Guo and Keyu Chen and Sen Liang and Yongjin Liu and Hujun Bao and Juyong Zhang},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/sha17/adnerf
cd adnerf

# INSTALLATION README WIP!!!
# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python train.py trainer.gpus=0

# train on GPU
PL_TORCH_DISTRIBUTED_BACKEND=gloo python train.py trainer.gpus=1

# train on Multi GPUs
# At the validation/test steps, the predicted frames at *CPU* processes are gathered.
# NCCL does not support CPU-tensor gathering, so use GLOO.
PL_TORCH_DISTRIBUTED_BACKEND=gloo python train.py trainer.gpus=2 +trainer.strategy=ddp

# test on Multi GPUs
PL_TORCH_DISTRIBUTED_BACKEND=gloo python test.py trainer.gpus=4 +trainer.strategy=ddp ckpt_path=logs/experiments/runs/default/2022-05-10_05-07-23/checkpoints/last.ckpt

# didn't test it yet but it should work as well on the multi-node environment.
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```
