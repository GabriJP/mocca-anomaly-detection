#!/usr/bin/env bash

pip3 uninstall -y torch torchvision

#pip3 install -U --user https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+410ce96a.nv22.12-cp38-cp38-linux_aarch64.whl
pip3 install -U --user https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+936e9305.nv22.11-cp38-cp38-linux_aarch64.whl

[ ! -d vision ] && git clone https://github.com/pytorch/vision.git

cd vision

git reset --hard

git chechout v0.14.0

export BUILD_VERSION=0.14.0 FORCE_CUDA=1 CUDA_VISIBLE_DEVICES=0 CUDA_HOME='/usr/local/cuda' USE_CUDA=1 USE_CUDNN=1

python3 setup.py install --user
