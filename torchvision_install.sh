#!/usr/bin/env bash

pip3 uninstall -y torch torchvision

# Torch
[ ! -d pytorch ] && https://github.com/pytorch/pytorch.git
cd pytorch || exit
git reset --hard
git checkout v1.13.1
export USE_NCCL=0 USE_DISTRIBUTED=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 TORCH_CUDA_ARCH_LIST="7.2;8.7" PYTORCH_BUILD_VERSION=1.13.1 PYTORCH_BUILD_NUMBER=1
#nano torch/utils/cpp_extension.py -> https://gist.github.com/dusty-nv/ce51796085178e1f38e3c6a1663a93a1#file-pytorch-1-11-jetpack-5-0-patch
python3 setup.py bdist_wheel
pip3 install --user -U dist/torch-1.13.1-cp38-cp38-linux_aarch64.whl

# Torchvision
[ ! -d vision ] && git clone https://github.com/pytorch/vision.git
cd vision || exit
git reset --hard
git checkout v0.14.1
export BUILD_VERSION=0.14.1 FORCE_CUDA=1 CUDA_VISIBLE_DEVICES=0 CUDA_HOME='/usr/local/cuda' USE_CUDA=1 USE_CUDNN=1
python3 setup.py install --user
