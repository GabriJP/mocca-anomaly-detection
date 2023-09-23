#!/usr/bin/env bash

pip3 uninstall -y torch torchvision

TORCH_RELEASE="2.0.1"
VISION_RELEASE="0.15.2"

sudo apt install -y clang-8 python3-pip cmake libopenblas-dev libopenmpi-dev
sudo ln -s /usr/bin/clang-8 /usr/bin/clang
sudo ln -s /usr/bin/clang++-8 /usr/bin/clang++

# Torch
[ ! -d pytorch ] && git clone "https://github.com/pytorch/pytorch.git"
cd pytorch || exit
git reset --hard
git pull
git submodule sync
git submodule update --init --recursive
git checkout "v${TORCH_RELEASE}"
export CC=clang CXX=clang++ USE_NCCL=0 USE_DISTRIBUTED=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 TORCH_CUDA_ARCH_LIST="7.2;8.7" PYTORCH_BUILD_VERSION=${TORCH_RELEASE} PYTORCH_BUILD_NUMBER=1
export PATH=/usr/lib/ccache:$PATH
#nano torch/utils/cpp_extension.py -> https://gist.github.com/dusty-nv/ce51796085178e1f38e3c6a1663a93a1#file-pytorch-1-11-jetpack-5-0-patch
#make triton
pip3 install -r requirements.txt
pip3 install scikit-build
pip3 install ninja
python3 setup.py bdist_wheel
pip3 install --user -U "dist/torch-${TORCH_RELEASE}-cp38-cp38-linux_aarch64.whl"

# Torchvision
[ ! -d vision ] && git clone "https://github.com/pytorch/vision.git"
cd vision || exit
git reset --hard
git pull
git checkout "v${VISION_RELEASE}"
export BUILD_VERSION=${VISION_RELEASE} FORCE_CUDA=1 CUDA_VISIBLE_DEVICES=0 CUDA_HOME='/usr/local/cuda' USE_CUDA=1 USE_CUDNN=1
python3 setup.py bdist_wheel
pip3 install --user -U "dist/torchvision-${TORCH_RELEASE}-cp38-cp38-linux_aarch64.whl"
