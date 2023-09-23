#!/usr/bin/env bash

pip3 uninstall -y torch torchvision

TORCH_RELEASE="2.0.1"
VISION_RELEASE="0.15.2"

sudo apt install -y clang python3-pip cmake libopenblas-dev libopenmpi-dev

# Torch
[ ! -d pytorch ] && git clone "https://github.com/pytorch/pytorch.git"
cd pytorch || exit
git reset --hard
git checkout main
git pull
git submodule sync
git submodule update --init --recursive
git checkout "v${TORCH_RELEASE}"
export BUILD_CAFFE2_OPS=OFF BUILD_TEST=OFF CC=clang CUDACXX=/usr/local/cuda/bin/nvcc CXX=clang++ USE_CUDA=ON
export USE_CUDNN=ON USE_DISTRIBUTED=OFF USE_FBGEMM=OFF USE_FAKELOWP=OFF USE_MKLDNN=OFF USE_NCCL=OFF USE_NNPACK=OFF
export USE_OPENCV=OFF USE_PYTORCH_QNNPACK=OFF USE_QNNPACK=OFF USE_SYSTEM_NCCL=OFF USE_XNNPACK=OFF
export PYTORCH_BUILD_VERSION=${TORCH_RELEASE} PYTORCH_BUILD_NUMBER=1 TORCH_CUDA_ARCH_LIST="7.2;8.7"
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
