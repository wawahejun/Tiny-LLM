#!/bin/bash

# 进入 cuda_kernels 目录
cd src/cuda_kernels

# 编译 CUDA 内核文件
nvcc -ptx matmul_transb_kernel.cu -o ../../matmul_kernel.ptx
nvcc -ptx rms_norm_kernel.cu -o ../../rms_norm_kernel.ptx
nvcc -ptx rope_kernel.cu -o ../../rope_kernel.ptx
