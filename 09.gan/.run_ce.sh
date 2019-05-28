#!/bin/bash
#This file is only used for continuous evaluation.
export FLAGS_cudnn_deterministic=True
export CUDA_VISIBLE_DEVICES=0
python dc_gan.py --enable_ce true --epoch 1  --use_gpu True | python _ce.py
 
