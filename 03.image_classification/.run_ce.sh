#!/bin/bash
#This file is only used for continuous evaluation.
export FLAGS_cudnn_deterministic=true
export CUDA_VISIBLE_DEVICES=0
python train.py --num_epochs 1 --use_gpu 1  --enable_ce | python _ce.py
 
