#!/bin/bash
#This file is only used for continuous evaluation.
export FLAGS_cudnn_deterministic=true
export CUDA_VISIBLE_DEVICES=0
python train_conv.py --use_gpu 1 --num_epochs=1  --enable_ce | python _ce.py
python train_dyn_rnn.py --use_gpu 1 --num_epochs=1  --enable_ce | python _ce.py 
python train_stacked_lstm.py --use_gpu 1 --num_epochs=1  --enable_ce | python _ce.py

 
