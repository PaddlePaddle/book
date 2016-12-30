#!/bin/bash
set -e

paddle train \
       --config ngram.py \
       --use_gpu=1 \
       --dot_period=100 \
       --log_period=3000 \
       --test_period=0 \
       --save_dir=model \
       --num_passes=30
