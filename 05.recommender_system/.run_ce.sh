#!/bin/bash
#This file is only used for continuous evaluation.
python train.py --enable_ce | python _ce.py
 
