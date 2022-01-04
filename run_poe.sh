#!/bin/bash

python train.py --batch_size=1024 --data_path='./data/Multi_2dim_log_spiral' --save_dir='./output/1230/' --device='1' --epochs=50 --wandb --Foldstart=0 --Foldend=8