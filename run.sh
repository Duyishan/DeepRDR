#!/bin/bash
python train.py \
    --seed 123 \
    --task 'test1' \
    --fold 1 \
    --batch_size 32 \
    --epoch_num 10000 \
    --h_dim 512 \
    --pretrain '/home/mnt/data/x/r11/save_model/train_task=test1_seed=123_batch=64_epoch=1217_param.pkl'