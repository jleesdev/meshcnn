#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/adni2_1000 \
--dataset_mode classification \
--ninput_edges 1500 \
--name adni2_1000 \
--ncf 16 32 64 128 256 \
--pool_res 750 600 450 375 300 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--batch_size 12 \
--gpu_ids -1 \
