#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/adni2 \
--dataset_mode classification \
--ninput_edges 14100 \
--name adni2 \
--ncf 32 32 32 32 32 64 64 64 64 128 128 128 128 256 \
--pool_res 5400 3600 2400 1800 1500 1350 1200 1050 960 840 720 600 450 300 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--batch_size 12 \
--gpu_ids -1 \
