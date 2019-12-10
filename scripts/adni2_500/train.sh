#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/adni2_500 \
--dataset_mode classification \
--ninput_edges 750 \
--name adni2_500_v2 \
--ncf 64 128 256 128 \
--pool_res 600 450 300 270 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 1 \
--niter_decay 100 \
--batch_size 8 \
--gpu_ids -1 \
