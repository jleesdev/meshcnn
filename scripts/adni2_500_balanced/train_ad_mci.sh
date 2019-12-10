#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/adni2_500_balanced_ad_mci \
--dataset_mode classification \
--ninput_edges 750 \
--name adni2_500_balanced_ad_mci \
--ncf 64 128 256 256 \
--pool_res 600 480 360 300 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 4 \
--niter_decay 100 \
--batch_size 16 \
--gpu_ids 0 \
