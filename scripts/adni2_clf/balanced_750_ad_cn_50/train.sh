#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/adni2_clf/balanced_750_ad_cn_50 \
--dataset_mode classification \
--ninput_edges 750 \
--name adni2_clf-balanced_750_ad_cn_50 \
--ncf 64 128 256 256 \
--pool_res 600 450 300 150 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 1 \
--niter_decay 200 \
--batch_size 12 \
--gpu_ids 1 \
--niter 300 \
--save_epoch_freq 1 \
