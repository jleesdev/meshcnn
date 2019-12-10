#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/adni2_seg/750_balanced \
--name adni2_seg_750_balanced_ad_cn \
--arch meshunet \
--dataset_mode autoencoder \
--ncf 32 64 128 256 \
--ninput_edges 750 \
--pool_res 600 450 300 \
--resblocks 1 \
--batch_size 24 \
--lr 0.001 \
--num_aug 5 \
--slide_verts 0.2 \
--gpu_id 1 \
--niter 200 \
--niter_decay 100 \