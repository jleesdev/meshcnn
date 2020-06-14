#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/adni2_clf/dp2191_ac141_fac/ \
--dataset_mode classification \
--ninput_edges 2190 \
--name _lr5e-3_bs16_aug0.2_adam \
--ncf 16 32 64 128 \
--pool_res 1200 720 450 300 \
--fc_n 32 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--scale_verts true \
--lr 0.005 \
--num_aug 20 \
--niter_decay 100 \
--batch_size 16 \
--gpu_ids 1 \
--niter 100 \
--save_epoch_freq 1 \
--optim Adam \
--reg_weight 0 \
--serial_batches false \
