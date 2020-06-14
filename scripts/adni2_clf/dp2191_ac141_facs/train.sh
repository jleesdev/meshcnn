#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/adni2_clf/dp2191_ac141_facs/ \
--dataset_mode classification \
--ninput_edges 2191 \
--name _lr5e-3_bs16_adam_rw5e-4 \
--ncf 16 16 32 \
--pool_res 1200 720 450 \
--fc_n 32 \
--norm group \
--resblocks 1 \
--flip_edges 0 \
--slide_verts 0 \
--scale_verts false \
--lr 0.005 \
--num_aug 20 \
--niter_decay 100 \
--batch_size 16 \
--gpu_ids 1 \
--niter 100 \
--save_epoch_freq 1 \
--optim Adam \
--reg_weight 0.0005 \
--serial_batches false \
