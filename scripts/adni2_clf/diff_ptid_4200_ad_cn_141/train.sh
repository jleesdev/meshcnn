#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/adni2_clf/diff_ptid_4200_ad_cn_141 \
--dataset_mode classification \
--ninput_edges 4200 \
--name adni2_clf-diff_ptid_4200_ad_cn_141_lr_0.0005_bs_16_no_aug-fcs_adam \
--ncf 64 128 256 512 1024 \
--pool_res 2800 2100 1400 700 560 \
--fc_n 8192 1024 64 \
--norm group \
--resblocks 1 \
--flip_edges 0 \
--lr 0.0005 \
--num_aug 1 \
--niter_decay 100 \
--batch_size 16 \
--gpu_ids 1 \
--niter 100 \
--save_epoch_freq 1 \
--optim Adam \
