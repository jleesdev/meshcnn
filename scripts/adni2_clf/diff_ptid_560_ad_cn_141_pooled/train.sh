#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/adni2_clf/diff_ptid_560_ad_cn_141_pooled \
--dataset_mode classification \
--ninput_edges 560 \
--name adni2_clf-diff_ptid_560_ad_cn_141_pooled_lr_0.0005_bs_16_no_aug-fcs_adam_l1reg \
--ncf 128 256 512 1024 \
--pool_res 560 450 300 150 \
--fc_n 256 16 \
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
--reg_weight 0.00005 \