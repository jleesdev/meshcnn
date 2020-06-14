#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/adni2_clf/diff_ptid_4200_ad_cn_141 \
--dataset_mode classification \
--ninput_edges 4200 \
--name adni2_clf-diff_ptid_e4200_adcn141_lr0.0005_bs16_aug0.1-fcs_adam_l1reg5e-5_2dp0.3 \
--ncf 128 256 512 1024 \
--pool_res 2800 1400 700 560 \
--fc_n 256 64 \
--norm group \
--resblocks 1 \
--flip_edges 0.1 \
--slide_verts 0.1 \
--scale_verts true \
--lr 0.0005 \
--num_aug 5 \
--niter_decay 100 \
--batch_size 16 \
--gpu_ids 1 \
--niter 100 \
--save_epoch_freq 1 \
--optim Adam \
--reg_weight 0.00005 \
--dropout_p 0.3 \
