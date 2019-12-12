#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot datasets/adni2_clf/diff_ptid_4200_ad_cn_141 \
--dataset_mode autoencoder \
--arch meshae \
--ninput_edges 4200 \
--name adni2_ae-diff_ptid_4200_ad_cn_141_lr_0.001_bs_16_no_aug-fcs_adam_chamfer \
--ncf 128 256 512 1024 \
--pool_res 2100 1400 700 560 \
--fc_n 128 512 \
--norm group \
--resblocks 1 \
--export_folder meshes \
--gpu_ids 1 \
--batch_size 5 \