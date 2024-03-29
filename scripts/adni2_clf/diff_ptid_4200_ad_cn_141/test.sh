#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot datasets/adni2_clf/diff_ptid_4200_ad_cn_141 \
--name adni2_clf-diff_ptid_e4200_adcn141_lr0.0005_bs16_aug0.1-fcs_adam_l1reg5e-5_2dp0.3 \
--ncf 128 256 512 1024 \
--pool_res 2800 1400 700 560 \
--fc_n 256 64 \
--norm group \
--resblocks 1 \
--export_folder meshes \
--gpu_ids 1 \
--ninput_edges 4200 \
--phase train \