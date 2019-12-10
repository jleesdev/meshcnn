#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot datasets/adni2_clf/diff_ptid_4200_ad_cn_141 \
--name adni2_clf-diff_ptid_4200_ad_cn_141_lr_0.0005_bs_16_no_aug-fcs \
--ncf 64 128 256 256 256 \
--pool_res 2800 2100 1400 700 560 \
--fc_n 8192 1024 64 \
--norm group \
--resblocks 1 \
--export_folder meshes \
--gpu_ids 1 \
--ninput_edges 4200 \