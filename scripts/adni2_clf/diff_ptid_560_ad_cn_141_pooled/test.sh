#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot datasets/adni2_clf/diff_ptid_560_ad_cn_141_pooled \
--name adni2_clf-diff_ptid_560_ad_cn_141_pooled_lr_0.0005_bs_16_no_aug-fcs_adam \
--ncf 128 256 512 1024 \
--pool_res 560 450 300 150 \
--fc_n 256 16 \
--norm group \
--resblocks 1 \
--export_folder meshes \
--gpu_ids 1 \
--ninput_edges 560 \
--phase train \