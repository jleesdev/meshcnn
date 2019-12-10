#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot datasets/adni2 \
--name adni2 \
--dataset_mode classification \
--ninput_edges 14100 \
--ncf 32 64 128 256 \
--pool_res 9600 7200 3600 1800 \
--norm group \
--resblocks 1 \
--export_folder meshes \
--gpu_id -1 \