#!/usr/bin/env bash

# test invertibility
python3 main.py --exp --test_invertibility --dataset cityscapes --direction label2photo \
                --img_size 256 256 \
                --model improved_1_b_maps --use_bmaps \
                --n_block 4 --n_flow 2 2 2 2
