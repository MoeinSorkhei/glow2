#!/usr/bin/env bash

# test invertibility
python3 main.py --exp --test_invertibility --dataset cityscapes --direction label2photo \
                --img_size 256 256 \
                --model improved_2 \
                --n_block 4 --n_flow 16 12 10 10

