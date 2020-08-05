#!/usr/bin/env bash

python3 main.py --model improved_1 --dataset cityscapes --direction label2photo \
                --n_block 4 --n_flow 16 16 16 16 \
                --max_step 136000 \
                --use_comet
