#!/usr/bin/env bash

python3 main.py --model improved_2 --dataset cityscapes --direction label2photo \
                --n_block 4 --n_flow 32 16 8 8 \
                --max_step 136000 \
                --use_comet


# with b_maps
python3 main.py --model improved_2_bmaps --dataset cityscapes --direction label2photo --use_bmaps \
                --n_block 4 --n_flow 32 16 8 8 \
                --max_step 136000 \
                --use_comet


# with b_maps and ceiling
python3 main.py --model improved_2_bmaps_ceil --dataset cityscapes --direction label2photo --use_bmaps --do_ceil \
                --n_block 4 --n_flow 32 16 8 8 \
                --max_step 136000 \
                --use_comet
