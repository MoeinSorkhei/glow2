#!/usr/bin/env bash

python3 main.py --model improved_1 --dataset cityscapes --direction label2photo \
                --n_block 4 --n_flow 16 16 16 16 \
                --max_step 136000 \
                --use_comet



# with b_maps
python3 main.py --model improved_1_bmaps --dataset cityscapes --direction label2photo --use_bmaps \
                --n_block 4 --n_flow 16 16 16 16 \
                --resume_train --last_optim_step 11000 \
                --max_step 136000 \
                --use_comet  --prev_exp_id 0640daf9685d4d3fb0ffbc3d552bac3f



# with b_maps and ceiling
python3 main.py --model improved_1_bmaps_ceil --dataset cityscapes --direction label2photo --use_bmaps --do_ceil \
                --n_block 4 --n_flow 16 16 16 16 \
                --resume_train --last_optim_step 11000 \
                --max_step 136000 \
                --use_comet --prev_exp_id 50850c6ff5424e9d8dbf0521f6ecf880