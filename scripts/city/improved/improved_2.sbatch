#!/usr/bin/env bash

python3 main.py --model improved_2 --dataset cityscapes --direction label2photo \
                --n_block 4 --n_flow 32 16 8 8 \
                --resume_train --last_optim_step 1000 \
                --max_step 136000 \
                --use_comet
