#!/usr/bin/env bash


# eval fcn
python3 main.py --exp --eval_fcn --all_sampling_rounds --temp 1.0 \
                --dataset cityscapes --direction label2photo \
                --model dual_glow_shallow --n_block 4 --n_flow 2 \
                --last_optim_step 30000


# eval ssim
python3 main.py --exp --eval_ssim --all_sampling_rounds --temp 1.0 \
                --dataset cityscapes --direction label2photo \
                --model dual_glow_shallow --n_block 4 --n_flow 2 \
                --last_optim_step 30000

