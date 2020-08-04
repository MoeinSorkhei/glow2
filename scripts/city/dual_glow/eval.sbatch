#!/usr/bin/env bash

# inference
conda activate tf2
python3 main.py --exp --infer_on_val --all_sampling_rounds --temp 1.0 \
                --model dual_glow --dataset cityscapes --direction label2photo \
                --n_block 4 --n_flow 16 \
                --last_optim_step 136000 \
                --bsize 20


# evaluate fcn
conda activate caffe
python3 main.py --exp --eval_fcn --all_sampling_rounds --temp 1.0 \
                --model dual_glow --dataset cityscapes --direction label2photo \
                --n_block 4 --n_flow 16 \
                --last_optim_step 136000


# eval ssim
conda activate workshop
python3 main.py --exp --eval_ssim --all_sampling_rounds --temp 1.0 \
                --dataset cityscapes --direction label2photo \
                --model dual_glow --n_block 4 --n_flow 16 \
                --last_optim_step 136000

