#!/usr/bin/env bash


# inference
conda activate workshop
python3 main.py --exp --infer_on_val --all_sampling_round --temp 1.0 \
                --model c_glow_v2 --dataset cityscapes --direction photo2label \
                --n_block 4 --n_flow 16 \
                --last_optim_step 136000 \
                --bsize 20


# evaluate fcn
conda activate workshop
python3 main.py --exp --eval_fcn --all_sampling_rounds --temp 1.0 \
                --model c_glow_v3 --dataset cityscapes --direction photo2label \
                --img_size 256 256 \
                --n_block 4 --n_flow 16 \
                --lr 1e-4 \
                --last_optim_step 136000


# eval ssim
python3 main.py --exp --eval_ssim --sampling_round 1 --temp 1.0 \
                --dataset cityscapes --direction label2photo \
                --model c_glow_v2 --n_block 4 --n_flow 16 \
                --last_optim_step 136000
