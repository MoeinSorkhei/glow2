#!/usr/bin/env bash

# ========= Running SSIM evaluation on all the models at once
# This script should be run from ouside of glow2 folder (home directory of Midgard)

# shellcheck disable=SC2164
cd glow2/src/

# ======= segment condition
# baseline
python3 main.py --exp --eval_ssim --dataset cityscapes --model c_flow --cond_mode segment \
                --img_size 256 256 \
                --n_block 4 --n_flow 16 \
                --lr 1e-4 \
                --last_optim_step 128000


# act conditional
python3 main.py --exp --eval_ssim --dataset cityscapes --model c_flow --cond_mode segment \
                --act_conditional \
                --img_size 256 256 \
                --n_block 4 --n_flow 16 \
                --lr 1e-4 \
                --last_optim_step 133000


# w conditional
python3 main.py --exp --eval_ssim --dataset cityscapes --model c_flow --cond_mode segment \
                --w_conditional \
                --img_size 256 256 \
                --n_block 4 --n_flow 16 \
                --lr 1e-4 \
                --last_optim_step 131000

# w + act conditional
python3 main.py --exp --eval_ssim --dataset cityscapes --model c_flow --cond_mode segment \
                --w_conditional --act_conditional \
                --img_size 256 256 \
                --n_block 4 --n_flow 16 \
                --lr 1e-4 \
                --last_optim_step 135000


# w + act conditional + coupling net
python3 main.py --exp --eval_ssim --dataset cityscapes --model c_flow --cond_mode segment \
                --w_conditional --act_conditional --coupling_cond_net \
                --img_size 256 256 \
                --n_block 4 --n_flow 16 \
                --lr 1e-4 \
                --last_optim_step 130000



# ======= segment_boundary condition
# add if else for segment and segment_boundary