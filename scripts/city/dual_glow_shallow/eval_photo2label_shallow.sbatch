#!/usr/bin/env bash

# infer
conda activate tf2
python3 main.py --exp --infer_on_val --sampling_round 1 --temp 1.0 \
                --model dual_glow_shallow --dataset cityscapes --direction photo2label \
                --img_size 256 256 --bsize 1 \
                --n_block 4 --n_flow 2 \
                --lr 1e-4 \
                --last_optim_step 1000


# evaluate
python3 main.py --exp --eval_fcn --sampling_round 1 --temp 1.0 \
                --model dual_glow_shallow --dataset cityscapes --direction photo2label \
                --img_size 256 256 --bsize 1 \
                --n_block 4 --n_flow 2 \
                --lr 1e-4 \
                --last_optim_step 1000
