#!/usr/bin/env bash

# inference
python3 main.py --exp --infer_on_val --all_sampling_rounds --temp 1.0 \
                --dataset cityscapes --direction label2photo --use_bmaps \
                --model c_flow --w_conditional --act_conditional --coupling_cond_net \
                --n_block 4 --n_flow 16 \
                --last_optim_step 136000 \
                --bsize 20



python3 main.py --exp --infer_on_val --all_sampling_rounds --temp 1.0 \
                --dataset cityscapes --direction label2photo \
                --model c_flow --w_conditional \
                --n_block 4 --n_flow 16 \
                --last_optim_step 136000 \
                --bsize 20



python3 main.py --exp --infer_on_val --all_sampling_rounds --temp 1.0 \
                --dataset cityscapes --direction photo2label \
                --model c_flow --coupling_cond_net \
                --n_block 4 --n_flow 16 \
                --last_optim_step 10000 \
                --bsize 20