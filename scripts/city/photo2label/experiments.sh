#!/usr/bin/env bash


# ============ inference on validation set (with a specific temperature)
python3 main.py --exp --infer_on_val --temp 0.7 \
                --dataset cityscapes --model c_flow --direction photo2label \
                --act_conditional --w_conditional --coupling_cond_net \
                --img_size 256 256 \
                --n_block 4 --n_flow 16 \
                --lr 1e-4 \
                --last_optim_step 83000



# ============ evaluate images on the validation set (with a specific temperature)
python3 main.py --exp --evaluate --temp 0.7 \
                --dataset cityscapes --model c_flow --direction photo2label \
                --act_conditional --w_conditional --coupling_cond_net \
                --img_size 256 256 \
                --n_block 4 --n_flow 16 \
                --lr 1e-4 \
                --last_optim_step 83000


# ============ evaluate images on the validation set (with all temperatures)
python3 main.py --exp --eval_complete \
                --dataset cityscapes --model c_flow --direction photo2label \
                --act_conditional --w_conditional --coupling_cond_net \
                --img_size 256 256 \
                --n_block 4 --n_flow 16 \
                --lr 1e-4 \
                --last_optim_step 83000