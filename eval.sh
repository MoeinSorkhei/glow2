#!/usr/bin/env bash

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
# shellcheck disable=SC1090
. ~/miniconda3/etc/profile.d/conda.sh
conda activate caffe  # caffe environment which has caffe-gpu and scipy=1.0.0

cd glow2/src/

# from scratch, lr=1e-5 -- CHECKED
# echo "running the first"
python3 main.py --eval_complete --dataset cityscapes --model c_flow --cond_mode segment --lr 1e-5 \
                --last_optim_step 26000 --bsize 20
# echo "running the first done"
sleep 5m # Waits 5 minutes.


# from scratch, lr=1e-4 -- CHECKED
python3 main.py --eval_complete --dataset cityscapes --model c_flow --cond_mode segment --lr 1e-4 \
                --last_optim_step 28000 --bsize 20
sleep 5m # Waits 5 minutes.


# pre-trained freezed, left_step=10K, lr=1e-5 -- CHECKED
python3 main.py --eval_complete --dataset cityscapes --model c_flow --cond_mode segment --left_pretrained \
                --left_lr 1e-4 --left_step 10000 --lr 1e-5 --last_optim_step 26000 --bsize 20
sleep 5m # Waits 5 minutes.


# pre-trained freezed, left_step=30K, lr=5e-5, step=13000 -- CHECKED
python3 main.py --eval_complete --dataset cityscapes --model c_flow --cond_mode segment --left_pretrained \
                --left_lr 1e-4 --left_step 30000 --lr 5e-5 --last_optim_step 13000 --bsize 20
sleep 5m # Waits 5 minutes.


# freezed, left_step=30K, lr=5e-5, step=14000 -- CHECKED
python3 main.py --eval_complete --dataset cityscapes --model c_flow --cond_mode segment --left_pretrained \
                --left_lr 1e-4 --left_step 30000 --lr 5e-5 --last_optim_step 14000 --bsize 20
sleep 5m # Waits 5 minutes.


# unfreezed, left_step=30K, lr=5e-5, step=19000 -- CHECKED
python3 main.py --eval_complete --dataset cityscapes --model c_flow --cond_mode segment --left_pretrained \
                --left_unfreeze --left_lr 1e-4 --left_step 30000 --lr 5e-5 --last_optim_step 19000 --bsize 20
sleep 5m # Waits 5 minutes.


# unfreezed, left_step=30K, lr=1e-5, step=9000 -- CHECKED
python3 main.py --eval_complete --dataset cityscapes --model c_flow --cond_mode segment --left_pretrained \
                --left_unfreeze --left_lr 1e-4 --left_step 30000 --lr 1e-5 --last_optim_step 9000 --bsize 20
sleep 5m # Waits 5 minutes.


# unfreezed, left_step=30K, lr=1e-5, step=10000 -- CHECKED
python3 main.py --eval_complete --dataset cityscapes --model c_flow --cond_mode segment --left_pretrained \
                --left_unfreeze --left_lr 1e-4 --left_step 30000 --lr 1e-5 --last_optim_step 10000 --bsize 20
sleep 5m # Waits 5 minutes.
