#!/bin/bash

# shellcheck disable=SC2164
# cd src
# echo 'In [runs.sh]: Changed dir to the "src" directory'

# pkill -9 python
# echo 'In [runs.sh]: Killed all python processes'
# echo ''  # just a new line


# ================ salloc
salloc --gres=gpu:1 --mem=5GB --cpus-per-task=1 --constrain=shire
# conda activate workshop

# first screen, then activate environment
screen
conda activate workshop

# ================ srun
srun --gres=gpu:1 --mem=5GB --cpus-per-task=1 --time=16:00:00 --constrain=shire \
     python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --resume_train --last_optim_step 6000 \
     --use_comet \
     python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --use_comet



# ================ kill all python processes
ps aux |grep python3 |awk '{print $2}' |xargs kill -9

# ================ cancel all jobs
scancel --user=${USER}


# ================================ transferring datasets to compute nodes
# on the compute node
mkdir -p /local_storage/datasets/moein
cp -a /Midgard/home/sorkhei/glow2/data/cityscapes/ /local_storage/datasets/moein/
cd /local_storage/datasets/moein/cityscapes/
unzip -q gtFine_trainvaltest.zip -d gtFine_trainvaltest
unzip -q leftImg8bit_trainvaltest.zip -d leftImg8bit_trainvaltest
rm gtFine_trainvaltest.zip
rm leftImg8bit_trainvaltest.zip


# moving boundaries from Midgard to compute node
rsync -Pav /Midgard/Data/moein/cityscapes/boundaries/ /local_storage/datasets/moein/cityscapes/gtFine_trainvaltest/


# ================ training glow
# train glow (unconditional) on real images
python3 main.py --dataset cityscapes --model glow --bsize 10 --lr 1e-4 --use_comet

# train glow (unconditional) on real images with larger images
python3 main.py --dataset cityscapes --model glow --img_size 128 256 --bsize 1 --lr 1e-4 --use_comet

# train glow on segmentations only
python3 main.py --dataset cityscapes --model glow --train_on_segment \
                --img_size 256 512 --bsize 1 \
                --lr 1e-4 \
                --use_comet

# train glow on segmentations only with larger images
python3 main.py --dataset cityscapes --model glow --train_on_segment --img_size 128 256 --bsize 1 --lr 1e-4 --use_comet



# ================================ training c_flow
#  train c-flow with freezed pre-trained left glow
python3 main.py --dataset cityscapes --model c_flow --cond_mode segment \
                --img_size 256 512 --bsize 1 \
                --n_block 4 --n_flow 14 \
                --left_pretrained --left_lr 1e-4 --left_step 136000 \
                --lr 1e-4 \
                --resume_train --last_optim_step 106000 \
                --use_comet



# train c-flow with unfreezed pre-trained left glow
python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --left_pretrained \
                --left_lr 1e-4 --left_unfreeze --left_step 30000 \
                --lr 1e-5 --use_comet


# training c_flow with sanity check (revision in sanity check may be needed)
python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --lr 1e-4 --sanity_check




# =====================================================================================================================
# ================================ inference on validation set
# infer for c_flow (default temperature)
python3 main.py --infer_on_val --dataset cityscapes --model c_flow --cond_mode segment --lr 5e-5 \
                --last_optim_step 19000


# infer for c_flow with non-default temperature - could be with larger batch size since it is evaluation (not tried)
python3 main.py --infer_on_val --dataset cityscapes --model c_flow --cond_mode segment --lr 5e-5 \
                --last_optim_step 19000 --temp 0



# ================================ resizing the generated images (after inferring in validation set)
# resize for original images
python3 main.py --resize_for_fcn --gt --dataset cityscapes

# resize for c_flow
python3 main.py --resize_for_fcn --dataset cityscapes --model c_flow --cond_mode segment --lr 5e-5 \
                --last_optim_step 19000

# resize for c_flow with different temperature
python3 main.py --resize_for_fcn --dataset cityscapes --model c_flow --cond_mode segment --lr 5e-5 \
                --last_optim_step 19000 --temp 0



# ================================ evaluating images
# evaluate for ground-truth data
python3 main.py --evaluate --gt --dataset cityscapes



# evaluate for c_flow
python3 main.py --evaluate --dataset cityscapes --model c_flow --cond_mode segment --lr 5e-5 \
                --last_optim_step 19000


# evaluate for c_flow with different temperature
python3 main.py --evaluate --dataset cityscapes --model c_flow --cond_mode segment --lr 5e-5 \
                --last_optim_step 19000 --temp 0



python3 main.py --eval_complete --dataset cityscapes --model c_flow --cond_mode segment --lr 1e-5 \
        --last_optim_step 26000 --bsize 20
# python3 main.py --infer_on_val --dataset cityscapes --model c_flow --cond_mode segment --lr 1e-5 --last_optim_step 26000




# create boundary maps (needs huge memory: 50GB)
python3 main.py --dataset cityscapes --create_boundaries


# =========== REVISE THE ARGS FOR EXPERIMENTS
# ================ resume training (now throws NotImplementedError)
# =========== resume training (pay attention to the optim_step)
# python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --resume_train --last_optim_step 19000
# --use_comet
# --dataset mnist --resume_train --last_optim_step 21000 --use_comet
# --dataset cityscapes --model c_flow --cond_mode segment --resume_train --last_optim_step 6000

# ================ interpolation
# --dataset mnist --exp --interp --last_optim_step 12000

# ================  new conditioning
# --dataset mnist --exp --new_cond --last_optim_step 12000

# ================  resampling
# --dataset mnist --exp --resample --last_optim_step 12000

# ================  sample trials conditional on real segmentations (with c_flow)
# --dataset cityscapes --exp --sample_c_flow --conditional --last_optim_step 27000 \
# --model c_flow --cond_mode segment --trials 5


# ================  synthesize segmentations (with c_flow)
# --dataset cityscapes --exp --sample_c_flow --syn_segs --last_optim_step 27000 --model c_flow --cond_mode segment
# --trial 5


# ================ evaluation of validation bits per dimension
python3 main.py --compute_val_bpd --dataset cityscapes --model c_flow --cond_mode segment \
                --img_size 256 512 \
                --n_block 4 --n_flow 14 \
                --lr 1e-4 \
                --last_optim_step 67000 \
                --bsize 1  # important