#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --mem  15GB
#SBATCH --cpus-per-task 1
#SBATCH --constrain "shire|rivendell|khazadum"
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user sorkhei@kth.se
#SBATCH --time 10:00:00
#SBATCH --output /Midgard/home/%u/glow2/logs/%J.out
#SBATCH --error  /Midgard/home/%u/glow2/logs/%J.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi

# shellcheck disable=SC1090
. ~/miniconda3/etc/profile.d/conda.sh
conda activate caffe

# shellcheck disable=SC2164
cd glow2/src/

# NOTE: No need to resize if it was already 256x256
# pre-trained freezed, left_step=10K, lr=1e-5 -- CHECKED
python3 main.py --eval_complete --dataset cityscapes --model c_flow --cond_mode segment \
                --img_size 256 512 \
                --n_block 4 --n_flow 14 \
                --left_pretrained --left_lr 1e-4 --left_step 136000 \
                --lr 1e-4 \
                --last_optim_step 103000 \
                --bsize 20