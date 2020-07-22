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

# should be ready
python3 main.py --exp --eval_complete --dataset cityscapes --model c_flow --cond_mode segment_boundary \
                --w_conditional --act_conditional \
                --img_size 256 256 \
                --n_block 4 --n_flow 16 \
                --lr 1e-4 \
                --last_optim_step 130000 \
                --bsize 20