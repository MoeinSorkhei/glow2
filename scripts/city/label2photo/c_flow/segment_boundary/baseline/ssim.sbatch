#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --cpus-per-task 1
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user sorkhei@kth.se
#SBATCH --time 10:00:00
#SBATCH --output /Midgard/home/%u/glow2/logs/%J.out
#SBATCH --error  /Midgard/home/%u/glow2/logs/%J.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

# shellcheck disable=SC1090
. ~/miniconda3/etc/profile.d/conda.sh
conda activate caffe

# shellcheck disable=SC2164
cd glow2/src/

# baseline
python3 main.py --exp --eval_ssim --dataset cityscapes --model c_flow --cond_mode segment_boundary \
                --img_size 256 256 \
                --n_block 4 --n_flow 16 \
                --lr 1e-4 \
                --last_optim_step 128000
