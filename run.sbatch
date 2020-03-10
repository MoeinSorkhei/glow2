#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --mem  5GB
#SBATCH --cpus-per-task 1
#SBATCH --constrain "shire"
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user sorkhei@kth.se
#SBATCH --time 16:00:00
#SBATCH --output /Midgard/home/%u/glow2/logs/%J.out
#SBATCH --error  /Midgard/home/%u/glow2/logs/%J.err


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate workshop

cd glow2/src/
# ps aux |grep python3 |awk '{print $2}' |xargs kill -9
python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --resume_train --last_optim_step 7000 --use_comet