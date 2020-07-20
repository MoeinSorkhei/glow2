#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --mem  20GB
#SBATCH --cpus-per-task 1
#SBATCH --constrain "belegost"
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user sorkhei@kth.se
#SBATCH --time 48:00:00
#SBATCH --output /Midgard/home/%u/glow2/logs/%J.out
#SBATCH --error  /Midgard/home/%u/glow2/logs/%J.err


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi

# shellcheck disable=SC1090
. ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2  # tf version 1

# Note: no need to change dir beforehand, just ssh moria and run this sbatch file: sbatch glow2/run.sbatch
# shellcheck disable=SC2164
cd glow2/src/

python3 main.py --dataset cityscapes --model dual_glow --direction label2photo --cond_mode segment \
                --img_size 256 256 --bsize 1 \
                --n_block 4 --n_flow 16 \
                --resume_train --last_optim_step 104000 --max_step 136000 \
                --use_comet --prev_exp_id 0ec3868bbbd24a7b902d970ee25c92fb
