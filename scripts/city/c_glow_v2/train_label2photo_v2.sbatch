#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --mem  10GB
#SBATCH --cpus-per-task 1
#SBATCH --constrain "shire|rivendell|khazadum|belegost"
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user sorkhei@kth.se
#SBATCH --time 48:00:00
#SBATCH --output /Midgard/home/%u/glow2/logs/%J.out
#SBATCH --error  /Midgard/home/%u/glow2/logs/%J.err


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
# shellcheck disable=SC1090
. ~/miniconda3/etc/profile.d/conda.sh
conda activate workshop

# Note: no need to change dir beforehand, just ssh moria and run this sbatch file: sbatch glow2/run.sbatch
# shellcheck disable=SC2164
cd glow2/src/

python3 main.py --dataset cityscapes --model c_glow_v2 --direction label2photo \
                --use_comet
                # --resume_train --last_optim_step 114000  --max_step 136000 \
                # --use_comet --prev_exp_id bd4c274b2c74437889973c5264c81ab4
