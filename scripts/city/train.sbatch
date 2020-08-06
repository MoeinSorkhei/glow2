#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --mem  15GB
#SBATCH --cpus-per-task 1
#SBATCH --constrain "shire|rivendell|smaug|balrog"  # khazadum does not have tf data
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user sorkhei@kth.se
#SBATCH --time 48:00:00
#SBATCH --output /Midgard/home/%u/glow2/logs/%J.out
#SBATCH --error  /Midgard/home/%u/glow2/logs/%J.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi

. ~/miniconda3/etc/profile.d/conda.sh
# conda activate caffe  # for label2photo fcn evaluation
# conda activate tf2  # for dual-glow inference
conda activate workshop  # for all other cases

cd glow2/src/


python3 main.py --model improved_2_bmaps_ceil --dataset cityscapes --direction label2photo --use_bmaps --do_ceil \
                --n_block 4 --n_flow 32 16 8 8 \
                --resume_train --last_optim_step 10000 \
                --max_step 136000 \
                --use_comet --prev_exp_id c0d89b441b274c11baf939b897f231a7
