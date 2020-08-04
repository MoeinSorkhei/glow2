#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --mem  15GB
#SBATCH --cpus-per-task 1
#SBATCH --constrain "shire|rivendell"  # khazadum does not have tf data
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user sorkhei@kth.se
#SBATCH --time 10:00:00
#SBATCH --output /Midgard/home/%u/glow2/logs/%J.out
#SBATCH --error  /Midgard/home/%u/glow2/logs/%J.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi

. ~/miniconda3/etc/profile.d/conda.sh
# conda activate caffe  # for label2photo fcn evaluation
# conda activate tf2  # for dual-glow inference
conda activate workshop  # for all other cases

cd glow2/src/

python3 main.py --exp --eval_ssim --all_sampling_rounds --temp 1.0 \
                --dataset cityscapes --direction photo2label \
                --model c_flow --w_conditional --act_conditional --coupling_cond_net \
                --n_block 4 --n_flow 16 \
                --last_optim_step 136000
