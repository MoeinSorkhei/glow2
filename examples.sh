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
     python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --resume_train --last_optim_step 6000 --use_comet
     python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --use_comet



# ================ kill python processes
ps aux |grep python3 |awk '{print $2}' |xargs kill -9

# ================ cancel all jobs
scancel --user=${USER}



# ================ training glow
# 2. train glow (unconditional) on real images
python3 main.py --dataset cityscapes --model glow --bsize 10 --use_comet
# train glow on segmentations only
python3 main.py --dataset cityscapes --model glow --train_on_segment --bsize 10 --use_comet


# ================ training c_flow
# train c_flow on cityscapes
python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --use_comet

# training c_flow with sanity check (revision in sanity check may be needed)
python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --sanity_check


# 1. train c-flow with pre-trained left glow, unfreezing its layers so it is trainable as well
python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --left_pretrained \
                --left_lr 1e-4 --left_unfreeze --left_step 30000 \
                --lr 1e-5 --use_comet

# 3. train c-flow with pre-trained left glow, freezing its layers so it is not trainable
python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --left_pretrained \
                --left_lr 1e-4 --left_step 30000 \
                --lr 5e-5 --use_comet

# 4.
python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --left_pretrained \
                --left_lr 1e-4 --left_unfreeze --left_step 30000 \
                --lr 5e-5 --use_comet


# =========== REVISE THE ARGS FOR EXPERIMENTS
# ================ resume training (now throws NotImplementedError)
# =========== resume training (pay attention to the optim_step)
# python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --resume_train --last_optim_step 19000 --use_comet
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
# --dataset cityscapes --exp --sample_c_flow --syn_segs --last_optim_step 27000 --model c_flow --cond_mode segment --trial 5
