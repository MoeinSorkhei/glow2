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