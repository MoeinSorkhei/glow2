# shellcheck disable=SC2164
# cd src
# echo 'In [runs.sh]: Changed dir to the "src" directory'

# pkill -9 python
# echo 'In [runs.sh]: Killed all python processes'
# echo ''  # just a new line


# ================ salloc
salloc --gres=gpu:1 --mem=5GB --cpus-per-task=1 --constrain=gondor
# conda activate workshop

conda activate workshop

# ================ salloc
srun --gres=gpu:1 --mem=5GB --cpus-per-task=1 --time=16:00:00 --constrain=gondor \
     python3 main.py --dataset cityscapes --model c_flow --cond_mode segment --use_comet

# ================ kill python processes
ps aux |grep python3 |awk '{print $2}' |xargs kill -9