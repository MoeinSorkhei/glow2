# shellcheck disable=SC2164
cd src
echo 'In [runs.sh]: Changed dir to the "src" directory'

pkill -9 python
echo 'In [runs.sh]: Killed all python processes'

python3 main.py --dataset cityscapes --cond_mode segment_id --use_comet

echo ''  # just a new line