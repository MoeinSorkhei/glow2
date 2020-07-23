python3 main.py --clean_midgard --dataset cityscapes --model c_glow_v2 --direction label2photo --cond_mode segment
python3 main.py --clean_midgard --dataset cityscapes --model c_glow_v2 --direction photo2label

python3 main.py --clean_midgard --dataset cityscapes --model c_glow_v3 --direction label2photo --cond_mode segment

python3 main.py --clean_midgard --dataset maps --model c_flow --direction map2photo
python3 main.py --clean_midgard --dataset maps --model c_flow --direction photo2map