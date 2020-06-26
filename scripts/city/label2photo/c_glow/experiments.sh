#!/usr/bin/env bash

# random samples with specific temperature
python3 main.py --exp --random_samples --n_samples 20 --temp 0.7 \
                --dataset cityscapes --model c_glow --direction label2photo --cond_mode segment \
			          --img_size 256 256 \
			          --n_block 4 --n_flow 16 \
			          --lr 1e-4 \
			          --last_optim_step 34000

