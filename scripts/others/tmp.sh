python3 main.py --exp --eval_ssim --dataset cityscapes --model c_flow --cond_mode segment \
                --act_conditional \
                --img_size 256 256 \
                --n_block 4 --n_flow 16 \
                --lr 1e-4 \
                --last_optim_step 133000