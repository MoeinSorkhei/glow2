# ================ random samples for segment
python3 main.py --exp --random_samples --n_samples 20 \
                --dataset cityscapes --model c_flow --cond_mode segment \
			          --img_size 256 512 \
			          --n_block 4 --n_flow 14 \
			          --lr 1e-4 \
			          --last_optim_step 277000



# ================ random samples for segment_boundary
python3 main.py --exp --random_samples --n_samples 20 \
                --dataset cityscapes --model c_flow --cond_mode segment_boundary \
			          --img_size 256 512 \
			          --n_block 4 --n_flow 14 \
			          --lr 1e-4 \
			          --last_optim_step 178000





# ================ new condition for segment
python3 main.py --exp --new_condition \
                --dataset cityscapes --model c_flow --cond_mode segment \
			          --img_size 256 512 \
			          --n_block 4 --n_flow 14 \
			          --lr 1e-4 \
			          --last_optim_step 277000



# ================ new condition for segment_boundary
python3 main.py --exp --new_condition \
                --dataset cityscapes --model c_flow --cond_mode segment_boundary \
			          --img_size 256 512 \
			          --n_block 4 --n_flow 14 \
			          --lr 1e-4 \
			          --last_optim_step 178000
