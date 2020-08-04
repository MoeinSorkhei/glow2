#!/usr/bin/env bash
# ================ sanity and invertibility checks

cd glow2/src

# glow
echo "============================== Invertibility  Check for Glow =============================="
python3 main.py --exp --test_invertibility --dataset cityscapes --model glow \
                --img_size 256 256 \
                --n_block 4 --n_flow 10



# c_flow with segment
echo "============================== Invertibility  Check for C-Flow with segment =============================="
python3 main.py --exp --test_invertibility --dataset cityscapes --model improved_1 --direction label2photo \
                --img_size 256 256 \
                --n_block 4 --n_flow 16 12 10 10



# c_flow with segment_boundary
echo "============================== Invertibility  Check for C-Flow with segment_boundary =============================="
python3 main.py --exp --test_invertibility --dataset cityscapes --model c_flow --direction label2photo --cond_mode segment_boundary \
                --act_conditional --w_conditional --coupling_cond_net \
                --img_size 256 256 \
                --n_block 4 --n_flow 10




# c_flow with for photo2label
echo "============================== Invertibility  Check for C-Flow for phot2label =============================="
python3 main.py --exp --test_invertibility --dataset cityscapes --model c_flow --direction photo2label \
                --act_conditional --w_conditional --coupling_cond_net \
                --img_size 256 256 \
                --n_block 4 --n_flow 10


# c_flow for transient dataset daylight2night
echo "============================== Invertibility  Check for C-Flow for transient daylight2night =============================="
python3 main.py --exp --test_invertibility --dataset transient --model c_flow --direction daylight2night \
                --act_conditional --w_conditional --coupling_cond_net \
                --img_size 256 256 \
                --n_block 4 --n_flow 10
