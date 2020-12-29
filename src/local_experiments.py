import helper
from helper import *
import models
import os
import shutil
import numpy as np
from PIL import Image
import experiments
from torchvision import utils
import argparse


def transfer(args, params):
    content_basepath = '../data/cityscapes_complete_downsampled/all_reals'
    cond_basepath = '../data/cityscapes_complete_downsampled/all_segments'

    # pure_content = 'jena_000011_000019'
    # pure_new_cond = 'jena_000066_000019'

    # pure_content = 'aachen_000028_000019'
    # pure_new_cond = 'jena_000011_000019'

    # pure_content = 'jena_000011_000019'
    # pure_new_cond = 'aachen_000010_000019'

    pure_content = 'aachen_000034_000019'
    pure_new_cond = 'bochum_000000_016260'

    content = f'{content_basepath}/{pure_content}.png'  # content image
    condition = f'{cond_basepath}/{pure_content}.png'   # corresponding condition needed to extract z
    new_cond = f'{cond_basepath}/{pure_new_cond}.png'   # new condition

    save_basepath = '../samples/content_transfer_local'
    helper.make_dir_if_not_exists(save_basepath)
    file_path = f'{save_basepath}/content={pure_content}_condition={pure_new_cond}.png'
    experiments.transfer_content(args, params, content, condition, new_cond, file_path)


def diverse_samples(args, params):
    """
    command:
    python3 main.py --local --run diverse --image_name strasbourg_000001_061472 --temp 0.9 --model improved_so_large --last_optim_step 276000 \
                    --img_size 512 1024 --dataset cityscapes --direction label2photo \
                    --n_block 4 --n_flow 10 10 10 10 --do_lu --reg_factor 0.0001 --grad_checkpoint
    """
    optim_step = args.last_optim_step
    temperature = args.temp
    n_samples = 10
    img_size = [512, 1024]
    n_blocks = args.n_block
    image_name = args.image_name

    image_cond_path = f'/local_storage/datasets/moein/cityscapes/gtFine_trainvaltest/gtFine/train/{image_name.split("_")[0]}/{image_name}_gtFine_color.png'
    image_gt_path = f'/local_storage/datasets/moein/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/{image_name.split("_")[0]}/{image_name}_leftImg8bit.png'

    model_paths = helper.compute_paths(args, params)
    output_folder = os.path.join(model_paths['diverse_path'], image_name, f'temp={temperature}')
    print(f'out folder: {output_folder}')

    helper.make_dir_if_not_exists(output_folder)

    shutil.copyfile(image_cond_path, os.path.join(output_folder, 'segment.png'))
    shutil.copyfile(image_gt_path, os.path.join(output_folder, 'real.png'))
    print('Copy segment and real images: done')

    model = models.init_model(args, params)
    model = helper.load_checkpoint(model_paths['checkpoints_path'], optim_step, model, optimizer=None, resume_train=False)[0]

    step = 2
    for i in range(0, 10, step):
        paths_list = [os.path.join(output_folder, f'sample_{(i + 1) + j}.png') for j in range(step)]
        experiments.take_multiple_samples(model, n_blocks, temperature, step, img_size, image_cond_path, paths_list)


def main(args, params):  # called from main.py
    if args.run == 'transfer':
        transfer(args, params)
    elif args.run == 'diverse':
        diverse_samples(args, params)

