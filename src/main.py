from helper import load_checkpoint, init_comet  # helper should be first imported because of Comet
from model import Glow, init_glow
from train import train
from experiments import interpolate, new_condition, resample_latent, get_image

import argparse
import json
import torch
from torch import nn, optim


# this device is accessible in all the functions in this file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_params_and_args():
    parser = argparse.ArgumentParser(description='Glow trainer')
    parser.add_argument('--batch', default=2, type=int, help='batch size')  # 256 => 2, 128 => 8, 64 => 16
    parser.add_argument('--dataset', type=str, help='the name of the dataset')
    parser.add_argument('--img_size', default=256, type=int, help='image size')
    parser.add_argument('--conditional', action='store_true')
    parser.add_argument('--use_comet', action='store_true')
    parser.add_argument('--resume_train', action='store_true')
    parser.add_argument('--last_optim_step', type=int)

    # args mainly for the experiment mode
    parser.add_argument('--exp', action='store_true')
    parser.add_argument('--interp', action='store_true')
    parser.add_argument('--new_cond', action='store_true')
    parser.add_argument('--resample', action='store_true')

    arguments = parser.parse_args()

    # reading params from the json file
    with open('../params.json', 'r') as f:
        parameters = json.load(f)[arguments.dataset]  # parameters related to the wanted dataset

    return arguments, parameters


def run_training(args, params):
    if args.resume_train:
        raise NotImplementedError('Need to take care of model single.')

    model = init_glow(params)
    # model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    tracker = None
    if args.use_comet:
        tracker = init_comet(params)
        print("Comet experiment initialized...")

    reverse_cond = ('mnist', 1, params['n_samples']) if args.conditional else None

    # resume training
    if args.resume_train:
        optim_step = args.last_optim_step
        mode = 'conditional' if args.conditional else 'unconditional'
        model, optimizer, _ = load_checkpoint(params['checkpoints_path'][mode], optim_step, model, optimizer, device)

        train(args, params, model, optimizer,
              device, tracker, resume=True, last_optim_step=optim_step, reverse_cond=reverse_cond)

    # train from scratch
    else:
        train(args, params, model, optimizer, device, tracker, reverse_cond=reverse_cond)


def run_interp_experiments(args, params):
    cond_config_0 = {
        'reverse_cond': ('mnist', 0, 1),
        'img_index1': 1,
        'img_index2': 51
    }

    cond_config_1 = {
        'reverse_cond': ('mnist', 1, 1),
        'img_index1': 14,  # start of interpolation
        'img_index2': 6
    }

    cond_config_2 = {
        'reverse_cond': ('mnist', 2, 1),
        'img_index1': 16,
        'img_index2': 25
    }

    cond_config_3 = {
        'reverse_cond': ('mnist', 3, 1),
        'img_index1': 27,
        'img_index2': 44
    }

    cond_config_4 = {
        'reverse_cond': ('mnist', 4, 1),
        'img_index1': 9,
        'img_index2': 58
    }

    cond_config_5 = {
        'reverse_cond': ('mnist', 5, 1),
        'img_index1': 0,
        'img_index2': 35
    }

    cond_config_6 = {
        'reverse_cond': ('mnist', 6, 1),
        'img_index1': 241,
        'img_index2': 62
    }

    cond_config_7 = {
        'reverse_cond': ('mnist', 7, 1),
        'img_index1': 38,
        'img_index2': 91
    }

    cond_config_8 = {
        'reverse_cond': ('mnist', 8, 1),
        'img_index1': 31,
        'img_index2': 144
    }

    cond_config_9 = {
        'reverse_cond': ('mnist', 9, 1),
        'img_index1': 87,
        'img_index2': 110
    }

    interp_conf_limited = {'type': 'limited', 'steps': 9, 'axis': 'all'}
    interp_conf_unlimited = {'type': 'unlimited', 'steps': 20, 'increment': 0.1, 'axis': 'z3'}

    # chosen running configs
    # c_config = cond_config_1
    i_config = interp_conf_limited

    configs = [cond_config_0, cond_config_1, cond_config_2, cond_config_3, cond_config_4, cond_config_5,
               cond_config_6, cond_config_7, cond_config_8, cond_config_9]
    for c_config in configs:
        interpolate(c_config, i_config, params, args, device)
        print('In [run_interp_experiments]: interpolation done for config with digit:', c_config['reverse_cond'][1])


def run_new_cond_experiments(args, params):
    # img_list = [2, 9, 26, 58]  # images to be conditioned on separately
    # img_list = [14, 12, 23, 34]
    img_list = [i for i in range(30)]  # all the first 30 images
    # new_cond = ('mnist', 8, 1)

    new_condition(img_list, params, args, device)


def run_resample_experiments(args, params):
    img_indices = range(30)
    labels = [get_image(idx, params['data_folder'], args.img_size, ret_type='2d_img')[1] for idx in img_indices]

    for i in range(len(img_indices)):
        img_info = {'img': img_indices[i], 'label': labels[i]}

        all_resample_lst = [[], ['z1'], ['z2'], ['z3'], ['z1', 'z2'], ['z2', 'z3']]
        resample_latent(img_info, all_resample_lst, params, args, device)


def main():
    args, params = read_params_and_args()
    print('In [main]: arguments:', args)

    if args.exp and args.interp:  # experiments
        run_interp_experiments(args, params)

    elif args.exp and args.new_cond:
        run_new_cond_experiments(args, params)

    elif args.exp and args.resample:
        run_resample_experiments(args, params)

    else:  # training
        run_training(args, params)


if __name__ == '__main__':
    main()

# --dataset mnist --batch 128 --img_size 24
# --dataset mnist --conditional --batch 128 --img_size 24 --resume_train --last_optim_step 21000 --use_comet

# --dataset mnist --batch 128 --img_size 24 --conditional
# --dataset cityscapes_segmentation


# for interp_prev
# --dataset mnist --exp --img_size 24 --last_optim_step 12000 --conditional

# for the second interp
# --dataset mnist --exp --interp --img_size 24 --last_optim_step 12000

# for new conditioning
# --dataset mnist --exp --new_cond --img_size 24 --last_optim_step 12000

# for resampling
# --dataset mnist --exp --resample --img_size 24 --last_optim_step 12000
