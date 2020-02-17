from helper import load_checkpoint, init_comet  # helper should be first imported because of Comet
from model import Glow, init_glow
from train import train
from experiments import interp_prev, interpolate, new_condition

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

    arguments = parser.parse_args()

    # reading params from the json file
    with open('../params.json', 'r') as f:
        parameters = json.load(f)[arguments.dataset]  # parameters related to the wanted dataset

    return arguments, parameters


def run_training(args, params):
    # initializing the model and the optimizer
    # RGB images, if image is PNG, the alpha channel will be removed
    # in_channels = params['channels']

    if args.resume_train:
        raise NotImplementedError('Need to take care of model single.')

    '''if args.resume_train:
        model_single = Glow(
            in_channels, params['n_flow'], params['n_block'], do_affine=params['affine'], conv_lu=params['lu']
        )
        model = nn.DataParallel(model_single)'''

    # model = Glow(
    #     in_channels, params['n_flow'], params['n_block'], do_affine=params['affine'], conv_lu=params['lu']
    # )
    model = init_glow(params)

    # model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    tracker = None
    if args.use_comet:
        # experiment = Experiment(api_key="QLZmIFugp5kqZjA4XE2yNS0iZ", project_name="glow", workspace="moeinsorkhei")
        # run_params = {'data_folder': params['datasets'][args.dataset]}
        # tracker = init_comet(run_params)
        tracker = init_comet(params)
        print("Comet experiment initialized...")

    reverse_cond = ('mnist', 1, params['n_samples']) if args.conditional else None

    # resume training
    if args.resume_train:
        optim_step = args.last_optim_step
        # model_single, model, optimizer = \
        #     load_model_and_optimizer(model, model_single, optimizer, model_path, optim_path, device)

        # resume_train(optim_step, args, params, device)
        # resume_train(model, optimizer, optim_step, args, params, in_channels, tracker)
        # save_checkpoint('checkpoints', optim_step, model, optimizer, loss=5.994)
        mode = 'conditional' if args.conditional else 'unconditional'

        # pth = params['checkpoints_path']['conditional'] if args.conditional \
        #    else params['checkpoints_path']['unconditional']

        model, optimizer, _ = load_checkpoint(params['checkpoints_path'][mode], optim_step, model, optimizer, device)
        train(args, params, model, optimizer,
              device, tracker, resume=True, last_optim_step=optim_step, reverse_cond=reverse_cond)

    # train from scratch
    else:
        train(args, params, model, optimizer, device, tracker, reverse_cond=reverse_cond)


def run_interp_experiments(args, params):
    cond_config_1 = {
        'reverse_cond': ('mnist', 1, 1),
        'img_index1': 14,  # start of interpolation
        'img_index2': 6
    }

    cond_config_0 = {
        'reverse_cond': ('mnist', 0, 1),
        'img_index1': 1,
        'img_index2': 51
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

    interp_conf_limited = {'type': 'limited', 'steps': 20, 'axis': 'z3'}
    interp_conf_unlimited = {'type': 'unlimited', 'steps': 20, 'increment': 0.1, 'axis': 'z3'}

    # chosen running configs
    c_config = cond_config_3
    i_config = interp_conf_limited

    interpolate(c_config, i_config, params, args, device)


def run_new_cond_experiments(args, params):
    img_list = [2, 9, 26, 58]  # images to be conditioned on separately
    img_list = [14, 12, 23, 34]
    img_list = [i for i in range(30)]
    # new_cond = ('mnist', 8, 1)

    new_condition(img_list, params, args, device)


def main():
    args, params = read_params_and_args()
    print('In [main]: arguments:', args)

    if args.exp and args.interp:  # experiments
        run_interp_experiments(args, params)

    elif args.exp and args.new_cond:
        run_new_cond_experiments(args, params)

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
