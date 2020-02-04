from model import Glow
from helper import load_checkpoint, init_comet
from train import train

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
    parser.add_argument('--use_comet', action='store_true')
    parser.add_argument('--resume_train', action='store_true')
    parser.add_argument('--last_optim_step', type=int)
    arguments = parser.parse_args()

    # reading params from the json file
    with open('../params.json', 'r') as f:
        parameters = json.load(f)[arguments.dataset]  # parameters related to the wanted dataset

    return arguments, parameters


def main():
    args, params = read_params_and_args()
    print('In [main]: arguments:', args)

    # initializing the model and the optimizer
    # RGB images, if image is PNG, the alpha channel will be removed
    in_channels = params['channels']
    model_single = Glow(
        in_channels, params['n_flow'], params['n_block'], do_affine=params['affine'], conv_lu=params['lu']
    )
    model = nn.DataParallel(model_single)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    tracker = None
    if args.use_comet:
        # experiment = Experiment(api_key="QLZmIFugp5kqZjA4XE2yNS0iZ", project_name="glow", workspace="moeinsorkhei")
        run_params = {'data_folder': params['datasets'][args.dataset]}
        tracker = init_comet(run_params)
        print("Comet experiment initialized...")

    # resume training
    if args.resume_train:
        optim_step = args.last_optim_step
        model_path = f'checkpoints/model_{str(optim_step).zfill(6)}.pt'
        optim_path = f'checkpoints/optim_{str(optim_step).zfill(6)}.pt'

        # model_single, model, optimizer = \
        #     load_model_and_optimizer(model, model_single, optimizer, model_path, optim_path, device)

        # resume_train(optim_step, args, params, device)
        # resume_train(model, optimizer, optim_step, args, params, in_channels, tracker)
        # save_checkpoint('checkpoints', optim_step, model, optimizer, loss=5.994)
        model, optimizer, _ = load_checkpoint(params['checkpoints_path'], optim_step, model, optimizer, device)
        train(args, params, model, model_single, optimizer, params['channels'],
              device, tracker, resume=True, last_optim_step=optim_step)

    # train from scratch
    else:
        train(args, params, model, model_single, optimizer, in_channels, device, tracker)


if __name__ == '__main__':
    main()