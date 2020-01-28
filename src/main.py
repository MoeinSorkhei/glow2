from src.model import Glow
from src.helper import init_comet
from src.train import train, resume_training

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
    arguments = parser.parse_args()

    # reading params from the json file
    with open('params.json', 'r') as f:
        parameters = json.load(f)

    return arguments, parameters


def main():
    args, params = read_params_and_args()
    print('In [main]: arguments:', args)

    # initializing the model and the optimizer
    # RGB images, if image is PNG, the alpha channel will be removed
    in_channels = 3
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
        optim_step = 6000
        resume_training(model, optimizer, optim_step, args, params, in_channels, tracker)

    # train from scratch
    else:
        train(args, params, model, model_single, optimizer, in_channels, device, tracker)


if __name__ == '__main__':
    main()
