import json

from .paths import *


def print_and_wait(to_be_printed):
    print(to_be_printed)
    print('======== Waiting for input...')
    input()


def read_params(params_path):
    with open(params_path, 'r') as f:  # reading params from the json file
        parameters = json.load(f)
    return parameters


def print_info(args, params, model, which_info='all'):
    if which_info == 'params' or which_info == 'all':
        # printing important running params
        # print(f'{"=" * 50} \n'
        #       f'In [print_info]: Important params: \n'
        #       f'model: {args.model} \n'
        #       # f'lr: {args.lr if args.lr is not None else params["lr"]} \n'
        #       f'lr: {params["lr"]} \n'
        #       f'batch_size: {params["batch_size"]} \n'
        #       f'temperature: {params["temperature"]} \n'
        #       f'last_optim_step: {args.last_optim_step} \n'
        #       f'left_lr: {args.left_lr} \n'
        #       f'left_step: {args.left_step} \n'
        #       f'cond: {args.cond_mode} \n\n')

        # printing paths
        paths = compute_paths(args, params)
        print(f'Paths:')
        for path_name, path_addr in paths.items():
            print(f'{path_name}: {path_addr}')
        print(f'{"=" * 50}\n')

    if which_info == 'model' or which_info == 'all':
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'{"=" * 50}\n'
              'In [print_info]: Using model with the following info:\n'
              f'Total parameters: {total_params:,} \n'
              f'Trainable parameters: {trainable_params:,} \n'
              f'n_flow: {params["n_flow"]} \n'
              f'n_block: {params["n_block"]} \n'
              f'{"=" * 50}\n')


def read_file_to_list(filename):
    lines = []
    if os.path.isfile(filename):
        with open(filename) as f:
            lines = f.read().splitlines()
    return lines
