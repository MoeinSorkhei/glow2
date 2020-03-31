from helper import load_checkpoint, init_comet  # helper should be first imported because of Comet
from helper import read_params, calc_cond_shapes
from models import Glow, init_glow, TwoGlows
from train import train
from experiments import interpolate, new_condition, resample_latent, get_image
from data_handler import create_segment_cond
import experiments
import helper

import argparse
import json
import torch
from torch import nn, optim


# this device is accessible in all the functions in this file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_params_and_args():
    parser = argparse.ArgumentParser(description='Glow trainer')
    # parser.add_argument('--batch', default=2, type=int, help='batch size')  # 256 => 2, 128 => 8, 64 => 16
    parser.add_argument('--dataset', type=str, help='the name of the dataset')
    parser.add_argument('--use_comet', action='store_true')
    parser.add_argument('--resume_train', action='store_true')
    parser.add_argument('--last_optim_step', type=int)
    parser.add_argument('--bsize', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--left_step', type=int)  # left glow optim_step (pretrained) in c_flow
    # Note: left_lr is str since it is used only for finding the checkpoints path of the left glow
    parser.add_argument('--left_lr', type=str)  # the lr using which the left glow was trained
    parser.add_argument('--left_pretrained', action='store_true')  # use pre-trained left glow inf c_flow
    parser.add_argument('--left_unfreeze', action='store_true')  # freeze the left glow of unfreeze it
    parser.add_argument('--left_cond', type=str)  # condition used for training left glow, if any

    # args for Cityscapes
    parser.add_argument('--cond_mode', type=str, help='the type of conditioning in Cityscapes')
    parser.add_argument('--model', type=str, default='glow', help='which model to be used: glow, c_flow, ...')
    parser.add_argument('--sanity_check', action='store_true')
    parser.add_argument('--train_on_segment', action='store_true')  # train/synthesis with vanilla Glow on segmentations

    # args mainly for the experiment mode
    parser.add_argument('--exp', action='store_true')
    parser.add_argument('--interp', action='store_true')
    parser.add_argument('--new_cond', action='store_true')
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--sample_c_flow', action='store_true')
    parser.add_argument('--conditional', action='store_true')
    parser.add_argument('--syn_segs', action='store_true')  # segmentation synthesis for c_flow
    parser.add_argument('--trials', type=int)

    arguments = parser.parse_args()
    parameters = read_params('../params.json')[arguments.dataset]  # parameters related to the wanted dataset

    return arguments, parameters


def prepare_reverse_cond(args, params, device):
    if args.dataset == 'mnist':
        reverse_cond = ('mnist', 1, params['n_samples'])
        return reverse_cond

    else:
        if args.cond_mode is None:  # vanilla Glow on real/segments without condition
            return None

        samples_path = helper.compute_paths(args, params)['samples_path']
        segmentations, id_repeats_batch, real_imgs = create_segment_cond(params['n_samples'],
                                                                         params['data_folder'],
                                                                         params['img_size'],
                                                                         device,
                                                                         save_path=samples_path)
        # condition is segmentation
        if args.cond_mode == 'segment':
            if args.model == 'glow':
                reverse_cond = ('city_segment', segmentations)

            elif args.model == 'c_flow':
                # here reverse_cond is equivalent to x_a, the actual condition will be made inside the reverse function
                if args.sanity_check:
                    reverse_cond = (segmentations, real_imgs)
                else:
                    reverse_cond = segmentations
            else:
                raise NotImplementedError

        # condition is segmentation + ID's
        elif args.cond_mode == 'segment_id':
            reverse_cond = ('city_segment_id', segmentations, id_repeats_batch)
        else:
            raise NotImplementedError

        # calculating condition shape (needed to init the model) -- maybe better to move this part somewhere else
        cond_shapes = calc_cond_shapes(segmentations.shape[1:],
                                       params['channels'],
                                       params['img_size'],
                                       params['n_block'],
                                       args.cond_mode)
        return reverse_cond, cond_shapes


def run_training(args, params):
    # ======== preparing reverse condition and initializing models
    if args.dataset == 'mnist':
        model = init_glow(params)
        reverse_cond = prepare_reverse_cond(args, params, device)

    elif args.model == 'glow':
        if args.cond_mode is None:
            reverse_cond = None
            model = init_glow(params)

        else:
            reverse_cond, cond_shapes = prepare_reverse_cond(args, params, device)
            model = init_glow(params, cond_shapes)

    # ======== init c_flow
    elif args.model == 'c_flow':
        reverse_cond, _ = prepare_reverse_cond(args, params, device)

        if args.left_pretrained:  # use pre-trained left Glow
            # pth = f"/Midgard/home/sorkhei/glow2/checkpoints/city_model=glow_image=segment"
            left_glow_path = helper.compute_paths(args, params)['left_glow_path']
            pre_trained_left_glow = init_glow(params)  # init the model
            pretrained_left_glow, _, _ = load_checkpoint(path_to_load=left_glow_path,
                                                         optim_step=args.left_step,
                                                         model=pre_trained_left_glow,
                                                         optimizer=None,
                                                         device=device,
                                                         resume_train=args.left_unfreeze)  # load from checkpoint
            model = TwoGlows(params, pretrained_left_glow)
        else:  # also train left Glow
            model = TwoGlows(params)
    else:
        raise NotImplementedError

    # ======== preparing model and optimizer
    model.to(device)
    lr = args.lr if args.lr is not None else params['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ======== setting comet tracker
    tracker = None
    if args.use_comet:
        tracker = init_comet(params)
        print("In [run_training]: Comet experiment initialized...")

    # ======== training
    if args.resume_train:
        if args.dataset == 'mnist':
            raise NotImplementedError('In [run_training]: consider the checkpoint path for MNIST...')

        optim_step = args.last_optim_step
        checkpoints_path = helper.compute_paths(args, params)['checkpoints_path']
        model, optimizer, _ = load_checkpoint(checkpoints_path, optim_step, model, optimizer, device)

        train(args, params, model, optimizer,
              device, tracker, resume=True, last_optim_step=optim_step, reverse_cond=reverse_cond)

    # train from scratch
    else:
        train(args, params, model, optimizer, device, tracker, reverse_cond=reverse_cond)


def main():
    args, params = read_params_and_args()
    print('In [main]: arguments:', args)

    if args.exp and args.interp:  # experiments
        experiments.run_interp_experiments(args, params)

    elif args.exp and args.new_cond:
        experiments.run_new_cond_experiments(args, params)

    elif args.exp and args.resample:
        experiments.run_resample_experiments(args, params)

    elif args.exp and args.sample_c_flow:
        # run_c_flow_trials(args, params)
        experiments.sample_trained_c_flow(args, params, device)

    else:  # training
        run_training(args, params)


if __name__ == '__main__':
    main()
