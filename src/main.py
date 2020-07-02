from helper import load_checkpoint, init_comet  # helper should be first imported because of Comet
from helper import read_params
import trainer
import experiments
import helper
import models
import evaluation

import argparse
import torch
from torch import optim


# this device is accessible in all the functions in this file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_params_and_args():
    parser = argparse.ArgumentParser(description='Glow trainer')
    parser.add_argument('--dataset', type=str, help='the name of the dataset')
    parser.add_argument('--use_comet', action='store_true')
    parser.add_argument('--resume_train', action='store_true')
    parser.add_argument('--last_optim_step', type=int)
    parser.add_argument('--sample_freq', type=int)
    parser.add_argument('--checkpoint_freq', type=int)
    parser.add_argument('--prev_exp_id', type=str)
    parser.add_argument('--max_step', type=int)
    parser.add_argument('--n_samples', type=int)

    parser.add_argument('--n_flow', type=int)
    parser.add_argument('--n_block', type=int)
    parser.add_argument('--img_size', nargs='+', type=int)  # in height width order: e.g. --img_size 128 256
    parser.add_argument('--bsize', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--temp', type=float)  # temperature

    # Note: left_lr is str since it is used only for finding the checkpoints path of the left glow
    parser.add_argument('--left_step', type=int)  # left glow optim_step (pretrained) in c_flow
    parser.add_argument('--left_lr', type=str)  # the lr using which the left glow was trained
    parser.add_argument('--left_pretrained', action='store_true')  # use pre-trained left glow inf c_flow
    parser.add_argument('--w_conditional', action='store_true')
    parser.add_argument('--act_conditional', action='store_true')
    parser.add_argument('--coupling_cond_net', action='store_true')
    parser.add_argument('--no_validation', action='store_true')
    # parser.add_argument('--batch', default=2, type=int, help='batch size')  # 256 => 2, 128 => 8, 64 => 16
    # parser.add_argument('--do_ceil', action='store_true')
    # parser.add_argument('--left_unfreeze', action='store_true')  # freeze the left glow of unfreeze it
    # parser.add_argument('--use_bmap', action='store_true')  # use boundary maps when training
    # parser.add_argument('--nearest_neighbor', action='store_true')

    # not used anymore
    parser.add_argument('--left_cond', type=str)  # condition used for training left glow, if any

    # args for Cityscapes
    parser.add_argument('--model', type=str, default='glow', help='which model to be used: glow, c_flow, ...')
    parser.add_argument('--cond_mode', type=str, help='the type of conditioning in Cityscapes')
    parser.add_argument('--direction', type=str, default='label2photo')

    parser.add_argument('--train_on_segment', action='store_true')  # train/synthesis with vanilla Glow on segmentations
    parser.add_argument('--sanity_check', action='store_true')
    parser.add_argument('--test_invertibility', action='store_true')

    # preparation
    parser.add_argument('--create_boundaries', action='store_true')
    parser.add_argument('--clean_midgard', action='store_true')

    # evaluation
    parser.add_argument('--infer_on_val', action='store_true')
    parser.add_argument('--resize_for_fcn', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--infer_and_evaluate_c_glow', action='store_true')
    parser.add_argument('--eval_complete', action='store_true')
    parser.add_argument('--eval_ssim', action='store_true')
    parser.add_argument('--gt', action='store_true')  # used only as --evaluate --gt for evaluating ground-truth images

    # args mainly for the experiment mode: --exp should be used for the following (should be revised though)
    parser.add_argument('--random_samples', action='store_true')
    parser.add_argument('--new_condition', action='store_true')
    parser.add_argument('--seg_image', type=str)  # name of the segmentation used for random sampling

    parser.add_argument('--compute_val_bpd', action='store_true')
    parser.add_argument('--exp', action='store_true')
    parser.add_argument('--interp', action='store_true')
    parser.add_argument('--new_cond', action='store_true')
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--sample_c_flow', action='store_true')
    parser.add_argument('--conditional', action='store_true')
    parser.add_argument('--syn_segs', action='store_true')  # segmentation synthesis for c_flow
    parser.add_argument('--trials', type=int)

    arguments = parser.parse_args()

    if arguments.clean_midgard:
        parameters = None
    else:
        parameters = read_params('../params.json')[arguments.dataset]  # parameters related to the wanted dataset

    return arguments, parameters


def adjust_params(args, params):
    """
    Change the default params if specified in the arguments.
    :param args:
    :param params:
    :return:
    """
    if args.n_block is not None:
        params['n_block'] = args.n_block

    if args.n_flow is not None:
        params['n_flow'] = args.n_flow

    if args.bsize is not None:
        params['batch_size'] = args.bsize

    if args.lr is not None:
        params['lr'] = args.lr

    if args.temp is not None:
        params['temperature'] = args.temp

    if args.img_size is not None:
        params['img_size'] = args.img_size

    if args.sample_freq is not None:
        params['sample_freq'] = args.sample_freq

    if args.checkpoint_freq is not None:
        params['checkpoint_freq'] = args.checkpoint_freq

    if args.no_validation:
        params['monitor_val'] = False

    if args.max_step:
        params['iter'] = args.max_step

    if args.n_samples:
        params['n_samples'] = args.n_samples

    print('In [adjust_params]: params adjusted')
    return params


def run_training(args, params):
    # ======== preparing model and optimizer
    model, reverse_cond = models.init_model(args, params)

    lr = params['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ======== setting comet tracker
    tracker = None
    if args.use_comet:
        tracker = init_comet(args, params)
        print("In [run_training]: Comet experiment initialized...")

    # ======== training
    if args.resume_train:
        if args.dataset == 'mnist':
            raise NotImplementedError('In [run_training]: consider the checkpoint path for MNIST...')

        optim_step = args.last_optim_step
        checkpoints_path = helper.compute_paths(args, params)['checkpoints_path']
        model, optimizer, _ = load_checkpoint(checkpoints_path, optim_step, model, optimizer)

        trainer.train(args, params, model, optimizer, tracker, resume=True, last_optim_step=optim_step,
                      reverse_cond=reverse_cond)

    # train from scratch
    else:
        trainer.train(args, params, model, optimizer, tracker, reverse_cond=reverse_cond)


def main():
    args, params = read_params_and_args()
    params = adjust_params(args, params)

    if args.clean_midgard:
        helper.clean_midgard()
        return

    # show important params and the paths (could be refactored based on --exp)
    if not args.exp:
        if not args.eval_complete and not args.create_boundaries and not args.random_samples and not args.new_condition:
            helper.print_info(args, params, model=None, which_info='params')  # print run info (mainly for training)

    if not args.exp:
        run_training(args, params)

    # ================ preparation
    if args.create_boundaries:
        helper.create_boundary_maps(params, device)

    # ================ evaluation
    elif args.exp and args.eval_complete:
        evaluation.eval_city_with_all_temps(args, params)

    elif args.exp and args.infer_on_val:
        experiments.infer_on_validation_set(args, params)

    elif args.exp and args.resize_for_fcn:
        helper.resize_for_fcn(args, params)

    elif args.exp and args.evaluate:
        evaluation.eval_city_with_temp(args, params)

    elif args.exp and args.infer_and_evaluate_c_glow:  # inference and evaluate
        evaluation.infer_and_evaluate_c_glow(args, params)

    elif args.exp and args.eval_ssim:
        evaluation.compute_ssim_all(args, params)

    # ================ experiments
    if args.exp and args.test_invertibility:
        models.verify_invertibility(args, params)

    elif args.exp and args.random_samples:
        experiments.take_random_samples(args, params)

    elif args.exp and args.new_condition:
        experiments.sample_with_new_condition(args, params)

    elif args.exp and args.interp:
        experiments.run_interp_experiments(args, params)

    elif args.exp and args.new_cond:  # is not used, should be refactored
        experiments.run_new_cond_experiments(args, params)

    elif args.exp and args.resample:
        experiments.run_resample_experiments(args, params)

    elif args.exp and args.sample_c_flow:
        # run_c_flow_trials(args, params)
        experiments.sample_trained_c_flow(args, params)

    elif args.exp and args.compute_val_bpd:
        evaluation.compute_val_bpd(args, params)


if __name__ == '__main__':
    main()
