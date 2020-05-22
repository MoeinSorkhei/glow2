from torch import optim

from .glow import *
from .two_glows import *
import helper
from data_handler import *
from globals import desired_real_imgs, device


def prepare_reverse_cond(args, params, run_mode='train'):
    if args.dataset == 'mnist':
        reverse_cond = ('mnist', 1, params['n_samples'])
        return reverse_cond

    else:
        if args.cond_mode is None:
            return None

        samples_path = helper.compute_paths(args, params)['samples_path']
        save_path = samples_path if run_mode == 'train' else None  # no need for reverse_cond at inference time
        segmentations, id_repeats_batch, \
            real_imgs, boundaries = create_cond(params, fixed_conds=desired_real_imgs, save_path=save_path)

        # ======= only support for c_flow mode now
        b_maps = boundaries if args.cond_mode == 'segment_boundary' else None
        reverse_cond = {'segment': segmentations, 'boundary': b_maps}
        return reverse_cond


def init_model(args, params, device, run_mode='train'):
    # ======== preparing reverse condition and initializing models
    if args.dataset == 'mnist':
        model = init_glow(params)
        reverse_cond = prepare_reverse_cond(args, params, run_mode)  # NOTE: needs to be refactored

    # ======== init glow
    elif args.model == 'glow':  # glow with no condition
        reverse_cond = None
        model = init_glow(params)

    # ======== init c_flow
    elif args.model == 'c_flow':
        reverse_cond = None if args.exp else prepare_reverse_cond(args, params, run_mode)  # only for training
        mode = args.cond_mode  # 'segment', 'segment_boundary', etc.

        # only two Blocks with conditional w
        w_conditionals = [True, True, False, False] if args.w_conditional else None
        act_conditionals = [True, True, False, False] if args.act_conditional else None
        coupling_use_cond_nets = [True, True, True, True] if args.coupling_cond_net else None

        if args.left_pretrained:  # use pre-trained left Glow
            # pth = f"/Midgard/home/sorkhei/glow2/checkpoints/city_model=glow_image=segment"
            left_glow_path = helper.compute_paths(args, params)['left_glow_path']
            pre_trained_left_glow = init_glow(params)  # init the model
            pretrained_left_glow, _, _ = helper.load_checkpoint(path_to_load=left_glow_path,
                                                                optim_step=args.left_step,
                                                                model=pre_trained_left_glow,
                                                                optimizer=None,
                                                                device=device,
                                                                resume_train=False)  # left-glow always freezed

            model = TwoGlows(params, mode,
                             pretrained_left_glow=pretrained_left_glow,
                             w_conditionals=w_conditionals,
                             act_conditionals=act_conditionals,
                             use_coupling_cond_nets=coupling_use_cond_nets)
        else:  # also train left Glow
            model = TwoGlows(params, mode,
                             w_conditionals=w_conditionals,
                             act_conditionals=act_conditionals,
                             use_coupling_cond_nets=coupling_use_cond_nets)
    else:
        raise NotImplementedError
    return model.to(device), reverse_cond


def init_and_load(args, params, run_mode):
    checkpoints_path = helper.compute_paths(args, params)['checkpoints_path']
    optim_step = args.last_optim_step
    model, reverse_cond = init_model(args, params, device, run_mode)

    if run_mode == 'infer':
        model, _, _ = helper.load_checkpoint(checkpoints_path, optim_step, model, None,
                                             device, resume_train=False)
        print(f'In [init_and_load]: returned model for inference')
        return model

    else:  # train
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        print(f'In [init_and_load]: returned model and optimizer for training')
        model, optimizer, _ = helper.load_checkpoint(checkpoints_path, optim_step, model,
                                                     optimizer, device, resume_train=True)
        return model, optimizer
