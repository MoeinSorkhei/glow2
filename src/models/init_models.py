from .glow import *
from .two_glows import *
import helper
from data_handler import *


def prepare_reverse_cond(args, params, device, run_mode='train'):
    if args.dataset == 'mnist':
        reverse_cond = ('mnist', 1, params['n_samples'])
        return reverse_cond

    else:
        if args.cond_mode is None:
            return None

        samples_path = helper.compute_paths(args, params)['samples_path']
        save_path = samples_path if run_mode == 'train' else None  # no need for reverse_cond at inference time
        segmentations, id_repeats_batch, \
            real_imgs, boundaries = create_cond(params['n_samples'],
                                                params['data_folder'],
                                                params['img_size'],
                                                device,
                                                save_path=save_path)  # do_ceil always True

        # ======= only support for c_flow mode now
        b_maps = boundaries if args.cond_mode == 'segment_boundary' else None
        reverse_cond = {'segment': segmentations, 'boundary': b_maps}
        return reverse_cond


def init_model(args, params, device, run_mode='train'):
    # ======== preparing reverse condition and initializing models
    if args.dataset == 'mnist':
        model = init_glow(params)
        reverse_cond = prepare_reverse_cond(args, params, device, run_mode)  # NOTE: needs to be refactored

    # ======== init glow
    elif args.model == 'glow':  # glow with no condition
        reverse_cond = None
        model = init_glow(params)

    # ======== init c_flow
    elif args.model == 'c_flow':
        reverse_cond = prepare_reverse_cond(args, params, device, run_mode)
        mode = args.cond_mode  # 'segment', 'segment_boundary', etc.

        # only two Blocks with conditional w
        w_conditionals = [True, True, False, False] if args.w_conditional else None
        act_conditionals = [True, True, False, False] if args.act_conditional else None

        if args.left_pretrained:  # use pre-trained left Glow
            # pth = f"/Midgard/home/sorkhei/glow2/checkpoints/city_model=glow_image=segment"
            left_glow_path = helper.compute_paths(args, params)['left_glow_path']
            pre_trained_left_glow = init_glow(params)  # init the model
            pretrained_left_glow, _, _ = helper.load_checkpoint(path_to_load=left_glow_path,
                                                                optim_step=args.left_step,
                                                                model=pre_trained_left_glow,
                                                                optimizer=None,
                                                                device=device,
                                                                resume_train=args.left_unfreeze)  # load from checkpoint

            model = TwoGlows(params, mode, pretrained_left_glow=pretrained_left_glow,
                             w_conditionals=w_conditionals, act_conditionals=act_conditionals)
        else:  # also train left Glow
            model = TwoGlows(params, mode,
                             w_conditionals=w_conditionals, act_conditionals=act_conditionals)
    else:
        raise NotImplementedError
    return model.to(device), reverse_cond

