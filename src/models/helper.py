from .glow import *
from .two_glows import *
import helper
from data_handler import *


def prepare_reverse_cond(args, params, device, run_mode='train'):
    if args.dataset == 'mnist':
        reverse_cond = ('mnist', 1, params['n_samples'])
        return reverse_cond

    else:
        if args.cond_mode is None:  # vanilla Glow on real/segments without condition
            return None

        samples_path = helper.compute_paths(args, params)['samples_path']
        save_path = samples_path if run_mode == 'train' else None  # no need for reverse_cond at inference time
        segmentations, id_repeats_batch, real_imgs = create_segment_cond(params['n_samples'],
                                                                         params['data_folder'],
                                                                         params['img_size'],
                                                                         device,
                                                                         save_path=save_path)
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


def init_model(args, params, device, run_mode='train'):
    # ======== preparing reverse condition and initializing models
    if args.dataset == 'mnist':
        model = init_glow(params)
        reverse_cond = prepare_reverse_cond(args, params, device, run_mode)

    elif args.model == 'glow':
        if args.cond_mode is None:
            reverse_cond = None
            model = init_glow(params)

        else:
            reverse_cond, cond_shapes = prepare_reverse_cond(args, params, device, run_mode)
            model = init_glow(params, cond_shapes)

    # ======== init c_flow
    elif args.model == 'c_flow':
        reverse_cond, _ = prepare_reverse_cond(args, params, device, run_mode)

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
            model = TwoGlows(params, pretrained_left_glow)
        else:  # also train left Glow
            model = TwoGlows(params)
    else:
        raise NotImplementedError
    return model.to(device), reverse_cond

