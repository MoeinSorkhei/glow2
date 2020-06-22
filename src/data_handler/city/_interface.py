import helper
from .cityscapes_loader import *
from globals import desired_real_imgs


def prepare_city_reverse_cond(args, params, run_mode='train'):
    if args.dataset == 'mnist':
        reverse_cond = ('mnist', 1, params['n_samples'])
        return reverse_cond

    else:
        if args.model == 'glow' and args.cond_mode is None:  # IMPROVEMENT: SHOULD BE FIXED
            return None

        samples_path = helper.compute_paths(args, params)['samples_path']
        save_path = samples_path if run_mode == 'train' else None  # no need for reverse_cond at inference time
        direction = args.direction
        segmentations, id_repeats_batch, real_imgs, boundaries = create_cond(params,
                                                                             fixed_conds=desired_real_imgs,
                                                                             save_path=save_path,
                                                                             direction=direction)
        # ======= only support for c_flow mode now
        if direction == 'label2photo':
            b_maps = boundaries if args.cond_mode == 'segment_boundary' else None
            reverse_cond = {'segment': segmentations, 'boundary': b_maps}

        elif direction == 'photo2label':  # 'photo2label'
            reverse_cond = {'real_cond': real_imgs}
        else:
            raise NotImplementedError('Direction not implemented')
        return reverse_cond
