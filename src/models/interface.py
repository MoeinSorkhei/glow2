from torch import optim
import torch

from .glow import *
from .two_glows import TwoGlows
from .interface_c_glow import *
from globals import real_conds_abs_path
import data_handler
import helper


def do_forward(args, params, model, img_batch, segment_batch, boundary_batch):
    """
    Does the forward operation of the model based on the model and the data batches.
    :param args:
    :param params:
    :param model:
    :param img_batch:
    :param segment_batch:
    :param boundary_batch:
    :return:
    """
    n_bins = 2. ** params['n_bits']

    if args.model == 'c_flow' or 'improved' in args.model:
        if args.dataset == 'cityscapes' and args.direction == 'label2photo':
            left_glow_outs, right_glow_outs = model(x_a=noise_added(segment_batch, n_bins),
                                                    x_b=noise_added(img_batch, n_bins),
                                                    b_map=boundary_batch)

        elif args.dataset == 'cityscapes' and args.direction == 'photo2label':
            left_glow_outs, right_glow_outs = model(x_a=noise_added(img_batch, n_bins),
                                                    x_b=noise_added(segment_batch, n_bins))

        elif args.dataset == 'maps' and args.direction == 'map2photo':
            left_glow_outs, right_glow_outs = model(x_a=noise_added(segment_batch, n_bins),
                                                    x_b=noise_added(img_batch, n_bins))

        elif args.dataset == 'maps' and args.direction == 'photo2map':
            left_glow_outs, right_glow_outs = model(x_a=noise_added(img_batch, n_bins),
                                                    x_b=noise_added(segment_batch, n_bins))

        else:
            raise NotImplementedError
        # the rest is the same for any configuration
        log_p_left, log_det_left = left_glow_outs['log_p'].mean(), left_glow_outs['log_det'].mean()
        log_p_right, log_det_right = right_glow_outs['log_p'].mean(), right_glow_outs['log_det'].mean()
        return log_p_left, log_det_left, log_p_right, log_det_right

    elif 'c_glow' in args.model:
        if args.dataset == 'cityscapes' and args.direction == 'label2photo':
            z, loss = model(x=noise_added(segment_batch, n_bins),
                            y=noise_added(img_batch, n_bins))
            return z, loss

        elif args.dataset == 'cityscapes' and args.direction == 'photo2label':
            z, loss = model(x=noise_added(img_batch, n_bins),
                            y=noise_added(segment_batch, n_bins))
            return z, loss
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def noise_added(batch, n_bins):  # add uniform noise
    return batch + torch.rand_like(batch) / n_bins


def take_samples(args, params, model, reverse_cond, n_samples=None):
    with torch.no_grad():
        if args.model == 'c_flow' or 'improved' in args.model:  # here reverse_cond is x_a
            num_samples = n_samples if n_samples is not None else params['n_samples']
            z_samples = sample_z(n_samples=num_samples,
                                 temperature=params['temperature'],
                                 channels=params['channels'],
                                 img_size=params['img_size'],
                                 n_block=params['n_block'],
                                 split_type=model.split_type)

            if args.direction == 'label2photo':
                sampled_images = model.reverse(x_a=reverse_cond['segment'],
                                               b_map=reverse_cond['boundary'],
                                               z_b_samples=z_samples).cpu().data

            elif args.direction == 'photo2label':  # 'photo2label'
                sampled_images = model.reverse(x_a=reverse_cond['real_cond'],
                                               z_b_samples=z_samples).cpu().data

            elif args.dataset == 'maps':
                sampled_images = model.reverse(x_a=reverse_cond,  # reverse cond is already extracted based on direction
                                               z_b_samples=z_samples).cpu().data
            else:
                raise NotImplementedError

        elif 'c_glow' in args.model:
            temp = params['temperature']
            if args.direction == 'label2photo':
                # the forward function with reverse=True takes samples from prior
                sampled_images, _ = model(x=reverse_cond['segment'], reverse=True, eps_std=temp)

            elif args.direction == 'photo2label':
                sampled_images, _ = model(x=reverse_cond['real_cond'], reverse=True, eps_std=temp)

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return sampled_images


def batch2revcond(args, img_batch, segment_batch, boundary_batch):  # only used in cityscapes experiments
    """
    Takes batches of data and arranges them as reverse condition based on the args.
    :param args:
    :param img_batch:
    :param segment_batch:
    :param boundary_batch:
    :return:
    """
    # ======= only support for c_flow mode now
    if args.direction == 'label2photo':
        reverse_cond = {'segment': segment_batch, 'boundary': boundary_batch}

    elif args.direction == 'photo2label':  # 'photo2label'
        reverse_cond = {'real_cond': img_batch}

    else:
        raise NotImplementedError('Direction not implemented')
    return reverse_cond


def verify_invertibility(args, params):
    # imgs = [real_conds_abs_path[0]]  # one image for now
    # segmentations, _, real_imgs, boundaries = data_handler.create_cond(params, fixed_conds=imgs, save_path=None)
    segmentations, real_imgs, boundaries = [torch.rand((1, 3, 256, 256)).to(device)] * 3
    model = init_model(args, params)
    x_a_rec, x_b_rec = model.reconstruct_all(x_a=segmentations, x_b=real_imgs, b_map=boundaries)
    sanity_check(segmentations, real_imgs, x_a_rec, x_b_rec)


def init_model_configs(args):
    assert 'improved' in args.model  # otherwise not implemented yet
    left_configs = {'all_conditional': False, 'split_type': 'regular', 'do_lu': False}  # default
    right_configs = {'all_conditional': True, 'split_type': 'regular', 'do_lu': False, 'condition': 'left'}  # default condition from left glow

    if 'improved' in args.model:
        if 'regular' in args.model:
            left_configs['split_type'], right_configs['split_type'] = 'regular', 'regular'
        else:
            left_configs['split_type'], right_configs['split_type'] = 'special', 'special'
            left_configs['split_sections'], right_configs['split_sections'] = [3, 9], [3, 9]

        if args.do_lu:
            left_configs['do_lu'] = True
            right_configs['do_lu'] = True

        if args.use_bmaps:
            right_configs['condition'] = 'left + b_maps'

        if args.use_bmaps and args.do_ceil:
            right_configs['condition'] = 'left + b_maps_ceil'

    print(f'In [init_configs]: configs init done: \nleft_configs: {left_configs} \nright_configs: {right_configs}\n')
    return left_configs, right_configs


def init_model(args, params):
    assert len(params['n_flow']) == params['n_block']

    if args.model == 'c_flow' or 'improved' in args.model:
        left_configs, right_configs = init_model_configs(args)
        model = TwoGlows(params, left_configs, right_configs)

    elif 'c_glow' in args.model:
        model = init_c_glow(args, params)

    else:
        raise NotImplementedError

    print('In [init_model]: init model done')
    helper.print_info(args, params, model, which_info='model')
    return model.to(device)


def init_and_load(args, params, run_mode):
    checkpoints_path = helper.compute_paths(args, params)['checkpoints_path']
    optim_step = args.last_optim_step
    model = init_model(args, params)

    if run_mode == 'infer':
        model, _, _ = helper.load_checkpoint(checkpoints_path, optim_step, model, None, resume_train=False)
        print(f'In [init_and_load]: returned model for inference')
        return model

    else:  # train
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        print(f'In [init_and_load]: returned model and optimizer for training')
        model, optimizer, _ = helper.load_checkpoint(checkpoints_path, optim_step, model, optimizer, resume_train=True)
        return model, optimizer

