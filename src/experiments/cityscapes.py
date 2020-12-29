import torch
from torchvision import utils

import helper

from models import sample_z, calc_z_shapes
import models
import data_handler
from globals import device, sampling_real_imgs


def sample_c_flow_conditional(args, params, model):
    raise NotImplementedError('Needs code refactoring')
    trials_pth = params['samples_path']['real'][args.cond_mode][args.model] + \
                 f'/trials/optim_step={args.last_optim_step}'
    helper.make_dir_if_not_exists(trials_pth)

    segmentations, _, real_imgs = \
        _create_cond(params['n_samples'],
                     params['data_folder'],
                     params['img_size'],
                     device,
                     save_path=trials_pth)

    z_shapes = calc_z_shapes(params['channels'], params['img_size'], params['n_block'])
    # split into tensors of 5 img: better for visualization
    seg_splits = torch.split(segmentations, split_size_or_sections=5, dim=0)
    real_splits = torch.split(real_imgs, split_size_or_sections=5, dim=0)

    for i in range(len(seg_splits)):
        print(f'====== Doing for the {i}th tensor in seg_splits and real_splits')
        n_samples = seg_splits[i].shape[0]
        # ============ different temperatures
        for temp in [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]:
            all_imgs = torch.cat([seg_splits[i].cpu().data, real_splits[i].cpu().data], dim=0)

            # ============ different trials with different z samples
            for trial in range(args.trials):  # sample for trials times
                z_samples = sample_z(z_shapes, n_samples, temp, device)
                with torch.no_grad():
                    sampled_images = model.reverse(x_a=seg_splits[i],
                                                   z_b_samples=z_samples).cpu().data

                    # all_imgs.append(sampled_images)
                    all_imgs = torch.cat([all_imgs, sampled_images], dim=0)
                # utils.save_image(sampled_images, f'{trials_pth}/trial={trial}.png', nrow=10)
                print(f'Temp={temp} - Trial={trial}: done')

            # save the images for the given temperature
            path = f'{trials_pth}/i={i}'
            helper.make_dir_if_not_exists(path)
            utils.save_image(all_imgs, f'{path}/temp={temp}.png', nrow=n_samples)


def syn_new_segmentations(args, params, model):
    raise NotImplementedError('Needs code refactoring')
    # only one trial for now
    z_shapes = calc_z_shapes(params['channels'], params['img_size'], params['n_block'])
    path = params['samples_path']['segment'][args.cond_mode][args.model] + \
           f'/syn_segs/optim_step={args.last_optim_step}'
    # helper.make_dir_if_not_exists(path)

    n_samples = 2
    for trial in range(args.trials):
        trial_path = f'{path}/i={trial}'
        helper.make_dir_if_not_exists(trial_path)

        for temp in [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]:
            z_a_samples = sample_z(z_shapes, n_samples, temp, device)
            syn_segmentations = model.reverse(z_a_samples=z_a_samples)

            z_b_samples = sample_z(z_shapes, n_samples, temp, device)
            syn_reals = model.reverse(x_a=syn_segmentations, z_b_samples=z_b_samples)
            all_imgs = torch.cat([syn_segmentations, syn_reals], dim=0)
            utils.save_image(all_imgs, f'{trial_path}/temp={temp}.png', nrow=n_samples)

            print(f'Temp={temp}: done')
        print(f'Trial={trial}: done')


def prep_for_sampling(args, params, img_name, additional_info):
    """
    :param args:
    :param params:
    :param img_name: the (path of the) real image whose segmentation which will be used for conditioning.
    :param additional_info:
    :return:
    """
    # ========== specifying experiment path
    paths = helper.compute_paths(args, params, additional_info)
    if additional_info['exp_type'] == 'random_samples':
        experiment_path = paths['random_samples_path']

    elif additional_info['exp_type'] == 'new_cond':
        experiment_path = paths['new_cond_path']

    else:
        raise NotImplementedError

    # ========== make the condition a single image
    fixed_conds = [img_name]
    # ========== create condition and save it to experiment path
    # no need to save for new_cond type
    path_to_save = None if additional_info['exp_type'] == 'new_cond' else experiment_path
    seg_batch, _, real_batch, boundary_batch = data_handler._create_cond(params,
                                                                         fixed_conds=fixed_conds,
                                                                         save_path=path_to_save)  # (1, C, H, W)
    # ========== duplicate condition for n_samples times (not used by all exp_modes)
    seg_batch_dup = seg_batch.repeat((params['n_samples'], 1, 1, 1))  # duplicated: (n_samples, C, H, W)
    boundary_dup = boundary_batch.repeat((params['n_samples'], 1, 1, 1))
    real_batch_dup = real_batch.repeat((params['n_samples'], 1, 1, 1))

    if additional_info['exp_type'] == 'random_samples':
        seg_rev_cond = seg_batch_dup  # (n_samples, C, H, W) - duplicate for random samples
        bmap_rev_cond = boundary_dup
        real_rev_cond = real_batch_dup

    elif additional_info['exp_type'] == 'new_cond':
        seg_rev_cond = seg_batch  # (1, C, H, W) - no duplicate needed
        bmap_rev_cond = boundary_batch

    else:
        raise NotImplementedError

    # ========== create reverse cond
    if not args.use_bmaps:
        rev_cond = {'segment': seg_rev_cond, 'boundary': None}

    elif args.use_bmaps:
        rev_cond = {'segment': seg_rev_cond, 'boundary': bmap_rev_cond}

    elif args.direction == 'photo2label':
        rev_cond = {'real': real_rev_cond}

    else:
        raise NotImplementedError

    # ========== specifying paths for saving samples
    if additional_info['exp_type'] == 'random_samples':
        exp_path = paths['random_samples_path']
        save_paths = [f'{exp_path}/sample {i + 1}.png' for i in range(params['n_samples'])]

    elif additional_info['exp_type'] == 'new_cond':
        save_paths = experiment_path
    else:
        raise NotImplementedError
    return save_paths, rev_cond, real_batch


def sample_c_flow(args, params, model):  # with the specified temperature
    for img_name in sampling_real_imgs:
        img_pure_name = img_name.split('/')[-1][:-len('_leftImg8bit.png')]
        additional_info = {'cond_img_name': img_pure_name, 'exp_type': 'random_samples'}

        save_paths, rev_cond, _ = prep_for_sampling(args, params, img_name, additional_info)
        print(f'In [sample_c_flow]: doing for images: {img_pure_name} {"=" * 50}')

        # ========== take samples from the model
        samples = models.take_samples(args, params, model, rev_cond)  # (n_samples, C, H, W)
        helper.save_one_by_one_old(samples, save_paths)
        print(f'In [sample_c_flow]: for images: {img_pure_name}: done {"=" * 50}\n\n')


