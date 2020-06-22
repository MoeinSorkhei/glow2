import torch
import matplotlib.pyplot as plt
from torchvision import utils
import os

import helper
from data_handler import CityDataset
from data_handler import create_cond
from helper import load_checkpoint

from models import sample_z, calc_z_shapes
import models
import data_handler
from globals import device, sampling_real_imgs, new_cond_reals


def visualize_img(img_path, data_folder, dataset_name, desired_size):
    """
    :param img_path: Should be relative to the data_folder (will be appended to that)
    :param data_folder:
    :param dataset_name:
    :param desired_size:
    :return:
    """
    dataset = CityDataset(data_folder, dataset_name, desired_size, remove_alpha=True)
    img_full_path = data_folder + '/' + img_path

    img = dataset[dataset.image_ids.index(img_full_path)]  # get the processed image - shape: (3, H, W)
    print(f'In [visualize_img]: visualizing image "{img_path}" of shape: {img.shape}')
    print('Pixel values:')
    print(img)
    print('Min and max values (in image * 255):', torch.min(img * 255), torch.max(img * 255))

    # plotting
    plt.title(f'Size: {desired_size}')
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


def sample_trained_c_flow(args, params):
    checkpt_pth = params['checkpoints_path']['real'][args.cond_mode][args.model]
    # model = TwoGlows(params, do_ceil=args.do_ceil)  # initializing the model
    model = models.init_model(args, params, run_mode='infer')
    model, _, _ = load_checkpoint(checkpt_pth, args.last_optim_step, model, None, False)  # loading the model

    if args.conditional:
        sample_c_flow_conditional(args, params, model)

    if args.syn_segs:
        syn_new_segmentations(args, params, model)


def sample_c_flow_conditional(args, params, model):
    trials_pth = params['samples_path']['real'][args.cond_mode][args.model] + \
                 f'/trials/optim_step={args.last_optim_step}'
    helper.make_dir_if_not_exists(trials_pth)

    segmentations, _, real_imgs = \
        create_cond(params['n_samples'],
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
                                                   z_b_samples=z_samples,
                                                   mode='sample_x_b').cpu().data

                    # all_imgs.append(sampled_images)
                    all_imgs = torch.cat([all_imgs, sampled_images], dim=0)
                # utils.save_image(sampled_images, f'{trials_pth}/trial={trial}.png', nrow=10)
                print(f'Temp={temp} - Trial={trial}: done')

            # save the images for the given temperature
            path = f'{trials_pth}/i={i}'
            helper.make_dir_if_not_exists(path)
            utils.save_image(all_imgs, f'{path}/temp={temp}.png', nrow=n_samples)


def syn_new_segmentations(args, params, model):
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
            syn_segmentations = model.reverse(z_a_samples=z_a_samples, mode='sample_x_a')

            z_b_samples = sample_z(z_shapes, n_samples, temp, device)
            syn_reals = model.reverse(x_a=syn_segmentations, z_b_samples=z_b_samples, mode='sample_x_b')
            all_imgs = torch.cat([syn_segmentations, syn_reals], dim=0)
            utils.save_image(all_imgs, f'{trial_path}/temp={temp}.png', nrow=n_samples)

            print(f'Temp={temp}: done')
        print(f'Trial={trial}: done')


def infer_on_validation_set(args, params):
    """
    The model name and paths should be equivalent to the name used in the resize_for_fcn function
    in evaluation.third_party.prepare.py module.
    :param args:
    :param params:
    :param device:
    :return:
    """
    with torch.no_grad():
        paths = helper.compute_paths(args, params)
        checkpt_path, val_path = paths['checkpoints_path'], paths['val_path']

        print(f'In [infer_on_validation_set]:\n====== checkpt_path: "{checkpt_path}" \n====== val_path: "{val_path}" \n')
        helper.make_dir_if_not_exists(val_path)

        # init and load model
        model, _ = models.init_model(args, params, run_mode='infer')  # init model based on args and params
        optim_step = args.last_optim_step
        model, _, _ = load_checkpoint(checkpt_path, optim_step, model, None)
        print(f'In [infer_on_validation_set]: init model and load checkpoint: done')

        # validation loader
        batch_size = params['batch_size']
        loader_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 0}
        _, val_loader = data_handler.init_city_loader(data_folder=params['data_folder'],
                                                      image_size=(params['img_size']),
                                                      remove_alpha=True,  # removing the alpha channel
                                                      loader_params=loader_params)
        print('In [infer_on_validation_set]: loaded val_loader of len:', len(val_loader))
        helper.print_info(args, params, model)

        print('In [infer_on_validation_set]: starting inference on validation set')
        for i_batch, batch in enumerate(val_loader):
            img_batch = batch['real'].to(device)
            segment_batch = batch['segment'].to(device)
            boundary_batch = batch['boundary'].to(device) if args.cond_mode == 'segment_boundary' else None
            real_paths = batch['real_path']  # list: used to save samples with the same name as original images
            seg_paths = batch['segment_path']

            # z_shapes = helper.calc_z_shapes(params['channels'], params['img_size'], params['n_block'])
            # z_samples = sample_z(z_shapes, batch_size, params['temperature'], device)  # batch_size samples in each iter

            # create reverse conditions and samples based on args.direction
            rev_cond = models.arrange_rev_cond(args, img_batch, segment_batch, boundary_batch)
            samples = models.take_samples(args, params, model, rev_cond)

            # take samples
            '''if args.model == 'c_flow':
                # change here if else
                if args.direction == 'label2photo':
                    samples = model.reverse(x_a=segment_batch,
                                            b_map=boundary_batch,
                                            z_b_samples=z_samples,
                                            mode='sample_x_b').cpu().data
                elif args.direction == 'photo2label':
                    samples = model.reverse(x_a=img_batch,
                                            z_b_samples=z_samples,
                                            mode='samples_x_b').cpu().data

                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError('In [infer_on_validation_set]: no support for the desired model')'''

            # save inferred images separately
            paths_list = real_paths if args.direction == 'label2photo' else seg_paths
            save_one_by_one(samples, paths_list, val_path)

            if i_batch > 0 and i_batch % 20 == 0:
                print(f'In [infer_on_validation_set]: done for the {i_batch}th batch out of {len(val_loader)} batches')
        print(f'In [infer_on_validation_set]: all done. Inferred images could be found at: {val_path} \n')


def save_one_by_one(imgs_batch, paths_list, save_path):
    bsize = imgs_batch.shape[0]
    for i in range(bsize):
        tensor = imgs_batch[i].unsqueeze(dim=0)  # make it a batch of size 1 so we can save it

        if save_path is not None:  # explicitly get the image name and save it to the desired location
            image_name = paths_list[i].split('/')[-1]  # e.g.: lindau_000023_000019_leftImg8bit.png
            full_path = f'{save_path}/{image_name}'

        else:  # full path is already provided in the path list
            full_path = paths_list[i]

        utils.save_image(tensor, full_path, nrow=1, padding=0)


def save_one_by_one2(imgs_batch, paths_list):
    bsize = imgs_batch.shape[0]
    for i in range(bsize):
        tensor = imgs_batch[i].unsqueeze(dim=0)  # make it a batch of size 1 so we can save it
        path = paths_list[i]
        utils.save_image(tensor, path, nrow=1, padding=0)


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
    seg_batch, _, real_batch, boundary_batch = data_handler.create_cond(params,
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
    if args.cond_mode == 'segment':
        rev_cond = {'segment': seg_rev_cond, 'boundary': None}

    elif args.cond_mode == 'segment_boundary':
        rev_cond = {'segment': seg_rev_cond, 'boundary': bmap_rev_cond}

    elif args.direction == 'photo2label':
        rev_cond = {'real_cond': real_rev_cond}

    else:
        raise NotImplementedError

    # ========== specifying paths for saving samples
    if additional_info['exp_type'] == 'random_samples':
        exp_path = paths['random_samples_path']
        save_paths = [f'{exp_path}/sample {i + 1}.png' for i in range(params['n_samples'])]

    elif additional_info['exp_type'] == 'new_cond':
        save_paths = experiment_path
        # orig_path = paths['orig_path']
        #exp_path = paths['new_cond_path']
        #save_paths = {
            # 'orig': orig_path,
        #    'exp_path': exp_path,   # where all the samples will be saved
            # 'new_cond_samples': [f'{exp_path}/sample {i + 1}.png' for i in range(params['n_samples'])]
        #}

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
        # z_shapes = models.calc_z_shapes(params['channels'], params['img_size'], params['n_block'])
        # z_samples = models.sample_z(z_shapes, params['n_samples'], params['temperature'], device)
        samples = models.take_samples(args, params, model, rev_cond)  # (n_samples, C, H, W)
        save_one_by_one2(samples, save_paths)
        print(f'In [sample_c_flow]: for images: {img_pure_name}: done {"=" * 50}\n\n')


def take_random_samples(args, params):
    """
    This function takes random samples from a model specified in the args. See scripts for example usage of this function.
    Uses the globals.py file for obtaining the conditions for sampling.
    :param args:
    :param params:
    :return:
    """
    model = models.init_and_load(args, params, run_mode='infer')
    print(f'In [take_random_samples]: model init: done')

    # temperature specified
    if args.temp:
        sample_c_flow(args, params, model)
        print(f'In [take_random_samples]: temperature = {params["temperature"]}: done')

    # try different temperatures
    else:
        for temp in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            print(f'In [take_random_samples]: doing for temperature = {temp}')
            params['temperature'] = temp

            sample_c_flow(args, params, model)
            print(f'In [take_random_samples]: temperature = {temp}: done \n')

    print(f'In [take_random_samples]: all done \n')


def sample_with_new_condition(args, params):
    """
    In this function, temperature has no effect here as we have no random sampling.
    :param args:
    :param params:
    :return:
    """
    model = models.init_and_load(args, params, run_mode='infer')
    orig_real_name = new_cond_reals['orig_img']  # image paths of the original image (with the desired style)
    orig_pure_name = orig_real_name.split('/')[-1][:-len('_leftImg8bit.png')]
    print(f'In [sample_with_new_cond]: orig cond is: "{orig_pure_name}" \n')
    # orig_seg_name = orig_real_name[:-len('_leftImg8bit.png')] + '_gtFine_color.png'
    # boundary_name = orig_img_name[:-len('_leftImg8bit.png')] + '_gtFine_boundary.png'

    # ========= get the original segmentation and real image
    orig_seg_batch, _, orig_real_batch, orig_bmap_batch = \
        data_handler.create_cond(params, fixed_conds=[orig_real_name], save_path=None)  # (1, C, H, W)

    # make b_maps None is not needed
    if args.cond_mode == 'segment':
        orig_bmap_batch = None

    # ========= real_img_name: the real image whose segmentation which will be used for conditioning.
    for new_cond_name in new_cond_reals['cond_imgs']:
        new_cond_pure_name = new_cond_name.split('/')[-1][:-len('_leftImg8bit.png')]
        additional_info = {
            'orig_pure_name': orig_pure_name,  # original condition city name
            'new_cond_pure_name': new_cond_pure_name,   # new condition city name
            'exp_type': 'new_cond'
        }
        print(f'In [sample_with_new_cond]: doing for images: "{new_cond_pure_name}" {"=" * 50} \n')

        # ========= getting new segment cond and real image batch
        exp_path, new_rev_cond, new_real_batch = prep_for_sampling(args, params, new_cond_name, additional_info)

        # ========= new_cond segment and bmap batches
        new_seg_batch = new_rev_cond['segment']  # (1, C, H, W)
        new_bmap_batch = new_rev_cond['boundary']

        if args.cond_mode == 'segment':
            new_bmap_batch = None

        # ========= save new segmentation and the corresponding real imgs
        helper.make_dir_if_not_exists(exp_path)

        utils.save_image(new_seg_batch.clone(), f"{exp_path}/new_seg.png", nrow=1, padding=0)
        utils.save_image(new_real_batch.clone(), f"{exp_path}/new_real.png", nrow=1, padding=0)
        # if new_bmap_batch is not None:
        #    utils.save_image(new_bmap_batch.clone(), f"{exp_path}/new_bmap.png", nrow=1, padding=0)

        # ========= save the original segmentation and real image
        utils.save_image(orig_seg_batch.clone(), f"{exp_path}/orig_seg.png", nrow=1, padding=0)
        utils.save_image(orig_real_batch.clone(), f"{exp_path}/orig_real.png", nrow=1, padding=0)
        # if orig_bmap_batch is not None:
        #    utils.save_image(orig_bmap_batch.clone(), f"{exp_path}/orig_bmap.png", nrow=1, padding=0)
        print(f'In [sample_with_new_cond]: saved original and new segmentation images')

        # =========== getting z_real corresponding to the desired style (x_b) from the orig real images
        left_glow_outs, right_glow_outs = model(x_a=orig_seg_batch, x_b=orig_real_batch, b_map=orig_bmap_batch)
        z_real = right_glow_outs['z_outs']  # desired style

        # =========== apply the new condition to the desired style
        new_real_syn = model.reverse(x_a=new_seg_batch,  # new segmentation (condition)
                                     b_map=new_bmap_batch,
                                     z_b_samples=z_real,  # desired style
                                     mode='new_condition')  # (1, C, H, W)
        utils.save_image(new_real_syn.clone(), f"{exp_path}/new_real_syn.png", nrow=1, padding=0)
        print(f'In [sample_with_new_cond]: save synthesized real image')

        # =========== save all images of combined in a grid
        all_together = torch.cat([orig_seg_batch, new_seg_batch, orig_real_batch, new_real_batch, new_real_syn], dim=0)
        utils.save_image(all_together.clone(), f"{exp_path}/all.png", nrow=2, padding=10)

        exp_pure_path = exp_path.split('/')[-1]
        all_together_path = os.path.split(exp_path)[0] + '/all'  # 'all' dir in the previous dir
        helper.make_dir_if_not_exists(all_together_path)

        # print(f'all together path: {all_together_path}')
        # input()

        utils.save_image(all_together.clone(), f"{all_together_path}/{exp_pure_path}.png", nrow=2, padding=10)
        print(f'In [sample_with_new_cond]: for images: "{new_cond_pure_name}": done {"=" * 50} \n')
