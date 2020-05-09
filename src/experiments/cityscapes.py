import torch
import matplotlib.pyplot as plt
from torchvision import utils

import helper
from data_handler import CityDataset
from data_handler import create_cond
from train import sample_z
from helper import calc_z_shapes, load_checkpoint
from models import TwoGlows
import models
import data_handler


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


def sample_trained_c_flow(args, params, device):
    checkpt_pth = params['checkpoints_path']['real'][args.cond_mode][args.model]
    # model = TwoGlows(params, do_ceil=args.do_ceil)  # initializing the model
    model = models.init_model(args, params, device, run_mode='infer')
    model, _, _ = load_checkpoint(checkpt_pth, args.last_optim_step, model, None, device, False)  # loading the model

    if args.conditional:
        sample_c_flow_conditional(args, params, model, device)

    if args.syn_segs:
        syn_new_segmentations(args, params, model, device)


def sample_c_flow_conditional(args, params, model, device):
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


def syn_new_segmentations(args, params, model, device):
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


def infer_on_validation_set(args, params, device):
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
        model, _ = models.init_model(args, params, device, run_mode='infer')  # init model based on args and params
        optim_step = args.last_optim_step
        model, _, _ = load_checkpoint(checkpt_path, optim_step, model, None, device)
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

        for i_batch, batch in enumerate(val_loader):
            segment_batch = batch['segment'].to(device)
            real_paths = batch['real_path']  # list: used to save samples with the same name as original images
            z_shapes = helper.calc_z_shapes(params['channels'], params['img_size'], params['n_block'])
            z_samples = sample_z(z_shapes, batch_size, params['temperature'], device)  # batch_size samples in each iter

            # take samples
            if args.model == 'c_flow':
                samples = model.reverse(x_a=segment_batch, z_b_samples=z_samples, mode='sample_x_b').cpu().data

            else:
                raise NotImplementedError

            # save inferred images separately
            save_one_by_one(samples, real_paths, val_path)

            if i_batch > 0 and i_batch % 20 == 0:
                print(f'In [infer_on_validation_set]: done for the {i_batch}th batch out of {len(val_loader)} batches')
        print(f'In [infer_on_validation_set]: all done. Inferred images could be found at: {val_path}')


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
