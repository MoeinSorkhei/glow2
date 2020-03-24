from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from torchvision import utils

import helper
from data_handler import CityDataset
from data_handler import create_segment_cond
from train import sample_z
from helper import calc_z_shapes, load_checkpoint
from models import TwoGlows


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
    model = TwoGlows(params)  # initializing the model
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
        create_segment_cond(params['n_samples'],
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
