import torch
from torchvision import utils

from . import loader
from .loader import *
from globals import device
import helper


def init_maps_loaders(args, params):
    dataset_params = {
        'data_folder': params['data_folder'],
        'img_size': params['img_size']
        # 'direction': args.direction
    }

    loader_params = {
        "batch_size": params['batch_size'],
        'num_workers': 0
    }

    loaders = loader.init_loaders(loader_params, dataset_params)
    return loaders


def create_rev_cond(args, params, fixed_conds, also_save=True):
    trans = loader.init_transformers(params['img_size'])
    # only supports reading from train data now
    images = read_imgs(params['data_folder'], split='train', fixed_conds=fixed_conds)

    conds_as_list = []
    real_as_list = []

    for image in images:
        h, w = params['img_size']
        # get half of the image corresponding to either the photo or the map
        photo = trans(Image.open(image))[:, :, :w]  # if args.direction == 'photo2map' else trans(Image.open(image))[:, :, w:]
        the_map = trans(Image.open(image))[:, :, w:]

        if args.direction == 'map2photo':
            cond = the_map
            real = photo
        else:
            cond = photo
            real = the_map

        conds_as_list.append(cond)
        real_as_list.append(real)

    conds_as_tensor = torch.stack(conds_as_list, dim=0).to(device)
    real_as_tensor = torch.stack(real_as_list, dim=0).to(device)

    if also_save:
        save_path = helper.compute_paths(args, params)['samples_path']
        helper.make_dir_if_not_exists(save_path)

        if 'conditions.png' not in os.listdir(save_path):
            utils.save_image(conds_as_tensor, os.path.join(save_path, 'conditions.png'))
            print(f'In [create_rev_cond]: saved reverse conditions')

        if 'real.png' not in os.listdir(save_path):
            utils.save_image(real_as_tensor, os.path.join(save_path, 'real.png'))
            print(f'In [create_rev_cond]: saved real images')

    print(f'In [create_rev_cond]: returning cond_as_tensor of shape: {conds_as_tensor.shape}')
    return conds_as_tensor


