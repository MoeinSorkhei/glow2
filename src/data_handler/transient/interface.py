from PIL import Image
import torch
from torchvision import utils
import os

from . import loader
from . import util
from globals import device
import helper


def init_transient_loaders(args, params):
    """
    This inits the data loaders to be used by outer modules.
    :param params:
    :return:
    """
    dataset_params = {
        'data_folder': params['paths']['data_folder'],
        'annotations_path': params['paths']['annotations'],
        'img_size': params['img_size'],
        'direction': args.direction,
        # 'left_attr': params['left_attr'],
        # 'right_attr': params['right_attr']
    }

    loader_params = {
        "batch_size": params['batch_size'],
        'num_workers': 0
    }

    loaders = loader.init_loaders(loader_params, dataset_params)
    return loaders


def create_rev_cond(args, params, also_save=True):
    trans = util.init_transformer(params['img_size'])

    # if args.direction == 'daylight2night':
    names = util.fixed_conds[args.direction]
    conds = [trans(Image.open(f"{params['paths']['data_folder']}/{name}")) for name in names]

    # else:
    #    raise NotImplementedError

    conditions = torch.stack(conds, dim=0).to(device)  # (B, C, H, W)

    if also_save:
        save_path = helper.compute_paths(args, params)['samples_path']
        helper.make_dir_if_not_exists(save_path)

        if 'conditions.png' not in os.listdir(save_path):
            utils.save_image(conditions, f'{save_path}/conditions.png')
            print(f'In [create_rev_cond]: saved reverse conditions')
    return conditions
