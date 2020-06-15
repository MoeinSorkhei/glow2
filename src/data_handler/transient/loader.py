import torch
from torch.utils import data
from PIL import Image

from . import util


class TransientDataset(data.Dataset):
    def __init__(self, data_folder, annotations_path, img_size, direction, split):
        super().__init__()

        self.data_folder = data_folder
        self.img_size = img_size
        # self.fixed_conds = util.fixed_conds[f'{left_attr}2{right_attr}']
        self.direction = direction
        self.fixed_conds = util.fixed_conds[direction]
        self.trans = util.init_transformer(img_size)
        self.pairs = util.create_multi_pairs(annotations_path, direction, split)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        left_path = f'{self.data_folder}/{self.pairs[item][0]}'
        right_path = f'{self.data_folder}/{self.pairs[item][1]}'

        left = self.trans(Image.open(left_path))
        right = self.trans(Image.open(right_path))
        return {'left': left, 'right': right}


def init_loaders(loader_params, dataset_params):
    loaders = []
    for split in ['train', 'val', 'test']:
        dataset_params['split'] = split
        loader_params['shuffle'] = True if split == 'train' else False

        dataset = TransientDataset(**dataset_params)
        data_loader = data.DataLoader(dataset, **loader_params)
        loaders.append(data_loader)
    return loaders



