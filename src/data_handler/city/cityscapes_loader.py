import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import os
from PIL import Image

from helper import read_image_ids
import helper


class CityDataset(data.Dataset):
    def __init__(self, data_folder, img_size, fixed_cond=None, limited=False):
        self.data_folder = data_folder

        # resize only if the desired size is different from the original size
        if img_size != [1024, 2048]:
            self.transforms = helper.get_transform(img_size)
        else:
            self.transforms = helper.get_transform()

        # finding the path for real images
        if not fixed_cond:
            self.real_img_paths = read_image_ids(self.data_folder['real'], 'cityscapes_leftImg8bit')
        else:
            print(f'In CityDataset [__init__]: using the fixed conditions...')
            self.real_img_paths = fixed_cond

        if limited:  # for debugging
            self.real_img_paths = self.real_img_paths[:10]

        # get list of (city, id) pairs from real image paths - (city, id) will be used to retrieve the corresponding segmentation
        cities_and_ids = self.cities_and_ids()

        # finding the path for the segmentation images
        self.seg_img_paths = \
            [self.data_folder['segment'] + f'/{city}/{Id}_gtFine_color.png' for (city, Id) in cities_and_ids]

        # finding the path to instance ID maps
        self.instance_paths = \
            [self.data_folder['segment'] + f'/{city}/{Id}_gtFine_instanceIds.png' for (city, Id) in cities_and_ids]

        # boundary paths
        self.boundary_paths = \
            [self.data_folder['segment'] + f'/{city}/{Id}_gtFine_boundary.png' for (city, Id) in cities_and_ids]

    def cities_and_ids(self):
        return [(img_path.split('/')[-2], img_path.split('/')[-1][:-len('_leftImg8bit.png')])
                for img_path in self.real_img_paths]

    def boundaries_exist(self):
        """
        Checks if the boundary files are available in the gtFine folder (just beside other segmentation files.)
        Note: boundary paths are always available (they are simply string variable denoting the paths to the boundaries)
        but the actual files might not. In such a case, one should call the "create_boundary_maps" function which uses
        these boundary paths to save the boundary maps.
        If boundaries are not available, None will be returned in the __getitem__ function for the 'boundary' key.
        """
        if os.path.isfile(self.boundary_paths[0]):  # check if the first file in the paths exists
            return True
        return False

    def __len__(self):
        return len(self.real_img_paths)

    def __getitem__(self, index):
        real_path, segment_path, instance_path, boundary_path = \
            self.real_img_paths[index], self.seg_img_paths[index], self.instance_paths[index], self.boundary_paths[index]

        # read and transform images
        real_img = self.transforms(Image.open(real_path))
        segment_img = self.transforms(Image.open(segment_path))
        boundary = self.transforms(Image.open(boundary_path)) if self.boundaries_exist() else None

        # removing the alpha channel (if it exists) by throwing away the fourth channels
        real_img = helper.remove_alpha_channel(real_img)
        segment_img = helper.remove_alpha_channel(segment_img)
        if boundary is not None:
            boundary = helper.remove_alpha_channel(boundary)

        return {
            'real': real_img,
            'segment': segment_img,
            'boundary': boundary,
            'real_path': real_path,
            'segment_path': segment_path
        }

