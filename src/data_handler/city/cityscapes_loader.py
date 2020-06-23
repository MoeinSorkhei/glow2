import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import os
from PIL import Image

from helper import read_image_ids
from city_utility import *


class CityDataset(data.Dataset):
    def __init__(self, data_folder, img_size, remove_alpha, ret_type='for_train', fixed_cond=None):
        """
        Initializes a dataset to be given to a DataLoader.
        :param data_folder: the folder of the dataset whose images are to be read. Please note that this constructor
        expects the path relative to the 'data' folder whe all the data lie, so it automatically prepends the 'data/'
        folder to the name of the folder given.

        :param img_size: the desired image size to be transformed to. Resizing operation is ignored if the desired size
        is exactly the same as the original size.

        :param ret_type: could be 'for_train' or 'all'. If set to 'for_train', it returns only a batch of things that
         are needed during training.

        :param fixed_cond: list of real image paths whose segmentations are used as fixed conditions. If not specified,
        random images will be taken from the dataset.

        Image IDs are the pure image names which, once joined with the corresponding data folder, can be used to
        retrieve both the real and segmentation images (and any other file corresponding to that ID).
        """
        self.data_folder = data_folder
        self.remove_alpha = remove_alpha
        self.ret_type = ret_type

        if img_size != [1024, 2048]:  # resize only if the desired size is different from the original size
            self.transforms = transforms.Compose([transforms.Resize(img_size),
                                                  transforms.ToTensor()])
            self.boundary_transform = transforms.Compose([transforms.Resize(img_size),
                                                          transforms.ToTensor()])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])
            self.boundary_transform = transforms.Compose([transforms.ToTensor()])

        # finding the path for real images
        if not fixed_cond:
            self.real_img_paths = read_image_ids(self.data_folder['real'], 'cityscapes_leftImg8bit')
        else:
            print(f'In CityDataset [__init__]: using the fixed conditions...')
            self.real_img_paths = fixed_cond

        # get list of (city, id) pairs
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
        :return:
        """
        if os.path.isfile(self.boundary_paths[0]):  # check if the first file in the paths exists
            return True
        return False

    def __len__(self):
        return len(self.real_img_paths)

    def __getitem__(self, index):
        """
        NOTE: the index has the full_path to the image itself.
        :param index:
        :return:
        """
        real_path, segment_path, instance_path, boundary_path = \
            self.real_img_paths[index], self.seg_img_paths[index], self.instance_paths[index], self.boundary_paths[index]

        real_img = self.transforms(Image.open(real_path))
        segment_img = self.transforms(Image.open(segment_path))
        boundary = self.boundary_transform(Image.open(boundary_path)) if self.boundaries_exist() else None
        boundary = torch.from_numpy(np.ceil(boundary.numpy()))  # ceiling values to 1

        # removing the alpha channel by throwing away the fourth channels
        if self.remove_alpha:
            real_img = real_img[0:3, :, :]
            segment_img = segment_img[0:3, :, :]
            if boundary is not None:
                boundary = boundary[0:3, :, :]

        # =========== return only the required things for training (saves CPU memory)
        if self.ret_type == 'for_train':
            return {'real': real_img,
                    'segment': segment_img,
                    'boundary': boundary,
                    'real_path': real_path,
                    'segment_path': segment_path}

        else:  # =========== otherwise, pass return everything
            instance_map = self.transforms(Image.open(instance_path))
            if self.remove_alpha:
                instance_map = instance_map[0:3, :, :]

            # getting object IDs with their repetitions in the image
            json_path = segment_path[:-len('color.png')] + 'polygons.json'
            id_repeats = id_repeats_to_cond(info_from_json(json_path)['id_repeats'],
                                            h=segment_img.shape[1],
                                            w=segment_img.shape[2])  # tensor of shape (34, h, w)
            return {'real': real_img,
                    'segment': segment_img,
                    'instance': instance_map,
                    'boundary': boundary,
                    'real_path': real_path,
                    'segment_path': segment_path,
                    'instance_path': instance_path,
                    'boundary_path': boundary_path,
                    'id_repeats': id_repeats}


def id_repeats_to_cond(id_repeats, h, w):
    """
    Transforms the list containing repetitions of object IDs to a tensor used as conditioning in the network.
    :param id_repeats: The list containing the repetitions of the object IDs.
    :param h
    :param w
    :return:
    """
    cond = torch.zeros((len(id_repeats), h, w))
    for i in range(len(id_repeats)):
        cond[i, :, :] = id_repeats[i]
    return cond
