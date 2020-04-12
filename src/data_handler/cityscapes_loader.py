import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import os
from PIL import Image
from torchvision import utils
from helper import make_dir_if_not_exists, read_image_ids
from city_utility import *


class CityDataset(data.Dataset):
    def __init__(self, data_folder, img_size, remove_alpha):
        """
        Initializes a dataset to be given to a DataLoader.
        :param data_folder: the folder of the dataset whose images are to be read. Please note that this constructor
        expects the path relative to the 'data' folder whe all the data lie, so it automatically prepends the 'data/'
        folder to the name of the folder given.
        :param img_size: the desired image size to be transformed to.

        Image IDs are the pure image names which, once joined with the corresponding data folder, can be used to
        retrieve both the real and segmentation images (and any other file corresponding to that ID).
        """
        self.data_folder = data_folder
        self.remove_alpha = remove_alpha
        self.transforms = transforms.Compose([transforms.Resize(img_size),
                                              transforms.ToTensor()])

        # finding the path for real images
        self.real_img_paths = read_image_ids(self.data_folder['real'], 'cityscapes_leftImg8bit')
        cities_and_ids = self.cities_and_ids()
        # finding the path for the segmentation images
        self.seg_img_paths = \
            [self.data_folder['segment'] + f'/{city}/{Id}_gtFine_color.png' for (city, Id) in cities_and_ids]

        print('In [__init__]: created Cityscapes Dataset of size:', len(self.real_img_paths))

    def cities_and_ids(self):
        return [(img_path.split('/')[-2], img_path.split('/')[-1][:-len('_leftImg8bit.png')])
                for img_path in self.real_img_paths]

    def __len__(self):
        return len(self.real_img_paths)

    def __getitem__(self, index):
        """
        NOTE: the index has the full_path to the image itself.
        :param index:
        :return:
        """
        real_path, segment_path = self.real_img_paths[index], self.seg_img_paths[index]
        real_img = self.transforms(Image.open(real_path))
        segment_img = self.transforms(Image.open(segment_path))

        # getting object IDs with their repetitions in the image
        json_path = segment_path[:-len('color.png')] + 'polygons.json'
        id_repeats = id_repeats_to_cond(info_from_json(json_path)['id_repeats'],
                                        h=segment_img.shape[1], w=segment_img.shape[2])  # tensor of shape (34, h, w)

        # removing the alpha channel by throwing away the fourth channels
        if self.remove_alpha:
            real_img = real_img[0:3, :, :]
            segment_img = segment_img[0:3, :, :]

        return {'real': real_img,
                'segment': segment_img,
                'real_path': self.real_img_paths[index],
                'segment_path': self.seg_img_paths[index],
                'id_repeats': id_repeats}


def init_city_loader(data_folder, image_size, remove_alpha, loader_params):
    """
    Initializes and returns a data loader based on the data_folder and dataset_name.
    :param remove_alpha:
    :param image_size:
    :param data_folder: the folder to read the images from
    :param loader_params: a dictionary containing the DataLoader parameters such batch_size and so on.
    :return: the initialized data loader
    """
    # train data loader
    train_df = {'real': data_folder['real'] + '/train',  # adjusting the paths for the train data folder
                'segment': data_folder['segment'] + '/train'}
    train_dataset = CityDataset(train_df, image_size, remove_alpha)
    train_loader = data.DataLoader(train_dataset, **loader_params)

    # val data loader
    val_df = {'real': data_folder['real'] + '/val',  # adjusting the paths for the validation data folder
              'segment': data_folder['segment'] + '/val'}
    val_dataset = CityDataset(val_df, image_size, remove_alpha)

    loader_params['shuffle'] = False  # no need to shuffle for the val set
    val_loader = data.DataLoader(val_dataset, **loader_params)
    return train_loader, val_loader


def create_segment_cond(n_samples, data_folder, img_size, device, save_path=None):
    # currently the conditions are taken from the train set - could be changed later
    train_df = {'segment': data_folder['segment'] + '/train',
                'real': data_folder['real'] + '/train'}
    city_dataset = CityDataset(train_df, img_size, remove_alpha=True)

    segs = [city_dataset[i]['segment'] for i in range(n_samples)]
    reals = [city_dataset[i]['real'] for i in range(n_samples)]
    seg_paths = [city_dataset[i]['segment_path'] for i in range(n_samples)]
    real_paths = [city_dataset[i]['real_path'] for i in range(n_samples)]

    n_channels = segs[0].shape[0]

    segmentations = torch.zeros((n_samples, n_channels, img_size[0], img_size[1]))
    real_imgs = torch.zeros((n_samples, n_channels, img_size[0], img_size[1]))
    id_repeats_batch = torch.zeros((n_samples, 34, img_size[0], img_size[1]))  # 34 different IDs

    for i in range(len(segs)):
        segmentations[i] = segs[i]
        real_imgs[i] = reals[i]

    for i in range(id_repeats_batch.shape[0]):
        json_path = seg_paths[i][:-len('color.png')] + 'polygons.json'
        id_repeats = id_repeats_to_cond(info_from_json(json_path)['id_repeats'],
                                        h=img_size[0], w=img_size[1])  # tensor (34, h, w)
        id_repeats_batch[i] = id_repeats
        # print('id_repeats:', info_from_json(json_path)['id_repeats'])

    if save_path:
        make_dir_if_not_exists(save_path)
        utils.save_image(segmentations, f'{save_path}/condition.png', nrow=10)
        utils.save_image(real_imgs, f'{save_path}/real_imgs.png', nrow=10)
        print(f'In [create_segment_cond]: saved the condition and real images to: "{save_path}"')

        with open(f'{save_path}/img_paths.txt', 'a') as f:
            f.write("==== SEGMENTATIONS PATHS \n")
            for item in seg_paths:
                f.write("%s\n" % item)

            f.write("==== REAL IMAGES PATHS \n")
            for item in real_paths:
                f.write("%s\n" % item)
        print('In [create_segment_cond]: saved the image paths \n')

    return segmentations.to(device), id_repeats_batch.to(device), real_imgs.to(device)


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
