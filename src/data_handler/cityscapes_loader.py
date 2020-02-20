import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import os
from PIL import Image
from torchvision import utils
from helper import make_dir_if_not_exists


class CityDataset(data.Dataset):
    def __init__(self, data_folder, img_size, remove_alpha):
        """
        Initializes a dataset to be given to a DataLoader.
        :param data_folder: the folder of the dataset whose images are to be read. Please note that this constructor
        expects the path relative to the 'data' folder whe all the data lie, so it automatically prepends the 'data/'
        folder to the name of the folder given.
        :param img_size: the desired image size to be transformed to. Due to large images size, it is currently set
        to 32 x 32 as default.

        Image IDs are the pure image names which, once joined with the corresponding data folder, can be used to
        retrieve both the real and segmentation images (and any other file corresponding to that ID).
        """
        self.data_folder = data_folder
        self.remove_alpha = remove_alpha
        self.transforms = transforms.Compose([transforms.Resize(img_size),
                                              transforms.ToTensor()])

        self.real_img_paths = read_image_ids(self.data_folder['real'], 'cityscapes_leftImg8bit')  # real images
        cities_and_ids = self.cities_and_ids()
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

        # removing the alpha channel by throwing away the fourth channels
        if self.remove_alpha:
            real_img = real_img[0:3, :, :]
            segment_img = segment_img[0:3, :, :]

        return {'real': real_img,
                'segment': segment_img,
                'real_path': self.real_img_paths[index],
                'segment_path': self.seg_img_paths[index]}


def init_city_loader(data_folder, image_size, remove_alpha, loader_params):
    """
    Initializes and returns a data loader based on the data_folder and dataset_name.
    :param remove_alpha:
    :param image_size:
    :param data_folder: the folder to read the images from
    :param dataset_name: the dataset_name (refer to the Dataset class)
    :param loader_params: a dictionary containing the DataLoader parameters such batch_size and so on.
    :return: the initialized data loader
    """
    dataset = CityDataset(data_folder, image_size, remove_alpha)
    data_loader = data.DataLoader(dataset, **loader_params)
    return data_loader


def read_image_ids(data_folder, dataset_name):
    """
    It reads all the image names (id's) in the given data_folder, and returns the image names needed according to the
    given dataset_name.

    :param data_folder: to folder to read the images from. NOTE: This function expects the data_folder to exist in the
    'data' directory.

    :param dataset_name: the name of the dataset (is useful when there are extra unwanted images in data_folder, such as
    reading the segmentations)

    :return: the list of the image names.
    """
    img_ids = []
    if dataset_name == 'cityscapes_segmentation':
        suffix = '_color.png'
    elif dataset_name == 'cityscapes_leftImg8bit':
        suffix = '_leftImg8bit.png'
    else:
        raise NotImplementedError('In [read_image_ids] of Dataset: the wanted dataset is not implemented yet')

    # all the files in all the subdirectories
    for city_name, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(suffix):  # read all the images in the folder with the desired suffix
                img_ids.append(os.path.join(city_name, file))

    # print(f'In [read_image_ids]: found {len(img_ids)} images')
    return img_ids


def create_segment_cond(n_samples, data_folder, img_size, save_path=None):
    city_dataset = CityDataset(data_folder, img_size, remove_alpha=True)
    segs = [city_dataset[i]['segment'] for i in range(n_samples)]
    reals = [city_dataset[i]['real'] for i in range(n_samples)]
    seg_paths = [city_dataset[i]['segment_path'] for i in range(n_samples)]
    real_paths = [city_dataset[i]['real_path'] for i in range(n_samples)]

    n_channels = segs[0].shape[0]

    segmentations= torch.zeros((n_samples, n_channels, img_size[0], img_size[1]))
    real_imgs = torch.zeros((n_samples, n_channels, img_size[0], img_size[1]))

    for i in range(len(segs)):
        segmentations[i] = segs[i]
        real_imgs[i] = reals[i]

    if save_path:
        make_dir_if_not_exists(save_path)
        utils.save_image(segmentations, f'{save_path}/condition.png', nrow=10)
        utils.save_image(real_imgs, f'{save_path}/real_imgs.png', nrow=10)
        print('In [create_segment_cond]: saved the condition and real images')

        with open(f'{save_path}/img_paths.txt', 'a') as f:
            f.write("==== SEGMENTATIONS PATHS \n")
            for item in seg_paths:
                f.write("%s\n" % item)

            f.write("==== REAL IMAGES PATHS \n")
            for item in real_paths:
                f.write("%s\n" % item)
        print('In [create_segment_cond]: saved the image paths')

    return segmentations
