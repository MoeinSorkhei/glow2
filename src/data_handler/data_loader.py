import torch
from torch.utils import data
from torchvision import transforms

import os
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, data_folder, dataset_name, img_size, remove_alpha):
        """
        Initializes a dataset to be given to a DataLoader.
        :param data_folder: the folder of the dataset whose images are to be read. Please note that this constructor
        expects the path relative to the 'data' folder whe all the data lie, so it automatically prepends the 'data/'
        folder to the name of the folder given.
        :param dataset_name: the name of the dataset. Currently it can only be 'cityscapes_segmentation' for reading
        segmentation images in a folder.
        :param img_size: the desired image size to be transformed to. Due to large images size, it is currently set
        to 32 x 32 as default.
        """
        self.data_folder = data_folder
        self.dataset_name = dataset_name
        self.image_ids = read_image_ids(self.data_folder, self.dataset_name)
        self.transforms = transforms.Compose([transforms.Resize((img_size, img_size)),
                                              transforms.ToTensor()])
        self.remove_alpha = remove_alpha

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        NOTE: the index has the full_path to the image itself.
        :param index:
        :return:
        """
        # read the image and apply the transforms to it (and convert it to a tensor)
        img_id = self.image_ids[index]
        img = Image.open(img_id)
        img_tensor = self.transforms(img)

        # removing the alpha channel by throwing away the fourth channels
        if self.remove_alpha:
            img_tensor = img_tensor[0:3, :, :]
        return img_tensor


def init_data_loader(data_folder, dataset_name, image_size, remove_alpha, loader_params):
    """
    Initializes and returns a data loader based on the data_folder and dataset_name.
    :param data_folder: the folder to read the images from
    :param dataset_name: the dataset_name (refer to the Dataset class)
    :param loader_params: a dictionary containing the DataLoader parameters such batch_size and so on.
    :return: the initialized data loader
    """
    dataset = Dataset(data_folder, dataset_name, image_size, remove_alpha)
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
        # could it be a better, more-related exception?
        raise ValueError('In [read_image_ids] of Dataset: the wanted dataset is not implemented yet')

    # all the files in all the subdirectories
    for city_name, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(suffix):  # read all the images in the folder with the desired suffix
                img_ids.append(os.path.join(city_name, file))

    print(f'In [read_image_ids]: found {len(img_ids)} images')
    return img_ids
