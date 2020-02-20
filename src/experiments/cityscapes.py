from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

from data_handler import CityDataset


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


