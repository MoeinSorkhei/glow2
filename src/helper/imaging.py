import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch

from .paths import *
from data_handler import CityDataset


def show_images(img_list):
    if len(img_list) != 2:
        raise NotImplementedError('Showing more than two images not implemented yet')

    f, ax_arr = plt.subplots(1, 2)
    ax_arr[0].imshow(img_list[0].permute(1, 2, 0))  # permute: making it (H, W, channel)
    ax_arr[1].imshow(img_list[1].permute(1, 2, 0))

    plt.show()


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


def open_and_resize_image(path, for_model=None):
    image = Image.open(path).resize((256, 256))  # read image and resize
    image_array = (np.array(image)[:, :, :3] / 255).astype(np.float32)  # remove alpha channel

    if for_model == 'dual_glow':
        return np.expand_dims(image_array, axis=(0, 1))  # expand for dual_glow model
    return image_array


def rescale_image(image):
    # just the same as torch save_image function
    return np.clip((image * 255) + 0.5, a_min=0, a_max=255).astype(np.uint8)


def rescale_and_save_image(image, path):
    rescaled = rescale_image(image)
    Image.fromarray(rescaled).save(path)


def resize_imgs(path_to_load, path_to_save, h=256, w=256, package='pil'):
    imgs = read_image_ids(path_to_load, dataset_name='cityscapes_leftImg8bit')
    print(f'In [resize_imgs]: read {len(imgs)} from: "{path_to_load}"')
    print(f'In [resize_imgs]: will save resized imgs to: "{path_to_save}"')
    make_dir_if_not_exists(path_to_save)

    for i in range(len(imgs)):
        if i > 0 and i % 50 == 0:
            print(f'In [resize_imgs]: done for the {i}th image')

        img_full_name = imgs[i].split('/')[-1]
        city = img_full_name.split('_')[0]
        image = Image.open(imgs[i])

        if package == 'pil':
            resized = image.resize((w, h))
            resized.save(f'{path_to_save}/{img_full_name}')

        else:  # package == 'scipy' => only works for scipy=1.0.0
            import scipy.misc
            image = np.array(image)
            resized = scipy.misc.imresize(image, (h, w))
            scipy.misc.imsave(f'{path_to_save}/{img_full_name}', resized)

    print('In [resize_imgs]: All done')


def resize_for_fcn(args, params):
    if args.gt:  # photo2label only
        load_path = '/local_storage/datasets/moein/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val'
        save_path = '/Midgard/home/sorkhei/glow2/data/cityscapes/resized/val'
    else:
        paths = compute_paths(args, params)
        load_path, save_path = paths['val_path'], paths['resized_path']

    resize_imgs(load_path, save_path, package='scipy')


