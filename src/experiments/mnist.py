import data_handler
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import utils, transforms
from copy import deepcopy
from PIL import Image

from train import calc_z_shapes, sample_z
from helper import load_checkpoint, label_to_tensor, make_dir_if_not_exists
from model import init_glow
from data_handler import MnistDataset


def get_image(img_index, mnist_folder, img_size, ret_type):
    mnist_dataset = MnistDataset(mnist_folder, img_size)
    data_item = mnist_dataset[img_index]
    img = data_item['image']
    label = data_item['label2']
    label_tensor = data_item['label']

    # img, label = ['image'], mnist_dataset[img_index]['label']

    if ret_type == '2d_img':
        return img.squeeze(dim=0), label
    elif ret_type == 'tensor':
        return img, label_tensor
    else:   # return as batch of size 1, determined by ret_type='batch'
        return img.unsqueeze(dim=0), label_tensor.unsqueeze(dim=0)


def get_image_prev(mnist_folder, img_index):
    # mnist_folder = 'data/mnist'
    imgs, labels = data_handler.read_mnist(mnist_folder)

    # index = 6  # index of the image to be visualized
    img = imgs[img_index].reshape(28, 28) / 255
    label = labels[img_index]

    tensor_img = Image.fromarray(img)
    print(tensor_img.shape)
    input()
    return img, label


def visualize_mnist(mnist_folder, img_index=0):
    img, label = get_image(mnist_folder, img_index)

    # plotting
    plt.title(f'Label is: {label}')
    plt.imshow(img, cmap='gray')
    plt.show()


def save_mnist_rgb(mnist_folder, save_folder):
    imgs, labels = data_handler.read_mnist(mnist_folder)

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # save 1000 images
    for index in range(1000):
        img = imgs[index].reshape(28, 28) / 255

        rgb = np.zeros((28, 28, 3))
        rgb[:, :, 0] = img
        rgb[:, :, 1] = img
        rgb[:, :, 2] = img
        plt.imsave(save_folder + f'/{index}.jpg', rgb)


def change_linear(z_samples, model, rev_cond, rep, val, save_path, mode):
    z1 = z_samples[0]
    for i in range(rep):  # linearly changing z1
        z1 = z1 + val if mode == 'increment' else z1 - val
        change = round((i + 1) * val, 1)
        print(f'In [mnist_interpolate]: z1 linearly changed ({mode}) {change}')

        sampled_imgs = model.reverse(z_samples, cond=rev_cond).cpu().data

        sign = '+' if mode == 'increment' else '-'
        utils.save_image(sampled_imgs, f'{save_path}/{sign}{change}.png', nrow=10)


def interp_prev(args, params, reverse_cond, optim_step, device, mode='conditional'):
    """
    The goal of this experiment is to investigate how different z's in latent space contribute to the style of the
    generated image. In order to do so, ...

    Some info:
        - image size: 24x24
        - Number of blocks (z's): 3

    Experiment 1:
    Sample all the z's from their corresponding Gaussians. Keeping 2 of the z's constant and change the third by
    sampling differently or linear change. In order to do, ...

    Experiment 2:
    Do the same thing as in Experiment 1 with different conditions (digits).

    Experiment 3:
    Given an new random image, extract its latent vectors and generate a new image using those latent vectors on another
    digit. The goal is to use the style of an image to generate a new image of another digit (condition)
    :return:

    Experiment 4:
    Generate different samples and digits (Fig. 6, 7 of the paper) - think more.
    """

    # load checkpoint and model
    # mode = 'conditional' if conditional
    # checkpoint_pth = \
    #    params['checkpoints_path']['conditional'] if conditional else params['checkpoints_path']['unconditional']

    checkpoint_pth = params['checkpoints_path'][mode]
    save_path = params['samples_path'][mode] + f'/interpolation/{str(optim_step)}'

    # create save_path directory
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        print(f'In [mnist_interpolate]: path "{save_path}" created.')

    # init model and load checkpoint
    model = init_glow(params)
    model, _, _ = load_checkpoint(checkpoint_pth, optim_step, model, None, device, resume_train=False)

    # sample z1, z2, z3
    z_shapes = calc_z_shapes(params['channels'], args.img_size, params['n_flow'], params['n_block'])
    z_samples = sample_z(z_shapes, params['n_samples'], params['temperature'], device)

    # save image with current z samples
    sampled_imgs = model.reverse(z_samples, cond=reverse_cond).cpu().data
    utils.save_image(sampled_imgs, f'{save_path}/0.png', nrow=10)  # WHAT IS 0 IN THE IMAGE NAME TO SAVE????????????

    # print(z_samples)
    # input()

    repetition, change = 10, 0.1
    change_linear(deepcopy(z_samples), model, reverse_cond, repetition, change, save_path, mode='decrement')
    change_linear(deepcopy(z_samples), model, reverse_cond, repetition, change, save_path, mode='increment')


def interpolate(cond_config, interp_config, params, args, device, mode='conditional'):
    # image and label are of type 'batch'
    img_index1, img_index2, rev_cond = cond_config['img_index1'], cond_config['img_index2'], cond_config['reverse_cond']
    img1, label1 = get_image(img_index1, params['data_folder'], args.img_size, ret_type='batch')
    img2, label2 = get_image(img_index2, params['data_folder'], args.img_size, ret_type='batch')

    checkpoint_pth = params['checkpoints_path'][mode]
    optim_step = args.last_optim_step
    save_path = params['samples_path'][mode] + f'/interp'
    make_dir_if_not_exists(save_path)

    # init model and load checkpoint
    model = init_glow(params)
    model, _, _ = load_checkpoint(checkpoint_pth, optim_step, model, None, device, resume_train=False)

    # assumption: the two images are of the same condition (label), so I am only using label1
    forward_cond = (args.dataset, label1)

    _, _, z_list1 = model(img1, forward_cond)
    _, _, z_list2 = model(img2, forward_cond)

    z_diff = [z_list2[i] - z_list1[i] for i in range(len(z_list1))]

    coeff = 0
    steps = interp_config['steps']
    for step in range(steps + 1):
        if interp_config['type'] == 'limited':
            coeff = step / steps  # this is the increment factor: e.g. 1/5, 2/5, ..., 5/5
        else:
            coeff = step * interp_config['increment']

        if interp_config['axis'] == 'all':  # full interpolation in all axes
            z_list_inter = [z_list1[i] + coeff * z_diff[i] for i in range(len(z_diff))]

        else:  # interpolation in only the fist axis and keeping others untouched
            axis = 0 if interp_config['axis'] == 'z1' else 1 if interp_config['axis'] == 'z2' else 2
            z_list_inter = [z_list1[i] for i in range(len(z_list1))]  # deepcopy not available for these tensors
            z_list_inter[axis] = z_list1[axis] + coeff * z_diff[axis]

        sampled_img = model.reverse(z_list_inter, reconstruct=True, cond=rev_cond).cpu().data
        # make naming consistent and easy to sort
        coeff_name = '%.2f' % coeff if interp_config['type'] == 'limited' else round(coeff, 2)

        utils.save_image(sampled_img, f'{save_path}/{coeff_name}.png', nrow=10)  # WHAT IS NROWS = 10?
        print(f'In [interpolate]: done for coeff {coeff_name}')


def new_condition(img_list, params, args, device):
    checkpoint_pth = params['checkpoints_path']['conditional']  # always conditional
    optim_step = args.last_optim_step
    save_path = params['samples_path']['conditional'] + f'/new_condition'
    make_dir_if_not_exists(save_path)

    # init model and load checkpoint
    model = init_glow(params)
    model, _, _ = load_checkpoint(checkpoint_pth, optim_step, model, None, device, resume_train=False)

    for img_num in img_list:
        all_sampled = []
        img, label = get_image(img_num, params['data_folder'], args.img_size, ret_type='batch')

        # get the latent vectors of the image
        forward_cond = (args.dataset, label)
        _, _, z_list = model(img, forward_cond)  # get the latent vectors corresponding to the style of the chosen image

        for digit in range(10):
            new_cond = ('mnist', digit, 1)
            # pass the new cond along with the extracted latent vectors
            # apply it to a new random image with another condition (another digit)
            sampled_img = model.reverse(z_list, reconstruct=True, cond=new_cond)
            all_sampled.append(sampled_img.squeeze(dim=0))  # removing the batch dimension (=1) for the sampled image
            print(f'In [new_condition]: sample with digit={digit} done.')

        utils.save_image(all_sampled, f'{save_path}/img={img_num}.png')
        print(f'In [new_condition]: done for img_num {img_num}')
