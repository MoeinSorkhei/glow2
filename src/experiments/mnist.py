import data_handler
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import utils

from models import sample_z
from helper import load_checkpoint, make_dir_if_not_exists
from models import init_glow
from data_handler.mnist import MnistDataset


def get_image(img_index, mnist_folder, img_size, ret_type):
    mnist_dataset = MnistDataset(mnist_folder, img_size)
    data_item = mnist_dataset[img_index]
    img = data_item['image']
    label = data_item['label2']
    label_tensor = data_item['label']

    if ret_type == '2d_img':
        return img.squeeze(dim=0), label
    elif ret_type == 'tensor':
        return img, label_tensor
    else:   # return as batch of size 1, determined by ret_type='batch'
        return img.unsqueeze(dim=0), label_tensor.unsqueeze(dim=0)


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

        sampled_imgs = model.reverse(z_samples, coupling_conds=rev_cond).cpu().data

        sign = '+' if mode == 'increment' else '-'
        utils.save_image(sampled_imgs, f'{save_path}/{sign}{change}.png', nrow=10)


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
    model, _, _ = load_checkpoint(checkpoint_pth, optim_step, model, None, resume_train=False)

    # assumption: the two images are of the same condition (label), so I am only using label1
    forward_cond = (args.dataset, label1)

    _, _, z_list1 = model(img1, forward_cond)
    _, _, z_list2 = model(img2, forward_cond)

    z_diff = [z_list2[i] - z_list1[i] for i in range(len(z_list1))]

    coeff = 0
    steps = interp_config['steps']
    all_sampled = []

    for step in range(steps + 1):
        if interp_config['type'] == 'limited':
            coeff = step / steps  # this is the increment factor: e.g. 1/5, 2/5, ..., 5/5
        else:
            coeff = step * interp_config['increment']

        if interp_config['axis'] == 'all':  # full interpolation in all axes
            z_list_inter = [z_list1[i] + coeff * z_diff[i] for i in range(len(z_diff))]

        else:  # interpolation in only the fist axis and keeping others untouched
            axis = 0 if interp_config['axis'] == 'z1' else 1 if interp_config['axis'] == 'z2' else 2
            # print(f'{interp_config["axis"]} shape: {z_list1[axis].shape}')
            # input()
            z_list_inter = [z_list1[i] for i in range(len(z_list1))]  # deepcopy not available for these tensors
            z_list_inter[axis] = z_list1[axis] + coeff * z_diff[axis]

        sampled_img = model.reverse(z_list_inter, reconstruct=True, coupling_conds=rev_cond).cpu().data
        all_sampled.append(sampled_img.squeeze(dim=0))
        # make naming consistent and easy to sort
        coeff_name = '%.2f' % coeff if interp_config['type'] == 'limited' else round(coeff, 2)
        print(f'In [interpolate]: done for coeff {coeff_name}')

    utils.save_image(all_sampled, f'{save_path}/{img_index1}-to-{img_index2}_[{interp_config["axis"]}].png', nrow=10)


def new_condition(img_list, params, args, device):
    checkpoint_pth = params['checkpoints_path']['conditional']  # always conditional
    optim_step = args.last_optim_step
    save_path = params['samples_path']['conditional'] + f'/new_condition'
    make_dir_if_not_exists(save_path)

    # init model and load checkpoint
    model = init_glow(params)
    model, _, _ = load_checkpoint(checkpoint_pth, optim_step, model, None, resume_train=False)

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
            sampled_img = model.reverse(z_list, reconstruct=True, coupling_conds=new_cond)
            all_sampled.append(sampled_img.squeeze(dim=0))  # removing the batch dimension (=1) for the sampled image
            print(f'In [new_condition]: sample with digit={digit} done.')

        utils.save_image(all_sampled, f'{save_path}/img={img_num}.png', nrow=10)
        print(f'In [new_condition]: done for img_num {img_num}')


def resample_latent(img_info, all_resample_lst, params, args, device):
    model, save_path = prepare_experiment(params, args, device, exp_name='resample_latent')
    img, label = get_image(img_info['img'], params['data_folder'], args.img_size, ret_type='batch')

    forward_cond = (args.dataset, label)
    reverse_cond = ('mnist', img_info['label'], 1)
    _, _, z_lst = model(img, forward_cond)
    all_sampled_imgs = []

    for resample_lst in all_resample_lst:
        # reconstruct_lst = [False] * len(resample_lst)
        z_samples = sample_z([z.squeeze(0).shape for z in z_lst],
                             1, params['temperature'], device)  # squeeze to remove the batch dimension

        if resample_lst == ['z1']:
            reconstruct_lst = [False, True, True]
            z_lst[0] = z_samples[0]  # re-sampling this z

        elif resample_lst == ['z2']:
            reconstruct_lst = [True, False, True]
            z_lst[1] = z_samples[1]

        elif resample_lst == ['z3']:
            reconstruct_lst = [True, True, False]
            z_lst[2] = z_samples[2]

        elif resample_lst == ['z1', 'z2']:
            reconstruct_lst = [False, False, True]
            z_lst[0], z_lst[1] = z_samples[0], z_samples[1]

        elif resample_lst == ['z2', 'z3']:
            reconstruct_lst = [True, False, False]
            z_lst[1], z_lst[2] = z_samples[1], z_samples[2]

        elif resample_lst == ['z1', 'z2', 'z3']:
            reconstruct_lst = [False, False, False]  # sample all again
            z_lst = z_samples  # use the samples to generate the image

        elif not resample_lst:  # no resampling
            reconstruct_lst = [True, True, True]  # reconstruct all

        else:
            raise NotImplementedError

        sampled_img = model.reverse(z_lst, reconstruct=reconstruct_lst, coupling_conds=reverse_cond).cpu().data
        all_sampled_imgs.append(sampled_img.squeeze(0))

    utils.save_image(all_sampled_imgs, f'{save_path}/img={img_info["img"]}.png', nrow=10)


def prepare_experiment(params, args, device, exp_name):
    checkpoint_pth = params['checkpoints_path']['conditional']  # always conditional
    optim_step = args.last_optim_step
    save_path = params['samples_path']['conditional'] + f'/{exp_name}'
    make_dir_if_not_exists(save_path)

    # init model and load checkpoint
    model = init_glow(params)
    model, _, _ = load_checkpoint(checkpoint_pth, optim_step, model, None, resume_train=False)

    return model, save_path


def run_interp_experiments(args, params):
    cond_config_0 = {
        'reverse_cond': ('mnist', 0, 1),
        'img_index1': 1,
        'img_index2': 51
    }

    cond_config_1 = {
        'reverse_cond': ('mnist', 1, 1),
        'img_index1': 14,  # start of interpolation
        'img_index2': 6
    }

    cond_config_2 = {
        'reverse_cond': ('mnist', 2, 1),
        'img_index1': 16,
        'img_index2': 25
    }

    cond_config_3 = {
        'reverse_cond': ('mnist', 3, 1),
        'img_index1': 27,
        'img_index2': 44
    }

    cond_config_4 = {
        'reverse_cond': ('mnist', 4, 1),
        'img_index1': 9,
        'img_index2': 58
    }

    cond_config_5 = {
        'reverse_cond': ('mnist', 5, 1),
        'img_index1': 0,
        'img_index2': 35
    }

    cond_config_6 = {
        'reverse_cond': ('mnist', 6, 1),
        'img_index1': 241,
        'img_index2': 62
    }

    cond_config_7 = {
        'reverse_cond': ('mnist', 7, 1),
        'img_index1': 38,
        'img_index2': 91
    }

    cond_config_8 = {
        'reverse_cond': ('mnist', 8, 1),
        'img_index1': 31,
        'img_index2': 144
    }

    cond_config_9 = {
        'reverse_cond': ('mnist', 9, 1),
        'img_index1': 87,
        'img_index2': 110
    }

    interp_conf_limited = {'type': 'limited', 'steps': 9, 'axis': 'all'}
    interp_conf_unlimited = {'type': 'unlimited', 'steps': 20, 'increment': 0.1, 'axis': 'z3'}

    # chosen running configs
    # c_config = cond_config_1
    i_config = interp_conf_limited

    configs = [cond_config_0, cond_config_1, cond_config_2, cond_config_3, cond_config_4, cond_config_5,
               cond_config_6, cond_config_7, cond_config_8, cond_config_9]
    for c_config in configs:
        interpolate(c_config, i_config, params, args, device)
        print('In [run_interp_experiments]: interpolation done for config with digit:', c_config['reverse_cond'][1])


def run_new_cond_experiments(args, params):
    # img_list = [2, 9, 26, 58]  # images to be conditioned on separately
    # img_list = [14, 12, 23, 34]
    img_list = [i for i in range(30)]  # all the first 30 images
    # new_cond = ('mnist', 8, 1)

    new_condition(img_list, params, args, device)


def run_resample_experiments(args, params):
    img_indices = range(30)
    labels = [get_image(idx, params['data_folder'], args.img_size, ret_type='2d_img')[1] for idx in img_indices]

    for i in range(len(img_indices)):
        img_info = {'img': img_indices[i], 'label': labels[i]}

        all_resample_lst = [[], ['z1'], ['z2'], ['z3'], ['z1', 'z2'], ['z2', 'z3']]
        resample_latent(img_info, all_resample_lst, params, args, device)
