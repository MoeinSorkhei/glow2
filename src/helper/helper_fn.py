import torch
import numpy as np
import os
import json


def read_params(params_path):
    with open(params_path, 'r') as f:  # reading params from the json file
        parameters = json.load(f)
    return parameters


def save_checkpoint(path_to_save, optim_step, model, optimizer, loss):
    name = path_to_save + f'/optim_step={optim_step}.pt'
    checkpoint = {'loss': loss,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, name)


def load_checkpoint(path_to_load, optim_step, model, optimizer, device, resume_train=True):
    # path_to_load = translate_address(path_to_load, 'helper')
    name = path_to_load + f'/optim_step={optim_step}.pt'
    checkpoint = torch.load(name, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss = checkpoint['loss']

    print('In [load_checkpoint]: load state dict: done')

    # putting the model in the correct mode
    model.train() if resume_train else model.eval()  # model or model_single?
    # model_single.train() if resume_train else model_single.eval()

    # return model_single, model, optimizer, loss
    return model, optimizer, loss


def translate_address(path, package):
    """
    This function changes a path which is from the project directory to a path readable by a specific package in the
    folder. For instance, the datasets paths in params.json are from the project directory. Using this function, every
    function in the data_loader package can use that address by converting it to a relative address using this function.
    The benefit is that no function needs to translate the address itself directly, and all of them could use this
    function to do so.
    :param path:
    :param package:
    :return:
    """
    if package == 'data_handler' or package == 'helper':
        return '../' + path
    else:
        raise NotImplementedError('NOT IMPLEMENTED...')


def label_to_tensor(label, height, width, count=0):
    if count == 0:
        arr = np.zeros((10, height, width))
        arr[label] = 1

    else:
        arr = np.zeros((count, 10, height, width))
        arr[:, label, :, :] = 1

    return torch.from_numpy(arr.astype(np.float32))


def make_dir_if_not_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print(f'In [make_dir_if_not_exists]: created path "{directory}"')


def reshape_cond(img_condition, h, w):
    return img_condition.rehspae((-1, h, w))  # reshaping to it could be concatenated in the channel dimension


def calc_z_shapes(n_channel, input_size, n_block):
    """
    This function calculates z shapes given the desired number of blocks in the Glow model. After each block, the
    spatial dimension is halved and the number of channels is doubled.
    :param n_channel:
    :param input_size:
    :param n_flow:
    :param n_block:
    :return:
    """
    z_shapes = []
    for i in range(n_block - 1):
        input_size = input_size // 2 if type(input_size) is int else (input_size[0] // 2, input_size[1] // 2)
        n_channel *= 2

        shape = (n_channel, input_size, input_size) if type(input_size) is int else (n_channel, *input_size)
        z_shapes.append(shape)

    # for the very last block where we have no split operation
    input_size = input_size // 2 if type(input_size) is int else (input_size[0] // 2, input_size[1] // 2)
    shape = (n_channel * 4, input_size, input_size) if type(input_size) is int else (n_channel * 4, *input_size)
    z_shapes.append(shape)

    return z_shapes


def calc_cond_shapes(orig_shape, in_channels, img_size, n_block):
    z_shapes = calc_z_shapes(in_channels, img_size, n_block)
    cond_shapes = []

    for z_shape in z_shapes:
        h, w = z_shape[1], z_shape[2]
        channels = (orig_shape[0] * orig_shape[1] * orig_shape[2]) // (h * w)  # new channels with new h and w
        cond_shapes.append((channels, h, w))

    return cond_shapes
