from torchvision import transforms
import numpy as np
import torch
from copy import deepcopy

from globals import device


def make_cond_dict(act_cond, w_cond, coupling_cond):
    return {'act_cond': act_cond, 'w_cond': w_cond, 'coupling_cond': coupling_cond}


def extract_conds(conditions, level, all_conditional):
    if all_conditional:
        act_cond = conditions['act_cond'][level]
        w_cond = conditions['w_cond'][level]
        coupling_cond = conditions['coupling_cond'][level]
    else:
        w_cond, act_cond, coupling_cond = None, None, None
    return act_cond, w_cond, coupling_cond


def to_dict(module, output):
    assert module == 'flow'
    return {
        'act_out': output[0],
        'w_out': output[1],
        'out': output[2],
        'log_det': output[3]
    }


def unsqueeze_tensor(inp):
    b_size, n_channel, height, width = inp.shape
    unsqueezed = inp.view(b_size, n_channel // 4, 2, 2, height, width)
    unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
    unsqueezed = unsqueezed.contiguous().view(
        b_size, n_channel // 4, height * 2, width * 2
    )
    return unsqueezed


def squeeze_tensor(inp):
    b_size, in_channel, height, width = inp.shape
    squeezed = inp.view(b_size, in_channel, height // 2, 2, width // 2, 2)  # squeezing height and width
    squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)  # putting 3, 5 at first to index the height and width easily
    out = squeezed.contiguous().view(b_size, in_channel * 4, height // 2, width // 2)  # squeeze into extra channels
    return out


def sample_z(n_samples, temperature, channels, img_size, n_block, split_type):
    z_shapes = calc_z_shapes(channels, img_size, n_block, split_type)
    z_samples = []
    for z in z_shapes:  # temperature squeezes the Gaussian which is sampled from
        z_new = torch.randn(n_samples, *z) * temperature
        z_samples.append(z_new.to(device))
    return z_samples


def calc_z_shapes(n_channel, image_size, n_block, split_type):
    # calculates shapes of z's after SPLIT operation (after Block operations) - e.g. channels: 6, 12, 24, 96
    z_shapes = []
    for i in range(n_block - 1):
        image_size = (image_size[0] // 2, image_size[1] // 2)
        n_channel = n_channel * 2 if split_type == 'regular' else 9  # now only supports split_sections [3, 9]

        shape = (n_channel, *image_size)
        z_shapes.append(shape)

    # for the very last block where we have no split operation
    image_size = (image_size[0] // 2, image_size[1] // 2)
    shape = (n_channel * 4, *image_size) if split_type == 'regular' else (12, *image_size)
    z_shapes.append(shape)
    return z_shapes


def calc_inp_shapes(n_channels, image_size, n_blocks, split_type):
    # calculates z shapes (inputs) after SQUEEZE operation (before Block operations) - e.g. channels: 12, 24, 48, 96
    z_shapes = calc_z_shapes(n_channels, image_size, n_blocks, split_type)
    input_shapes = []
    for i in range(len(z_shapes)):
        if i < len(z_shapes) - 1:
            channels = z_shapes[i][0] * 2 if split_type == 'regular' else 12  # now only supports split_sections [3, 9]
            input_shapes.append((channels, z_shapes[i][1], z_shapes[i][2]))
        else:
            input_shapes.append((z_shapes[i][0], z_shapes[i][1], z_shapes[i][2]))
    return input_shapes


def calc_cond_shapes(n_channels, image_size, n_blocks, split_type, condition):
    # computes additional channels dimensions based on additional conditions: left input + condition
    input_shapes = calc_inp_shapes(n_channels, image_size, n_blocks, split_type)
    cond_shapes = []
    for block_idx in range(len(input_shapes)):
        shape = [input_shapes[block_idx][0], input_shapes[block_idx][1], input_shapes[block_idx][2]]  # from left glow
        if 'b_maps' in condition:
            shape[0] += 3   # down-sampled image with 3 channels
        cond_shapes.append(tuple(shape))
    return cond_shapes


def sanity_check(x_a_ref, x_b_ref, x_a_rec, x_b_rec):
    x_a_diff = torch.mean(torch.abs(x_a_ref - x_a_rec))
    x_b_diff = torch.mean(torch.abs(x_b_ref - x_b_rec))

    ok_or_not_a = 'OK!' if x_a_diff < 1e-5 else 'NOT OK!'
    ok_or_not_b = 'OK!' if x_b_diff < 1e-5 else 'NOT OK!'

    print('=' * 100)
    print(f'In [sanity_check]: mean x_a_diff: {x_a_diff} ===> {ok_or_not_a}')
    print(f'In [sanity_check]: mean x_b_diff: {x_b_diff} ===> {ok_or_not_b}')
    print('=' * 100, '\n')
