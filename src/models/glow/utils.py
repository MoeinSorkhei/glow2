from torchvision import transforms
import numpy as np
import torch
from copy import deepcopy

from globals import device


def make_cond_dict(act_cond, w_cond, coupling_cond):
    return {'act_cond': act_cond, 'w_cond': w_cond, 'coupling_cond': coupling_cond}


def extract_conds(conditions, level, layers_conditional):
    if layers_conditional:
        act_cond = conditions['act_cond'][level]
        w_cond = conditions['w_cond'][level]
        coupling_cond = conditions['coupling_cond'][level]
    else:
        w_cond, act_cond, coupling_cond = None, None, None
    return act_cond, w_cond, coupling_cond


def prep_conds(left_glow_out, direction):
    left_glow_w_outs = left_glow_out['all_w_outs']
    left_glow_act_outs = left_glow_out['all_act_outs']
    left_coupling_outs = left_glow_out['all_flows_outs']
    conditions = make_cond_dict(left_glow_act_outs, left_glow_w_outs, left_coupling_outs)

    if direction == 'reverse':  # reverse lists
        conditions['act_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['act_cond']))]  # reverse 2d list
        conditions['w_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['w_cond']))]
        conditions['coupling_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['coupling_cond']))]
    return conditions


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


def sample_z(n_samples, temperature, channels, img_size, n_block):
    z_shapes = calc_z_shapes(channels, img_size, n_block)
    z_samples = []
    for z in z_shapes:  # temperature squeezes the Gaussian which is sampled from
        z_new = torch.randn(n_samples, *z) * temperature
        z_samples.append(z_new.to(device))
    return z_samples


def calc_z_shapes(n_channel, input_size, n_block):
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


def calc_cond_shapes(params, mode):
    in_channels, img_size, n_block = params['channels'], params['img_size'], params['n_block']
    z_shapes = calc_z_shapes(in_channels, img_size, n_block)

    if mode == 'z_outs':  # the condition is has the same shape as the z's themselves
        return z_shapes

    for i in range(len(z_shapes)):
        z_shapes[i] = list(z_shapes[i])  # converting the tuple to list
        if i < len(z_shapes) - 1:
            z_shapes[i][0] = z_shapes[i][0] * 2  # extra channel dim for zA coming from the left glow
            if mode is not None and mode == 'segment_boundary':
                z_shapes[i][0] += 12  # extra channel dimension for the boundary

        elif mode is not None and mode == 'segment_boundary':  # last layer - adding dim only for boundaries
            # no need to have z_shapes[i][0] * 2 since this layer does not have split
            z_shapes[i][0] += 12  # extra channel dimension for the boundary

        z_shapes[i] = tuple(z_shapes[i])  # convert back to tuple

    return z_shapes


def compute_inp_shapes(n_channels, input_size, n_blocks):
    z_shapes = calc_z_shapes(n_channels, input_size, n_blocks)
    input_shapes = []
    for i in range(len(z_shapes)):
        if i < len(z_shapes) - 1:
            input_shapes.append((z_shapes[i][0] * 2, z_shapes[i][1], z_shapes[i][2]))
        else:
            input_shapes.append((z_shapes[i][0], z_shapes[i][1], z_shapes[i][2]))
    return input_shapes


def sanity_check(x_a_ref, x_b_ref, x_a_rec, x_b_rec):
    x_a_diff = torch.mean(torch.abs(x_a_ref - x_a_rec))
    x_b_diff = torch.mean(torch.abs(x_b_ref - x_b_rec))

    ok_or_not_a = 'OK!' if x_a_diff < 1e-5 else 'NOT OK!'
    ok_or_not_b = 'OK!' if x_b_diff < 1e-5 else 'NOT OK!'

    print('=' * 100)
    print(f'In [sanity_check]: mean x_a_diff: {x_a_diff} ===> {ok_or_not_a}')
    print(f'In [sanity_check]: mean x_b_diff: {x_b_diff} ===> {ok_or_not_b}')
    print('=' * 100, '\n')
