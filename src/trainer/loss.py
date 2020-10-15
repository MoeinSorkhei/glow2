from math import log
import torch
import numpy as np

import data_handler
import models
from globals import device


def forward_and_loss(args, params, model, left_batch, right_batch, extra_cond_batch):
    n_bins = 2. ** params['n_bits']

    if 'c_glow' in args.model:
        z, nll = model(x=noise_added(left_batch, n_bins),
                       y=noise_added(right_batch, n_bins))
        loss = torch.mean(nll)
        return {'loss': loss, 'z': z}

    else:
        left_glow_outs, right_glow_outs = model(x_a=noise_added(left_batch, n_bins),
                                                x_b=noise_added(right_batch, n_bins),
                                                extra_cond=extra_cond_batch)
        log_p_left, log_det_left = left_glow_outs['log_p'].mean(), left_glow_outs['log_det'].mean()
        log_p_right, log_det_right = right_glow_outs['log_p'].mean(), right_glow_outs['log_det'].mean()

        loss_left, _, _ = calc_loss(log_p_left, log_det_left, params['img_size'], n_bins)
        loss_right, _, _ = calc_loss(log_p_right, log_det_right, params['img_size'], n_bins)
        loss = args.reg_factor * loss_left + loss_right
        return {'loss': loss, 'loss_left': loss_left, 'loss_right': loss_right}


def noise_added(batch, n_bins):  # add uniform noise
    return batch + torch.rand_like(batch) / n_bins


def calc_loss(log_p, logdet, image_size, n_bins):
    """
    :param log_p:
    :param logdet:
    :param image_size:
    :param n_bins:
    :return:

    Note: by having 8 bits, that is, discretization level of 1/256:
    loss = -log(n_bins) * n_pixel  ==> this line computes -c: log(1/256) = -log(256)
    Then this values is added to log likelihood, and finally in the return the value is negated to denote
    the negative log-likelihood.
    """
    n_pixel = image_size * image_size * 3 if type(image_size) is int else image_size[0] * image_size[1] * 3

    loss = -log(n_bins) * n_pixel  # -c in Eq. 2 of the Glow paper, discretization level = 1/256 (for instance)
    loss = loss + logdet + log_p  # log_x = logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),  # make it negative log likelihood
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def compute_val_bpd(args, params, model, val_loader):
    val_loss_mean, loss_right_mean = calc_val_loss(args, params, model, val_loader)
    print(f'====== In [train]: val_loss mean: {round(val_loss_mean, 3)} - loss_right_mean: {round(loss_right_mean, 3)}')
    print('waiting for input')
    input()


def calc_val_loss(args, params, model, val_loader):
    print(f'In [calc_val_loss]: computing validation loss for data loader of len: {len(val_loader)} '
          f'and batch size: {params["batch_size"]}')

    with torch.no_grad():
        val_list, loss_right_list = [], []
        for i_batch, batch in enumerate(val_loader):
            left_batch, right_batch, extra_cond_batch = data_handler.extract_batches(batch, args)
            forward_output = forward_and_loss(args, params, model, left_batch, right_batch, extra_cond_batch)
            loss, loss_left, loss_right = forward_output['loss'], forward_output['loss_left'], forward_output['loss_right']
            val_list.append(loss.item())
            loss_right_list.append(loss_right.item())
        # return np.mean(val_list), np.std(val_list)
        return np.mean(val_list), np.mean(loss_right_list)
