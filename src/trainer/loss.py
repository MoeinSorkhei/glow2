from math import log
import torch
import numpy as np

import models
from .util import *
from globals import device


def forward_and_loss(args, params, model, img_batch, segment_batch, boundary_batch=None):
    n_bins = 2. ** params['n_bits']

    # log_p_left, log_det_left, log_p_right, log_det_right = \
    # forward_out = models.do_forward(args, params, model, img_batch, segment_batch, boundary_batch=boundary_batch)
    if args.model == 'c_flow':
        log_p_left, log_det_left, log_p_right, log_det_right = \
             models.do_forward(args, params, model, img_batch, segment_batch, boundary_batch=boundary_batch)

        loss_left, log_p_left, _ = calc_loss(log_p_left, log_det_left, params['img_size'], n_bins)
        loss_right, log_p_right, _ = calc_loss(log_p_right, log_det_right, params['img_size'], n_bins)
        loss = loss_left + loss_right

        return loss, loss_left, loss_right

    # elif args.model == 'c_glow':
    elif 'c_glow' in args.model:
        z, nll = models.do_forward(args, params, model, img_batch, segment_batch, boundary_batch=boundary_batch)
        loss = torch.mean(nll)
        return z, loss

    else:
        raise NotImplementedError


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


def calc_val_loss(args, params, model, val_loader):
    print(f'In [calc_val_loss]: computing validation loss for data loader of len: {len(val_loader)} '
          f'and batch size: {params["batch_size"]}')

    with torch.no_grad():
        val_list = []

        for i_batch, batch in enumerate(val_loader):
            # boundary_batch is None for datasets other than cityscapes
            img_batch, segment_batch, boundary_batch = extract_batches(batch, args)

            if args.model == 'glow':
                cond = None
                # TO BE REFACTORED FOR GLOW
                loss, log_p, log_det = models.do_forward(args, params, model, img_batch, segment_batch, cond)

            elif args.model == 'c_flow':
                loss, loss_left, loss_right = \
                    forward_and_loss(args, params, model, img_batch, segment_batch, boundary_batch)

            # elif args.model == 'c_glow':
            elif 'c_glow' in args.model:
                z, loss = forward_and_loss(args, params, model, img_batch, segment_batch)

            else:
                raise NotImplementedError
            val_list.append(loss.item())
        return np.mean(val_list), np.std(val_list)