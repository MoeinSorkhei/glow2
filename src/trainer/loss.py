from math import log
import torch
import numpy as np

import models


def forward_and_loss(args, params, model, right_img_batch, left_img_batch, boundary_batch=None):
    n_bins = 2. ** params['n_bits']

    log_p_left, log_det_left, log_p_right, log_det_right = \
        models.do_forward(args, params, model, right_img_batch, left_img_batch, boundary_batch=boundary_batch)

    loss_left, log_p_left, _ = calc_loss(log_p_left, log_det_left, params['img_size'], n_bins)
    loss_right, log_p_right, _ = calc_loss(log_p_right, log_det_right, params['img_size'], n_bins)
    loss = loss_left + loss_right

    return loss, loss_left, loss_right


def calc_loss(log_p, logdet, image_size, n_bins):  # how does it work
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


def calc_val_loss(args, params, device, model, val_loader):
    print(f'In [calc_val_loss]: computing validation loss for data loader of len: {len(val_loader)} '
          f'and batch size: {params["batch_size"]}')

    with torch.no_grad():
        val_list = []
        for i_batch, batch in enumerate(val_loader):  # IMRPVE: SHOULD USE make_batch function
            if args.dataset == 'cityscapes':
                right_img_batch = batch['real'].to(device)
                left_img_batch = batch['segment'].to(device)
                boundary_batch = batch['boundary'].to(device) if args.cond_mode == 'segment_boundary' else None

            elif args.dataset == 'maps':
                right_img_batch = batch['photo'].to(device)
                left_img_batch = batch['the_map'].to(device)
                boundary_batch = None

            else:
                raise NotImplementedError

            # elif args.dataset == 'transient':  # IMRPVE
            #     right_img_batch = batch['right'].to(device)
            #     left_img_batch = batch['left'].to(device)
            #     boundary_batch = None

            # ======== creating the condition __ NOT NEEDED; IMPROVE
            if args.cond_mode is None:  # use: only vanilla Glow on segmentations for now
                cond = None  # no condition for training Glow on segmentations

            else:  # use: c_flow with cond_mode = segmentation --- TO BE REFACTORED
                cond_name = 'city_segment' if args.model == 'glow' else 'c_flow'
                # cond = (cond_name, left_img_batch)

            if args.model == 'glow':
                # TO BE REFACTORED FOR GLOW
                loss, log_p, log_det = models.do_forward(args, params, model, right_img_batch, left_img_batch, cond)

            elif args.model == 'c_flow':
                loss, loss_left, loss_right = \
                    forward_and_loss(args, params, model, right_img_batch, left_img_batch, boundary_batch)

            else:
                raise NotImplementedError
            # val_loss += loss.item()
            val_list.append(loss.item())

        # return val_loss / len(val_loader)
        return np.mean(val_list), np.std(val_list)