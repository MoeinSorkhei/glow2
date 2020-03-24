from math import log
import data_handler
import helper
import os
import torch
from torchvision import utils
from torch import nn
import numpy as np


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
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3 if type(image_size) is int else image_size[0] * image_size[1] * 3

    loss = -log(n_bins) * n_pixel  # -c in Eq. 2 of the Glow paper, discretization level = 1/256 (for instance)
    loss = loss + logdet + log_p  # log_x = logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),  # make it negative log likelihood
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def sample_z(z_shapes, n_samples, temperature, device):
    z_samples = []
    for z in z_shapes:  # temperature squeezes the Gaussian which is sampled from
        z_new = torch.randn(n_samples, *z) * temperature
        z_samples.append(z_new.to(device))

    return z_samples


def calc_val_loss(args, params, device, model, val_loader):
    # at the moment: only for Cityscapes and 'segment' condition
    if args.dataset != 'cityscapes' or args.cond_mode != 'segment':
        raise NotImplementedError

    val_loss = 0.
    for i_batch, batch in enumerate(val_loader):
        img_batch = batch['real'].to(device)
        segment_batch = batch['segment'].to(device)
        cond_name = 'city_segment' if args.model == 'glow' else 'c_flow'
        cond = (cond_name, segment_batch)

        if args.model == 'glow':
            loss, log_p, log_det = do_forward(args, params, model, img_batch, segment_batch, cond)

        elif args.model == 'c_flow':
            loss, loss_left, log_p_left, log_det_left, \
                loss_right, log_p_right, log_det_right = \
                do_forward(args, params, model, img_batch, segment_batch, cond)
        else:
            raise NotImplementedError
        val_loss += loss.item()

    return val_loss / len(val_loader)


def do_forward(args, params, model, img_batch, segment_batch, cond):
    n_bins = 2. ** params['n_bits']

    if args.model == 'glow':
        log_p, logdet, _ = model(img_batch + torch.rand_like(img_batch) / n_bins, cond)
        log_p = log_p.mean()
        logdet = logdet.mean()  # logdet and log_p: tensors of shape torch.Size([5])
        loss, log_p, log_det = calc_loss(log_p, logdet, params['img_size'], n_bins)
        return loss, log_p, log_det

    elif args.model == 'c_flow':
        left_glow_outs, right_glow_outs = model(segment_batch + torch.rand_like(segment_batch) / n_bins,
                                                img_batch + torch.rand_like(img_batch) / n_bins)

        log_p_left, log_det_left = left_glow_outs['log_p'].mean(), left_glow_outs['log_det'].mean()
        log_p_right, log_det_right = right_glow_outs['log_p'].mean(), right_glow_outs['log_det'].mean()

        loss_left, log_p_left, _ = calc_loss(log_p_left, log_det_left, params['img_size'], n_bins)
        loss_right, log_p_right, _ = calc_loss(log_p_right, log_det_right, params['img_size'], n_bins)
        loss = loss_left + loss_right
        return loss, loss_left, log_p_left, log_det_left, loss_right, log_p_right, log_det_right


def train(args, params, model, optimizer, device, comet_tracker=None,
          resume=False, last_optim_step=0, reverse_cond=None):
    # ============ setting params
    batch_size = params['batch_size'] if args.dataset == 'mnist' else params['batch_size'][args.cond_mode]
    loader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}

    # ============ initializing data loaders
    if args.dataset == 'cityscapes':
        train_loader, \
            val_loader = data_handler.init_city_loader(data_folder=params['data_folder'],
                                                       image_size=(params['img_size']),
                                                       remove_alpha=True,  # removing the alpha channel
                                                       loader_params=loader_params)
    else:  # MNIST
        data_loader = data_handler.init_mnist_loader(mnist_folder=params['data_folder'],
                                                     img_size=params['img_size'],
                                                     loader_params=loader_params)
    # ============ sampling z's
    # sampled z's used to show evolution of the generated images during training
    z_shapes = helper.calc_z_shapes(params['channels'], params['img_size'], params['n_block'])
    z_samples = sample_z(z_shapes, params['n_samples'], params['temperature'], device)

    # ============ adjusting optim step
    optim_step = last_optim_step + 1 if resume else 0
    max_optim_steps = params['iter']

    if resume:
        print(f'In [train]: resuming training from optim_step={optim_step}')

    if args.dataset == 'cityscapes':  # printing this only for cityscapes
        print(f'In [train]: training with a data loaders of size: \n'
              f'train_loader: {len(train_loader)} \n'
              f'val_loader: {len(val_loader)} \n'
              f'and batch_size of: {params["batch_size"][args.cond_mode]}')

    # ============ optimization
    while optim_step < max_optim_steps:
        for i_batch, batch in enumerate(train_loader):
            # conditional, using labels
            if args.dataset == 'mnist':
                img_batch = batch['image'].to(device)
                label_batch = batch['label'].to(device)
                cond = (args.dataset, label_batch)

            # ============ creating the data batches and the conditions
            elif args.dataset == 'cityscapes':
                img_batch = batch['real'].to(device)
                segment_batch = batch['segment'].to(device)

                # ============ segment condition
                if args.cond_mode == 'segment':  # could be refactored later so args.cond_mode is exactly 'city_segment'
                    cond_name = 'city_segment' if args.model == 'glow' else 'c_flow'
                    cond = (cond_name, segment_batch)
                    # cond = ('city_segment', segment_batch)

                # ============ segment_id condition
                elif args.cond_mode == 'segment_id':  # not updated for now
                    id_repeats_batch = batch['id_repeats'].to(device)
                    cond = ('city_segment_id', segment_batch, id_repeats_batch)

                else:
                    raise NotImplementedError('The desired condition is not implemented.')

            # unconditional, without using any labels
            else:
                raise NotImplementedError('Now only conditional implemented.')

            # ============ forward pass calculating loss
            if args.model == 'glow':
                loss, log_p, log_det = do_forward(args, params, model, img_batch, segment_batch, cond)

            elif args.model == 'c_flow':
                loss, loss_left, log_p_left, log_det_left, \
                    loss_right, log_p_right, log_det_right = \
                    do_forward(args, params, model, img_batch, segment_batch, cond)
            else:
                raise NotImplementedError

            # ============ backward pass
            model.zero_grad()
            loss.backward()

            # ============ optimizer step
            warmup_lr = params['lr']
            optimizer.param_groups[0]['lr'] = warmup_lr  # this could be removed if lr is not changing
            optimizer.step()

            if args.model == 'c_flow':
                print(f'\nloss_left: {loss_left.item():.3f} - loss_right: {loss_right.item():.3f}')

            print(f'Step: {optim_step} => loss: {loss.item():.3f}')

            # ============ validation loss
            if optim_step % params['val_freq'] == 0:
                with torch.no_grad():
                    val_loss = calc_val_loss(args, params, device, model, val_loader)
                    print(f'==== val_loss: {round(val_loss, 3)}')

            # ============ tracking by comet
            if args.use_comet:
                comet_tracker.track_metric('loss', round(loss.item(), 3), optim_step)
                # also track the val_loss at the desired frequency
                if optim_step % params['val_freq'] == 0:
                    comet_tracker.track_metric('val_loss', round(val_loss, 3), optim_step)

                if args.model == 'glow':
                    comet_tracker.track_metric('log_p', round(log_p.item(), 3), optim_step)

                if args.model == 'c_flow':
                    comet_tracker.track_metric('loss_left', round(loss_left.item(), 3), optim_step)
                    comet_tracker.track_metric('loss_right', round(loss_right.item(), 3), optim_step)

                    comet_tracker.track_metric('log_p_left', round(log_p_left.item(), 3), optim_step)
                    comet_tracker.track_metric('log_p_right', round(log_p_right.item(), 3), optim_step)

            # ============ saving samples
            if optim_step % params['sample_freq'] == 0:
                if args.dataset == 'cityscapes':
                    pth = params['samples_path']['real'][args.cond_mode][args.model]  # only 'real' for now
                else:
                    pth = params['samples_path']['conditional']

                # ============ create path if not available
                if not os.path.exists(pth):
                    os.makedirs(pth)
                    print(f'In [train]: created path "{pth}"')

                # ============ get the samples by calling reverse()
                with torch.no_grad():
                    if args.model == 'glow':
                        sampled_images = model.reverse(z_samples, cond=reverse_cond).cpu().data

                    elif args.model == 'c_flow':  # here reverse_cond is x_a
                        # ============ only sanity check for c_flow: make sure we can reconstruct the images
                        if args.sanity_check:
                            x_a, x_b = reverse_cond[0], reverse_cond[1]
                            x_a_rec, x_b_rec = model.reverse(x_a=x_a, x_b=x_b, mode='reconstruct_all')
                            x_a_rec, x_b_rec = x_a_rec.cpu().data, x_b_rec.cpu().data
                            sampled_images = torch.cat([x_a_rec, x_b_rec], dim=0)

                        # ============ real sampling
                        else:
                            sampled_images = \
                                model.reverse(x_a=reverse_cond, z_b_samples=z_samples, mode='sample_x_b').cpu().data

                    else:
                        raise NotImplementedError

                    utils.save_image(sampled_images, f'{pth}/{str(optim_step).zfill(6)}.png', nrow=10)

                print("\nSample saved at iteration", optim_step, '\n')

            # ============ saving checkpoint
            if optim_step > 0 and optim_step % params['checkpoint_freq'] == 0:
                if args.dataset == 'cityscapes':
                    pth = params['checkpoints_path']['real'][args.cond_mode][args.model]  # only 'real' for now

                else:
                    pth = params['checkpoints_path']['conditional']

                if not os.path.exists(pth):
                    os.makedirs(pth)
                    print(f'In [train]: created path "{pth}"')

                helper.save_checkpoint(pth, optim_step, model, optimizer, loss)
                print("Checkpoint saved at iteration", optim_step, '\n')

            optim_step += 1

