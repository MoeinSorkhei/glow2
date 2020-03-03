from math import log
import data_handler
import helper
import os
import torch
from torchvision import utils
from torch import nn
import numpy as np


def calc_loss(log_p, logdet, image_size, n_bins):  # how does it work
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3 if type(image_size) is int else image_size[0] * image_size[1] * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def sample_z(z_shapes, n_samples, temperature, device):
    z_samples = []
    for z in z_shapes:  # temperature squeezes the Gaussian which is sampled from
        z_new = torch.randn(n_samples, *z) * temperature
        z_samples.append(z_new.to(device))

    return z_samples


def train(args, params, model, optimizer, device, comet_tracker=None,
          resume=False, last_optim_step=0, reverse_cond=None):
    batch_size = params['batch_size'] if args.dataset == 'mnist' else params['batch_size'][args.cond_mode]
    loader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}

    if args.dataset == 'cityscapes':
        data_loader = data_handler.init_city_loader(data_folder=params['data_folder'],
                                                    image_size=(params['img_size']),
                                                    remove_alpha=True,  # removing the alpha channel
                                                    loader_params=loader_params)
    else:  # MNIST
        data_loader = data_handler.init_mnist_loader(mnist_folder=params['data_folder'],
                                                     img_size=params['img_size'],
                                                     loader_params=loader_params)
    in_channels = params['channels']
    n_bins = 2. ** params['n_bits']
    z_shapes = helper.calc_z_shapes(in_channels, params['img_size'], params['n_block'])

    # sampled z's used to show evolution of the generated images during training
    z_samples = sample_z(z_shapes, params['n_samples'], params['temperature'], device)

    optim_step = last_optim_step + 1 if resume else 0
    max_optim_steps = params['iter']

    if resume:
        print(f'In [train]: resuming training from optim_step={optim_step}')

    print(f'In [train]: training with a data loader of size: {len(data_loader)} '
          f'and batch size of: {params["batch_size"]}')

    while optim_step < max_optim_steps:
        for i_batch, batch in enumerate(data_loader):
            # conditional, using labels
            if args.dataset == 'mnist':
                img_batch = batch['image'].to(device)
                label_batch = batch['label'].to(device)
                cond = (args.dataset, label_batch)

            elif args.dataset == 'cityscapes':
                img_batch = batch['real'].to(device)
                segment_batch = batch['segment'].to(device)

                if args.cond_mode == 'segment':
                    cond = ('city_segment', segment_batch)

                elif args.cond_mode == 'segment_id':
                    id_repeats_batch = batch['id_repeats'].to(device)
                    cond = ('city_segment_id', segment_batch, id_repeats_batch)

                    # print('in train: id_repeats_batch shape:', id_repeats_batch.shape, 'segment_batch shape:', segment_batch.shape)
                    # input()

            # unconditional, without using any labels
            else:
                raise NotImplementedError('Now only conditional implemented.')
                # img_batch = batch.to(device)
                # cond = None

            # I do not know this
            if optim_step == 0:
                with torch.no_grad():  # why
                    log_p, logdet, _ = model(img_batch + torch.rand_like(img_batch) / n_bins, cond)
                    optim_step += 1
                    continue
            else:
                log_p, logdet, _ = model(img_batch + torch.rand_like(img_batch) / n_bins, cond)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, params['img_size'], n_bins)
            model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = params['lr']
            optimizer.param_groups[0]['lr'] = warmup_lr
            optimizer.step()

            print(f'Step: {optim_step} => loss: {loss.item():.3f}, log_p: {log_p.item():.3f}')

            if args.use_comet:
                comet_tracker.track_metric('loss', round(loss.item(), 3), optim_step)

            if optim_step % 100 == 0:
                if args.dataset == 'cityscapes':
                    pth = params['samples_path']['real'][args.cond_mode]  # only 'real' for now
                else:
                    pth = params['samples_path']['conditional']

                if not os.path.exists(pth):
                    os.makedirs(pth)
                    print(f'In [train]: created path "{pth}"')

                with torch.no_grad():
                    sampled_images = model.reverse(z_samples, cond=reverse_cond).cpu().data
                    utils.save_image(sampled_images, f'{pth}/{str(optim_step).zfill(6)}.png', nrow=10)

                    '''utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f'sample/{str(optim_step).zfill(6)}.png',
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )'''
                print("\nSample saved at iteration", optim_step, '\n')

            if optim_step % 1000 == 0:
                if args.dataset == 'cityscapes':
                    pth = params['checkpoints_path']['real'][args.cond_mode]  # only 'real' for now
                else:
                    pth = params['checkpoints_path']['conditional']

                if not os.path.exists(pth):
                    os.makedirs(pth)
                    print(f'In [train]: created path "{pth}"')

                helper.save_checkpoint(pth, optim_step, model, optimizer, loss)
                print("Checkpoint saved at iteration", optim_step, '\n')

            optim_step += 1

