from math import log
import data_handler
import helper
import os
import torch
from torchvision import utils
from torch import nn


def calc_loss(log_p, logdet, image_size, n_bins):  # how does it work
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
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
        input_size //= 2
        n_channel *= 2
        z_shapes.append((n_channel, input_size, input_size))

    # for the very last block where we have no split operation
    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))
    return z_shapes


def train(args, params, model, model_single, optimizer, in_channels,
          device, comet_tracker=None, resume=False, last_optim_step=0):
    loader_params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': 0}

    if args.dataset == 'cityscapes_segmentation' or args.dataset == 'cityscapes_leftImg8bit':
        data_loader = data_handler.init_city_loader(data_folder=params['data_folder'],
                                                    dataset_name=args.dataset,
                                                    image_size=args.img_size,
                                                    remove_alpha=True,  # removing the alpha channel
                                                    loader_params=loader_params)
    else:  # MNIST
        data_loader = data_handler.init_mnist_loader(mnist_folder=params['data_folder'],
                                                     img_size=args.img_size,
                                                     loader_params=loader_params)

    n_bins = 2. ** params['n_bits']
    z_sample = []  # sampled z's used to show evolution of the generated images during training
    z_shapes = calc_z_shapes(in_channels, args.img_size, params['n_flow'], params['n_block'])

    for z in z_shapes:  # temperature squeezes the Gaussian which is sampled from
        z_new = torch.randn(params['n_samples'], *z) * params['temperature']
        z_sample.append(z_new.to(device))

    optim_step = last_optim_step + 1 if resume else 0
    max_optim_steps = params['iter']

    if resume:
        print(f'In [train]: resuming training from optim_step={optim_step}')

    print(f'In [train]: training with a data loader of size: {len(data_loader)}')
    while optim_step < max_optim_steps:
        for i_batch, batch in enumerate(data_loader):
            batch = batch.to(device)

            # I do not know this
            if optim_step == 0:
                with torch.no_grad():  # why
                    log_p, logdet, _ = model.module(batch + torch.rand_like(batch) / n_bins)
                    optim_step += 1
                    continue
            else:
                log_p, logdet, _ = model(batch + torch.rand_like(batch) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
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
                if not os.path.exists('samples'):
                    os.mkdir('samples')
                    print('In [train]: created path "samples"...')

                with torch.no_grad():
                    sampled_images = model_single.reverse(z_sample).cpu().data  # why .CPU?, why model_single?
                    utils.save_image(sampled_images, f'samples/{str(optim_step).zfill(6)}.png', nrow=10)

                    '''utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f'sample/{str(optim_step).zfill(6)}.png',
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )'''
                print("\nSample saved at iteration", optim_step, '\n')

            if optim_step % 1000 == 0:
                if not os.path.exists('checkpoints'):
                    os.mkdir('checkpoints')
                    print('In [train]: created path "checkpoints"...')

                # torch.save(model.state_dict(), f'checkpoints/model_{str(optim_step).zfill(6)}.pt')
                # torch.save(optimizer.state_dict(), f'checkpoints/optim_{str(optim_step).zfill(6)}.pt')

                helper.save_checkpoint(params['checkpoints_path'], model, optimizer, optim_step, loss)
                print("Checkpoint saved at iteration", optim_step, '\n')

            optim_step += 1


# def resume_train(optim_step, arguments, params, device):
#     # optim_step = 9000
#     model_path = f'checkpoints/model_{str(optim_step).zfill(6)}.pt'
#     optim_path = f'checkpoints/optim_{str(optim_step).zfill(6)}.pt'
#     model_single, model, optimizer = \
#         helper.load_model_and_optimizer(model_path, optim_path, params, device, resume_train=True)
#
#     train(arguments, params, model, model_single, optimizer, params['channels'],
#           device, comet_tracker=None, resume=True, last_optim_step=optim_step)
#
#
# def resume_training_prev(glow, opt, optim_step, arguments, parameters, input_channels, comet_tracker):
#     model_path = f'checkpoints/model_{str(optim_step).zfill(6)}.pt'
#     optim_path = f'checkpoints/optim_{str(optim_step).zfill(6)}.pt'
#
#     glow.load_state_dict(torch.load(model_path))
#     glow = nn.DataParallel(glow)
#     # print('model load')
#     # input()
#
#     opt.load_state_dict(torch.load(optim_path))
#
#     # print('optim load')
#     # input()
#
#     # need to also save the z variables if we want to resume with exactly the same samples +
#     # optim_step, or epoch + last train loss
#     glow.train()  # set to train mode
#     train(arguments, parameters, glow, opt, input_channels, comet_tracker, resume=True, last_optim_step=optim_step)
