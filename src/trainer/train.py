from torchvision import utils

import data_handler
from data_handler import transient
import helper
from .loss import *


def track_metrics(args, params, comet_tracker, metrics, optim_step):
    # ============ tracking by comet
    comet_tracker.track_metric('loss', round(metrics['loss'].item(), 3), optim_step)
    # also track the val_loss at the desired frequency
    if params['monitor_val'] and optim_step % params['val_freq'] == 0:
        comet_tracker.track_metric('val_loss', round(metrics['val_loss'], 3), optim_step)

    if args.model == 'glow':
        comet_tracker.track_metric('log_p', round(metrics['log_p'].item(), 3), optim_step)

    if args.model == 'c_flow':
        comet_tracker.track_metric('loss_right', round(metrics['loss_right'].item(), 3), optim_step)
        # comet_tracker.track_metric('log_p_right', round(metrics['log_p_right'].item(), 3), optim_step)


def train(args, params, model, optimizer, device, comet_tracker=None,
          resume=False, last_optim_step=0, reverse_cond=None):
    # ============ setting params
    batch_size = params['batch_size']
    loader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}

    # ============ initializing data loaders
    if args.dataset == 'cityscapes':  # IMPROVE HERE, in a separate function
        train_loader, \
            val_loader = data_handler.init_city_loader(data_folder=params['data_folder'],
                                                       image_size=(params['img_size']),
                                                       remove_alpha=True,  # removing the alpha channel
                                                       loader_params=loader_params)

    elif args.dataset == 'maps':
        train_loader, val_loader = data_handler.maps.init_maps_loaders(args, params)

    elif args.dataset == 'transient':
        train_loader, val_loader, _ = transient.init_transient_loaders(args, params)

    elif args.dataset == 'mnist':  # MNIST
        train_loader = data_handler.init_mnist_loader(mnist_folder=params['data_folder'],
                                                      img_size=params['img_size'],
                                                      loader_params=loader_params)
    else:
        raise NotImplementedError

    print(f'In [train]: training with data loaders of size: \n'
          f'train_loader: {len(train_loader):,} \n'
          f'val_loader: {len(val_loader):,} \n'
          f'and batch_size of: {batch_size}')

    # ============ adjusting optim step
    optim_step = last_optim_step + 1 if resume else 0
    max_optim_steps = params['iter']

    # ============ show model params
    helper.print_info(args, params, model, which_info='model')

    if resume:
        print(f'In [train]: resuming training from optim_step={optim_step} - max_step: {max_optim_steps}')

    # ============ computing paths for samples, checkpoints, etc. based on the args and params
    paths = helper.compute_paths(args, params)

    # ============ optimization
    while optim_step < max_optim_steps:
        for i_batch, batch in enumerate(train_loader):
            if optim_step > max_optim_steps:
                print(f'In [train]: reaching max_step: {max_optim_steps}. Terminating...')
                return  # ============ terminate training if max steps reached

            # ============ forward pass, calculating loss
            # need to re-write here for MNIST
            if args.model == 'glow':
                right_img_batch = batch['real'].to(device)
                left_img_batch = batch['segment'].to(device)
                # right_img_batch, left_img_batch = batches['right_img_batch'], batches['left_img_batch']
                loss, log_p, log_det = models.do_forward(args, params, model, right_img_batch, left_img_batch)
                metrics = {'loss': loss, 'log_p': log_p}

            elif args.model == 'c_flow':
                if args.dataset == 'cityscapes':
                    right_img_batch = batch['real'].to(device)
                    left_img_batch = batch['segment'].to(device)

                    if args.direction == 'label2photo':
                        boundary_batch = batch['boundary'].to(device) if args.cond_mode == 'segment_boundary' else None
                    else:  # 'photo2label'
                        boundary_batch = None

                elif args.dataset == 'maps':
                    right_img_batch = batch['photo'].to(device)
                    left_img_batch = batch['the_map'].to(device)
                    boundary_batch = None

                elif args.dataset == 'transient':
                    right_img_batch = batch['right'].to(device)
                    left_img_batch = batch['left'].to(device)
                    boundary_batch = None

                else:
                    raise NotImplementedError

                loss, loss_left, loss_right = \
                    forward_and_loss(args, params, model, right_img_batch, left_img_batch, boundary_batch)

                metrics = {'loss': loss,
                           'loss_right': loss_right}

                if args.left_pretrained:
                    pass
                else:  # normal c_flow OR pre-trained left glow unfreezed
                    metrics['loss_left'] = loss_left
            else:
                raise NotImplementedError

            # ============ backward pass and optimizer step
            model.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'In [train]: Step: {optim_step} => loss: {loss.item():.3f}')

            # ============ validation loss
            if params['monitor_val'] and optim_step % params['val_freq'] == 0:
                val_loss_mean, _ = calc_val_loss(args, params, device, model, val_loader)
                metrics['val_loss'] = val_loss_mean
                print(f'====== In [train]: val_loss mean: {round(val_loss_mean, 3)}')

            # ============ tracking metrics
            if args.use_comet:
                track_metrics(args, params, comet_tracker, metrics, optim_step)

            # ============ saving samples
            if optim_step % params['sample_freq'] == 0:
                samples_path = paths['samples_path']
                helper.make_dir_if_not_exists(samples_path)
                sampled_images = models.take_samples(args, params, model, reverse_cond)
                utils.save_image(sampled_images, f'{samples_path}/{str(optim_step).zfill(6)}.png', nrow=10)
                print(f'\nIn [train]: Sample saved at iteration {optim_step} to: \n"{samples_path}"\n')

            # ============ saving checkpoint
            if optim_step > 0 and optim_step % params['checkpoint_freq'] == 0:
                checkpoints_path = paths['checkpoints_path']
                helper.make_dir_if_not_exists(checkpoints_path)
                helper.save_checkpoint(checkpoints_path, optim_step, model, optimizer, loss)
                print("In [train]: Checkpoint saved at iteration", optim_step, '\n')

            optim_step += 1
