from torchvision import utils

import data_handler
from data_handler import transient
import helper
from .loss import *


def train(args, params, model, optimizer, comet_tracker=None, resume=False, last_optim_step=0, reverse_cond=None):
    # getting data loaders
    train_loader, val_loader = data_handler.init_data_loaders(args, params)

    # adjusting optim step
    optim_step = last_optim_step + 1 if resume else 0
    max_optim_steps = params['iter']
    paths = helper.compute_paths(args, params)

    if resume:
        print(f'In [train]: resuming training from optim_step={optim_step} - max_step: {max_optim_steps}')

    # optimization loop
    while optim_step < max_optim_steps:
        for i_batch, batch in enumerate(train_loader):
            if optim_step > max_optim_steps:
                print(f'In [train]: reaching max_step: {max_optim_steps}. Terminating...')
                return  # ============ terminate training if max steps reached

            # forward pass
            img_batch, segment_batch, boundary_batch = extract_batches(batch, args)
            forward_output = forward_and_loss(args, params, model, img_batch, segment_batch, boundary_batch)
            loss = forward_output['loss']
            metrics = {'loss': loss}

            # also add left and right loss if available
            if 'loss_left' in forward_output.keys():
                metrics = {'loss_right': forward_output['loss_right'], 'loss_left': forward_output['loss_left']}

            # backward pass and optimizer step
            model.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'In [train]: Step: {optim_step} => loss: {loss.item():.3f}')

            # validation loss
            if params['monitor_val'] and optim_step % params['val_freq'] == 0:
                val_loss_mean, _ = calc_val_loss(args, params, model, val_loader)
                metrics['val_loss'] = val_loss_mean
                print(f'====== In [train]: val_loss mean: {round(val_loss_mean, 3)}')

            # tracking metrics
            if args.use_comet:
                for key, value in metrics.items():  # track all metric values
                    comet_tracker.track_metric(key, round(value.item(), 3), optim_step)

            # saving samples
            if optim_step % params['sample_freq'] == 0:
                samples_path = paths['samples_path']
                helper.make_dir_if_not_exists(samples_path)
                sampled_images = models.take_samples(args, params, model, reverse_cond)
                utils.save_image(sampled_images, f'{samples_path}/{str(optim_step).zfill(6)}.png', nrow=10)
                print(f'\nIn [train]: Sample saved at iteration {optim_step} to: \n"{samples_path}"\n')

            # saving checkpoint
            if optim_step > 0 and optim_step % params['checkpoint_freq'] == 0:
                checkpoints_path = paths['checkpoints_path']
                helper.make_dir_if_not_exists(checkpoints_path)
                helper.save_checkpoint(checkpoints_path, optim_step, model, optimizer, loss)
                print("In [train]: Checkpoint saved at iteration", optim_step, '\n')

            optim_step += 1
