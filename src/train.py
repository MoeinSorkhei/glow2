from math import log
import data_handler
import helper
import torch
from torchvision import utils
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
    print(f'In [calc_val_loss]: computing validation loss for data loader of len: {len(val_loader)} '
          f'and batch size: {params["batch_size"]}')

    with torch.no_grad():
        # at the moment: only for Cityscapes
        if args.dataset != 'cityscapes':  # or args.cond_mode != 'segment':
            raise NotImplementedError

        # val_loss = 0.
        val_list = []
        for i_batch, batch in enumerate(val_loader):
            img_batch = batch['real'].to(device)
            segment_batch = batch['segment'].to(device)
            boundary_batch = batch['boundary'].to(device) if args.cond_mode == 'segment_boundary' else None

            # ======== creating the condition
            if args.cond_mode is None:  # use: only vanilla Glow on segmentations for now
                cond = None  # no condition for training Glow on segmentations

            else:  # use: c_flow with cond_mode = segmentation --- TO BE REFACTORED
                cond_name = 'city_segment' if args.model == 'glow' else 'c_flow'
                # cond = (cond_name, segment_batch)

            if args.model == 'glow':
                loss, log_p, log_det = do_forward(args, params, model, img_batch, segment_batch, cond)

            elif args.model == 'c_flow':
                loss, loss_left, log_p_left, log_det_left, \
                    loss_right, log_p_right, log_det_right = \
                    do_forward(args, params, model, img_batch, segment_batch, boundary_batch=boundary_batch)

            else:
                raise NotImplementedError
            # val_loss += loss.item()
            val_list.append(loss.item())

        # return val_loss / len(val_loader)
        return np.mean(val_list), np.std(val_list)


def do_forward(args, params, model, img_batch, segment_batch, boundary_batch=None):
    n_bins = 2. ** params['n_bits']

    if args.model == 'glow':
        if args.train_on_segment:  # vanilla Glow on segmentation
            log_p, logdet, _ = model(inp=segment_batch + torch.rand_like(segment_batch) / n_bins)

        else:  # vanilla Glow on real images
            log_p, logdet, _ = model(inp=img_batch + torch.rand_like(img_batch) / n_bins)

        log_p = log_p.mean()
        logdet = logdet.mean()  # logdet and log_p: tensors of shape torch.Size([5])
        loss, log_p, log_det = calc_loss(log_p, logdet, params['img_size'], n_bins)
        return loss, log_p, log_det

    elif args.model == 'c_flow':
        left_glow_outs, right_glow_outs = model(x_a=segment_batch + torch.rand_like(segment_batch) / n_bins,
                                                x_b=img_batch + torch.rand_like(img_batch) / n_bins,
                                                b_map=boundary_batch)

        log_p_left, log_det_left = left_glow_outs['log_p'].mean(), left_glow_outs['log_det'].mean()
        log_p_right, log_det_right = right_glow_outs['log_p'].mean(), right_glow_outs['log_det'].mean()

        loss_left, log_p_left, _ = calc_loss(log_p_left, log_det_left, params['img_size'], n_bins)
        loss_right, log_p_right, _ = calc_loss(log_p_right, log_det_right, params['img_size'], n_bins)
        loss = loss_left + loss_right
        return loss, loss_left, log_p_left, log_det_left, loss_right, log_p_right, log_det_right


def take_samples(args, model, z_samples, reverse_cond):
    with torch.no_grad():
        if args.model == 'glow':
            sampled_images = model.reverse(z_samples, coupling_conds=reverse_cond).cpu().data

        elif args.model == 'c_flow':  # here reverse_cond is x_a
            # ============ only sanity check for c_flow: make sure we can reconstruct the images
            if args.sanity_check:
                x_a, x_b = reverse_cond[0], reverse_cond[1]
                x_a_rec, x_b_rec = model.reverse(x_a=x_a, x_b=x_b, mode='reconstruct_all')
                x_a_rec, x_b_rec = x_a_rec.cpu().data, x_b_rec.cpu().data
                sampled_images = torch.cat([x_a_rec, x_b_rec], dim=0)
            # ============ real sampling
            else:
                sampled_images = model.reverse(x_a=reverse_cond['segment'],
                                               b_map=reverse_cond['boundary'],
                                               z_b_samples=z_samples,
                                               mode='sample_x_b').cpu().data
        else:
            raise NotImplementedError
        return sampled_images


def take_samples2():
    pass


def track_metrics(args, params, comet_tracker, metrics, optim_step):
    # ============ tracking by comet
    comet_tracker.track_metric('loss', round(metrics['loss'].item(), 3), optim_step)
    # also track the val_loss at the desired frequency
    if params['monitor_val'] and optim_step % params['val_freq'] == 0:
        comet_tracker.track_metric('val_loss', round(metrics['val_loss'], 3), optim_step)

    if args.model == 'glow':
        comet_tracker.track_metric('log_p', round(metrics['log_p'].item(), 3), optim_step)

    if args.model == 'c_flow':
        # if args.left_unfreeze:  # track left metrics only if left_glow is trainable
        #    comet_tracker.track_metric('loss_left', round(metrics['loss_left'].item(), 3), optim_step)
        #    comet_tracker.track_metric('log_p_left', round(metrics['log_p_left'].item(), 3), optim_step)

        comet_tracker.track_metric('loss_right', round(metrics['loss_right'].item(), 3), optim_step)
        comet_tracker.track_metric('log_p_right', round(metrics['log_p_right'].item(), 3), optim_step)


def prepare_batch(args, batch, device):
    # conditional, using labels
    if args.dataset == 'mnist':
        img_batch = batch['image'].to(device)
        label_batch = batch['label'].to(device)
        cond = (args.dataset, label_batch)

        return {'img_batch': img_batch, 'label_batch': label_batch, 'cond': cond}

    # ============ creating the data batches and the conditions
    elif args.dataset == 'cityscapes':
        img_batch = batch['real'].to(device)
        segment_batch = batch['segment'].to(device)
        boundary_batch = batch['boundary'].to(device)

        if args.model == 'glow':
            raise NotImplementedError('Needs to be refactored for Glow')

        # ======================== Assumption: working with c_flow
        # ============ segment condition
        # if args.cond_mode == 'segment':  # could be refactored later so args.cond_mode is exactly 'city_segment'
            # cond_name = 'city_segment' if args.model == 'glow' else 'c_flow'  # NOTE: this should be refactored
            # cond = (cond_name, segment_batch)
            # cond = ('city_segment', segment_batch)
        # elif args.cond_mode == 'segment_boundary':  # NOTE: this should be refactored
        #   cond_name = 'c_flow'  # only support for c_flow now
        #    bmap = None
        #    cond = (cond_name, torch.cat([segment_batch, bmap], dim=0))  # channel-wise concat of segment and bmap

        # ============ segment_id condition: NEEDS TO BE REFACTORED
        if args.cond_mode == 'segment_id':  # not updated for now
            id_repeats_batch = batch['id_repeats'].to(device)
            cond = ('city_segment_id', segment_batch, id_repeats_batch)

        # ============ unconditional (for vanilla Glow) - no condition
        else:  # NEEDS TO BE REFACTORED
            # if args.model == 'c_flow':
            #    raise NotImplementedError('The desired condition is not implemented.')
            cond = None

        return {'img_batch': img_batch,
                'segment_batch': segment_batch,
                'boundary_batch': boundary_batch}
                # 'cond': cond}

    # unsupported dataset
    else:
        raise NotImplementedError('Dataset should be specified/supported.')


def train(args, params, model, optimizer, device, comet_tracker=None,
          resume=False, last_optim_step=0, reverse_cond=None):
    # ============ setting params
    batch_size = params['batch_size']
    loader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}

    # ============ initializing data loaders
    if args.dataset == 'cityscapes':
        train_loader, \
            val_loader = data_handler.init_city_loader(data_folder=params['data_folder'],
                                                       image_size=(params['img_size']),
                                                       remove_alpha=True,  # removing the alpha channel
                                                       loader_params=loader_params)
        print(f'In [train]: training with data loaders of size: \n'
              f'train_loader: {len(train_loader)} \n'
              f'val_loader: {len(val_loader)} \n'
              f'and batch_size of: {batch_size}')
    else:  # MNIST
        train_loader = data_handler.init_mnist_loader(mnist_folder=params['data_folder'],
                                                      img_size=params['img_size'],
                                                      loader_params=loader_params)

    # ============ sampling z's (to show evolution of the generated images during training)
    z_shapes = helper.calc_z_shapes(params['channels'], params['img_size'], params['n_block'])
    z_samples = sample_z(z_shapes, params['n_samples'], params['temperature'], device)

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
                img_batch = batch['real'].to(device)
                segment_batch = batch['segment'].to(device)
                # img_batch, segment_batch = batches['img_batch'], batches['segment_batch']
                loss, log_p, log_det = do_forward(args, params, model, img_batch, segment_batch)
                metrics = {'loss': loss, 'log_p': log_p}

            elif args.model == 'c_flow':
                img_batch = batch['real'].to(device)
                segment_batch = batch['segment'].to(device)
                boundary_batch = batch['boundary'].to(device) if args.cond_mode == 'segment_boundary' else None
                # print(boundary_batch[0])

                loss, loss_left, log_p_left, log_det_left, loss_right, log_p_right, log_det_right = \
                    do_forward(args, params, model, img_batch, segment_batch, boundary_batch=boundary_batch)

                metrics = {'loss': loss,
                           'loss_right': loss_right,
                           'log_p_right': log_p_right}

                # if args.left_pretrained and not args.left_unfreeze:  # left glow pre-trained and freezed
                if args.left_pretrained:
                    pass
                else:  # normal c_flow OR pre-trained left glow unfreezed
                    metrics['loss_left'] = loss_left
                    metrics['log_p_left'] = log_p_left
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
                # val_loss_mean, val_loss_std = calc_val_loss(args, params, device, model, val_loader)
                # print(f'====== In [train]: val_loss mean: {round(val_loss_mean, 3)} - std: {round(val_loss_std, 3)}')
                metrics['val_loss'] = val_loss_mean
                print(f'====== In [train]: val_loss mean: {round(val_loss_mean, 3)}')

            # ============ tracking metrics
            if args.use_comet:
                track_metrics(args, params, comet_tracker, metrics, optim_step)

            # ============ saving samples
            if optim_step % params['sample_freq'] == 0:
                samples_path = paths['samples_path']
                helper.make_dir_if_not_exists(samples_path)
                sampled_images = take_samples(args, model, z_samples, reverse_cond)
                utils.save_image(sampled_images, f'{samples_path}/{str(optim_step).zfill(6)}.png', nrow=10)
                print(f'\nIn [train]: Sample saved at iteration {optim_step} to: \n"{samples_path}"\n')

            # ============ saving checkpoint
            if optim_step > 0 and optim_step % params['checkpoint_freq'] == 0:
                checkpoints_path = paths['checkpoints_path']
                helper.make_dir_if_not_exists(checkpoints_path)
                helper.save_checkpoint(checkpoints_path, optim_step, model, optimizer, loss)
                print("In [train]: Checkpoint saved at iteration", optim_step, '\n')

            optim_step += 1
