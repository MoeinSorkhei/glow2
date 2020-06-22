import torch

from globals import device


def do_forward(args, params, model, img_batch, segment_batch, boundary_batch=None):
    n_bins = 2. ** params['n_bits']

    if args.model == 'glow':
        if args.train_on_segment:  # vanilla Glow on segmentation
            log_p, logdet, _ = model(inp=segment_batch + torch.rand_like(segment_batch) / n_bins)

        else:  # vanilla Glow on real images
            log_p, logdet, _ = model(inp=img_batch + torch.rand_like(img_batch) / n_bins)

        log_p = log_p.mean()
        logdet = logdet.mean()  # logdet and log_p: tensors of shape torch.Size([5])
        # loss, log_p, log_det = calc_loss(log_p, logdet, params['img_size'], n_bins)
        # return loss, log_p, log_det
        return log_p, logdet

    # THIS CHEKCING SHOULD BE DONE OUTSIDE THIS FUNCTION
    elif args.model == 'c_flow':
        if args.direction == 'label2photo':  # IMPROVE: args.dataset should also be checked
            # IMPROVE: x_a=segment_batch + torch.rand_like(segment_batch) / n_bins ==> is used in if and else: take it out
            left_glow_outs, right_glow_outs = model(x_a=segment_batch + torch.rand_like(segment_batch) / n_bins,
                                                    x_b=img_batch + torch.rand_like(img_batch) / n_bins,
                                                    b_map=boundary_batch)
        elif args.direction == 'photo2label':
            left_glow_outs, right_glow_outs = model(x_a=img_batch + torch.rand_like(img_batch) / n_bins,
                                                    x_b=segment_batch + torch.rand_like(segment_batch) / n_bins)

        elif args.dataset == 'maps' and args.direction == 'map2photo':
            left_glow_outs, right_glow_outs = model(x_a=segment_batch + torch.rand_like(segment_batch) / n_bins,
                                                    x_b=img_batch + torch.rand_like(img_batch) / n_bins)

        elif args.dataset == 'maps' and args.direction == 'photo2map':
            left_glow_outs, right_glow_outs = model(x_a=img_batch + torch.rand_like(img_batch) / n_bins,
                                                    x_b=segment_batch + torch.rand_like(segment_batch) / n_bins)

        else:
            raise NotImplementedError

        # =========== the rest is the same for any configuration
        log_p_left, log_det_left = left_glow_outs['log_p'].mean(), left_glow_outs['log_det'].mean()
        log_p_right, log_det_right = right_glow_outs['log_p'].mean(), right_glow_outs['log_det'].mean()

        # loss_left, log_p_left, _ = calc_loss(log_p_left, log_det_left, params['img_size'], n_bins)
        # loss_right, log_p_right, _ = calc_loss(log_p_right, log_det_right, params['img_size'], n_bins)
        # loss = loss_left + loss_right
        # return loss, loss_left, log_p_left, log_det_left, loss_right, log_p_right, log_det_right
        return log_p_left, log_det_left, log_p_right, log_det_right


def take_samples(args, params, model, reverse_cond):
    z_samples = sample_z(params['n_samples'],
                         params['temperature'],
                         params['channels'],
                         params['img_size'],
                         params['n_block'])

    with torch.no_grad():
        if args.model == 'glow':
            sampled_images = model.reverse(z_samples, coupling_conds=reverse_cond).cpu().data

        elif args.model == 'c_flow':  # here reverse_cond is x_a
            if args.direction == 'label2photo':
                sampled_images = model.reverse(x_a=reverse_cond['segment'],
                                               b_map=reverse_cond['boundary'],
                                               z_b_samples=z_samples,
                                               mode='sample_x_b').cpu().data

            elif args.direction == 'photo2label':  # 'photo2label'
                sampled_images = model.reverse(x_a=reverse_cond['real_cond'],
                                               z_b_samples=z_samples,
                                               mode='sample_x_b').cpu().data

            elif args.dataset == 'maps':
                sampled_images = model.reverse(x_a=reverse_cond,  # reverse cond is already extracted based on direction
                                               z_b_samples=z_samples,
                                               mode='sample_x_b').cpu().data

            # elif args.dataset == 'maps' and args.direction == 'map2photo':
            #     sampled_images = model.reverse(x_a=reverse_cond)
            #
            # elif args.dataset == 'maps' and args.direction == 'photo2map':
            #     pass

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return sampled_images


def sample_z(n_samples, temperature, channels, img_size, n_block):
    # n_samples, temperature = params['n_samples'], params['temperature']
    # z_shapes = calc_z_shapes(params['channels'], params['img_size'], params['n_block'])
    z_shapes = calc_z_shapes(channels, img_size, n_block)
    z_samples = []
    for z in z_shapes:  # temperature squeezes the Gaussian which is sampled from
        z_new = torch.randn(n_samples, *z) * temperature
        z_samples.append(z_new.to(device))
    return z_samples


def calc_z_shapes(n_channel, input_size, n_block):
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

    # print_and_wait(f'z shapes: {z_shapes}')

    if mode == 'z_outs':  # the condition is has the same shape as the z's themselves
        return z_shapes

    # flows_outs are before split while z_shapes are calculated for z's after they are split
    # ===> channels should be multiplied by 2 (except for the last shape)
    # if mode == 'flows_outs' or mode == 'flows_outs + bmap':

    # REFACTORING NEEDED, I THINK THIS IF CONDITION IS NOT NEEDED
    # if mode == 'segment' or mode == 'segment_boundary' or mode == 'real_cond':
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
        # print(f'z[{i}] cond shape = {z_shapes[i]}')
        # input()

    # print_and_wait(f'cond shapes: {z_shapes}')
    return z_shapes

    # REFACTORING NEEDED: I THINK THIS PART IS NOT REACHABLE
    # for 'segment' or 'segment_id' modes
    '''cond_shapes = []
    for z_shape in z_shapes:
        h, w = z_shape[1], z_shape[2]
        if mode == 'segment':
            channels = (orig_shape[0] * orig_shape[1] * orig_shape[2]) // (h * w)  # new channels with new h and w

        elif mode == 'segment_id':
            channels = 34 + (orig_shape[0] * orig_shape[1] * orig_shape[2]) // (h * w)

        else:
            raise NotImplementedError

        cond_shapes.append((channels, h, w))

    return cond_shapes'''


def arrange_rev_cond(args, img_batch, segment_batch, boundary_batch):  # only used in cityscapes experiments
    # ======= only support for c_flow mode now
    if args.direction == 'label2photo':
        b_maps = boundary_batch if args.cond_mode == 'segment_boundary' else None
        reverse_cond = {'segment': segment_batch, 'boundary': b_maps}

    elif args.direction == 'photo2label':  # 'photo2label'
        reverse_cond = {'real_cond': img_batch}

    else:
        raise NotImplementedError('Direction not implemented')
    return reverse_cond

