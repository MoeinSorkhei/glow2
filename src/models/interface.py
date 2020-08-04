from torch import optim

from .glow import *
from .two_glows import TwoGlows
from .utility import *
from .interface_c_glow import *


def do_forward(args, params, model, img_batch, segment_batch, boundary_batch=None):
    """
    Does the forward operation of the model based on the model and the data batches.
    :param args:
    :param params:
    :param model:
    :param img_batch:
    :param segment_batch:
    :param boundary_batch:
    :return:
    """
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

    elif args.model == 'c_flow':
        if args.dataset == 'cityscapes' and args.direction == 'label2photo':
            left_glow_outs, right_glow_outs = model(x_a=noise_added(segment_batch, n_bins),
                                                    x_b=noise_added(img_batch, n_bins),
                                                    b_map=boundary_batch)

        elif args.dataset == 'cityscapes' and args.direction == 'photo2label':
            left_glow_outs, right_glow_outs = model(x_a=noise_added(img_batch, n_bins),
                                                    x_b=noise_added(segment_batch, n_bins))

        elif args.dataset == 'maps' and args.direction == 'map2photo':
            left_glow_outs, right_glow_outs = model(x_a=noise_added(segment_batch, n_bins),
                                                    x_b=noise_added(img_batch, n_bins))

        elif args.dataset == 'maps' and args.direction == 'photo2map':
            left_glow_outs, right_glow_outs = model(x_a=noise_added(img_batch, n_bins),
                                                    x_b=noise_added(segment_batch, n_bins))

        else:
            raise NotImplementedError

        # =========== the rest is the same for any configuration
        log_p_left, log_det_left = left_glow_outs['log_p'].mean(), left_glow_outs['log_det'].mean()
        log_p_right, log_det_right = right_glow_outs['log_p'].mean(), right_glow_outs['log_det'].mean()
        return log_p_left, log_det_left, log_p_right, log_det_right

    # elif args.model == 'c_glow':
    elif 'c_glow' in args.model:
        if args.dataset == 'cityscapes' and args.direction == 'label2photo':
            z, loss = model(x=noise_added(segment_batch, n_bins),
                            y=noise_added(img_batch, n_bins))
            return z, loss

        elif args.dataset == 'cityscapes' and args.direction == 'photo2label':
            z, loss = model(x=noise_added(img_batch, n_bins),
                            y=noise_added(segment_batch, n_bins))
            return z, loss
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def noise_added(batch, n_bins):  # add uniform noise
    return batch + torch.rand_like(batch) / n_bins


def take_samples(args, params, model, reverse_cond, n_samples=None):
    num_samples = n_samples if n_samples is not None else params['n_samples']
    z_samples = sample_z(n_samples=num_samples,
                         temperature=params['temperature'],
                         channels=params['channels'],
                         img_size=params['img_size'],
                         n_block=params['n_block'])

    with torch.no_grad():
        if args.model == 'glow':
            sampled_images = model.reverse(z_samples, coupling_conds=reverse_cond).cpu().data

        elif args.model == 'c_flow':  # here reverse_cond is x_a
            if args.direction == 'label2photo':
                sampled_images = model.reverse(x_a=reverse_cond['segment'],
                                               b_map=reverse_cond['boundary'],
                                               z_b_samples=z_samples).cpu().data

            elif args.direction == 'photo2label':  # 'photo2label'
                sampled_images = model.reverse(x_a=reverse_cond['real_cond'],
                                               z_b_samples=z_samples).cpu().data

            elif args.dataset == 'maps':
                sampled_images = model.reverse(x_a=reverse_cond,  # reverse cond is already extracted based on direction
                                               z_b_samples=z_samples).cpu().data
            else:
                raise NotImplementedError

        elif 'c_glow' in args.model:
            temp = params['temperature']
            if args.direction == 'label2photo':
                # the forward function with reverse=True takes samples from prior
                sampled_images, _ = model(x=reverse_cond['segment'], reverse=True, eps_std=temp)

            elif args.direction == 'photo2label':
                sampled_images, _ = model(x=reverse_cond['real_cond'], reverse=True, eps_std=temp)

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return sampled_images


def batch2revcond(args, img_batch, segment_batch, boundary_batch):  # only used in cityscapes experiments
    """
    Takes batches of data and arranges them as reverse condition based on the args.
    :param args:
    :param img_batch:
    :param segment_batch:
    :param boundary_batch:
    :return:
    """
    # ======= only support for c_flow mode now
    if args.direction == 'label2photo':
        b_maps = boundary_batch if args.use_bmaps else None
        reverse_cond = {'segment': segment_batch, 'boundary': b_maps}

    elif args.direction == 'photo2label':  # 'photo2label'
        reverse_cond = {'real_cond': img_batch}

    else:
        raise NotImplementedError('Direction not implemented')
    return reverse_cond


def verify_invertibility(args, params):
    assert args.model == 'c_flow'
    imgs = [real_conds_abs_path[0]]  # one image for now
    segmentations, _, real_imgs, boundaries = data_handler.create_cond(params, fixed_conds=imgs, save_path=None)
    b_map = boundaries if args.direction == 'label2photo' and args.use_bmaps else None

    model = init_model(args, params)
    x_a_rec, x_b_rec = model.reconstruct_all(x_a=segmentations, x_b=real_imgs)
    sanity_check(segmentations, real_imgs, x_a_rec, x_b_rec)


def init_model(args, params):
    # ======== init glow
    if args.model == 'glow':  # glow with no condition
        reverse_cond = None
        model = init_glow(params)

    elif args.model == 'c_flow':
        mode = 'segment'
        # model = TwoGlows(params, args.dataset, args.direction, mode=mode)
        model = TwoGlows(params, mode=mode)

    # ======== init c_flow
    # elif args.model == 'c_flow':  # change here for new datasets
    #     # only two Blocks with conditional w - IMPROVEMENT: THIS SHOULD BE MOVED TO GLOBALS.PY
    #     w_conditionals = [True, True, False, False] if args.w_conditional else None
    #     act_conditionals = [True, True, False, False] if args.act_conditional else None
    #     coupling_use_cond_nets = [True, True, True, True] if args.coupling_cond_net else None
    #     mode = None
    #     if args.direction == 'label2photo':
    #         mode = 'segment_boundary' if args.use_bmaps else 'segment'  # SHOULD REFACTOR THE WAY TWO_GLOWS IS INITIALIZED
    #     if args.left_pretrained:  # use pre-trained left Glow
    #         # pth = f"/Midgard/home/sorkhei/glow2/checkpoints/city_model=glow_image=segment"
    #         left_glow_path = helper.compute_paths(args, params)['left_glow_path']
    #         pre_trained_left_glow = init_glow(params)  # init the model
    #         pretrained_left_glow, _, _ = helper.load_checkpoint(path_to_load=left_glow_path, optim_step=args.left_step,
    #                                                             model=pre_trained_left_glow, optimizer=None,
    #                                                             resume_train=False)  # left-glow always freezed
    #
    #         model = TwoGlows(params, args.dataset, args.direction, mode,
    #                          pretrained_left_glow=pretrained_left_glow,
    #                          w_conditionals=w_conditionals,
    #                          act_conditionals=act_conditionals,
    #                          use_coupling_cond_nets=coupling_use_cond_nets)
    #     else:  # also train left Glow
    #         model = TwoGlows(params, args.dataset, args.direction, mode,
    #                          w_conditionals=w_conditionals,
    #                          act_conditionals=act_conditionals,
    #                          use_coupling_cond_nets=coupling_use_cond_nets)

    # elif args.model == 'c_glow':
    elif 'c_glow' in args.model:
        model = init_c_glow(args, params)

    else:
        raise NotImplementedError
    return model.to(device)


def init_and_load(args, params, run_mode):
    checkpoints_path = helper.compute_paths(args, params)['checkpoints_path']
    optim_step = args.last_optim_step
    model = init_model(args, params)

    if run_mode == 'infer':
        model, _, _ = helper.load_checkpoint(checkpoints_path, optim_step, model, None, resume_train=False)
        print(f'In [init_and_load]: returned model for inference')
        return model

    else:  # train
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        print(f'In [init_and_load]: returned model and optimizer for training')
        model, optimizer, _ = helper.load_checkpoint(checkpoints_path, optim_step, model, optimizer, resume_train=True)
        return model, optimizer

