from torch import optim

from .two_glows import *
from data_handler import *
import data_handler
from globals import desired_real_imgs, device


def prepare_reverse_cond(args, params, run_mode='train'):
    if args.dataset == 'mnist':
        reverse_cond = ('mnist', 1, params['n_samples'])
        return reverse_cond

    else:
        if args.model == 'glow' and args.cond_mode is None:  # IMPROVEMENT: SHOULD BE FIXED
            return None

        samples_path = helper.compute_paths(args, params)['samples_path']
        save_path = samples_path if run_mode == 'train' else None  # no need for reverse_cond at inference time
        direction = args.direction
        segmentations, id_repeats_batch, real_imgs, boundaries = create_cond(params,
                                                                             fixed_conds=desired_real_imgs,
                                                                             save_path=save_path,
                                                                             direction=direction)

        # IMRPVEMEN: THIS HOULS CALL ARRANGE REVERSE COND
        # ======= only support for c_flow mode now
        if direction == 'label2photo':
            b_maps = boundaries if args.cond_mode == 'segment_boundary' else None
            reverse_cond = {'segment': segmentations, 'boundary': b_maps}
        elif direction == 'photo2label':  # 'photo2label'
            reverse_cond = {'real_cond': real_imgs}
        else:
            raise NotImplementedError('Direction not implemented')
        return reverse_cond


def init_model(args, params, device, run_mode='train'):
    # IMPROVEMENT: FUNCTIONS THAT USE THIS AND THE LOAD THE CHECKPOINT THEMSELVES COULD USE INIT_AND_LOAD FUNCTION
    # ======== preparing reverse condition and initializing models
    if args.dataset == 'mnist':
        model = init_glow(params)
        reverse_cond = prepare_reverse_cond(args, params, run_mode)  # NOTE: needs to be refactored

    # ======== init glow
    elif args.model == 'glow':  # glow with no condition
        reverse_cond = None
        model = init_glow(params)

    # ======== init c_flow
    elif args.model == 'c_flow':
        if args.dataset == 'transient':
            reverse_cond = data_handler.create_rev_cond(args, params)
            # direction = args.direction
            mode = None

        else:  # cityscapes
            reverse_cond = None if args.exp else prepare_reverse_cond(args, params, run_mode)  # uses direction in itself
            # direction = args.direction
            # mode: 'segment', 'segment_boundary', etc.
            mode = args.cond_mode if args.direction == 'label2photo' else None  # no mode if 'photo2label'

        # only two Blocks with conditional w - IMPROVEMENT: THIS SHOULD BE MOVED TO GLOBALS.PY
        w_conditionals = [True, True, False, False] if args.w_conditional else None
        act_conditionals = [True, True, False, False] if args.act_conditional else None
        coupling_use_cond_nets = [True, True, True, True] if args.coupling_cond_net else None

        if args.left_pretrained:  # use pre-trained left Glow
            # pth = f"/Midgard/home/sorkhei/glow2/checkpoints/city_model=glow_image=segment"
            left_glow_path = helper.compute_paths(args, params)['left_glow_path']
            pre_trained_left_glow = init_glow(params)  # init the model
            pretrained_left_glow, _, _ = helper.load_checkpoint(path_to_load=left_glow_path,
                                                                optim_step=args.left_step,
                                                                model=pre_trained_left_glow,
                                                                optimizer=None,
                                                                device=device,
                                                                resume_train=False)  # left-glow always freezed

            model = TwoGlows(params, args.dataset, args.direction, mode,
                             pretrained_left_glow=pretrained_left_glow,
                             w_conditionals=w_conditionals,
                             act_conditionals=act_conditionals,
                             use_coupling_cond_nets=coupling_use_cond_nets)
        else:  # also train left Glow
            model = TwoGlows(params, args.dataset, args.direction, mode,
                             w_conditionals=w_conditionals,
                             act_conditionals=act_conditionals,
                             use_coupling_cond_nets=coupling_use_cond_nets)
    else:
        raise NotImplementedError
    return model.to(device), reverse_cond


def init_and_load(args, params, run_mode):
    checkpoints_path = helper.compute_paths(args, params)['checkpoints_path']
    optim_step = args.last_optim_step
    model, reverse_cond = init_model(args, params, device, run_mode)

    if run_mode == 'infer':
        model, _, _ = helper.load_checkpoint(checkpoints_path, optim_step, model, None,
                                             device, resume_train=False)
        print(f'In [init_and_load]: returned model for inference')
        return model

    else:  # train
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        print(f'In [init_and_load]: returned model and optimizer for training')
        model, optimizer, _ = helper.load_checkpoint(checkpoints_path, optim_step, model,
                                                     optimizer, device, resume_train=True)
        return model, optimizer


def sanity_check_c_flow(x_a_ref, x_b_ref, x_a_rec, x_b_rec):
    x_a_diff = torch.mean(torch.abs(x_a_ref - x_a_rec))
    x_b_diff = torch.mean(torch.abs(x_b_ref - x_b_rec))

    ok_or_not_a = 'OK!' if x_a_diff < 1e-5 else 'NOT OK!'
    ok_or_not_b = 'OK!' if x_b_diff < 1e-5 else 'NOT OK!'

    print('=' * 100)
    print(f'In [sanity_check]: mean x_a_diff: {x_a_diff} ===> {ok_or_not_a}')
    print(f'In [sanity_check]: mean x_b_diff: {x_b_diff} ===> {ok_or_not_b}')
    print('=' * 100, '\n')


def sanity_check_glow(x_ref, x_rec):
    diff = torch.mean(torch.abs(x_ref - x_rec))
    ok_or_not = 'OK!' if diff < 1e-5 else 'NOT OK!'
    print('=' * 100)
    print(f'In [sanity_check]: mean diff between reference and reconstructed image is: {diff} ==> {ok_or_not}')
    print('=' * 100, '\n')


def verify_invertibility(args, params, path=None):
    """
    :param args:
    :param params:
    :param path: If not None, the reference and reconstructed images will be saved to the path
    (e.g., '../tmp/invertibility').
    :return:
    """
    def verify_glow(glow, ref_img):
        # ========= forward to get z's
        z_list = glow(ref_img)['z_outs']
        # ========= reverse to reconstruct input
        inp_rec = model.reverse(z_list, reconstruct=True)
        return inp_rec

    def verify_c_flow(c_flow, x_a_ref, x_b_ref, b_map=None):
        x_a_rec, x_b_rec = c_flow.reverse(x_a=x_a_ref, x_b=x_b_ref, b_map=b_map)  # performs forward in itself
        return x_a_rec, x_b_rec

    # read from checkpoint, if specified
    '''if args.last_optim_step:
        model = init_and_load(args, params, run_mode='infer')
        print(f'In [verify_invertibility]: model loaded \n')'''
    # else:
    # init a new model
    model, rev_cond = init_model(args, params, device, run_mode='infer')
    print(f'In [verify_invertibility]: model initialized \n')

    imgs = [desired_real_imgs[0]]  # one image for now
    segmentations, _, real_imgs, boundaries = data_handler.create_cond(params,
                                                                       fixed_conds=imgs,
                                                                       save_path=path)
    glow_conf = {  # testing reconstruction for both real and segmentation image
        'real_img': real_imgs,
        'seg_img': segmentations
    }

    if args.model == 'glow':
        for name, inp in glow_conf.items():
            print(f'In [In [verify_invertibility]: doing for {name}')
            reconstructed = verify_glow(model, ref_img=inp)

            if path is not None:
                utils.save_image(inp.cpu().data, f'{path}/{name}.png', nrow=1, padding=0)
                utils.save_image(reconstructed.cpu().data, f'{path}/{name}_rec.png', nrow=1, padding=0)
                print(f'In [verify_invertibility]: saved inp and inp_rec to: "{path}"')

            # ========= sanity check for the difference between reference and reconstructed image
            sanity_check_glow(x_ref=inp, x_rec=reconstructed)

    else:
        b_map = boundaries if args.direction == 'label2photo' and args.cond_mode == 'segment_boundary' else None
        if args.direction == 'label2photo':
            x_a_rec, x_b_rec = verify_c_flow(model, x_a_ref=segmentations, x_b_ref=real_imgs, b_map=b_map)
            sanity_check_c_flow(segmentations, real_imgs, x_a_rec, x_b_rec)

        elif args.direction == 'photo2label':
            x_a_rec, x_b_rec = verify_c_flow(model, x_a_ref=real_imgs, x_b_ref=segmentations)
            sanity_check_c_flow(x_a_ref=real_imgs, x_b_ref=segmentations, x_a_rec=x_a_rec, x_b_rec=x_b_rec)
        else:
            raise NotImplementedError('Direction not implemented')

        if path is not None:
            raise NotImplementedError('Needs to rename the file names based on arg.direction')
            # utils.save_image(segmentations.cpu().data, f'{path}/x_a_ref.png', nrow=1, padding=0)
            # utils.save_image(real_imgs.cpu().data, f'{path}/x_b_ref.png', nrow=1, padding=0)
            # utils.save_image(x_a_rec.cpu().data, f'{path}/x_a_rec.png', nrow=1, padding=0)
            # utils.save_image(x_b_rec.cpu().data, f'{path}/x_b_rec.png', nrow=1, padding=0)
            # print(f'In [verify_invertibility]: saved inp and and reconstructed to: "{path}"')

        # ========= sanity check for the difference between reference and reconstructed image
        # sanity_check_c_flow(segmentations, real_imgs, x_a_rec, x_b_rec)



