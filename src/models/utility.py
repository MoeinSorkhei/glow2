from data_handler import *
import data_handler
from globals import desired_real_imgs
from ._interface import init_model


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
    model, rev_cond = init_model(args, params, run_mode='infer')
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



