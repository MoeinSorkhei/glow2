from data_handler import *
import data_handler
from globals import real_conds_abs_path
# from ._interface import init_model


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


# THIS FUNCTION DOES NOT WORK NOW - refactoring: this function should take model as argument rather than itialized it itself
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

    imgs = [real_conds_abs_path[0]]  # one image for now
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
        b_map = boundaries if args.direction == 'label2photo' and args.use_bmaps else None
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



