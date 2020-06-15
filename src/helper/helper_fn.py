import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from PIL import Image


def print_and_wait(to_be_printed):
    print(to_be_printed)
    print('======== Waiting for input...')
    input()


def show_images(img_list):
    if len(img_list) != 2:
        raise NotImplementedError('Showing more than two images not implemented yet')

    f, ax_arr = plt.subplots(1, 2)
    ax_arr[0].imshow(img_list[0].permute(1, 2, 0))  # permute: making it (H, W, channel)
    ax_arr[1].imshow(img_list[1].permute(1, 2, 0))

    plt.show()


def read_params(params_path):
    with open(params_path, 'r') as f:  # reading params from the json file
        parameters = json.load(f)
    return parameters


def save_checkpoint(path_to_save, optim_step, model, optimizer, loss):
    name = path_to_save + f'/optim_step={optim_step}.pt'
    checkpoint = {'loss': loss,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, name)
    print(f'In [save_checkpoint]: save state dict done at: "{name}"')


def load_checkpoint(path_to_load, optim_step, model, optimizer, device, resume_train=True):
    name = path_to_load + f'/optim_step={optim_step}.pt'
    checkpoint = torch.load(name, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss = checkpoint['loss']

    print(f'In [load_checkpoint]: load state dict done from: "{name}"')

    # putting the model in the correct mode
    if resume_train:
        model.train()
    else:
        model.eval()
        for param in model.parameters():  # freezing the layers when using only for evaluation
            param.requires_grad = False

    if optimizer is not None:
        return model.to(device), optimizer, loss
    return model.to(device), None, loss


def translate_address(path, package):
    """
    This function changes a path which is from the project directory to a path readable by a specific package in the
    folder. For instance, the datasets paths in params.json are from the project directory. Using this function, every
    function in the data_loader package can use that address by converting it to a relative address using this function.
    The benefit is that no function needs to translate the address itself directly, and all of them could use this
    function to do so.
    :param path:
    :param package:
    :return:
    """
    if package == 'data_handler' or package == 'helper':
        return '../' + path
    else:
        raise NotImplementedError('NOT IMPLEMENTED...')


def label_to_tensor(label, height, width, count=0):
    if count == 0:
        arr = np.zeros((10, height, width))
        arr[label] = 1

    else:
        arr = np.zeros((count, 10, height, width))
        arr[:, label, :, :] = 1

    return torch.from_numpy(arr.astype(np.float32))


def make_dir_if_not_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print(f'In [make_dir_if_not_exists]: created path "{directory}"')


def reshape_cond(img_condition, h, w):
    return img_condition.rehspae((-1, h, w))  # reshaping to it could be concatenated in the channel dimension

# # to be removed
# def calc_z_shapes(n_channel, input_size, n_block):
#     """
#     This function calculates z shapes given the desired number of blocks in the Glow model. After each block, the
#     spatial dimension is halved and the number of channels is doubled.
#     :param n_channel:
#     :param input_size:
#     :param n_flow:
#     :param n_block:
#     :return:
#     """
#     z_shapes = []
#     for i in range(n_block - 1):
#         input_size = input_size // 2 if type(input_size) is int else (input_size[0] // 2, input_size[1] // 2)
#         n_channel *= 2
#
#         shape = (n_channel, input_size, input_size) if type(input_size) is int else (n_channel, *input_size)
#         z_shapes.append(shape)
#
#     # for the very last block where we have no split operation
#     input_size = input_size // 2 if type(input_size) is int else (input_size[0] // 2, input_size[1] // 2)
#     shape = (n_channel * 4, input_size, input_size) if type(input_size) is int else (n_channel * 4, *input_size)
#     z_shapes.append(shape)
#
#     return z_shapes
#
# # to be removed
# def calc_cond_shapes(params, mode):
#     in_channels, img_size, n_block = params['channels'], params['img_size'], params['n_block']
#     z_shapes = calc_z_shapes(in_channels, img_size, n_block)
#
#     # print_and_wait(f'z shapes: {z_shapes}')
#
#     if mode == 'z_outs':  # the condition is has the same shape as the z's themselves
#         return z_shapes
#
#     # flows_outs are before split while z_shapes are calculated for z's after they are split
#     # ===> channels should be multiplied by 2 (except for the last shape)
#     # if mode == 'flows_outs' or mode == 'flows_outs + bmap':
#
#     # REFACTORING NEEDED, I THINK THIS IF CONDITION IS NOT NEEDED
#     # if mode == 'segment' or mode == 'segment_boundary' or mode == 'real_cond':
#     for i in range(len(z_shapes)):
#         z_shapes[i] = list(z_shapes[i])  # converting the tuple to list
#
#         if i < len(z_shapes) - 1:
#             z_shapes[i][0] = z_shapes[i][0] * 2  # extra channel dim for zA coming from the left glow
#             if mode is not None and mode == 'segment_boundary':
#                 z_shapes[i][0] += 12  # extra channel dimension for the boundary
#
#         elif mode is not None and mode == 'segment_boundary':  # last layer - adding dim only for boundaries
#             # no need to have z_shapes[i][0] * 2 since this layer does not have split
#             z_shapes[i][0] += 12  # extra channel dimension for the boundary
#
#         z_shapes[i] = tuple(z_shapes[i])  # convert back to tuple
#         # print(f'z[{i}] cond shape = {z_shapes[i]}')
#         # input()
#
#     # print_and_wait(f'cond shapes: {z_shapes}')
#     return z_shapes
#
#     # REFACTORING NEEDED: I THINK THIS PART IS NOT REACHABLE
#     # for 'segment' or 'segment_id' modes
#     '''cond_shapes = []
#     for z_shape in z_shapes:
#         h, w = z_shape[1], z_shape[2]
#         if mode == 'segment':
#             channels = (orig_shape[0] * orig_shape[1] * orig_shape[2]) // (h * w)  # new channels with new h and w
#
#         elif mode == 'segment_id':
#             channels = 34 + (orig_shape[0] * orig_shape[1] * orig_shape[2]) // (h * w)
#
#         else:
#             raise NotImplementedError
#
#         cond_shapes.append((channels, h, w))
#
#     return cond_shapes'''


def print_info(args, params, model, which_info='all'):
    if which_info == 'params' or which_info == 'all':
        # printing important running params
        print(f'{"=" * 50} \n'
              f'In [print_info]: Important params: \n'
              f'model: {args.model} \n'
              # f'lr: {args.lr if args.lr is not None else params["lr"]} \n'
              f'lr: {params["lr"]} \n'
              f'batch_size: {params["batch_size"]} \n'
              f'temperature: {params["temperature"]} \n'
              f'last_optim_step: {args.last_optim_step} \n'
              f'left_lr: {args.left_lr} \n'
              f'left_step: {args.left_step} \n'
              f'cond: {args.cond_mode} \n\n')

        # printing paths
        paths = compute_paths(args, params)
        print(f'Paths:')
        for path_name, path_addr in paths.items():
            print(f'{path_name}: {path_addr}')
        print(f'{"=" * 50}\n')

    if which_info == 'model' or which_info == 'all':
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'{"=" * 50}\n'
              'In [print_info]: Using model with the following info:\n'
              f'Total parameters: {total_params:,} \n'
              f'Trainable parameters: {trainable_params:,} \n'
              f'n_flow: {params["n_flow"]} \n'
              f'n_block: {params["n_block"]} \n'
              f'{"=" * 50}\n')


def scientific(float_num):
    # if float_num < 1e-4:
    #    return str(float_num)
    if float_num == 1e-5:
        return '1e-5'
    elif float_num == 5e-5:
        return '5e-5'
    elif float_num == 1e-4:
        return '1e-4'
    elif float_num == 1e-3:
        return '1e-3'
    raise NotImplementedError('In [scientific]: Conversion from float to scientific str needed.')


def compute_paths(args, params, additional_info=None):
    """
    Different paths computed in this function include:
        - eval_path: where the validation images and the results of evaluation is stored.
        - validation_path: where the samples are generated on the validation images
          (in the size the network was trained on).
        - resized_path: where the validation set samples are resized (to 256x256) to be used for cityscapes evaluation.
    :param additional_info:
    :param args:
    :param params:
    :return:
    """
    dataset = args.dataset
    model = args.model

    if model == 'glow':
        img = 'segment' if args.train_on_segment else 'real'  # --train_on_segment, if used, is always used with glow
        cond = None

    elif model == 'c_flow':
        if dataset == 'cityscapes' and args.direction == 'label2photo':
            img = 'real'
            cond = args.cond_mode

        elif dataset == 'cityscapes' and args.direction == 'photo2label':
            img = 'segment'
            cond = 'real'

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError('mode not implemented')

    # cond = args.cond_mode
    w_conditional = args.w_conditional
    act_conditional = args.act_conditional

    run_mode = 'infer' if args.exp else 'train'
    # run_mode = 'infer' if (args.exp or
    #                        args.infer_on_val or
    #                        args.random_samples or
    #                        args.new_condition or
    #                        args.evaluate or
    #                        args.eval_complete or
    #                        args.resize_for_fcn) else 'train'
    h, w = params['img_size'][0], params['img_size'][1]

    # used for checkpoints only
    coupling_str_checkpts = '/coupling_net' if args.coupling_cond_net else ''

    if args.dataset == 'cityscapes':
        if args.direction == 'label2photo':
            cond_with_ceil = 'segment_boundary/do_ceil=True' if args.cond_mode == 'segment_boundary' else 'segment'

        elif args.direction == 'photo2label':
            cond_with_ceil = 'real'
        else:
            raise NotImplementedError

    elif args.dataset == 'transient':
        cond_with_ceil = cond

    else:
        raise NotImplementedError

    # used for samples only
    cond_variant = 'baseline'
    if args.act_conditional:
        cond_variant += ' + act_cond'
    if args.w_conditional:
        cond_variant += ' + w_cond'
    if args.coupling_cond_net:
        cond_variant += ' + coupling_net'

    samples_base_dir = f'{params["samples_path"]}' \
                       f'/{dataset}' \
                       f'/{h}x{w}' \
                       f'/model={model}' \
                       f'/img={img}' \
                       f'/cond={cond}' \
                       f'/{cond_variant}'

    checkpoints_base_dir = f'{params["checkpoints_path"]}' \
                           f'/{dataset}' \
                           f'/{h}x{w}' \
                           f'/model={model}' \
                           f'/img={img}' \
                           f'/cond={cond_with_ceil}' \
                           f'/w_conditional={w_conditional}' \
                           f'/act_conditional={act_conditional}' \
                           f'{coupling_str_checkpts}'

    lr = scientific(params['lr'])
    paths = {}  # the paths dict be filled along the way

    # ========= c_flow paths
    if model == 'c_flow':
        c_flow_type = 'left_pretrained' if args.left_pretrained else 'from_scratch'
        samples_path = f'{samples_base_dir}/{c_flow_type}'
        checkpoints_path = f'{checkpoints_base_dir}/{c_flow_type}'

        # ========= only for left_pretrained
        if c_flow_type == 'left_pretrained':
            left_lr = args.left_lr  # example: left_pretrained/left_lr=1e-4/freezed/left_step=10000
            left_step = args.left_step  # optim_step of the left glow

            samples_path += f'/left_lr={left_lr}_left_step={left_step}'
            checkpoints_path += f'/left_lr={left_lr}/freezed/left_step={left_step}'  # always freezed

            # left glow checkpoint path
            left_glow_path = f'{params["checkpoints_path"]}/{dataset}/{h}x{w}/model=glow/img=segment/cond={args.left_cond}'
            left_glow_path += f'/lr={left_lr}'  # e.g.: model=glow/img=segment/cond=None/lr=1e-4
            paths['left_glow_path'] = left_glow_path

        samples_path += f'/{run_mode}'  # adding run mode # e.g.: left_lr=1e-4/freezed/left_step=10000/train
        checkpoints_path += f'/lr={lr}'  # adding lr only to checkpoints path (backward compatibility)

        # ========= infer: also adding optimization step (step is specified after lr)
        if run_mode == 'infer':
            optim_step = args.last_optim_step    # e.g.: left_lr=1e-4/freezed/left_step=10000/infer/lr=1e-5/step=1000
            samples_path += f'/step={optim_step}'

            eval_path_base = f'{samples_path}/eval'  # without including the temperature (for saving ssim results)
            eval_path = f'{samples_path}/eval/temp={params["temperature"]}'  # with temperature

            paths['eval_path'] = eval_path
            paths['val_path'] = eval_path + '/val_imgs'
            paths['resized_path'] = eval_path + '/val_imgs_resized'
            paths['eval_results'] = eval_path  # no need to a separate dir because we don not save segmented images
            paths['eval_path_base'] = eval_path_base

            # ========= adding random_samples_path only if the city name is given in additional info
            if additional_info is not None:
                if additional_info['exp_type'] == 'random_samples':
                    random_samples_path = f'{samples_path}' \
                                          f'/random_samples' \
                                          f'/{additional_info["cond_img_name"]}' \
                                          f'/temp={params["temperature"]}'
                    # adding to dict
                    paths['random_samples_path'] = random_samples_path

                elif additional_info['exp_type'] == 'new_cond':
                    new_cond_path = f'{samples_path}' \
                                f'/new_condition' \
                                f'/orig={additional_info["orig_pure_name"]}' \
                                f' - new_cond={additional_info["new_cond_pure_name"]}'
                    # adding to dict
                    paths['new_cond_path'] = new_cond_path

                    #new_cond_path = f'{orig_path}' \
                    #                f'/new_cond={additional_info["cond_img_name"]}' \
                    #                f'/temp={params["temperature"]}'
                    #paths['new_cond_path'] = new_cond_path



    # ========= glow paths
    else:  # e.g.: img=real/cond=segment/train/lr=1e-4
        samples_path = f'{samples_base_dir}/{run_mode}/lr={lr}'
        checkpoints_path = f'{checkpoints_base_dir}/lr={lr}'

        if run_mode == 'infer':
            step = args.last_optim_step
            samples_path += f'/step={step}'

    paths['samples_path'] = samples_path
    paths['checkpoints_path'] = checkpoints_path
    return paths


def read_image_ids(data_folder, dataset_name):
    """
    It reads all the image names (id's) in the given data_folder, and returns the image names needed according to the
    given dataset_name.

    :param data_folder: to folder to read the images from. NOTE: This function expects the data_folder to exist in the
    'data' directory.

    :param dataset_name: the name of the dataset (is useful when there are extra unwanted images in data_folder, such as
    reading the segmentations)

    :return: the list of the image names.
    """
    img_ids = []
    if dataset_name == 'cityscapes_segmentation':
        suffix = '_color.png'
    elif dataset_name == 'cityscapes_leftImg8bit':
        suffix = '_leftImg8bit.png'
    else:
        raise NotImplementedError('In [read_image_ids] of Dataset: the wanted dataset is not implemented yet')

    # all the files in all the subdirectories
    for city_name, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(suffix):  # read all the images in the folder with the desired suffix
                img_ids.append(os.path.join(city_name, file))

    # print(f'In [read_image_ids]: found {len(img_ids)} images')
    return img_ids


def resize_imgs(path_to_load, path_to_save, h=256, w=256, package='pil'):
    imgs = read_image_ids(path_to_load, dataset_name='cityscapes_leftImg8bit')
    print(f'In [resize_imgs]: read {len(imgs)} from: "{path_to_load}"')
    print(f'In [resize_imgs]: will save resized imgs to: "{path_to_save}"')
    make_dir_if_not_exists(path_to_save)

    for i in range(len(imgs)):
        if i > 0 and i % 50 == 0:
            print(f'In [resize_imgs]: done for the {i}th image')

        img_full_name = imgs[i].split('/')[-1]
        city = img_full_name.split('_')[0]
        image = Image.open(imgs[i])

        if package == 'pil':
            resized = image.resize((w, h))
            resized.save(f'{path_to_save}/{img_full_name}')

        else:  # package == 'scipy' => only works for scipy=1.0.0
            import scipy.misc
            image = np.array(image)
            resized = scipy.misc.imresize(image, (h, w))
            scipy.misc.imsave(f'{path_to_save}/{img_full_name}', resized)

    print('In [resize_imgs]: All done')


def resize_for_fcn(args, params):
    if args.gt:  # photo2label only
        load_path = '/local_storage/datasets/moein/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val'
        save_path = '/Midgard/home/sorkhei/glow2/data/cityscapes/resized/val'
    else:
        paths = compute_paths(args, params)
        load_path, save_path = paths['val_path'], paths['resized_path']

    resize_imgs(load_path, save_path, package='scipy')
