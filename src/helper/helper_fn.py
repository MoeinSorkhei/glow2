import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt


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


def calc_cond_shapes(orig_shape, in_channels, img_size, n_block, mode):
    z_shapes = calc_z_shapes(in_channels, img_size, n_block)

    if mode == 'z_outs':  # the condition is has the same shape as the z's themselves
        return z_shapes

    # flows_outs are before split => channels should be multiplied by 2 (except for the last shape)
    if mode == 'flows_outs':
        for i in range(len(z_shapes) - 1):
            z_shapes[i] = list(z_shapes[i])
            z_shapes[i][0] = z_shapes[i][0] * 2
            z_shapes[i] = tuple(z_shapes[i])
        return z_shapes

    cond_shapes = []
    for z_shape in z_shapes:
        h, w = z_shape[1], z_shape[2]
        if mode == 'segment':
            channels = (orig_shape[0] * orig_shape[1] * orig_shape[2]) // (h * w)  # new channels with new h and w

        elif mode == 'segment_id':
            channels = 34 + (orig_shape[0] * orig_shape[1] * orig_shape[2]) // (h * w)

        else:
            raise NotImplementedError

        cond_shapes.append((channels, h, w))

    return cond_shapes


def print_info(args, params, model, which_info='all'):
    if which_info == 'params' or which_info == 'all':
        # printing important running params
        print(f'{"=" * 50} \n'
              f'In [print_info]: Important params: \n'
              f'model: {args.model} \n'
              f'lr: {args.lr if args.lr is not None else params["lr"]} \n'
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
              f'{"=" * 50}\n')


def scientific(float_num):
    if float_num < 1e-4:
        return str(float_num)
    elif float_num == 1e-4:
        return '1e-4'
    elif float_num == 1e-3:
        return '1e-3'
    raise NotImplementedError('In [scientific]: Conversion from float to scientific str needed.')


def compute_paths(args, params):
    dataset = args.dataset
    model = args.model
    img = 'segment' if args.train_on_segment else 'real'  # now only training glow on segmentations
    cond = args.cond_mode
    run_mode = 'infer' if args.exp else 'train'

    # base paths - common between all models
    samples_base_dir = f'{params["samples_path"]}/{dataset}/model={model}/img={img}/cond={cond}'
    checkpoints_base_dir = f'{params["checkpoints_path"]}/{dataset}/model={model}/img={img}/cond={cond}'

    # specifying lr: from args if determined, otherwise default from params.json
    lr = scientific(args.lr if args.lr is not None else params['lr'])
    paths = {}  # the paths dict be filled along the way

    # ========= c_flow paths
    if model == 'c_flow':
        c_flow_type = 'left_pretrained' if args.left_pretrained else 'from_scratch'
        # paths common between c_flow models
        samples_path = f'{samples_base_dir}/{c_flow_type}'
        checkpoints_path = f'{checkpoints_base_dir}/{c_flow_type}'

        # details for left_pretrained paths
        if c_flow_type == 'left_pretrained':
            left_lr = args.left_lr  # example: left_pretrained/left_lr=1e-4/freezed/left_step=10000
            left_status = 'unfreezed' if args.left_unfreeze else 'freezed'
            left_step = args.left_step  # optim_step of the left glow

            samples_path += f'/left_lr={left_lr}/{left_status}/left_step={left_step}'
            checkpoints_path += f'/left_lr={left_lr}/{left_status}/left_step={left_step}'

            # left glow checkpoint path
            left_glow_path = f'{params["checkpoints_path"]}/{dataset}/model=glow/img=segment/cond={args.left_cond}'
            left_glow_path += f'/lr={left_lr}'  # e.g.: model=glow/img=segment/cond=None/lr=1e-4
            paths['left_glow_path'] = left_glow_path

        # common between left_pretrained and from_scratch
        samples_path += f'/{run_mode}/lr={lr}'  # e.g.: left_lr=1e-4/freezed/left_step=10000/train/lr=1e-5
        checkpoints_path += f'/lr={lr}'

        # infer: also adding optimization step (step is specified after lr)
        if run_mode == 'infer':
            optim_step = args.last_optim_step  # e.g.: left_lr=1e-4/freezed/left_step=10000/infer/lr=1e-5/step=1000
            samples_path += f'{samples_path}/step={optim_step}'

    # ========= glow paths
    else:  # e.g.: img=real/cond=segment/train/lr=1e-4
        samples_path = f'{samples_base_dir}/{run_mode}/lr={lr}'
        checkpoints_path = f'{checkpoints_base_dir}/lr={lr}'

        if run_mode == 'infer':
            step = args.last_optim_step
            samples_path += f'/step={step}'

    # bring everything together
    paths['samples_path'] = samples_path
    paths['checkpoints_path'] = checkpoints_path
    return paths

