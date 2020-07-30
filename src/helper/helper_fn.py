import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import glob

from globals import device


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


def load_checkpoint(path_to_load, optim_step, model, optimizer, resume_train=True):
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


def print_info(args, params, model, which_info='all'):
    if which_info == 'params' or which_info == 'all':
        # printing important running params
        # print(f'{"=" * 50} \n'
        #       f'In [print_info]: Important params: \n'
        #       f'model: {args.model} \n'
        #       # f'lr: {args.lr if args.lr is not None else params["lr"]} \n'
        #       f'lr: {params["lr"]} \n'
        #       f'batch_size: {params["batch_size"]} \n'
        #       f'temperature: {params["temperature"]} \n'
        #       f'last_optim_step: {args.last_optim_step} \n'
        #       f'left_lr: {args.left_lr} \n'
        #       f'left_step: {args.left_step} \n'
        #       f'cond: {args.cond_mode} \n\n')

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
    assert args.model != 'glow' and additional_info is None  # these cases not implemented yet
    if args.model == 'c_flow':
        return compute_paths_old(args, params)

    dataset = args.dataset
    direction = args.direction
    model = args.model
    image_size = f"{params['img_size'][0]}x{params['img_size'][1]}"
    step = args.last_optim_step
    temp = params['temperature']
    # run_mode = 'infer' if args.exp else 'train'

    # samples path for train
    samples_base_dir = f'{params["samples_path"]}'
    samples_path = os.path.join(samples_base_dir, dataset, image_size, model, direction, 'train')

    # generic infer path
    infer_path = os.path.join(samples_base_dir, dataset, image_size, model, direction, 'infer', f'step={step}')
    # path for evaluation
    eval_path = os.path.join(infer_path, 'eval', f'temp={temp}')
    val_path = os.path.join(eval_path, 'val_imgs')  # where inferred val images are stored inside the eval folder

    # checkpoints path
    checkpoints_base_dir = f'{params["checkpoints_path"]}'
    checkpoints_path = os.path.join(checkpoints_base_dir, dataset, image_size, model, direction)

    return {
        'samples_path': samples_path,
        'eval_path': eval_path,
        'val_path': val_path,
        'checkpoints_path': checkpoints_path
    }


def compute_paths_old(args, params, additional_info=None):
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

    # elif model == 'c_flow' or model == 'c_glow':  # only this part needs to be changed for new datasets
    # elif model == 'c_flow' or 'c_glow' in model:  # only this part needs to be changed for new datasets
    else:
        if dataset == 'cityscapes' and args.direction == 'label2photo':
            img = 'real'
            cond = args.cond_mode

        elif dataset == 'cityscapes' and args.direction == 'photo2label':
            img = 'segment'
            cond = 'real'

        elif dataset == 'maps' and args.direction == 'map2photo':
            img = 'photo'
            cond = 'map'

        elif dataset == 'maps' and args.direction == 'photo2map':
            img = 'map'
            cond = 'photo'

        else:
            raise NotImplementedError
    # else:
    #    raise NotImplementedError('model not implemented')

    # cond = args.cond_mode
    w_conditional = args.w_conditional
    act_conditional = args.act_conditional

    run_mode = 'infer' if args.exp else 'train'
    h, w = params['img_size'][0], params['img_size'][1]

    # used for checkpoints only
    coupling_str_checkpts = '/coupling_net' if args.coupling_cond_net else ''

    if args.dataset == 'cityscapes':
        if args.direction == 'label2photo':
            cond_with_ceil = 'segment_boundary/do_ceil=True' if args.use_bmaps else 'segment'
        else:
            cond_with_ceil = 'real'
    else:  # for all other datasets
        cond_with_ceil = cond

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

    # ========= other models paths
    else:  # e.g.: img=real/cond=segment/train/lr=1e-4
        samples_path = f'{samples_base_dir}/{run_mode}/lr={lr}'
        checkpoints_path = f'{checkpoints_base_dir}/lr={lr}'

    # ========= infer (regardless of the model): also adding optimization step (step is specified after lr)
    if run_mode == 'infer':
        optim_step = args.last_optim_step  # e.g.: left_lr=1e-4/freezed/left_step=10000/infer/lr=1e-5/step=1000
        samples_path += f'/step={optim_step}'

        eval_path_base = f'{samples_path}/eval'  # without including the temperature (for saving ssim results)
        eval_path = f'{samples_path}/eval/temp={params["temperature"]}'  # with temperature

        paths['eval_path'] = eval_path
        paths['val_path'] = eval_path + '/val_imgs'
        paths['resized_path'] = eval_path + '/val_imgs_resized'
        paths['eval_results'] = eval_path  # no need to a separate dir because we do not save segmented images
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

    paths['samples_path'] = samples_path
    paths['checkpoints_path'] = checkpoints_path

    if 'dual_glow' in model:
        paths['checkpoints_path'] = f'{params["checkpoints_path"]}' \
                                    f'/{dataset}' \
                                    f'/{h}x{w}' \
                                    f'/model={model}' \
                                    f'/img={img}' \
                                    f'/cond={cond_with_ceil}'

        paths['samples_path'] = f'{params["samples_path"]}' \
                                f'/{dataset}' \
                                f'/{h}x{w}' \
                                f'/model={model}' \
                                f'/img={img}' \
                                f'/cond={cond_with_ceil}'

        eval_path = f'{samples_path}/eval/temp={params["temperature"]}'  # with temperature
        paths['eval_path'] = eval_path
        paths['val_path'] = eval_path + '/val_imgs'
        paths['eval_results'] = eval_path  # no need to a separate dir because we do not save segmented images

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


def extend_val_path(val_path, number):
    return f'{val_path}_{number}'


# def indexed_eval_file(output_dir):
#     file = output_dir + '/evaluation_results_1.txt'  # first time evaluation
#     if os.path.isfile(output_dir + '/evaluation_results_1.txt'):  # second time evaluation
#         file = output_dir + '/evaluation_results_2.txt'
#
#     if os.path.isfile(output_dir + '/evaluation_results_2.txt'):  # third time evaluation
#         file = output_dir + '/evaluation_results_3.txt'
#     return file


def files_with_suffix(directory, suffix):
    files = [os.path.abspath(path) for path in glob.glob(f'{directory}/**/*{suffix}', recursive=True)]  # full paths
    return files


def pure_name(path):
    return os.path.split(path)[1]


def replace_suffix(name, direction):
    if direction == 'segment_to_real':
        return name.replace('_gtFine_color.png', '_leftImg8bit.png')
    return name.replace('_leftImg8bit.png', '_gtFine_color.png')


def get_all_data_folder_images(path, partition, image_type):
    pattern = '_color.png' if image_type == 'segment' else '_leftImg8bit.png'
    files = [os.path.abspath(path) for path in glob.glob(f'{path}/{partition}/**/*{pattern}', recursive=True)]  # full paths
    return files


def get_all_partition_files(dataset, base_data_folder, img_type, partition):  # NOT VALID
    assert dataset == 'cityscapes' and img_type == 'segment'

    # partition_path = os.path.join(params['data_folder'][img_type], partition)
    partition_path = os.path.join(base_data_folder[img_type], partition)
    files = [os.path.abspath(path) for path in glob.glob(f'{partition_path}/**/*_color.png', recursive=True)]  # full paths

    print(f'In [get_all_validation_files]: found {len(files)} images in the partition_path: "{partition_path}"')
    return files


def open_and_resize_image(path, for_model=None):
    image = Image.open(path).resize((256, 256))  # read image and resize
    image_array = (np.array(image)[:, :, :3] / 255).astype(np.float32)  # remove alpha channel

    if for_model == 'dual_glow':
        return np.expand_dims(image_array, axis=(0, 1))  # expand for dual_glow model
    return image_array


def rescale_image(image):
    # just the same as torch save_image function
    return np.clip((image * 255) + 0.5, a_min=0, a_max=255).astype(np.uint8)


def rescale_and_save_image(image, path):
    rescaled = rescale_image(image)
    Image.fromarray(rescaled).save(path)


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


def clean_midgard(args):
    dataset, model = args.dataset, args.model

    if args.direction == 'label2photo':
        img, cond = 'real', args.cond_mode
    elif args.direction == 'photo2label':
        img, cond = 'segment', 'real'
    elif args.direction == 'map2photo':
        img, cond = 'photo', 'map'
    elif args.direction == 'photo2map':
        img, cond = 'map', 'photo'
    else:
        raise NotImplementedError

    # base_path = '/Midgard/home/sorkhei/glow2/checkpoints/cityscapes/256x256/model=c_flow/img=real/cond=segment_boundary/do_ceil=True'
    base_path = f'/Midgard/home/sorkhei/glow2/checkpoints/{dataset}/256x256/' \
                f'model={model}/img={img}/cond={cond}'

    # if args.dataset == 'cityscapes':
    max_step = 130000
    # else:
    #     max_step = 80000

    if model == 'c_flow':
        if cond == 'segment_boundary':
            base_path += '/do_ceil=True'

        if dataset == 'cityscapes':
            run_paths = [
                'w_conditional=False/act_conditional=False/from_scratch/lr=1e-4',
                'w_conditional=False/act_conditional=True/from_scratch/lr=1e-4',
                'w_conditional=True/act_conditional=False/from_scratch/lr=1e-4',
                'w_conditional=True/act_conditional=True/from_scratch/lr=1e-4',
                'w_conditional=True/act_conditional=True/coupling_net/from_scratch/lr=1e-4'
            ]
        elif dataset == 'maps':
            run_paths = ['w_conditional=False/act_conditional=False/coupling_net/from_scratch/lr=1e-4']
        else:
            raise NotImplementedError

    elif 'c_glow' in model:
        run_paths = ['w_conditional=False/act_conditional=False/lr=1e-4']

    elif 'dual_glow' in model:
        raise NotImplementedError('Need to understand how tensorflow saves checkpoints')

    else:
        raise NotImplementedError

    for run_path in run_paths:
        checkpoints = sorted(os.listdir(os.path.join(base_path, run_path)))
        print('Available checkpoints: \n', "\n".join(checkpoints),
              f'\nThe ones less than {max_step} will be deleted. Press Enter to confirm.')
        input()

        for checkpoint in checkpoints:
            step = int(checkpoint.split('=')[-1][:-3])

            if step < max_step:
                full_path = os.path.join(base_path, run_path, checkpoint)
                os.remove(full_path)
                print(f'Removed: "{full_path}"')
        print(f'Cleaning for folder "{run_path}": done')


