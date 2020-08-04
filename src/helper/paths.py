import glob
import os


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
            cond = 'segment_boundary' if args.use_bmaps else 'segment'

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

    lr = _scientific(params['lr'])
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


def _scientific(float_num):
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


def extend_val_path(val_path, number):
    return f'{val_path}_{number}'


def make_dir_if_not_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print(f'In [make_dir_if_not_exists]: created path "{directory}"')


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


def files_with_suffix(directory, suffix):
    files = [os.path.abspath(path) for path in glob.glob(f'{directory}/**/*{suffix}', recursive=True)]  # full paths
    return files


def get_file_with_name(directory, filename):
    """
    Finds the file with a specific name in a hierarchy of directories.
    :param directory:
    :param filename:
    :return:
    """
    files = [os.path.abspath(path) for path in glob.glob(f'{directory}/**/{filename}', recursive=True)]  # full paths
    return files[0]  # only one file should be found with a specific name


def absolute_paths(directory):
    # return [os.path.abspath(filepath) for filepath in os.listdir(directory)]
    return [os.path.abspath(path) for path in glob.glob(f'{directory}/**/*', recursive=True)]  # full paths


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

