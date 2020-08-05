from torchvision import utils

import helper
from .cityscapes_loader import *
from globals import real_conds_abs_path, device


def init_city_loader(data_folder, image_size, remove_alpha, loader_params, ret_type='for_train'):
    """
    Initializes and returns a data loader based on the data_folder and dataset_name.
    :param remove_alpha:
    :param image_size:
    :param data_folder: the folder to read the images from
    :param loader_params: a dictionary containing the DataLoader parameters such batch_size and so on.
    :param ret_type
    :return: the initialized data loader
    """
    # train data loader
    train_df = {'real': data_folder['real'] + '/train',  # adjusting the paths for the train data folder
                'segment': data_folder['segment'] + '/train'}
    train_dataset = CityDataset(train_df, image_size, remove_alpha, ret_type=ret_type)
    train_loader = data.DataLoader(train_dataset, **loader_params)

    # val data loader
    val_df = {'real': data_folder['real'] + '/val',  # adjusting the paths for the validation data folder
              'segment': data_folder['segment'] + '/val'}
    val_dataset = CityDataset(val_df, image_size, remove_alpha, ret_type=ret_type)

    loader_params['shuffle'] = False  # no need to shuffle for the val set
    val_loader = data.DataLoader(val_dataset, **loader_params)

    print(f'In [init_city_loader]: created train and val loaders of len: {len(train_loader)} and {len(val_loader)}')
    return train_loader, val_loader


def prepare_city_reverse_cond(args, params, run_mode='train'):
    samples_path = helper.compute_paths(args, params)['samples_path']
    save_path = samples_path if run_mode == 'train' else None  # no need for reverse_cond at inference time
    direction = args.direction
    segmentations, id_repeats_batch, real_imgs, boundaries = create_cond(params,
                                                                         fixed_conds=real_conds_abs_path,
                                                                         save_path=save_path,
                                                                         direction=direction)
    # ======= only support for c_flow mode now
    if direction == 'label2photo':
        reverse_cond = {'segment': segmentations, 'boundary': boundaries}

    elif direction == 'photo2label':  # 'photo2label'
        reverse_cond = {'real_cond': real_imgs}
    else:
        raise NotImplementedError('Direction not implemented')
    return reverse_cond


def create_cond(params, fixed_conds=None, save_path=None, direction='label2photo'):
    """
    :param direction:
    :param params:
    :param fixed_conds:
    :param save_path:
    :return:

    Notes:
        - The use of .clones() for saving in this function is very important. In utils.save_image() the float values
          will be normalized to [0-255] integer and the values of the tensor change in-place, so the tensor does not
          have valid float values after this operation, unless we use .clone() to make a new copy of it for saving.
    """
    n_samples = params['n_samples'] if fixed_conds is None else len(fixed_conds)
    data_folder = params['data_folder']
    img_size = params['img_size']

    # this will not be used if fixed_conds is given
    train_df = {'segment': data_folder['segment'] + '/train',
                'real': data_folder['real'] + '/train'}

    city_dataset = CityDataset(train_df, img_size, remove_alpha=True, fixed_cond=fixed_conds)
    print(f'In [create_cond]: created dataset of len {len(city_dataset)}')

    segs = [city_dataset[i]['segment'] for i in range(n_samples)]
    reals = [city_dataset[i]['real'] for i in range(n_samples)]
    b_maps = [city_dataset[i]['boundary'] for i in range(n_samples)]
    seg_paths = [city_dataset[i]['segment_path'] for i in range(n_samples)]
    real_paths = [city_dataset[i]['real_path'] for i in range(n_samples)]

    n_channels = segs[0].shape[0]
    segmentations = torch.zeros((n_samples, n_channels, img_size[0], img_size[1]))
    real_imgs = torch.zeros((n_samples, n_channels, img_size[0], img_size[1]))
    boundaries = torch.zeros((n_samples, n_channels, img_size[0], img_size[1]))
    id_repeats_batch = torch.zeros((n_samples, 34, img_size[0], img_size[1]))  # 34 different IDs
    # id_repeats_batch = None

    for i in range(len(segs)):
        segmentations[i] = segs[i]
        real_imgs[i] = reals[i]
        boundaries[i] = b_maps[i]

    # for i in range(id_repeats_batch.shape[0]):
    #     json_path = seg_paths[i][:-len('color.png')] + 'polygons.json'
    #     id_repeats = id_repeats_to_cond(info_from_json(json_path)['id_repeats'],
    #                                     h=img_size[0], w=img_size[1])  # tensor (34, h, w)
    #     id_repeats_batch[i] = id_repeats

    if save_path:
        helper.make_dir_if_not_exists(save_path)
        # seg_img_name = 'condition' if direction == 'label2photo' else 'real_imgs'
        utils.save_image(segmentations.clone(), f'{save_path}/segmentation.png', nrow=10)  # .clone(): very important

        # real_img_name = 'real_imgs' if direction == 'label2photo' else 'condition'
        utils.save_image(real_imgs.clone(), f'{save_path}/real_img.png', nrow=10)

        if direction == 'label2photo':
            utils.save_image(boundaries.clone(), f'{save_path}/boundary.png', nrow=10)

        print(f'In [create_cond]: saved the condition and real images to: "{save_path}"')

        with open(f'{save_path}/img_paths.txt', 'a') as f:
            f.write("==== SEGMENTATIONS PATHS \n")
            for item in seg_paths:
                f.write("%s\n" % item)

            f.write("==== REAL IMAGES PATHS \n")
            for item in real_paths:
                f.write("%s\n" % item)
        print('In [create_cond]: saved the image paths \n')

    return segmentations.to(device), id_repeats_batch.to(device), real_imgs.to(device), boundaries.to(device)


def create_boundary_maps(params):
    """
    This function should be called once before using the boundary maps in the generative process. Currently, it requires
    to have GPU access (the "get_edges" functions needs it). Otherwise it would be too slow.
    :param params:
    :return:
    Notes:
        - do_ceil is not needed here because the generated boundary maps are of the original size.
    """
    batch_size = 40
    loader_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 1}
    train_loader, val_loader = \
        init_city_loader(data_folder=params['data_folder'],
                                      image_size=[1024, 2048],  # keeping original size
                                      remove_alpha=True,  # removing the alpha channel
                                      loader_params=loader_params,
                                      ret_type='all')  # return everything in the batch

    print(f'In [create_boundary_maps]: performing with data loaders of size: \n'
          f'train_loader: {len(train_loader)} \n'
          f'val_loader: {len(val_loader)} \n'
          f'and batch_size of: {batch_size} \n')

    for loader_name, loader in {'train_loader': train_loader, 'val_loader': val_loader}.items():
        print(f'In [create_boundary_maps]: creating for {loader_name}')
        for i_batch, batch in enumerate(loader):
            if i_batch % 1 == 0:
                print(f'Doing for the batch {i_batch}')

            instance_maps = batch['instance'].to(device)
            boundaries = helper.get_edges(instance_maps)
            boundary_paths = batch['boundary_path']
            # save one by one in the same location as gtFine images
            helper.save_one_by_one(boundaries, boundary_paths, save_path=None)  # saving to boundary_paths
        print(f'In [create_boundary_maps]: done for {loader_name}')
    print('In [create_boundary_maps]: all done')


def recreate_boundary_map(instance_path, boundary_path):
    from PIL import Image
    from torchvision import transforms

    trans = transforms.Compose([transforms.ToTensor()])
    instance = trans(Image.open(instance_path)).unsqueeze(0).to(device)
    boundary = helper.get_edges(instance)
    print('boundary shape:', boundary.shape)
    helper.save_one_by_one(boundary, [boundary_path], save_path=None)  # saving to boundary_paths
    print(f'In [recreate_boundary_map]: save the recreated boundary to: "{boundary_path}"')
