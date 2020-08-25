from . import city, maps, mnist, transient
from globals import maps_fixed_conds, device


def retrieve_rev_cond(args, params, run_mode='train'):
    if args.dataset == 'cityscapes':
        reverse_cond = None if args.exp else city.prepare_city_reverse_cond(args, params, run_mode)
    elif args.dataset == 'maps':
        reverse_cond = maps.create_rev_cond(args, params, fixed_conds=maps_fixed_conds, also_save=True)
    else:
        raise NotImplementedError('Dataset not implemented')
    return reverse_cond


def extract_batches(batch, args):
    """
    This function depends onf the dataset and direction.
    :param batch:
    :param args:
    :return:
    """
    if args.dataset == 'cityscapes':
        img_batch = batch['real'].to(device)
        segment_batch = batch['segment'].to(device)
        boundary_batch = batch['boundary'].to(device)

        if args.direction == 'label2photo':
            left_batch = segment_batch
            right_batch = img_batch
            extra_cond_batch = boundary_batch if args.use_bmaps else None

        elif args.direction == 'photo2label':
            left_batch = img_batch
            right_batch = segment_batch
            extra_cond_batch = None

        elif args.direction == 'bmap2label':
            left_batch = boundary_batch
            right_batch = segment_batch
            extra_cond_batch = None

        else:
            raise NotImplementedError

    elif args.dataset == 'maps':
        img_batch = batch['photo'].to(device)
        segment_batch = batch['the_map'].to(device)
        boundary_batch = None
        raise NotImplementedError('Should refactor this part')

    else:
        raise NotImplementedError
    return left_batch, right_batch, extra_cond_batch


def init_data_loaders(args, params):
    batch_size = params['batch_size']
    if args.dataset == 'cityscapes':
        loader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
        train_loader, \
            val_loader = city.init_city_loader(data_folder=params['data_folder'],
                                               image_size=(params['img_size']),
                                               loader_params=loader_params)
    elif args.dataset == 'maps':
        train_loader, val_loader = maps.init_maps_loaders(args, params)

    else:
        raise NotImplementedError

    print(f'\nIn [init_data_loaders]: training with data loaders of size: \n'
          f'train_loader: {len(train_loader):,} \n'
          f'val_loader: {len(val_loader):,} \n'
          f'and batch_size of: {batch_size}\n')
    return train_loader, val_loader
