from . import city, maps, mnist, transient
from globals import maps_fixed_conds


def retrieve_rev_cond(args, params, run_mode='train'):
    if args.dataset == 'cityscapes':
        reverse_cond = None if args.exp else city.prepare_city_reverse_cond(args, params, run_mode)
    elif args.dataset == 'maps':
        reverse_cond = maps.create_rev_cond(args, params, fixed_conds=maps_fixed_conds, also_save=True)
    else:
        raise NotImplementedError('Dataset not implemented')
    return reverse_cond


def init_data_loaders(args, params):
    batch_size = params['batch_size']
    if args.dataset == 'cityscapes':
        loader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
        train_loader, \
            val_loader = city.init_city_loader(data_folder=params['data_folder'],
                                               image_size=(params['img_size']),
                                               remove_alpha=True,  # removing the alpha channel
                                               loader_params=loader_params)
    elif args.dataset == 'maps':
        train_loader, val_loader = maps.init_maps_loaders(args, params)

    else:
        raise NotImplementedError

    print(f'\n\nIn [init_data_loaders]: training with data loaders of size: \n'
          f'train_loader: {len(train_loader):,} \n'
          f'val_loader: {len(val_loader):,} \n'
          f'and batch_size of: {batch_size}\n\n')
    return train_loader, val_loader
