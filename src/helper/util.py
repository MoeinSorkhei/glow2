import torch

import data_handler
import experiments


def get_edges(t):
    """
    This function is taken from: https://github.com/NVIDIA/pix2pixHD.
    :param t:
    :return:
    """
    edge = torch.cuda.ByteTensor(t.size()).zero_()
    # comparing with the left pixels
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1]).type(torch.uint8)
    # comparing with the right pixels
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1]).type(torch.uint8)
    # comparing with the lower pixels
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).type(torch.uint8)
    # comparing with upper  pixels
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).type(torch.uint8)
    return edge.float()


def create_boundary_maps(params, device):
    """
    This function should be called once before using the boundary maps in the generative process. Currently, it requires
    to have GPU access (the "get_edges" functions needs it). Otherwise it would be too slow.
    :param params:
    :param device:
    :return:
    Notes:
        - do_ceil is not needed here because the generated boundary maps are of the original size.
    """
    batch_size = 40
    loader_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 1}
    train_loader, val_loader = \
        data_handler.init_city_loader(data_folder=params['data_folder'],
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
            boundaries = get_edges(instance_maps)
            boundary_paths = batch['boundary_path']
            # save one by one in the same location as gtFine images
            experiments.save_one_by_one(boundaries, boundary_paths, save_path=None)  # saving to boundary_paths
        print(f'In [create_boundary_maps]: done for {loader_name}')
    print('In [create_boundary_maps]: all done')


def recreate_boundary_map(instance_path, boundary_path, device):
    from PIL import Image
    from torchvision import transforms

    trans = transforms.Compose([transforms.ToTensor()])
    instance = trans(Image.open(instance_path)).unsqueeze(0).to(device)
    boundary = get_edges(instance)
    print('boundary shape:', boundary.shape)
    experiments.save_one_by_one(boundary, [boundary_path], save_path=None)  # saving to boundary_paths
    print(f'In [recreate_boundary_map]: save the recreated boundary to: "{boundary_path}"')
