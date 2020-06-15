import torch

from . import third_party
import helper
import experiments
import data_handler
from trainer import calc_val_loss
from globals import device
import helper
import models


def evaluate_city(args, params):
    if args.gt:  # for ground-truth images (photo2label only)
        paths = {'resized_path': '/Midgard/home/sorkhei/glow2/data/cityscapes/resized/val',
                 'eval_results': '/Midgard/home/sorkhei/glow2/gt_eval_results'}
    else:
        paths = helper.compute_paths(args, params)

    output_dir = paths['eval_results']
    # no resize for 256x256, so we read from validation path directly
    result_dir = paths['resized_path'] if params['img_size'] != [256, 256] else paths['val_path']
    # third_party.evaluate(data_folder=params['data_folder']['base'], paths=paths, split='val')
    if args.direction == 'label2photo':
        third_party.evaluate(data_folder=params['data_folder']['base'],
                             output_dir=output_dir,
                             result_dir=result_dir,
                             split='val')
    elif args.direction == 'photo2label':
        raise NotImplementedError('Code to be written for segmentation evaluation')
    else:
        raise NotImplementedError
    print(f'In [evaluate_city]: evaluation done')


def eval_complete(args, params, device):
    for temp in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f'In [eval_complete]: evaluating for temperature = {temp}')
        params['temperature'] = temp

        # inference
        experiments.infer_on_validation_set(args, params, device)
        torch.cuda.empty_cache()  # very important
        print('In [eval_complete]: inference done \n')

        # resize if not 256x256
        if params['img_size'] == [256, 256]:
            print(f'In [eval_complete]: no resize since the image size already is {params["img_size"]} \n')
        else:
            helper.resize_for_fcn(args, params)
            print('In [eval_complete]: resize done \n')

        # evaluate
        evaluate_city(args, params)
        print(f'In [eval_complete]: evaluating for temperature = {temp}: done \n')

    torch.cuda.empty_cache()  # very important
    print('In [eval_complete]: all done \n')


def compute_val_bpd(args, params):
    loader_params = {'batch_size': params['batch_size'], 'shuffle': False, 'num_workers': 0}
    _, val_loader = data_handler.init_city_loader(data_folder=params['data_folder'],
                                                  image_size=(params['img_size']),
                                                  remove_alpha=True,  # removing the alpha channel
                                                  loader_params=loader_params)

    model = models.init_and_load(args, params, run_mode='infer')

    mean, std = calc_val_loss(args, params, device, model, val_loader)
    print(f'In [compute_val_bpd]: mean = {mean} - std = {std}')


def to_nearest_label_color(img):
    """
    Part of the code inspired and taken from: https://github.com/phillipi/pix2pix/issues/115.
    :param img:
    :return:
    """
    # 1. CHEKCK THESE RGB VALUES AGAIN
    # 2. what about the classes that are not used for evaluation? -- see get_score function
    # 3. / 255 again

    # classes used for evaluation, ordered by their trainIDs
    label_colors_as_list = [(128, 64, 128), # road
                     (244, 35, 232), # sidewalk
                     (70, 70, 70), # building
                     (102, 102, 156), # wall
                     (190, 153, 153), # fence
                     (153, 153, 153), # pole
                     (250, 170, 30), # traffic light
                     (220, 220, 0), # traffic sign
                     (107, 142, 35), # vegetation
                     (152, 251, 152), # terrain
                     (70, 130, 180), # sky
                     (220, 20, 60), # person
                     (255, 0, 0), # rider
                     (0, 0, 142), # car
                     (0, 0, 70), # truck
                     (0, 60, 100), # bus
                     (0, 80, 100), # train
                    (0, 0, 230), # motorcycle
                    (119, 11, 32)] # bicycle

    h, w = img.shape[1], img.shape[2]  # img (C, H, W)
    n_labels = len(label_colors_as_list)
    # label_colors = torch.FloatTensor(label_colors_as_list)  # shape (n_labels, 3)

    # already be resized
    img = img * 255  # normalize values to 0-255 interval

    # a 4-D tensor which has the RGB colors of all the classes, each of them having a tensor 3-D tensor filled with
    # their RGB colors
    label_colors_img = torch.zeros((n_labels, 3, h, w))
    for i in range(n_labels):
        label_colors_img[i, 0, :, :] = label_colors_as_list[i][0]  # fill R
        label_colors_img[i, 1, :, :] = label_colors_as_list[i][1]  # fill G
        label_colors_img[i, 2, :, :] = label_colors_as_list[i][2]  # fill B

    # difference with each class per pixel (average over RGB)
    dists = torch.ones((n_labels, h, w))  # (n_labels, H, W)
    for i in range(n_labels):
        dists[i] = ((img - label_colors_img[i]) ** 2).mean(dim=0)  # dist[i]: shape (H, W)

    min_val, min_indices = torch.min(dists, dim=0)  # min_indices (H, W)
    nearest_image = torch.zeros((3, h, w))  # the final nearest image (C, H, W)

    for i in range(n_labels):
        # (H, W), 1 whenever that class has min distance, elsewhere 0
        mask = (min_indices == i).int()
        # shape (3, 1, 1) - the RGB values of the corresponding color
        color = torch.FloatTensor(label_colors_as_list[i]).unsqueeze(1).unsqueeze(2)
        # (C, H, W), all the channels 1 in the (i, j) pixel where class i is nearest, elsewhere 0
        mask_unsqueezed = torch.zeros_like(nearest_image) + mask  # broadcast in channel dimension
        # (C, H, W), channels have RGB in (i, j) pixel where class i is nearest, elsewhere 0
        mask_unsqueezed = mask_unsqueezed * color  # broadcast to all (i, j) locations
        # add the colors for (i, j) pixels corresponding to class i to the whole image
        nearest_image += mask_unsqueezed

    return nearest_image
