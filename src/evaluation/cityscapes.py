import torch
from torchvision import utils

from . import third_party
from helper import *
import helper
import experiments
import data_handler
from trainer import calc_val_loss
from globals import device
import helper
import models

import numpy as np
import os
from PIL import Image
from torchvision import transforms


def eval_city_with_all_temps(args, params, device):
    """
    Performs steps needed for evaluation with all the temperatures.
    :param args:
    :param params:
    :param device:
    :return:
    """
    for temp in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f'In [eval_city_with_all_temps]: evaluating for temperature = {temp}')
        params['temperature'] = temp

        # inference
        experiments.infer_on_validation_set(args, params, device)
        torch.cuda.empty_cache()  # very important
        print('In [eval_city_with_all_temps]: inference done \n')

        # current implementation fo label2photo requires that generated images are resized and save to the folder
        if args.direction == 'label2photo':
            # resize if not 256x256
            if params['img_size'] == [256, 256]:
                print(f'In [eval_city_with_all_temps]: no resize since the image size already is {params["img_size"]} \n')
            else:
                helper.resize_for_fcn(args, params)
                print('In [eval_city_with_all_temps]: resize done \n')

        # evaluate
        eval_city_with_temp(args, params)
        print(f'In [eval_city_with_all_temps]: evaluating for temperature = {temp}: done \n')

    torch.cuda.empty_cache()  # very important
    print('In [eval_city_with_all_temps]: all done \n')


def eval_city_with_temp(args, params):
    """
    Evaluate the generated validation images wit the given temperature. This function is called from
    eval_city_with_all_temps function that tries different temperatures.
    If this function is called individually, the it used the temperature specified in params to find the correct path for
    the images that are to be evaluated.

    :param args:
    :param params:
    :return:
    """
    if args.direction == 'label2photo':
        if args.gt:  # for ground-truth images (photo2label only)
            paths = {'resized_path': '/Midgard/home/sorkhei/glow2/data/cityscapes/resized/val',
                     'eval_results': '/Midgard/home/sorkhei/glow2/gt_eval_results'}
        else:
            paths = helper.compute_paths(args, params)

        output_dir = paths['eval_results']
        # no resize for 256x256, so we read from validation path directly
        result_dir = paths['resized_path'] if params['img_size'] != [256, 256] else paths['val_path']

        third_party.evaluate_real_imgs_with_temp(data_folder=params['data_folder']['base'],
                                                 output_dir=output_dir,
                                                 result_dir=result_dir,
                                                 split='val')

    else:
        evaluate_segmentations_with_temp(args, params)
    print(f'In [eval_city_with_temp]: evaluation done')


def evaluate_segmentations_with_temp(args, params):
    paths = helper.compute_paths(args, params)
    # val images should already be inferred: concat val_path with image name to get full address
    syn_segs_names = os.listdir(paths['val_path'])  # e.g. munster_000109_000019_leftImg8bit.png
    syn_segs_paths = [os.path.join(paths['val_path'], file) for file in syn_segs_names]  # full path of the files

    # create cityscapes instance (needs the base dir to cityscapes data)
    cs = third_party.cityscapes(params['data_folder']['base'])
    n_cl = len(cs.classes)  # 19 classes used for evaluation
    hist_perframe = np.zeros((n_cl, n_cl))  # using exactly the same functions defined in third_party for evaluation

    for i in range(len(syn_segs_names)):  # each inferred image should end with _color.png
        syn_seg_name = syn_segs_names[i]  # e.g. munster_000109_000019_leftImg8bit.png
        syn_seg_path = syn_segs_paths[i]  # full path to the image in the val_path folder (used for inference)

        # extracting city_name: e.g. for munster_000109_000019_leftImg8bit.png -> city_name: lindau
        city_name = syn_segs_names[i].split('_')[0]
        ref_img_path = os.path.join(params['data_folder']['segment'], 'val', city_name, syn_seg_name)  # ref image from gt data
        hist = evaluate_single_segmentation_with_temp(syn_seg_path, ref_img_path, params['data_folder']['base'])
        hist_perframe += hist

        if i % 50 == 0:
            print(f'In [evaluate_segmentations_with_temp]: evaluation for {i} images: done - scores so far:')
            third_party.get_score_and_print(hist_perframe, cs.classes)

    third_party.get_score_and_print(hist_perframe, cs.classes, verbose=True, save_to=paths['eval_path'])
    print(f'[evaluate_segmentations_with_temp]: all done')


def evaluate_single_segmentation_with_temp(syn_img_path, ref_img_path, cityscapes_base_dir, verbose=False, save_path=None):
    """
    Procedure:
        - Each image should first be resized to the original segmentation image size.
        - Each generated segmentation is converted to have the nearest RGB colors of the 19 classes defined in labels of
          cityscapes.
        - From the color2trainId mapping defined in the cityscapes class of the third_party package, we retrieve the trainIds
          for each pixel based on the color.
        - We compute the scores using the reference image and through the functions provided in the third_party package.
    """
    trans = transforms.Compose([transforms.ToTensor()])  # transformation for PIL image to tensor
    ref_img = Image.open(ref_img_path)
    ref_img = (trans(ref_img)[:3] * 255).to(torch.uint8)  # remove alpha channel and normalize to 0-255 and convert to int

    # resize synthesized image to the original image size and getting nearest cityscapes class colors
    syn_img = Image.open(syn_img_path).resize((ref_img.shape[2], ref_img.shape[1]))  # PIL takes dim order as (W, H)
    syn_img = trans(syn_img)[:3] * 255  # should be of range 0-255 for to_nearest_label_color to work
    nearest = to_nearest_label_color(syn_img)  # returns result with int values

    if save_path:  # if we want to save: we should normalize values to 0-1
        utils.save_image(torch.clone(nearest), f'{save_path}/nearest.png', normalize=True, nrow=1, padding=0)
        utils.save_image(torch.clone(ref_img.float()), f'{save_path}/ref.png', normalize=True, nrow=1, padding=0)
        print(f'In [evaluate_single_segmentation_with_temp]: saved images to "{save_path}')

    cs = third_party.cityscapes(cityscapes_base_dir)  # cityscapes instance

    # converting color to trainIs using the dictionaries of the cityscapes class
    ref_label = color_to_train_id(ref_img.cpu().data.numpy(), cs)
    syn_label = color_to_train_id(nearest.cpu().data.numpy(), cs)

    # evaluate the single image
    n_cl = len(cs.classes)  # 19 classes used for evaluation
    hist = third_party.fast_hist(ref_label.flatten(), syn_label.flatten(), n_cl)

    if verbose:
        mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = third_party.get_scores(hist)
        print('Mean pixel accuracy: %f\n' % mean_pixel_acc)
        print('Mean class accuracy: %f\n' % mean_class_acc)
        print('Mean class IoU: %f\n' % mean_class_iou)

        for i, cl in enumerate(cs.classes):
            while len(cl) < 15:
                cl = cl + ' '  # adding spaces
            print('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))

    return hist


def color_to_train_id(img, cityscapes_instance):
    """
    :param cityscapes_instance: an instance of the cityscapes class define in the third party pakcage
    :param img: the image that is to be transformed to trainIds.
    :return: the label corresponding to the image where each pixel has the corresponding trainId.

    Procedure:
        - In the for loop, we compare 3 channels of each pixel with the RGB color of the train_id, and if all channels
          are equal (sum of the comparisons of the three channels for each pixel), that pixel will have train_id will.
          img == color_as_array: of shape (3, H, W)
          np.sum(img == color_as_array, axis=0): of shape (H, W)
          np.sum(img == color_as_array, axis=0) == 3: pixels for which all the RGB channels were equal to the give color

    """
    train_id_to_color_map = cityscapes_instance.trainId2color

    h, w = img.shape[1], img.shape[2]
    label = np.zeros((h, w), dtype=np.uint8)

    for train_id, color in train_id_to_color_map.items():
        color_as_array = np.array(color)[:, np.newaxis][:, np.newaxis]  # shape (3, 1, 1): the color of the train_id
        id_mask = (np.sum(img == color_as_array, axis=0) == 3) * train_id  # each pixel will have train_id if equal, otherwise zero
        id_mask = id_mask.astype(np.uint8)  # so it could be summed with label
        label += id_mask
    return label


def to_nearest_label_color(img):
    """
    Takes a synthesized image and converts the pixels to the nearest color defined in the Cityscapes classes (see the
    code for classes and their corresponding colors).
    Part of the code inspired and taken from: https://github.com/phillipi/pix2pix/issues/115.

    :param img:
    :return:

    Notes:
        - The input image is assumed to have values in the range 0-255.
        - The output image will have int values as defined in the label colors of Cityscapes.
    """
    # 19 classes used for evaluation, ordered by their trainIDs taken from original cityscapes classes
    label_colors_as_list = [(128, 64, 128),  # road
                     (244, 35, 232),  # sidewalk
                     (70, 70, 70),  # building
                     (102, 102, 156),  # wall
                     (190, 153, 153),  # fence
                     (153, 153, 153),  # pole
                     (250, 170, 30),  # traffic light
                     (220, 220, 0),  # traffic sign
                     (107, 142, 35),  # vegetation
                     (152, 251, 152),  # terrain
                     (70, 130, 180),  # sky
                     (220, 20, 60),  # person
                     (255, 0, 0),  # rider
                     (0, 0, 142),  # car
                     (0, 0, 70),  # truck
                     (0, 60, 100),  # bus
                     (0, 80, 100),  # train
                    (0, 0, 230),  # motorcycle
                    (119, 11, 32)]   # bicycle

    h, w = img.shape[1], img.shape[2]  # img (C, H, W)
    n_labels = len(label_colors_as_list)

    # 4-D tensor which has the RGB colors of all the classes, each of them having a tensor 3-D tensor filled with their RGB colors
    label_colors_img = torch.zeros((n_labels, 3, h, w))  # (19, 3, h, w)
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


def compute_val_bpd(args, params):
    loader_params = {'batch_size': params['batch_size'], 'shuffle': False, 'num_workers': 0}
    _, val_loader = data_handler.init_city_loader(data_folder=params['data_folder'],
                                                  image_size=(params['img_size']),
                                                  remove_alpha=True,  # removing the alpha channel
                                                  loader_params=loader_params)

    model = models.init_and_load(args, params, run_mode='infer')

    mean, std = calc_val_loss(args, params, device, model, val_loader)
    print(f'In [compute_val_bpd]: mean = {mean} - std = {std}')