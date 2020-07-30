import os
import tensorflow as tf
import numpy as np
from PIL import Image

import helper
from . import data_io
import globals


def retrieve_data(sess, hps, args, params):
    train_tfrecords_file = params['tfrecords_file']['train']
    val_tfrecords_file = params['tfrecords_file']['val']

    train_iter = data_io.read_tfrecords(train_tfrecords_file, args.dataset, args.direction, hps.batch_size, is_training=True)
    valid_iter = data_io.read_tfrecords(val_tfrecords_file, args.dataset, args.direction, hps.batch_size, is_training=False)

    first_batch = make_first_batch(sess, train_iter)

    dataset_name, direction = args.dataset, args.direction
    assert dataset_name == 'cityscapes'  # get from helper

    run_mode = 'infer' if args.exp else 'train'
    if run_mode == 'infer':
        if args.direction == 'label2photo':
            conditions_paths = helper.get_all_data_folder_images(path=params['data_folder']['segment'],
                                                                 partition='val',
                                                                 image_type='segment')
        else:
            conditions_paths = helper.get_all_data_folder_images(path=params['data_folder']['real'],
                                                                 partition='val',
                                                                 image_type='real')
        cond_list = create_conditions(conditions_paths, also_visual_grid=False)
        print(f'In [retrieve_data]: conditions_paths for inference is of len: {len(conditions_paths)}')

    else:
        conditions_paths = globals.seg_conds_abs_paths if direction == 'label2photo' else globals.real_conds_abs_path
        cond_list, visual_grid = create_conditions(conditions_paths)

        # save condition if not exists
        samples_path = helper.compute_paths(args, params)['samples_path']
        helper.make_dir_if_not_exists(samples_path)
        path = os.path.join(samples_path, 'conditions.png')

        if not os.path.isfile(path):
            Image.fromarray(visual_grid).save(path)
            print(f'In [retrieve_data]: saved conditions to: "{path}"')

    # make a list of dicts containing both image path and image array
    conditions = [{'image_path': conditions_paths[i], 'image_array': cond_list[i]} for i in range(len(cond_list))]
    return train_iter, valid_iter, first_batch, conditions


def create_conditions(conditions_paths, also_visual_grid=True):
    conds_list = []

    for cond_path in conditions_paths:
        cond_array = helper.open_and_resize_image(cond_path, for_model='dual_glow')
        conds_list.append(cond_array)

    if also_visual_grid:
        visual_grid = create_condition_grid(conditions_paths)
        return conds_list, visual_grid
    return conds_list


def create_condition_grid(conditions_paths):
    visual_cond_list = []
    for cond_path in conditions_paths:
        visual_img_array = np.array(Image.open(cond_path).resize((256, 256)))[:, :, :3]  # with int values
        visual_img_array = np.pad(visual_img_array, pad_width=((5, 5), (2, 2), (0, 0)), mode='constant', constant_values=0)
        visual_cond_list.append(np.expand_dims(visual_img_array, axis=0))  # add batch size

    visual_grid = np.concatenate(np.asarray(visual_cond_list), axis=0)  # (5, H, W, C)
    visual_grid = data_io.make_grid(visual_grid)
    return visual_grid


def make_first_batch(sess, iterator):
    data = iterator.get_next()
    left_img, right_img = sess.run(data)
    return left_img, right_img
