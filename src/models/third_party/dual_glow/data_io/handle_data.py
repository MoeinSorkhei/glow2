import os
import tensorflow as tf
import numpy as np
from PIL import Image

import helper
from . import tf_data
import globals


def retrieve_data(sess, hps, args, params):
    train_tfrecords_file = params['tfrecords_file']['train']
    val_tfrecords_file = params['tfrecords_file']['val']

    train_iter = tf_data.read_tfrecords(train_tfrecords_file, args.dataset, args.direction, hps.batch_size, is_training=True)
    valid_iter = tf_data.read_tfrecords(val_tfrecords_file, args.dataset, args.direction, hps.batch_size, is_training=False)

    first_batch = make_first_batch(sess, train_iter)
    cond_list, visual_cond_array = create_conditions(args.dataset, args.direction)

    # save condition if not exists
    samples_path = helper.compute_paths(args, params)['samples_path']
    helper.make_dir_if_not_exists(samples_path)
    path = os.path.join(samples_path, 'conditions.png')

    if not os.path.isfile(path):
        Image.fromarray(visual_cond_array).save(path)
        print(f'In [retrieve_data]: saved conditions to: "{path}"')

    return train_iter, valid_iter, first_batch, cond_list


def create_conditions(dataset_name, direction):
    conds_list = []
    visual_cond_list = []

    if dataset_name == 'cityscapes':
        conditions_paths = globals.seg_conds_abs_paths if direction == 'label2photo' else globals.real_conds_abs_path
    else:
        raise NotImplementedError

    for cond_path in conditions_paths:
        cond_array = np.array(Image.open(cond_path).resize((256, 256)))[:, :, :3] / 255
        cond_array = np.expand_dims(cond_array, axis=(0, 1)).astype(np.float32)  # (1, 1, H, W, C)
        conds_list.append(cond_array)

        visual_img_array = np.array(Image.open(cond_path).resize((256, 256)))[:, :, :3]  # with int values
        visual_img_array = np.pad(visual_img_array, pad_width=((5, 5), (2, 2), (0, 0)), mode='constant', constant_values=0)
        visual_cond_list.append(np.expand_dims(visual_img_array, axis=0))  # add batch size

    visual_cond_array = np.concatenate(np.asarray(visual_cond_list), axis=0)  # (5, H, W, C)
    visual_cond_array = tf_data.make_grid(visual_cond_array)
    return conds_list, visual_cond_array


def make_first_batch(sess, iterator):
    data = iterator.get_next()
    segment_img, real_img = sess.run(data)
    return segment_img, real_img
