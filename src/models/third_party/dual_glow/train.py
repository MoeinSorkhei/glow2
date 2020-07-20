import json
import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image

from . import model
import helper
from . import data_io


def count_trainable_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def count_trainable_params2():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


def take_sample(model, conditions, sample_path, iteration):
    y_list = [[None]] * len(conditions)  # conditions elements are of shape (1, H, W, C)
    eps_list = [[0.7]] * len(conditions)

    samples_list = []
    for i in range(len(conditions)):
        sample = model.sample(conditions[i], y_list[i], eps_list[i])  # (1, D, H, W, C), D = 1
        # sample = np.squeeze(np.squeeze(sample, axis=1), axis=0)  # (1, 1, H, W, C)  -> (H, W, C)
        sample = np.squeeze(sample, axis=1)  # (1, 1, H, W, C)  -> (1, H, W, C) - removed depth
        samples_list.append(sample)

    samples_array = np.concatenate(np.asarray(samples_list), axis=0)  # (5, H, W, C)
    # helper.print_and_wait(f'samples array type and shape: {type(samples_array)} - {samples_array.shape}')
    grid = data_io.make_grid(samples_array)
    # helper.print_and_wait(f'grid array type and shape: {type(grid)} - {grid.shape}')

    path = os.path.join(sample_path, f'step={iteration}.png')
    tf.keras.preprocessing.image.save_img(path, grid, data_format='channels_last', scale=True)
    # Image.fromarray(grid).save(path)
    print(f'In [take_sample]: sample saved to: "{path}"')


def create_tensorflow_session():
    """
    Create tensorflow session. This function was modified from the original repo to only use one GPU.
    :return:
    """
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = '0'  # only one GPU
    sess = tf.Session(config=config)
    return sess


def train(args, params, hps, sess, model, conditions, tracker):
    paths = helper.compute_paths(args, params)
    checkpoint_path = paths['checkpoints_path']
    samples_path = paths['samples_path']
    helper.make_dir_if_not_exists(checkpoint_path)
    helper.make_dir_if_not_exists(samples_path)

    sess.graph.finalize()

    iteration = 0 if not args.resume_train else args.last_optim_step + 1
    while iteration <= hps.train_its:
        lr = hps.lr
        train_results = model.train(lr)  # returns [local_loss, bits_x_u, bits_x_o, bits_y]
        train_loss = round(train_results[0], 3)
        print(f'Step {iteration} - train loss: {train_loss}')

        # take sample
        if iteration % hps.sample_freq == 0:
            take_sample(model, conditions, samples_path, iteration)

        # track train loss
        if tracker is not None:
            tracker.track_metric('train_loss', round(train_loss, 3), iteration)

        # compute val loss
        if iteration % hps.val_freq == 0:
            test_results = []
            print('Computing validation loss...')

            for _ in range(hps.val_its):  # one loop over all all validation examples
                test_results += [model.test()]

            test_results = np.mean(np.asarray(test_results), axis=0)  # get the mean of val loss
            val_loss = round(test_results[0], 3)
            print(f'Step {iteration} - validation loss: {val_loss}')

            # save checkpoint
            path = os.path.join(checkpoint_path, f"step={iteration}.ckpt")
            model.save(path)
            print(f'Checkpoint saved to: "{path}"')

            # track val loss
            if tracker is not None:
                tracker.track_metric('val_loss', round(val_loss, 3), iteration)

        iteration += 1


def init_dual_glow_and_train(args, params, hps, tracker):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sess = create_tensorflow_session()  # init session with GPU

    seed = int(time.time())
    tf.set_random_seed(seed)
    np.random.seed(seed)
    print(f'In [init_dual_glow_and_train]: Random seed set to: {seed}')

    train_iterator, test_iterator, data_init, conditions = data_io.retrieve_data(sess, hps, args, params)
    print('In [init_dual_glow_and_train]: Iterators init: done')

    # Create model
    dual_glow = model.init_model(sess, hps, train_iterator, test_iterator, data_init)
    print('In [init_dual_glow_and_train]: Model initialized')

    trainable_params = count_trainable_params()
    print(f'Total trainable params: {trainable_params:,}')

    train(args, params, hps, sess, dual_glow, conditions, tracker)
