import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image

from . import model_definition
import helper
from . import data_io


def take_samples_into_array(model, conditions, y_list, eps_list):
    samples_list = []
    for i in range(len(conditions)):
        sample = model.sample(conditions[i], y_list[i], eps_list[i])  # (1, D, H, W, C), D = 1
        sample = np.squeeze(sample, axis=1)  # (1, 1, H, W, C)  -> (1, H, W, C) - removed depth
        samples_list.append(sample)

    samples_array = np.concatenate(np.asarray(samples_list), axis=0)  # (5, H, W, C)
    return samples_array, samples_list


def take_sample(model, conditions, save_dir, mode, direction=None, iteration=None):
    y_list = [[None]] * len(conditions)  # conditions elements are of shape (1, H, W, C)
    eps_list = [[0.7]] * len(conditions)

    # samples_array (n_samples, H, H, C) - samples_list of len n_samples and items of shape (1, H, W, C)
    cond_arrays = [cond['image_array'] for cond in conditions]
    samples_array, samples_list = take_samples_into_array(model, cond_arrays, y_list, eps_list)

    if mode == 'infer':
        for i, cond in enumerate(conditions):
            suffix_direction = 'segment_to_real' if direction == 'label2photo' else 'real_to_segment'
            image_name = helper.replace_suffix(helper.pure_name(cond['image_path']), suffix_direction)
            path = os.path.join(save_dir, image_name)
            helper.rescale_and_save_image(np.squeeze(samples_list[i], axis=0), path)

            if i % 50 == 0:
                print(f'In [take_sample]: done for the {i}th image.')

    else:
        grid = data_io.make_grid(samples_array)
        path = os.path.join(save_dir, f'step={iteration}.png')
        helper.rescale_and_save_image(grid, path)
        print(f'In [take_sample]: sample saved to: "{path}"')


def compute_conditional_bpd(model, iteration, hps):
    test_results = []
    print('Computing validation loss...')

    for _ in range(hps.val_its):  # one loop over all all validation examples
        test_results += [model.test()]

    test_results = np.mean(np.asarray(test_results), axis=0)  # get the mean of val loss
    val_loss = round(test_results[0], 3)
    print(f'Step {iteration} - validation loss: {val_loss}')

    print('local_loss, bits_x_u, bits_x_o, bits_y:', test_results, 'conditional BPD:', round(test_results[1], 3))
    print('waiting for input')
    input()


def run_model(mode, args, params, hps, sess, model, conditions, tracker):
    sess.graph.finalize()

    # inference on validation set
    if mode == 'infer':
        # val_path = helper.compute_paths(args, params)['val_path']
        paths = helper.compute_paths(args, params)
        val_path = helper.extend_path(paths['val_path'], args.sampling_round)
        helper.make_dir_if_not_exists(val_path)

        take_sample(model, conditions, val_path, mode, direction=args.direction, iteration=None)
        print(f'In [run_model]: validation samples saved to: "{val_path}"')

    # training
    else:
        paths = helper.compute_paths(args, params)
        checkpoint_path = paths['checkpoints_path']
        samples_path = paths['samples_path']
        helper.make_dir_if_not_exists(checkpoint_path)
        helper.make_dir_if_not_exists(samples_path)

        # compute_conditional_bpd(model, args.last_optim_step, hps)

        iteration = 0 if not args.resume_train else args.last_optim_step + 1
        while iteration <= hps.train_its:
            lr = hps.lr
            train_results = model.train(lr)  # returns [local_loss, bits_x_u, bits_x_o, bits_y]
            train_loss = round(train_results[0], 3)
            print(f'Step {iteration} - train loss: {train_loss}')

            # take sample
            if iteration % hps.sample_freq == 0:
                take_sample(model, conditions, samples_path, mode, iteration)

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

