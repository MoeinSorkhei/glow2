import numpy as np
import os
import tensorflow as tf

from . import dual_glow
import helper


def train_dual_glow(args, params, tracker=None):
    hps = dual_glow.init_hps_for_dual_glow(args, params)
    dual_glow_model, sess, _, _, conditions = dual_glow.init_dual_glow_model(args, params, hps, tracker)
    dual_glow.run_model('train', args, params, hps, sess, dual_glow_model, conditions, tracker)


def infer_dual_glow(args, params):
    # init model
    hps = dual_glow.init_hps_for_dual_glow(args, params)
    dual_glow_model, sess, _, _, conditions = dual_glow.init_dual_glow_model(args, params, hps, tracker=None)

    if args.all_sampling_rounds:
        for sampling_round in [1, 2, 3]:
            print(f'In [infer_dual_glow]: doing for sampling round: {sampling_round}')
            args.sampling_round = sampling_round
            dual_glow.run_model('infer', args, params, hps, sess, dual_glow_model, conditions, tracker=None)
            print(f'In [infer_dual_glow]: done for sampling round: {sampling_round}\n\n')
    else:
        dual_glow.run_model('infer', args, params, hps, sess, dual_glow_model, conditions, tracker=None)


def create_tf_records(args, params):
    if args.dataset == 'cityscapes':
        # for train data
        tfrecords_file = params['tfrecords_file']['train']
        gt_fine_path = os.path.join(params['data_folder']['segment'], 'train')
        helper.make_dir_if_not_exists(os.path.split(tfrecords_file)[0])

        dual_glow.write_data_for_tf(tfrecords_file, gt_fine_path)
        print(f'In [create_tf_records]: creating tf_records for train data: done. Saved to: "{tfrecords_file}"')

        # for val data
        tfrecords_file = params['tfrecords_file']['val']
        gt_fine_path = os.path.join(params['data_folder']['segment'], 'val')

        dual_glow.write_data_for_tf(tfrecords_file, gt_fine_path)
        print(f'In [create_tf_records]: creating tf_records for val data: done. Saved to: "{tfrecords_file}"')

    else:
        raise NotImplementedError


def investigate_model(args, hps, write_tf_records=False):
    tfrecords_file = '../tmp/data.tfrecords'
    if write_tf_records:
        tf.enable_eager_execution()  # need to enable in the beginning
        gt_path = '../data/cityscapes/gtFine_trainvaltest/gtFine/train'
        dual_glow.data_io.write_data_for_tf(tfrecords_file, gt_path)
        print('writing tfrecords done')

    train_iter = dual_glow.data_io.read_tfrecords(tfrecords_file, args.dataset, args.direction, hps.batch_size, is_training=True)
    valid_iter = dual_glow.data_io.read_tfrecords(tfrecords_file, args.dataset, args.direction, hps.batch_size, is_training=False)

    config = tf.ConfigProto()
    sess = tf.Session(config=config)

    first_batch = np.zeros(shape=(1, 1, 256, 256, 3)), np.zeros(shape=(1, 1, 256, 256, 3))
    dual_glow_model = dual_glow.model_definition.init_model(sess, hps, train_iter, valid_iter, first_batch)
    trainable_params = dual_glow.count_trainable_params(verbose=True)
    print(f'Total trainable params: {trainable_params:,}')
