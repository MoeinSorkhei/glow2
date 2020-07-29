import os
import tensorflow as tf
import numpy as np

from . import dual_glow
import helper
import evaluation

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_dual_glow(args, params, tracker=None):
    hps = dual_glow.init_hps_for_dual_glow(args, params)
    dual_glow_model, sess, _, _, conditions = dual_glow.init_dual_glow_model(args, params, hps, tracker)
    dual_glow.run_model('train', args, params, hps, sess, dual_glow_model, conditions, tracker)


def infer_dual_glow(args, params):
    # init model
    hps = dual_glow.init_hps_for_dual_glow(args, params)
    dual_glow_model, sess, _, _, conditions = dual_glow.init_dual_glow_model(args, params, hps, tracker=None)
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
    hps.n_levels = 1
    hps.depth = [1]

    if write_tf_records:
        tf.enable_eager_execution()  # need to enable in the beginning
        # if write_tf_records:
        gt_path = '../data/cityscapes/gtFine_trainvaltest/gtFine/train'
        dual_glow.data_io.write_data_for_tf(tfrecords_file, gt_path)
        print('writing tfrecords done')

    train_iter = dual_glow.data_io.read_tfrecords(tfrecords_file, args.dataset, args.direction, hps.batch_size, is_training=True)
    valid_iter = dual_glow.data_io.read_tfrecords(tfrecords_file, args.dataset, args.direction, hps.batch_size, is_training=False)

    config = tf.ConfigProto()
    sess = tf.Session(config=config)

    # data = train_iter.get_next()
    # first_batch = sess.run(data)
    first_batch = np.zeros(shape=(1, 1, 256, 256, 3)), np.zeros(shape=(1, 1, 256, 256, 3))
    dual_glow_model = dual_glow.model.init_model(sess, hps, train_iter, valid_iter, first_batch)
    # sess.graph.finalize()
    print('done')
    trainable_params = dual_glow.count_trainable_params()
    print(f'Total trainable params: {trainable_params:,}')
