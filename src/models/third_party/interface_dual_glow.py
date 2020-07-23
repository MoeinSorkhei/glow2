import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from . import dual_glow
import helper


def init_hps_for_dual_glow(args, params):
    class hps(object):  # a helper class just for carrying attributes among functions
        pass

    # running params
    hps.inference = False
    if args.resume_train:
        hps.restore_path = os.path.join(helper.compute_paths(args, params)['checkpoints_path'],
                                        f"step={args.last_optim_step}.ckpt")
    else:
        hps.restore_path = None  # would be checkpoints path + step when resume training or inference

    # batch and image size
    hps.batch_size = 1
    hps.input_size = [256, 256, 3]
    hps.output_size = [256, 256, 3]
    hps.n_bits_x = 8

    # model config
    hps.n_levels = 4
    hps.depth = [16, 16, 16, 16]  # similar to our models
    # hps.depth = [8, 8, 8, 8]  # similar to our models
    # hps.depth = [1, 4, 8, 2]  # reduced to match model complexity
    hps.n_l = 1  # mlp basic layers, default: 1 by the paper
    hps.flow_permutation = 2  # 0: reverse (RealNVP), 1: shuffle, 2: invconv (Glow)"
    hps.flow_coupling = 1  # 0: additive, 1: affine

    hps.width = 512  # Width of hidden layers - default by the paper
    hps.eps_std = .7

    # other model configs
    hps.ycond = False  # Use y conditioning - default by the paper
    hps.learntop = True  # Learn spatial prior
    hps.n_y = 1  # always 1 in the original code
    hps.ycond_loss_type = 'l2'  # loss type of y inferred from z_in - default by the paper - not used by us as we do not have y conditioning

    # training config
    hps.train_its = 2000000 if args.max_step is None else args.max_step  # training iterations
    hps.val_its = 500  # 500 val iterations so we get full validation result with batch size 1 (val set size is 500)
    hps.val_freq = 1000  # get val result every 1000 iterations
    hps.sample_freq = 500
    hps.direct_iterator = True  # default by the paper
    hps.weight_lambda = 0.001  # Weight of log p(x_o|x_u) in weighted loss, default by the paper
    hps.weight_y = 0.01  # Weight of log p(y|x) in weighted loss, default by the paper - not used by us as we do not have y conditioning

    # adam params
    hps.optimizer = 'adam'
    hps.gradient_checkpointing = 1  # default
    hps.beta1 = .9
    hps.beta2 = .999
    hps.lr = 0.0001
    hps.weight_decay = 1.  # Switched off by default
    hps.polyak_epochs = 1   # default by the code - not used by us
    return hps


def init_and_train_dual_glow(args, params, tracker):
    hps = init_hps_for_dual_glow(args, params)
    dual_glow.init_model_and_train(args, params, hps, tracker)


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
