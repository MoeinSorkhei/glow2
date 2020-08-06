import tensorflow as tf
import time
import os
import numpy as np

from . import model_definition, data_handler

import helper


def count_trainable_params(verbose=False):
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        if verbose:
            print(f'variable name: {variable.name} - shape: {shape} - shape prod: {np.prod(shape)}')
        # shape is an array of tf.Dimension
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def count_trainable_params2():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


def init_hps_for_dual_glow(args, params):
    class hps(object):  # a helper class just for carrying attributes among functions
        pass

    # running params
    hps.inference = False
    if args.resume_train or args.exp:
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
    hps.n_levels, hps.depth = params['n_block'], params['n_flow']

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
    hps.sample_freq = 500 if args.sample_freq is None else args.sample_freq
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


def init_dual_glow_model(args, params, hps, tracker):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sess = create_tensorflow_session()  # init session with GPU

    seed = int(time.time())
    tf.set_random_seed(seed)
    np.random.seed(seed)
    print(f'In [init_dual_glow_and_train]: Random seed set to: {seed}')

    train_iterator, test_iterator, data_init, conditions = data_handler.retrieve_data(sess, hps, args, params)
    print('In [init_dual_glow_and_train]: Iterators init: done')

    # Create model
    dual_glow = model_definition.init_model(sess, hps, train_iterator, test_iterator, data_init)
    print('In [init_dual_glow_and_train]: Model initialized')

    trainable_params = count_trainable_params()
    print(f'Total trainable params: {trainable_params:,}')
    return dual_glow, sess, train_iterator, test_iterator, conditions
    # train(args, params, hps, sess, dual_glow, conditions, tracker)
