from .c_glow import *


def init_c_glow(args, params):
    h, w = params['img_size']

    # basic config
    config = {
        'x_size': (3, h, w),  # condition
        'y_size': (3, h, w),

        'learn_top': False,  # default by the paper
        'y_bins': 2 ** params['n_bits']
    }

    if args.model == 'c_glow':  # try the first config - **** NO LONGER USED SINCE THIS IS THE WEAKEST CONFIG ****
        config.update({
            'x_hidden_size': 16,  # reduced to fit into GPU
            'x_hidden_channels': 32,  # reduced to fit into GPU
            'y_hidden_channels': 32,  # reduced to fit into GPU

            'flow_depth': params['n_flow'],  # equal flows as in other models we tried
            'num_levels': params['n_block'],  # equal blocks as in other models we tried
        })

    elif args.model == 'c_glow_v2':  # the second config - kept the ratio as the default
        config.update({
            'x_hidden_size': 26,  # default was: 64
            'x_hidden_channels': 52,  # default was: 128
            'y_hidden_channels': 104,  # default was: 256

            'flow_depth': 8,  # default by the paper
            'num_levels': 3,  # default by the paper
        })

    elif args.model == 'c_glow_v3':  # the third config - similar to us, deeper glow but shallower cond_net
        config.update({
            'x_hidden_size': 10,  # default was: 64
            'x_hidden_channels': 20,  # default was: 128
            'y_hidden_channels': 512,  # same as us

            'flow_depth': params['n_flow'],   # same as us
            'num_levels': params['n_block'],  # same as us
        })

    else:
        raise NotImplementedError

    c_glow = CondGlowModel(config)
    print(f'In [init_c_glow]: CGlow initialized with config: \n{config}')
    return c_glow
