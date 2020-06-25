from .c_glow import *
import helper


def init_c_glow(args, params):
    h, w = params['img_size']

    config = {
        'x_size': (3, h, w),  # condition
        'y_size': (3, h, w),

        'x_hidden_size': 16,  # reduced to fit into GPU
        'x_hidden_channels': 32,  # reduced to fit into GPU
        'y_hidden_channels': 32,  # reduced to fit into GPU

        'flow_depth': params['n_flow'],
        'num_levels': params['n_block'],

        'learn_top': False,  # default by the paper
        'y_bins': 2 ** params['n_bits']
    }

    c_glow = CondGlowModel(config)
    print(f'In [init_c_glow]: CGlow initialized with config: \n{config}')
    # helper.print_info(args, params, c_glow, which_info='model')
    return c_glow

