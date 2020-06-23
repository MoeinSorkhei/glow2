from .c_glow import *


def init_c_glow():
    config = {
        'x_size': (3, 64, 64),
        'y_size': (3, 64, 64),
        'x_hidden_channels': 128,
        'x_hidden_size': 64,
        'y_hidden_channels': 256,
        'flow_depth': 8,
        'num_levels': 3,
        'learn_top': False,
        'y_bins': 2.0  # ?
    }
    # x_size = (3,64,64),
    # y_size = (3,64,64),
    # x_hidden_channels = 128,
    # x_hidden_size = 64,
    # y_hidden_channels = 256,
    # K = 8,
    # L = 3,
    # learn_top = False
    # y_bins = 2.0  # ?

    c_glow = CondGlowModel(config)
    print(f'In [init_c_glow]: CGlow initialized')

