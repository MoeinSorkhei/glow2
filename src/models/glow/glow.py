from torch import nn

from .block import Block
from .utils import *


class Glow(nn.Module):
    def __init__(self, in_channel, n_flows, n_blocks, coupling_cond_shapes=None, all_layers_conditional=False, input_shapes=None):
        super().__init__()
        self.all_layers_conditional = all_layers_conditional
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList()
        n_channel = in_channel

        # making None -> [None, None, ..., None]
        if coupling_cond_shapes is None:
            coupling_cond_shapes = [None] * n_blocks

        for i in range(n_blocks):
            inp_shape = None if input_shapes is None else input_shapes[i]
            coupling_cond_shape = coupling_cond_shapes[i]

            if all_layers_conditional:
                stride = 3 if i == 0 else 3 if i == 1 else 2 if i == 2 else 1
            else:
                stride = None

            # last Block does not have split
            do_split = False if i == (n_blocks - 1) else True

            # create Block
            block = Block(in_channel=n_channel,
                          n_flow=n_flows,
                          do_split=do_split,
                          coupling_cond_shape=coupling_cond_shape,
                          inp_shape=inp_shape,
                          conv_stride=stride,
                          all_layers_conditional=all_layers_conditional)
            self.blocks.append(block)

            # increase the channels for all the Blocks before the last Block
            if i != (n_blocks - 1):
                n_channel = n_channel * 2

    def forward(self, inp, conditions=None):
        log_p_sum = 0
        log_det = 0
        out = inp
        z_outs = []

        all_flows_outs = []  # a 2d list, each element of which corresponds to the flows_outs of each Block
        all_w_outs = []  # 2d list
        all_act_outs = []  # 2d list

        for i, block in enumerate(self.blocks):
            act_cond, w_cond, coupling_cond = extract_conds(conditions, i, self.all_layers_conditional)
            conds = make_cond_dict(act_cond, w_cond, coupling_cond)
            block_out = block(out, conds)

            out, det, log_p = block_out['out'], block_out['total_log_det'], block_out['log_p']
            z_new = block_out['z_new']

            # appending flows_outs - done by the left_glow
            flows_out = block_out['flows_outs']
            all_flows_outs.append(flows_out)

            # appending w_outs - done by the left_glow
            w_outs = block_out['w_outs']
            all_w_outs.append(w_outs)

            # appending act_outs - done by the left_glow
            act_outs = block_out['act_outs']
            all_act_outs.append(act_outs)

            z_outs.append(z_new)
            log_det = log_det + det
            log_p_sum = log_p_sum + log_p

        return {
            'all_act_outs': all_act_outs,
            'all_w_outs': all_w_outs,
            'all_flows_outs': all_flows_outs,
            'z_outs': z_outs,
            'log_p_sum': log_p_sum,
            'log_det': log_det
        }

    def reverse(self, z_list, reconstruct=False, conditions=None):
        inp = None
        rec_list = [reconstruct] * self.n_blocks  # make a list of True or False

        # Block reverse operations one by one
        for i, block in enumerate(self.blocks[::-1]):  # it starts from the last Block
            act_cond, w_cond, coupling_cond = extract_conds(conditions, i, self.all_layers_conditional)
            reverse_input = z_list[-1] if i == 0 else inp
            conds = make_cond_dict(act_cond, w_cond, coupling_cond)
            block_reverse = block.reverse(output=reverse_input,  # Block reverse operation
                                          eps=z_list[-(i + 1)],
                                          reconstruct=rec_list[-(i + 1)],
                                          conditions=conds)
            inp = block_reverse
        return inp


def init_glow(params, cond_shapes=None, all_layers_conditional=False):
    if all_layers_conditional:
        input_shapes = compute_inp_shapes(params['channels'], params['img_size'], params['n_block'])
        return Glow(
            in_channel=params['channels'],
            n_flows=params['n_flow'],
            n_blocks=params['n_block'],
            coupling_cond_shapes=cond_shapes,
            input_shapes=input_shapes,
            all_layers_conditional=all_layers_conditional
        )

    return Glow(
        in_channel=params['channels'],
        n_flows=params['n_flow'],
        n_blocks=params['n_block'],
        coupling_cond_shapes=cond_shapes
    )
