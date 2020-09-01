from torch import nn
from torch.autograd import gradcheck
import numpy as np

from .block import Block, PairedBlock
from .utils import *


class Glow(nn.Module):
    def __init__(self, n_blocks, n_flows, input_shapes, cond_shapes, configs):
        super().__init__()
        self.all_conditional = configs['all_conditional']
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList()

        # making None -> [None, None, ..., None]
        if cond_shapes is None:
            cond_shapes = [None] * n_blocks

        for i in range(n_blocks):
            inp_shape = input_shapes[i]
            cond_shape = cond_shapes[i]

            # last Block does not have split
            do_split = False if i == (n_blocks - 1) else True

            # create Block
            block = Block(n_flow=n_flows[i],
                          inp_shape=inp_shape,
                          cond_shape=cond_shape,
                          do_split=do_split,
                          configs=configs)
            self.blocks.append(block)

    def forward(self, inp, conditions=None):
        log_p_sum = 0
        log_det = 0
        out = inp
        z_outs = []

        all_flows_outs = []  # a 2d list, each element of which corresponds to the flows_outs of each Block
        all_w_outs = []  # 2d list
        all_act_outs = []  # 2d list

        for i, block in enumerate(self.blocks):
            act_cond, w_cond, coupling_cond = extract_conds(conditions, i, self.all_conditional)
            conds = make_cond_dict(act_cond, w_cond, coupling_cond)
            block_out = block(out, conds)

            # extracting output and preparing conditions
            out, det, log_p, z_new = block_out['out'], block_out['total_log_det'], block_out['log_p'], block_out['z_new']
            flows_out, w_outs, act_outs = block_out['flows_outs'], block_out['w_outs'], block_out['act_outs']
            all_flows_outs.append(flows_out)  # appending flows_outs - done by the left_glow
            all_w_outs.append(w_outs)  # appending w_outs - done by the left_glow
            all_act_outs.append(act_outs)  # appending act_outs - done by the left_glow
            z_outs.append(z_new)

            # objective
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
            act_cond, w_cond, coupling_cond = extract_conds(conditions, i, self.all_conditional)
            conds = make_cond_dict(act_cond, w_cond, coupling_cond)

            reverse_input = z_list[-1] if i == 0 else inp
            block_reverse = block.reverse(output=reverse_input,  # Block reverse operation
                                          eps=z_list[-(i + 1)],
                                          reconstruct=rec_list[-(i + 1)],
                                          conditions=conds)
            inp = block_reverse
        return inp


class PairedGLow(nn.Module):
    def __init__(self, n_blocks, n_flows, cond_shapes, input_shapes, configs):
        super().__init__()
        # self.all_conditional = configs['all_conditional']
        self.n_blocks = n_blocks
        self.n_flows = n_flows
        self.paired_blocks = nn.ModuleList()

        # making None -> [None, None, ..., None]
        if cond_shapes is None:
            cond_shapes = [None] * n_blocks

        for i in range(n_blocks):
            do_split = False if i == (n_blocks - 1) else True  # last Block does not have split
            # create Block
            self.paired_blocks.append(PairedBlock(n_flow=n_flows[i],
                                                  cond_shape=cond_shapes[i],
                                                  inp_shape=input_shapes[i],
                                                  do_split=do_split, configs=configs))

    def forward(self, back_prop_info, inp_left, inp_right):
        total_logdet_left = total_logdet_right = 0
        log_p_sum_left = log_p_sum_right = 0
        z_outs_left, z_outs_right = [], []
        out_left, out_right = inp_left, inp_right  # initialize output with the input
        # back_prop_info = {}  # shared by all layers for backprop

        for i_block, paired_block in enumerate(self.paired_blocks):
            # squeeze in the beginning of Block
            out_left, out_right = squeeze_tensor(out_left), squeeze_tensor(out_right)

            # apply Flows
            for i_flow, paired_flow in enumerate(paired_block.paired_flows):
                out_left, act_logdet_left, out_right, act_logdet_right = paired_flow.paired_actnorm(back_prop_info, out_left, out_right)
                out_left, w_logdet_left, out_right, w_logdet_right = paired_flow.paired_inv_conv(back_prop_info, out_left, out_right)
                out_left, coupling_logdet_left, out_right, coupling_logdet_right = paired_flow.paired_coupling(back_prop_info, out_left, out_right)
                total_logdet_left += (act_logdet_left + w_logdet_left + coupling_logdet_left)
                total_logdet_right += (act_logdet_right + w_logdet_right + coupling_logdet_right)
            
            # split after all Flows in a Block
            out_left, z_new_left, log_p_left = paired_block.possibly_split(out_left, 'left')
            out_right, z_new_right, log_p_right = paired_block.possibly_split(out_right, 'right')
            log_p_sum_left += log_p_left
            log_p_sum_right += log_p_right
            z_outs_left.append(z_new_left)
            z_outs_right.append(z_new_right)

        # prepare items for backprop
        back_prop_info.update({
            'left_activations': [out_left.data],
            'right_activations': [out_right.data],
            'z_outs_left': [z_outs_left[i].data for i in range(len(z_outs_left) - 1)],
            'z_outs_right': [z_outs_right[i].data for i in range(len(z_outs_right) - 1)],
            'marginal_flows_inds': [cumulative_flows - 1 for cumulative_flows in np.cumsum(self.n_flows)[:-1]],
            'current_i_flow': sum(self.n_flows) - 1
        })
        return z_outs_left, z_outs_right, log_p_sum_left, log_p_sum_right, total_logdet_left, total_logdet_right


def init_glow(n_blocks, n_flows, input_shapes, cond_shapes, configs):
    return Glow(
        n_blocks=n_blocks,
        n_flows=n_flows,
        input_shapes=input_shapes,
        cond_shapes=cond_shapes,
        configs=configs
    )
