from torch import nn

import helper
from ..interface import calc_z_shapes
from .block import Block, corresponding_coupling_cond, reverse_conditions


class Glow(nn.Module):
    def __init__(self, in_channel, n_flows, n_blocks, do_affine=True, conv_lu=True, coupling_cond_shapes=None,
                 w_conditionals=None, act_conditionals=None, use_coupling_cond_nets=None, input_shapes=None):
        super().__init__()

        self.w_conditionals = w_conditionals
        self.act_conditionals = act_conditionals
        self.use_coupling_cond_net = use_coupling_cond_nets

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        # making None -> [None, None, ..., None]
        coupling_cond_shapes = [None] * n_blocks if coupling_cond_shapes is None else coupling_cond_shapes

        for i in range(n_blocks):
            # for this Block: specifying if w and actnorm are conditional
            w_conditional = True if w_conditionals is not None and w_conditionals[i] else False
            act_conditional = True if act_conditionals is not None and act_conditionals[i] else False
            use_coupling_cond_net = True if use_coupling_cond_nets is not None and use_coupling_cond_nets[i] else False
            inp_shape = None if input_shapes is None else input_shapes[i]
            coupling_cond_shape = coupling_cond_shapes[i]

            # specifying the stride for the conditioning networks
            if w_conditionals or act_conditionals:
                stride = 3 if i == 0 else 3 if i == 1 else 2 if i == 2 else 1
            else:
                stride = None

            # last Block does not have split
            do_split = False if i == (n_blocks - 1) else True

            # create Block
            block = Block(in_channel=n_channel,
                          n_flow=n_flows,
                          do_split=do_split,
                          do_affine=do_affine,
                          conv_lu=conv_lu,
                          coupling_cond_shape=coupling_cond_shape,
                          w_conditionals=w_conditional,
                          act_conditionals=act_conditional,
                          inp_shape=inp_shape,
                          conv_stride=stride,
                          use_coupling_cond_net=use_coupling_cond_net)
            self.blocks.append(block)

            # increase the channels for all the Blocks before the last Block
            if i != (n_blocks - 1):
                n_channel = n_channel * 2

    def forward(self, inp, coupling_conds=None, left_glow_w_outs=None, left_glow_act_outs=None,
                return_flows_outs=False, return_w_outs=False, return_act_outs=False):
        """
        The forward functions take as input an image and performs all the operations to obtain z's: x->z
        :param return_act_outs:
        :param left_glow_act_outs:
        :param left_glow_w_outs:
        :param return_w_outs:
        :param coupling_conds:
        :param inp: the image to be encoded

        :param return_flows_outs
        :return: the extracted z's, log_det, and log_p_sum, the last two used to compute the log p(x) according to the
        change of variables theorem.
        """
        log_p_sum = 0
        log_det = 0
        out = inp
        z_outs = []

        all_flows_outs = []  # a 2d list, each element of which corresponds to the flows_outs of each Block
        all_w_outs = []  # 2d list
        all_act_outs = []  # 2d list

        for i, block in enumerate(self.blocks):
            coupling_conditions = corresponding_coupling_cond('block', i, coupling_conds)

            # specifying w_outs for the corresponding block
            left_block_w_outs, left_block_act_outs = None, None
            if self.w_conditionals is not None and self.w_conditionals[i]:  # used by the right glow
                left_block_w_outs = left_glow_w_outs[i]  # w_out for the corresponding Block indexed by i

            # specifying act_outs for the corresponding block
            if self.act_conditionals is not None and self.act_conditionals[i]:  # used by the right glow
                left_block_act_outs = left_glow_act_outs[i]

            block_out = block(out,
                              coupling_conds=coupling_conditions,
                              left_w_outs=left_block_w_outs,
                              left_act_outs=left_block_act_outs,
                              return_flows_outs=return_flows_outs,
                              return_w_outs=return_w_outs,
                              return_act_outs=return_act_outs)

            out, det, log_p = block_out['out'], block_out['total_log_det'], block_out['log_p']
            z_new = block_out['z_new']

            # appending flows_outs - done by the left_glow
            if return_flows_outs:
                flows_out = block_out['flows_outs']
                all_flows_outs.append(flows_out)

            # appending w_outs - done by the left_glow
            if return_w_outs:
                w_outs = block_out['w_outs']
                all_w_outs.append(w_outs)

            # appending act_outs - done by the left_glow
            if return_act_outs:
                act_outs = block_out['act_outs']
                all_act_outs.append(act_outs)

            z_outs.append(z_new)
            log_det = log_det + det
            log_p_sum = log_p_sum + log_p

        output_dict = {'log_p_sum': log_p_sum, 'log_det': log_det, 'z_outs': z_outs}
        if return_flows_outs:
            output_dict['all_flows_outs'] = all_flows_outs
        if return_w_outs:
            output_dict['all_w_outs'] = all_w_outs
        if return_act_outs:
            output_dict['all_act_outs'] = all_act_outs

        return output_dict

    def reverse(self, z_list, reconstruct=False, coupling_conds=None, left_glow_w_outs=None, left_glow_act_outs=None):
        """
        The reverse function performs the operations in the direction z->x.
        :param left_glow_act_outs:
        :param left_glow_w_outs:
        :param z_list: the list of z'a sampled from unit Gaussian (with temperature) for different Blocks. Each element
        in the list is a tensor of shape (B, C, H, W) and has a batch of samples for the corresponding Block.

        :param reconstruct: is set to True if one wants to reconstruct the image with its extracted latent variables.
        :param coupling_conds
        :return: the generated image.
        """
        # preparing for reconstruction
        inp = None
        rec_list = [reconstruct] * len(z_list) if reconstruct is True or reconstruct is False else reconstruct

        # reversing the conditions
        coupling_conds, left_glow_w_outs, left_glow_act_outs = reverse_conditions(coupling_conds,
                                                                                  left_glow_w_outs,
                                                                                  left_glow_act_outs,
                                                                                  self.w_conditionals,
                                                                                  self.act_conditionals)

        # Block reverse operations one by one
        for i, block in enumerate(self.blocks[::-1]):  # it starts from the last Block
            # preparing coupling conditions
            reversed_i = len(self.blocks) - 1 - i  # i = 0 ==> reversed_i = 3, i = 1 ==> reversed_i = 2, ...
            coupling_conditions = corresponding_coupling_cond('block',
                                                              level=i,
                                                              cond=coupling_conds,
                                                              in_reverse=True,
                                                              reverse_level=reversed_i)
            # preparing w and actnorm conditions
            left_block_w_outs, left_block_act_outs = None, None
            if self.w_conditionals is not None and self.w_conditionals[::-1][i]:
                left_block_w_outs = left_glow_w_outs[i]
            if self.act_conditionals is not None and self.act_conditionals[::-1][i]:
                left_block_act_outs = left_glow_act_outs[i]

            # Block reverse operation
            reverse_input = z_list[-1] if i == 0 else inp
            block_reverse = block.reverse(reverse_input,
                                          eps=z_list[-(i + 1)],
                                          reconstruct=rec_list[-(i + 1)],
                                          coupling_conds=coupling_conditions,
                                          left_block_w_outs=left_block_w_outs,
                                          left_block_act_outs=left_block_act_outs)
            inp = block_reverse
        return inp


def compute_inp_shapes(n_channels, input_size, n_blocks):
    z_shapes = calc_z_shapes(n_channels, input_size, n_blocks)
    input_shapes = []
    for i in range(len(z_shapes)):
        if i < len(z_shapes) - 1:
            input_shapes.append((z_shapes[i][0] * 2, z_shapes[i][1], z_shapes[i][2]))
        else:
            input_shapes.append((z_shapes[i][0], z_shapes[i][1], z_shapes[i][2]))
    return input_shapes


def init_glow(params, cond_shapes=None, w_conditionals=None, act_conditionals=None, use_coupling_cond_nets=None):
    if w_conditionals or act_conditionals or use_coupling_cond_nets:
        input_shapes = compute_inp_shapes(params['channels'], params['img_size'], params['n_block'])
        return Glow(
            params['channels'], params['n_flow'], params['n_block'], params['affine'], params['lu'],
            coupling_cond_shapes=cond_shapes,
            w_conditionals=w_conditionals,
            act_conditionals=act_conditionals,
            use_coupling_cond_nets=use_coupling_cond_nets,
            input_shapes=input_shapes
        )

    return Glow(
        params['channels'], params['n_flow'], params['n_block'], params['affine'], params['lu'], cond_shapes
    )
