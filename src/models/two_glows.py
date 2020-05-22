from torch import nn
import torch

from . import init_glow
from helper import calc_cond_shapes


class TwoGlows(nn.Module):
    def __init__(self, params, mode, pretrained_left_glow=None, w_conditionals=None, act_conditionals=None,
                 use_coupling_cond_nets=None):
        super().__init__()
        self.mode = mode
        self.w_conditionals = w_conditionals
        self.act_conditionals = act_conditionals
        self.cond_shapes = calc_cond_shapes(orig_shape=None,
                                            in_channels=params['channels'],
                                            img_size=params['img_size'],
                                            n_block=params['n_block'],
                                            mode=mode)  # shape (C, H, W)
        self.left_glow = init_glow(params) if pretrained_left_glow is None else pretrained_left_glow
        self.right_glow = init_glow(params, self.cond_shapes, w_conditionals, act_conditionals, use_coupling_cond_nets)

    def prep_coupling_conds(self, flows_outs_left, b_map):
        if self.mode == 'segment':
            cond = {'name': 'segment', 'segment': flows_outs_left}

        else:  # mode = 'segment_boundary'
            cond = {'name': 'segment_boundary', 'segment': flows_outs_left, 'boundary': b_map}
        return cond

    def left_glow_forward(self, x_a, b_map=None):
        """
        Performs forward only on the left glow and prepares the required conditions for the right glow.
        :param x_a:
        :param b_map:
        :return:
        """
        left_glow_out = self.left_glow(x_a,
                                       coupling_conds=None,
                                       return_flows_outs=True,
                                       return_w_outs=self.w_conditionals,
                                       return_act_outs=self.act_conditionals)

        log_p_sum_left, log_det_left = left_glow_out['log_p_sum'], left_glow_out['log_det']
        z_outs_left, flows_outs_left = left_glow_out['z_outs'], left_glow_out['all_flows_outs']

        left_glow_w_outs = left_glow_out['all_w_outs'] if self.w_conditionals is not None else None
        left_glow_act_outs = left_glow_out['all_act_outs'] if self.act_conditionals is not None else None
        coupling_conds = self.prep_coupling_conds(flows_outs_left, b_map)  # b_map could be None based on self.mode

        return z_outs_left, left_glow_w_outs, left_glow_act_outs, coupling_conds

    def forward(self, x_a, x_b, b_map=None):  # x_a: segmentation
        left_glow_out = self.left_glow(inp=x_a,
                                       coupling_conds=None,
                                       return_flows_outs=True,
                                       return_w_outs=self.w_conditionals,
                                       return_act_outs=self.act_conditionals)

        log_p_sum_left, log_det_left = left_glow_out['log_p_sum'], left_glow_out['log_det']
        z_outs_left, flows_outs_left = left_glow_out['z_outs'], left_glow_out['all_flows_outs']

        left_glow_w_outs = left_glow_out['all_w_outs'] if self.w_conditionals is not None else None
        left_glow_act_outs = left_glow_out['all_act_outs'] if self.act_conditionals is not None else None

        coupling_conds = self.prep_coupling_conds(flows_outs_left, b_map)  # b_map could be None based on self.mode

        right_glow_out = self.right_glow(x_b,
                                         coupling_conds=coupling_conds,
                                         left_glow_w_outs=left_glow_w_outs,
                                         left_glow_act_outs=left_glow_act_outs,
                                         return_flows_outs=True)  # return_flows_outs here unneeded - should be removed

        log_p_sum_right, log_det_right = right_glow_out['log_p_sum'], right_glow_out['log_det']
        z_outs_right, flows_outs_right = right_glow_out['z_outs'], right_glow_out['all_flows_outs']

        left_glow_outs = {'log_p': log_p_sum_left, 'log_det': log_det_left,
                          'z_outs': z_outs_left, 'flows_outs': flows_outs_left}

        right_glow_outs = {'log_p': log_p_sum_right, 'log_det': log_det_right,
                           'z_outs': z_outs_right, 'flows_outs': flows_outs_right}

        return left_glow_outs, right_glow_outs

    def reverse(self, x_a=None, x_b=None, b_map=None, z_a_samples=None, z_b_samples=None, mode='reconstruct_all'):
        """
        Performs the reverse operation for the model.
        :param z_b_samples:
        :param z_a_samples:
        :param b_map:
        :param x_a: -
        :param x_b: -
        :param: z_b_samples: a 2D list of z samples for x_b. See the reverse function of Glow for more info.

        :param mode: determines the behavior of the function. If set to 'reconstruct_all', both x_a and x_b should be
        give, and it reconstructs both images and performs a sanity check on the final values of the tensors. If set to
        'sample_x_b', it samples x_b conditioned on the flows_outs of x_a.

        Note regarding different modes:
            - 'new_condition': applies the given z_b_samples to the new condition x_a
            - 'reconstruct_all':
            - ...

        :return: the reconstructed/synthesized image(s). See the function for more details.
        ----------------------------
        NOTES:
            - flows_outs_left would be a 2D list of length n_block, whole elements are 1Ds list of len n_flow,
              whose elements are tensors of shape (B, C, H, W)
            - In most of the modes, the function itself performs the required forward pass to obtain the corresponding
              z's and conditions needed for the reverse operation (see the code for more details).
        """
        if mode == 'new_condition':  # x_a (possibly with b_map) and z_b_samples (for the desired style) are required
            # ========== performing the required forward
            _, left_glow_w_outs, left_glow_act_outs, coupling_conds = self.left_glow_forward(x_a=x_a,
                                                                                             b_map=b_map)
            x_b_rec = self.right_glow.reverse(z_b_samples,
                                              reconstruct=True,
                                              coupling_conds=coupling_conds,
                                              left_glow_w_outs=left_glow_w_outs,
                                              left_glow_act_outs=left_glow_act_outs)
            return x_b_rec

        elif mode == 'reconstruct_all':  # reconstructing: x_a (possibly with b_map) and x_b are required
            # ========== left glow and right glow forward
            z_outs_left, left_glow_w_outs, left_glow_act_outs, coupling_conds = self.left_glow_forward(x_a=x_a,
                                                                                                       b_map=b_map)

            right_glow_out = self.right_glow(inp=x_b,
                                             coupling_conds=coupling_conds,
                                             left_glow_w_outs=left_glow_w_outs,  # could be None based on --cond_mode
                                             left_glow_act_outs=left_glow_act_outs)
            z_outs_right = right_glow_out['z_outs']

            # ========== reverse operations
            x_a_rec = self.left_glow.reverse(z_outs_left, reconstruct=True)
            x_b_rec = self.right_glow.reverse(z_outs_right,
                                              reconstruct=True,
                                              coupling_conds=coupling_conds,
                                              left_glow_w_outs=left_glow_w_outs,
                                              left_glow_act_outs=left_glow_act_outs)
            return x_a_rec, x_b_rec

        if mode == 'sample_x_b':
            # CODE IMPROVEMENT: IT SHOULD DIRECTLY USED LEFT_GLOW_FORWARD HERE (LIKE OTHER MODES)
            left_glow_out = self.left_glow(x_a,
                                           coupling_conds=None,
                                           return_flows_outs=True,
                                           return_w_outs=self.w_conditionals,
                                           return_act_outs=self.act_conditionals)

            z_outs_left = left_glow_out['z_outs']
            flows_outs_left = left_glow_out['all_flows_outs']
            left_glow_w_outs = left_glow_out['all_w_outs'] if self.w_conditionals is not None else None
            left_glow_act_outs = left_glow_out['all_act_outs'] if self.act_conditionals is not None else None

            # preparing coupling conds
            coupling_conds = self.prep_coupling_conds(flows_outs_left, b_map)  # b_map could be None based on self.mode

            # sample x_b conditioned on x_a - z_b_samples: list of sampled z's
            x_b_syn = self.right_glow.reverse(z_b_samples,
                                              coupling_conds=coupling_conds,  # for coupling layer
                                              left_glow_w_outs=left_glow_w_outs,  # for w
                                              left_glow_act_outs=left_glow_act_outs)  # for actnorm
            return x_b_syn

        if mode == 'sample_x_a':
            # synthesize image with sampled z's
            x_a_syn = self.left_glow.reverse(z_a_samples, reconstruct=False, coupling_conds=None)
            return x_a_syn
