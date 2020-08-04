from copy import deepcopy

from ..glow import *
from ..utility import *


class TwoGlows(nn.Module):
    def __init__(self, params, mode, pretrained_left_glow=None):
        super().__init__()
        self.cond_shapes = calc_cond_shapes(params, mode=mode)  # shape (C, H, W)
        self.left_glow = init_glow(params) if pretrained_left_glow is None else pretrained_left_glow
        self.right_glow = init_glow(params, self.cond_shapes, all_layers_conditional=True)
        print(f'In [TwoGlows] __init__: Two Glows initialized')

    def forward(self, x_a, x_b, b_map=None):  # x_a: segmentation
        #  perform left glow forward
        left_glow_out = self.left_glow(x_a)
        # extract left outputs
        log_p_sum_left, log_det_left = left_glow_out['log_p_sum'], left_glow_out['log_det']
        z_outs_left, flows_outs_left = left_glow_out['z_outs'], left_glow_out['all_flows_outs']

        # perform right glow forward
        conditions = prep_conds(left_glow_out, direction='forward')
        right_glow_out = self.right_glow(x_b, conditions)

        # extract right outputs
        log_p_sum_right, log_det_right = right_glow_out['log_p_sum'], right_glow_out['log_det']
        z_outs_right, flows_outs_right = right_glow_out['z_outs'], right_glow_out['all_flows_outs']

        # gather left outputs together
        left_glow_outs = {'log_p': log_p_sum_left, 'log_det': log_det_left,
                          'z_outs': z_outs_left, 'flows_outs': flows_outs_left}

        #  gather right outputs together
        right_glow_outs = {'log_p': log_p_sum_right, 'log_det': log_det_right,
                           'z_outs': z_outs_right, 'flows_outs': flows_outs_right}

        return left_glow_outs, right_glow_outs

    def reverse(self, x_a=None, z_b_samples=None):
        left_glow_out = self.left_glow(x_a)
        conditions = prep_conds(left_glow_out, direction='reverse')
        x_b_syn = self.right_glow.reverse(z_b_samples, conditions=conditions)  # sample x_b conditioned on x_a
        return x_b_syn

    def new_condition(self, x_a, z_b_samples):
        left_glow_out = self.left_glow(x_a)
        conditions = prep_conds(left_glow_out, direction='reverse')
        x_b_rec = self.right_glow.reverse(z_b_samples, reconstruct=True, conditions=conditions)
        return x_b_rec

    def reconstruct_all(self, x_a, x_b):
        left_glow_out = self.left_glow(x_a)
        z_outs_left = left_glow_out['z_outs']
        print('left forward done')

        conditions = prep_conds(left_glow_out, direction='forward')  # preparing for right glow forward
        right_glow_out = self.right_glow(x_b, conditions)
        z_outs_right = right_glow_out['z_outs']
        print('right forward done')

        # reverse operations
        x_a_rec = self.left_glow.reverse(z_outs_left, reconstruct=True)
        print('left reverse done')

        conditions = prep_conds(left_glow_out, direction='reverse')  # prepare for right glow reverse
        x_b_rec = self.right_glow.reverse(z_outs_right, reconstruct=True, conditions=conditions)
        print('right reverse done')
        return x_a_rec, x_b_rec