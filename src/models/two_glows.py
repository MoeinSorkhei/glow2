from torch import nn
import torch

from . import init_glow
from helper import calc_cond_shapes, show_images


class TwoGlows(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.cond_shapes = calc_cond_shapes(orig_shape=None,
                                            in_channels=params['channels'],
                                            img_size=params['img_size'],
                                            n_block=params['n_block'],
                                            mode='flows_outs')
        self.left_glow = init_glow(params)  # no condition
        self.right_glow = init_glow(params, self.cond_shapes)

    def forward(self, x_a, x_b):
        log_p_sum_left, log_det_left, \
            z_outs_left, flows_outs_left = self.left_glow(x_a, cond=None, return_flows_outs=True)  # x_a: segmentation

        # print('left forward done')

        cond = ['c_flow', flows_outs_left]
        log_p_sum_right, log_det_right, \
            z_outs_right, flows_outs_right = self.right_glow(x_b, cond=cond, return_flows_outs=True)

        # print('right forward done')

        total_log_det = log_det_left + log_det_right
        total_log_p = log_p_sum_left + log_p_sum_right

        left_glow_outs = {'log_p': log_p_sum_left, 'log_det': log_det_left,
                          'z_outs': z_outs_left, 'flows_outs': flows_outs_left}

        right_glow_outs = {'log_p': log_p_sum_right, 'log_det': log_det_right,
                           'z_outs': z_outs_right, 'flows_outs': flows_outs_right}

        return left_glow_outs, right_glow_outs
        # return total_log_det, total_log_p, left_glow_outs, right_glow_outs

    def reverse(self, x_a=None, x_b=None, z_a_samples=None, z_b_samples=None, mode='reconstruct_all'):
        """
        Later it could be extended to sample from both glows: first generate a new segmentation and then sample a real
        image conditioned on that to generate novel scenes (so x_a is not given, but z_a_samples should be give).
        :param x_a: -
        :param x_b: -
        :param: z_b_samples: a 2D list of z samples for x_b. See the reverse function of Glow for more info.

        :param mode: determines the behavior of the function. If set to 'reconstruct_all', both x_a and x_b should be
        give, and it reconstructs both images and performs a sanity check on the final values of the tensors. If set to
        'sample_x_b', it samples x_b conditioned on the flows_outs of x_a.

        :return: see the function.
        ----------------------------
        NOTE:
        flows_outs_left would be a 2D list of length n_block, whole elements are 1Ds list of len n_flow,
        whose elements are tensors of shape (B, C, H, W)
        """
        if mode == 'reconstruct_all':  # reconstructing bot x_a and x_b (mostly for sanity check)
            log_p_sum_left, log_det_left, \
                z_outs_left, flows_outs_left = self.left_glow(x_a, cond=None, return_flows_outs=True)
            cond = ['c_flow', flows_outs_left]

            log_p_sum_right, log_det_right, \
                z_outs_right, flows_outs_right = self.right_glow(x_b, cond=cond, return_flows_outs=True)

            x_a_rec = self.left_glow.reverse(z_outs_left, reconstruct=True, cond=None, return_reverses=False)
            x_b_rec = self.right_glow.reverse(z_outs_right, reconstruct=True, cond=cond, return_reverses=False)
            sanity_check(x_a, x_b, x_a_rec, x_b_rec)

            return x_a_rec, x_b_rec

        if mode == 'sample_x_b':
            log_p_sum_left, log_det_left, \
                z_outs_left, flows_outs_left = self.left_glow(x_a, cond=None, return_flows_outs=True)
            cond = ['c_flow', flows_outs_left]

            # sample x_b conditioned on x_a. z_b_samples: list of sampled z's
            x_b_syn = self.right_glow.reverse(z_b_samples, reconstruct=False, cond=cond, return_reverses=False)
            return x_b_syn

        if mode == 'sample_x_a':
            # synthesize image with sampled z's
            x_a_syn = self.left_glow.reverse(z_a_samples, reconstruct=False, cond=None, return_reverses=None)
            return x_a_syn


def sanity_check(x_a, x_b, x_a_rec, x_b_rec):
    x_a_diff = torch.mean(torch.abs(x_a - x_a_rec))
    x_b_diff = torch.mean(torch.abs(x_b - x_b_rec))

    # print('x_a, x_rec_a:', x_a, x_a_rec)
    print('In [sanity_check]: mean x_a_diff over all the batch:', x_a_diff)
    print('In [sanity_check]: mean x_b_diff over all the batch:', x_b_diff)




