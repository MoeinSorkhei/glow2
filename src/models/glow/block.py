import torch
from torch import nn
from math import log, pi
from torchvision import transforms
import numpy as np

from .flow import Flow, ZeroInitConv2d
from globals import device
from .utils import *


def gaussian_log_p(x, mean, log_sd):  # computes the log density of a point according to the mean and sd of the Gaussian
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)  # CONFIRMED TO BE UNDERSTOOD


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    """
    Each of the Glow block.
    """
    def __init__(self, n_flow, inp_shape, cond_shape, do_split=True, all_conditional=False, conv_stride=None):
        super().__init__()

        chunk_channels = inp_shape[0] // 2 if do_split else inp_shape[0]  # channels after chunking the output
        self.do_split = do_split
        self.all_conditional = all_conditional

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(inp_shape=inp_shape,
                                   cond_shape=cond_shape,
                                   all_conditional=all_conditional,
                                   conv_stride=conv_stride))

        # gaussian: it is a "learned" prior, a prior whose parameters are optimized to give higher likelihood!
        self.gaussian = ZeroInitConv2d(in_channel=chunk_channels, out_channel=chunk_channels * 2)

    def forward(self, inp, conditions=None):
        # squeeze operation
        b_size, in_channel, height, width = inp.shape
        out = squeeze_tensor(inp)

        # outputs needed  from the left glow
        flows_outs = []  # list of tensor, each element of which is the output of the corresponding flow step
        w_outs = []
        act_outs = []

        # Flow operations
        total_log_det = 0
        for i, flow in enumerate(self.flows):
            act_cond, w_cond, coupling_cond = extract_conds(conditions, i, self.all_conditional)
            conds = make_cond_dict(act_cond, w_cond, coupling_cond)

            flow_output = flow(out, conds)  # Flow forward

            out, log_det = flow_output['out'], flow_output['log_det']
            total_log_det = total_log_det + log_det

            # appending flow_outs - done by the left glow
            flows_outs.append(out)

            # appending w_outs - done by the left glow
            w_out = flow_output['w_out']
            w_outs.append(w_out)

            # appending act_outs - done by the left glow
            act_out = flow_output['act_out']
            act_outs.append(act_out)

        # splitting operation
        if self.do_split:
            out, z_new = out.chunk(chunks=2, dim=1)  # split along the channel dimension
            mean, log_sd = self.gaussian(out).chunk(chunks=2, dim=1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zeros = torch.zeros_like(out)
            mean, log_sd = self.gaussian(zeros).chunk(chunks=2, dim=1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        # return output_dict
        return {
            'act_outs': act_outs,
            'w_outs': w_outs,
            'flows_outs': flows_outs,
            'out': out,
            'total_log_det': total_log_det,
            'log_p': log_p,
            'z_new': z_new
        }

    def reverse(self, output, eps=None, reconstruct=False, conditions=None):
        inp = output

        # specifying inp based on whether we are reconstructing or not
        if reconstruct:
            if self.do_split:
                inp = torch.cat([output, eps], 1)
            else:
                inp = eps
        else:
            if self.do_split:
                mean, log_sd = self.gaussian(inp).chunk(chunks=2, dim=1)
                z = gaussian_sample(eps, mean, log_sd)
                inp = torch.cat(tensors=[output, z], dim=1)
            else:
                zeros = torch.zeros_like(inp)
                mean, log_sd = self.gaussian(zeros).chunk(chunks=2, dim=1)
                z = gaussian_sample(eps, mean, log_sd)
                inp = z

        for i, flow in enumerate(self.flows[::-1]):
            act_cond, w_cond, coupling_cond = extract_conds(conditions, i, self.all_conditional)
            conds = make_cond_dict(act_cond, w_cond, coupling_cond)

            flow_reverse = flow.reverse(inp, conds)  # Flow reverse
            inp = flow_reverse

        # unsqueezing
        unsqueezed = unsqueeze_tensor(inp)
        return unsqueezed
