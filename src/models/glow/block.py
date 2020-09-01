import torch
from torch import nn
from math import log, pi
from torchvision import transforms
from torch.autograd import gradcheck
from torch.utils import checkpoint
import numpy as np
from collections import OrderedDict

from .flow import Flow, ZeroInitConv2d, PairedFlow
from globals import device
from .utils import *


def gaussian_log_p(x, mean, log_sd):  # computes the log density of a point according to the mean and sd of the Gaussian
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)  # CONFIRMED TO BE UNDERSTOOD


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


def compute_chunk_channels():
    pass


class Block(nn.Module):
    """
    Each of the Glow block.
    """
    def __init__(self, n_flow, inp_shape, cond_shape, do_split, configs):
        super().__init__()

        self.do_split = do_split
        self.all_conditional = configs['all_conditional']
        self.split_type = configs['split_type']
        self.split_sections = configs['split_sections'] if self.split_type == 'special' else None  # [3, 9]
        self.configs = configs

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(inp_shape=inp_shape,
                                   cond_shape=cond_shape,
                                   configs=configs))

        # gaussian prior: it is a "learned" prior, a prior whose parameters are optimized to give higher likelihood!
        in_channels, out_channels = self.compute_gaussian_channels(inp_shape)
        self.gaussian = ZeroInitConv2d(in_channels, out_channels)

    def compute_gaussian_channels(self, inp_shape):  # regardless of cond_shape, since this is based on Block outputs
        if self.do_split and self.split_type == 'regular':
            gaussian_in_channels = inp_shape[0] // 2
            gaussian_out_channels = inp_shape[0]

        elif self.do_split and self.split_type == 'special':
            gaussian_in_channels = self.split_sections[0]
            gaussian_out_channels = self.split_sections[1] * 2

        else:
            gaussian_in_channels = inp_shape[0]
            gaussian_out_channels = inp_shape[0] * 2
        return gaussian_in_channels, gaussian_out_channels

    def split_tensor(self, tensor):
        if self.split_type == 'special':
            return torch.split(tensor, split_size_or_sections=self.split_sections, dim=1)  # [3, 9]
        return torch.chunk(tensor, chunks=2, dim=1)

    def possibly_split(self, out):
        b_size = out.shape[0]
        if self.do_split:
            out, z_new = self.split_tensor(out)
            mean, log_sd = self.gaussian(out).chunk(chunks=2, dim=1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zeros = torch.zeros_like(out)
            mean, log_sd = self.gaussian(zeros).chunk(chunks=2, dim=1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out
        return out, z_new, log_p

    def forward(self, inp, conditions=None, grad_checking=False):
        # squeeze operation
        out = squeeze_tensor(inp)

        # outputs needed  from the left glow
        flows_outs = []  # list of tensor, each element of which is the output of the corresponding flow step
        w_outs = []
        act_outs = []

        # Flow operations
        total_log_det = 0
        for i, flow in enumerate(self.flows):
            act_cond, w_cond, coupling_cond = extract_conds(conditions, i, self.all_conditional)

            if self.configs['grad_checkpoint']:
                dummy_tensor = torch.ones(out.shape, dtype=torch.float32, requires_grad=True)  # needed so the output requires grad
                act_out, w_out, out, log_det = checkpoint.checkpoint(flow, out, act_cond, w_cond, coupling_cond, dummy_tensor)  # no longer works
            else:
                act_out, w_out, out, log_det = flow(out, act_cond, w_cond, coupling_cond)  # Flow forward

            total_log_det = total_log_det + log_det
            flows_outs.append(out)  # appending flow_outs - done by the left glow
            w_outs.append(w_out)  # appending w_outs - done by the left glow
            act_outs.append(act_out)  # appending act_outs - done by the left glow

        # splitting operation
        out, z_new, log_p = self.possibly_split(out)

        if grad_checking:
            return out, total_log_det, log_p, z_new
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
        # reverse of the flows
        for i, flow in enumerate(self.flows[::-1]):
            act_cond, w_cond, coupling_cond = extract_conds(conditions, i, self.all_conditional)
            conds = make_cond_dict(act_cond, w_cond, coupling_cond)
            flow_reverse = flow.reverse(inp, act_cond, w_cond, coupling_cond)  # Flow reverse
            inp = flow_reverse
        # unsqueezing
        unsqueezed = unsqueeze_tensor(inp)
        return unsqueezed

    def check_grad(self, inp, conditions=None):
        return gradcheck(func=self.forward, inputs=(inp, conditions, True), eps=1e-6)


def compute_gaussian_channels(inp_shape, do_split):  # regardless of cond_shape, since this is based on Block outputs
    if do_split:
        gaussian_in_channels = inp_shape[0] // 2
        gaussian_out_channels = inp_shape[0]
    else:
        gaussian_in_channels = inp_shape[0]
        gaussian_out_channels = inp_shape[0] * 2
    return gaussian_in_channels, gaussian_out_channels


class PairedBlock(nn.Module):
    def __init__(self, n_flow, cond_shape, inp_shape, do_split, configs):
        super().__init__()
        self.do_split = do_split
        self.configs = configs

        self.paired_flows = nn.ModuleList()
        for i in range(n_flow):
            self.paired_flows.append(PairedFlow(cond_shape, inp_shape, configs))

        in_channels, out_channels = compute_gaussian_channels(inp_shape, do_split)
        self.left_gaussian = ZeroInitConv2d(in_channels, out_channels)
        self.right_gaussian = ZeroInitConv2d(in_channels, out_channels)

    def possibly_split(self, out, left_or_right):
        b_size = out.shape[0]
        gaussian = self.left_gaussian if left_or_right == 'left' else self.right_gaussian

        if self.do_split:
            out, z_new = out.chunk(chunks=2, dim=1)
            mean, log_sd = gaussian(out).chunk(chunks=2, dim=1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
        else:
            zeros = torch.zeros_like(out)
            mean, log_sd = gaussian(zeros).chunk(chunks=2, dim=1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out
        return out, z_new, log_p

    def forward_not_to_be_used(self, activations, inp_left, inp_right):
        left_out, right_out = squeeze_tensor(inp_left), squeeze_tensor(inp_right)
        left_total_logdet = right_total_logdet = 0

        for i in range(len(self.paired_flows)):
            left_out, left_logdet, right_out, right_logdet = self.paired_flows[i](activations, left_out, right_out)
            left_total_logdet += left_logdet
            right_total_logdet += right_logdet

        left_out, left_z_new, left_log_p = self.possibly_split(left_out, 'left')
        right_out, right_z_new, right_log_p = self.possibly_split(right_out, 'right')
        return left_out, left_total_logdet, left_log_p, left_z_new, right_out, right_total_logdet, right_log_p, right_z_new

