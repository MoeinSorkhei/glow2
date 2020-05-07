import torch
from torch import nn
from math import log, pi
from torchvision import transforms
import numpy as np

from .flow import Flow, ZeroInitConv2d
from globals import device


def gaussian_log_p(x, mean, log_sd):  # computes the log density of a point according to the mean and sd of the Gaussian
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)  # CONFIRMED TO BE UNDERSTOOD


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


def prep_coupling_cond(module, level, cond, in_reverse=False, reverse_level=None):
    """
    This takes the condition corresponding to the given level. For obtaining the 'segment' condition, if it is used for
    Blocks, it takes the 1D list with the given index from the 2D list, and if used for Flows, it takes the condition
    tensor with the given index from the 1D list.
    Currently, the boundaries are specific for Block levels only. For obtaining the 'boundary' condition, if it is used
    for Blocks, it takes the corresponding (down-sampled) boundary from the 1D list. If used for Flows, it directly
    uses the boundary since all the Flows in a block use the same boundary at the moment.

    :param reverse_level:
    :param in_reverse:
    :param module: the module that is going to use the condition. Either 'block' or 'flow'.
    :param level: the level (index) of the Block or Flow which will use this condition.
    :param cond: the condition in the form of a dictionary from which the condition with the wanted level (index) will
    be obtained.
    :return: the condition corresponding to the given index in the form of a dictionary.

    Notes: The boundary condition now only works with batch size of 1.

    """
    def down_sample_and_squeeze_boundaries(boundaries):
        down_sampled_bmaps = []  # list of down-sampled tensors
        # down-sample each boundary map individually (any better way? It needs PIL)
        for i in range(boundaries.shape[0]):
            b_map = boundaries[i]
            # compute the down-sampling size for the corresponding block level
            h, w = b_map.shape[1], b_map.shape[2]

            factor = reverse_level if in_reverse else level
            down_size = (h // (2 ** factor), w // (2 ** factor))
            # define the needed transformations
            trans = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(down_size),
                                        transforms.ToTensor()])

            # level = 1 example: (3, 128, 256) ==> (3, 64, 128)
            b_map = trans(b_map.cpu()).to(device)
            # ceil the pixels so that they are exactly 1 or 0
            b_map = torch.from_numpy(np.ceil(b_map.cpu().numpy())).to(device)
            # squeeze the boundary in the same way the z in the beginning of each block is squeezed => (1, 12, 64, 128)
            in_channel, height, width = b_map.shape
            b_map = b_map.view(in_channel, height // 2, 2, width // 2, 2).permute(0, 2, 4, 1, 3) \
                .contiguous().view(in_channel * 4, height // 2, width // 2)

            # down_sampled_bmaps[i] = b_map
            down_sampled_bmaps.append(b_map)
        return torch.stack(down_sampled_bmaps, dim=0)  # convert the list to tensor

    if cond is None:
        condition = None

    elif cond['name'] == 'segment':
        condition = {'name': 'segment', 'segment': cond['segment'][level]}

    elif cond['name'] == 'segment_boundary':
        condition = {'name': 'segment_boundary', 'segment': cond['segment'][level]}

        if module == 'block':  # module = 'block' is always called before module = 'flow'
            boundary = cond['boundary']  # a tensor of shape (1, C, H, W) - assumption: batch size is 1
            down_sampled_squeezed = down_sample_and_squeeze_boundaries(boundary)
            condition['boundary'] = down_sampled_squeezed

        else:  # module = 'flow'
            condition['boundary'] = cond['boundary']  # all the flows in a block use the same boundary

    else:
        condition = cond
    return condition


def reverse_conditions(coupling_conds, left_glow_w_outs, left_glow_act_outs, w_is_conditional, act_is_conditional):
    # in the 'c_flow' case: reversing the condition so it matches the reverse direction for each Flow
    # [F1, F2, F3] => [F3, F2, F1] where each Fi denotes the condition tensor for the corresponding Flow
    # 'segment_boundary' does not need reverse here since all Flows use the same bmap
    if coupling_conds is not None \
            and (coupling_conds['name'] == 'segment' or coupling_conds['name'] == 'segment_boundary'):
        coupling_conds['segment'] = coupling_conds['segment'][::-1]

    if w_is_conditional:
        left_glow_w_outs = left_glow_w_outs[::-1]
        # reverse_w_conditional = self.w_conditional[::-1]
    if act_is_conditional:
        left_glow_act_outs = left_glow_act_outs[::-1]
    return coupling_conds, left_glow_w_outs, left_glow_act_outs


def unsqueeze_tensor(inp):
    b_size, n_channel, height, width = inp.shape
    unsqueezed = inp.view(b_size, n_channel // 4, 2, 2, height, width)
    unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
    unsqueezed = unsqueezed.contiguous().view(
        b_size, n_channel // 4, height * 2, width * 2
    )
    return unsqueezed


def squeeze_tensor(inp):
    b_size, in_channel, height, width = inp.shape
    squeezed = inp.view(b_size, in_channel, height // 2, 2, width // 2, 2)  # squeezing height and width
    squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)  # putting 3, 5 at first to index the height and width easily
    out = squeezed.contiguous().view(b_size, in_channel * 4, height // 2, width // 2)  # squeeze into extra channels
    return out


class Block(nn.Module):
    """
    Each of the Glow block.
    """
    def __init__(self, in_channel, n_flow, do_split=True, do_affine=True, conv_lu=True, coupling_cond_shape=None,
                 w_conditionals=False, act_conditionals=False, inp_shape=None, conv_stride=None):
        super().__init__()

        squeeze_dim = in_channel * 4
        self.do_split = do_split
        self.w_conditionals = w_conditionals
        self.act_conditionals = act_conditionals

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim,
                                   do_affine,
                                   conv_lu,
                                   coupling_cond_shape,
                                   w_conditional=w_conditionals,
                                   act_conditional=act_conditionals,
                                   inp_shape=inp_shape,
                                   conv_stride=conv_stride))

        # gaussian: it is a "learned" prior, a prior whose parameters are optimized to give higher likelihood!
        if self.do_split:
            self.gaussian = ZeroInitConv2d(in_channel=in_channel * 2, out_channel=in_channel * 4)
        else:
            self.gaussian = ZeroInitConv2d(in_channel=in_channel * 4, out_channel=in_channel * 8)

    def forward(self, inp, coupling_conds=None, left_w_outs=None, left_act_outs=None,
                return_flows_outs=False, return_w_outs=False, return_act_outs=False):
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
            # preparing conditions
            coupling_condition = prep_coupling_cond('flow', i, coupling_conds)
            left_w_out = left_w_outs[i] if self.w_conditionals else None  # done by the right glow
            left_act_out = left_act_outs[i] if self.act_conditionals else None

            # Flow forward
            flow_output = flow(out,
                               coupling_cond=coupling_condition,
                               w_left_out=left_w_out,
                               act_left_out=left_act_out,
                               return_w_out=return_w_outs,
                               return_act_out=return_act_outs)

            out, log_det = flow_output['out'], flow_output['log_det']
            total_log_det = total_log_det + log_det

            # appending flow_outs - done by the left glow
            if return_flows_outs:
                flows_outs.append(out)

            # appending w_outs - done by the left glow
            if return_w_outs:
                w_out = flow_output['w_out']
                w_outs.append(w_out)

            # appending act_outs - done by the left glow
            if return_act_outs:
                act_out = flow_output['act_out']
                act_outs.append(act_out)

        # splitting the out or not
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

        output_dict = {'out': out, 'total_log_det': total_log_det, 'log_p': log_p, 'z_new': z_new}
        if return_flows_outs:
            output_dict['flows_outs'] = flows_outs
        if return_w_outs:
            output_dict['w_outs'] = w_outs
        if return_act_outs:
            output_dict['act_outs'] = act_outs

        return output_dict

    def reverse(self, output, eps=None, reconstruct=False,
                coupling_conds=None, left_block_w_outs=None, left_block_act_outs=None):
        """
        The reverse operation in each Block.
        :param left_block_act_outs:
        :param left_block_w_outs:
        :param coupling_conds: in the case of 'c_flow', it is a list of length n_flow, each element of which is a tensor of shape
        (B, C, H, W). The list is ordered in the forward direction (flow outputs were extracted in the forward call).

        :param output: the input to the reverse function, latent variable from the previous Block.
        :param eps: could be the latent variable already extracted in the forward pass. It is used for reconstruction of
        an image with its extracted latent vectors. Otherwise (in the cases I uses) it is simply a sample of the unit
        Gaussian (with temperature).
        :param reconstruct: If true, eps is used for reconstruction.
        :return: -
        """
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

        # reversing the conditions (since we are moving in the reverse order)
        coupling_conds, left_glow_w_outs, left_glow_act_outs = reverse_conditions(coupling_conds,
                                                                                  left_block_w_outs,
                                                                                  left_block_act_outs,
                                                                                  self.w_conditionals,
                                                                                  self.act_conditionals)
        # reverse Flow operations one by one
        for i, flow in enumerate(self.flows[::-1]):
            # preparing conditions - for prep_coupling_cond: we do not need reverse_i as all Flows use the same b_map
            coupling_condition = prep_coupling_cond('flow', i, coupling_conds)
            left_w_out = left_block_w_outs[i] if self.w_conditionals else None
            left_act_out = left_block_act_outs[i] if self.act_conditionals else None

            flow_reverse = flow.reverse(inp,
                                        coupling_cond=coupling_condition,
                                        w_left_out=left_w_out,
                                        act_left_out=left_act_out)
            inp = flow_reverse

        # unsqueezing
        unsqueezed = unsqueeze_tensor(inp)
        return unsqueezed
