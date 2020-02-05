import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi

import numpy as np
from scipy import linalg as la
from helper import label_to_tensor


# this device is accessible in all the functions in this file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))  # this operation is done channel-wise
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))  # loc, scale: vectors applied to all channels
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, inp):
        with torch.no_grad():
            flatten = inp.permute(1, 0, 2, 3).contiguous().view(inp.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )

            # data dependent initialization
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, inp):
        _, _, height, width = inp.shape  # input of shape [bsize, in_channel, h, w]

        # data-dependent initialization of scale and shift
        if self.initialized.item() == 0:  # to be initialized the first time
            self.initialize(inp)
            self.initialized.fill_(1)

        # computing log determinant
        scale_logabs = logabs(self.scale)
        log_det = height * width * torch.sum(scale_logabs)

        return self.scale * (inp + self.loc), log_det

    def reverse(self, out):
        return (out / self.scale) - self.loc


class InvConv1x1(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        q, _ = torch.qr(torch.randn(in_channel, in_channel))
        w = q.unsqueeze(2).unsqueeze(3)  # why not unsqueeze(0).unsqueeze(1)? (ablation)
        self.weight = nn.Parameter(w)  # the weight matrix

    def forward(self, inp):
        _, _, height, width = inp.shape
        out = F.conv2d(inp, self.weight)

        log_w = torch.slogdet(self.weight.squeeze().double())[1].float()  # use of double() and float()? (ablation)
        log_det = height * width * log_w
        return out, log_det

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv1x1LU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        # kept fixed
        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))

        # learnable parameters
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, inp):
        _, _, height, width = inp.shape

        weight = self.calc_weight()

        out = F.conv2d(inp, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        # ablation: doing the following
        # why torch.exp(self.w_s)?
        # w_s is a matrix not a vector here?
        # s_sign changes while s is optimized, but is assumed to be fixed by register buffer. Why?
        # What is the role of mask and sign and eye matrices here?
        # What are squeeze unsqueeze doing here?
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroInitConv2d(nn.Module):
    """
    To be used in the Affine Coupling step:
    The last convolution of each NN(), according to the paper is initialized with zeros, such that the each affine layer
    initially performs an identity function (at the moment, I have difficulty understanding why this is helpful)
    """
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3)  # why kernels_size = 3?
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))  # why incorporating learnable scale? (ablation)

    def forward(self, inp):
        # padding with additional 1 in each side to keep the spatial dimension unchanged after the convolution operation
        out = F.pad(input=inp, pad=[1, 1, 1, 1], value=1)  # why padding=1 each side and value=1 (ablation)?
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)  # what is this for (ablation)?
        return out


class AffineCoupling(nn.Module):
    """
    This transforms part of the input tensor in a way that half of the output tensor in a way that half of the output
    tensor is a non-linear function of the other half. This non-linearity is obtained through the stacking some CNNs.
    """
    def __init__(self, in_channel, n_filters=512, do_affine=True, conditional=True):
        super().__init__()

        conv_channels = (in_channel // 2) + 10 if conditional else in_channel // 2  # extra 10 for class conditions
        self.do_affine = do_affine
        self.net = nn.Sequential(  # NN() in affine coupling
            nn.Conv2d(in_channels=conv_channels, out_channels=n_filters, kernel_size=3, padding=1),  # why such params? (ablation study)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=1),
            nn.ReLU(inplace=True),
            ZeroInitConv2d(in_channel=n_filters, out_channel=in_channel)  # out_channel equal to in_channel
        )

        # Initializing the params
        self.net[0].weight.data.normal_(0, 0.05)  # other ways of initialization?? (ablation study)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, inp, cond=None):
        inp_a, inp_b = inp.chunk(chunks=2, dim=1)  # chunk along the channel dimension
        if self.do_affine:
            if cond is not None:  # conditional
                if cond[0] == 'mnist':
                    # expects the cond to be of shape (B, 10, H, W) - B: batch size
                    # print('In [Block].[forward]: inp_a shape:', inp_a.shape)
                    # concatenate condition along channel: C -> C+10
                    # cond_tensor = label_to_tensor(label=cond[1], height=inp_a.shape[2], width=inp_a.shape[3])
                    cond_tensor = cond[1][:, :, :inp_a.shape[2], :inp_b.shape[3]]  # truncate spatial dimension (more explanation)

                    # print('cond_tensor shape:', cond_tensor.shape)
                    inp_a_conditional = torch.cat(tensors=[inp_a, cond_tensor], dim=1)
                    # print('In [Block].[forward]: inp_a_conditional shape:', inp_a_conditional.shape)
                    # input()

                    log_s, t = self.net(inp_a_conditional).chunk(chunks=2, dim=1)

                else:
                    raise NotImplementedError('In [Block] forward: Condition not implemented...')
            else:
                log_s, t = self.net(inp_a).chunk(chunks=2, dim=1)

            s = F.sigmoid(log_s + 2)  # why not exp(.)? why + 2? (ablation study)

            out_b = (inp_b + t) * s  # why first + then *??  (ablation study)
            log_det = torch.sum(torch.log(s).view(inp.shape[0], -1), 1)  # print and check shape (ablation)

        else:
            # note: ZeroConv2d(in_channel=n_filters, out_channel=in_channel) should also be changed for additive
            print('Not implemented... Use --affine')
            out_b, log_det = None, None

        return torch.cat(tensors=[inp_a, out_b], dim=1), log_det

    def reverse(self, output, cond=None):
        out_a, out_b = output.chunk(chunks=2, dim=1)  # here we know that out_a = inp_a (see the forward fn)
        if self.do_affine:
            if cond is not None:
                if cond[0] == 'mnist':
                    # concatenate with the same condition as in the forward pass
                    # print('In [Block].[reverse]: out_a shape:', out_a.shape)
                    # cond_tensor = label_to_tensor(label=cond[1], height=out_a.shape[2], width=out_a.shape[3])
                    label, n_samples = cond[1], cond[2]
                    cond_tensor = label_to_tensor(label, out_a.shape[2], out_a.shape[3], n_samples).to(device)
                    # cond_tensor = cond[1][:, :, :out_a.shape[2], :out_a.shape[3]]
                    out_a_conditional = torch.cat(tensors=[out_a, cond_tensor], dim=1)
                    # print('In [Block].[reverse]: out_a_conditional shape:', out_a_conditional.shape)
                    # input()

                    log_s, t = self.net(out_a_conditional).chunk(chunks=2, dim=1)
                else:
                    raise NotImplementedError('In [Block] reverse: Condition not implemented...')
            else:
                log_s, t = self.net(out_a).chunk(chunks=2, dim=1)

            s = F.sigmoid(log_s + 2)
            inp_b = (out_b / s) - t

        else:
            print('Not implemented... Use --affine')
            inp_b = None

        return torch.cat(tensors=[out_a, inp_b], dim=1)


class Flow(nn.Module):
    """
    The Flow module does not change the dimensionality of its input.
    """
    def __init__(self, in_channel, do_affine=True, conv_lu=True):
        super().__init__()

        self.act_norm = ActNorm(in_channel=in_channel)
        self.inv_conv = InvConv1x1LU(in_channel) if conv_lu else InvConv1x1(in_channel)
        self.coupling = AffineCoupling(in_channel=in_channel, do_affine=do_affine)

    def forward(self, inp, cond=None):
        out, act_logdet = self.act_norm(inp)
        out, conv_logdet = self.inv_conv(out)
        out, affine_logdet = self.coupling(out, cond=cond)

        log_det = act_logdet + conv_logdet + affine_logdet
        return out, log_det

    def reverse(self, output, cond=None):
        inp = self.coupling.reverse(output, cond=cond)
        inp = self.inv_conv.reverse(inp)
        inp = self.act_norm.reverse(inp)
        return inp


def gaussian_log_p(x, mean, log_sd):  # computes the log density of a point according to the mean and sd of the Gaussian
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    """
    Each of the Glow block.
    """
    def __init__(self, in_channel, n_flow, do_split=True, do_affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4
        self.do_split = do_split

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(in_channel=squeeze_dim, do_affine=do_affine, conv_lu=conv_lu))

        # gaussian: it is a "learned" prior, a prior whose parameters are optimized to give higher likelihood!
        if self.do_split:
            self.gaussian = ZeroInitConv2d(in_channel=in_channel * 2, out_channel=in_channel * 4)
        else:
            self.gaussian = ZeroInitConv2d(in_channel=in_channel * 4, out_channel=in_channel * 8)

    def forward(self, inp, cond=None):
        b_size, in_channel, height, width = inp.shape
        squeezed = inp.view(b_size, in_channel, height // 2, 2, width // 2, 2)  # squeezing height and width
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)  # putting 3, 5 at first to index the height and width easily
        out = squeezed.contiguous().view(b_size, in_channel * 4, height // 2, width // 2)  # squeeze into extra channels

        # flow operations
        total_log_det = 0
        for flow in self.flows:
            out, log_det = flow(out, cond=cond)  # each flow step keeps the dimension unchanged
            total_log_det = total_log_det + log_det

        # output shape [b_size, n_channel * 4, height // 2, width // 2]
        if self.do_split:
            out, z_new = out.chunk(chunks=2, dim=1)  # split along the channel dimension
            mean, log_sd = self.gaussian(out).chunk(chunks=2, dim=1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)  # why view (bsize, -1)? sum(1)? what is dimension of log_p? I should print it.

        else:
            zeros = torch.zeros_like(out)  # making zero output - why??
            mean, log_sd = self.gaussian(zeros).chunk(chunks=2, dim=1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, total_log_det, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False, cond=None):
        """
        The reverse operation in each Block.
        :param output: the input to the reverse function, latent variable from the previous Block.
        :param eps: could be the latent variable already extracted in the forward pass. It is used for reconstruction of
        an image with its extracted latent vectors. Otherwise (in the cases I uses) it is simply a sample of the unit
        Gaussian (with temperature).
        :param reconstruct: If true, eps is used for reconstruction.
        :return: -
        """
        inp = output
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

        for flow in self.flows[::-1]:
            inp = flow.reverse(inp, cond=cond)

        # unsqueezing
        b_size, n_channel, height, width = inp.shape
        unsqueezed = inp.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )
        return unsqueezed


class Glow(nn.Module):
    def __init__(self, in_channel, n_flows, n_blocks, do_affine=True, conv_lu=True):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_blocks - 1):
            self.blocks.append(Block(in_channel=n_channel, n_flow=n_flows, do_split=True,
                                     do_affine=do_affine, conv_lu=conv_lu))
            n_channel = n_channel * 2
        self.blocks.append(Block(in_channel=n_channel, n_flow=n_flows, do_split=False, do_affine=do_affine,
                                 conv_lu=conv_lu))

    def forward(self, inp, cond=None):
        """
        The forward functions take as input an image and performs all the operations to obtain z's: x->z
        :param inp: the image to be encoded
        :return: the extracted z's, log_det, and log_p_sum, the last two used to compute the log p(x) according to the
        change of variables theorem.
        """
        log_p_sum = 0
        log_det = 0
        out = inp
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out, cond=cond)
            z_outs.append(z_new)
            log_det = log_det + det
            log_p_sum = log_p_sum + log_p  # I am avoiding the if None condition as I do not know why it may be None

        return log_p_sum, log_det, z_outs

    def reverse(self, z_list, reconstruct=False, cond=None):
        """
        The reverse function performs the operations in the direction z->x.
        :param z_list: the list of z'a sampled from unit Gaussian (with temperature) for different Blocks.
        :param reconstruct: is set to True if one wants to reconstruct the image with its extracted latent variables.
        :return: the generated image.
        """
        inp = None
        for i, block in enumerate(self.blocks[::-1]):  # print to see what us ::-1
            if i == 0:
                inp = block.reverse(output=z_list[-1], eps=z_list[-1], reconstruct=reconstruct, cond=cond)
            else:
                inp = block.reverse(output=inp, eps=z_list[-(i + 1)], reconstruct=reconstruct, cond=cond)
        return inp
