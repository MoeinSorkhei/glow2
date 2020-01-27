import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi

import numpy as np
from scipy import linalg as la


logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    """
    Todo:
        - check the functions in test cases
        - document the code
        - is the parameter 'return_logdet=True' really needed?we should always return log determinant.
    """
    def __init__(self, in_channel, return_logdet=True):
        super().__init__()

        self.return_logdet = return_logdet
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))  # this operation is done channel-wise
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))  # loc, scale: vectors of shape [1, in_channel, 1, 1]
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, inp):
        # NOT VERIFIED
        with torch.no_grad():
            flatten = inp.permute(1, 0, 2, 3).contiguous().view(inp.shape[1], -1)
            # print(flatten.size())
            # print(flatten, '\n\n')
            # print(flatten.mean(1))

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

            #print("In ActNorm initialize: found \n mean={}, size: {} \n\n std={}, size: {}".
             #     format(mean, std, mean.shape, std.size()))
            # data dependent initialization
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, inp):  # GLOW FORWARD IS PYTORCH FORWARD???
        # print('In [ActNorm] forward')
        _, _, height, width = inp.shape  # input of shape [bsize, in_channel, h, w]

        # print(type(self.initialized.item()))
        if self.initialized.item() == 0:  # to be initialized the first time
            self.initialize(inp)
            self.initialized.fill_(1)

        # computing log determinant
        scale_logabs = logabs(self.scale)
        log_det = height * width * torch.sum(scale_logabs)

        # print('In [ActNorm] forward: done')
        if self.return_logdet:
            return self.scale * (inp + self.loc), log_det
        else:
            return self.scale * (inp + self.loc)

    def reverse(self, out):
        return (out / self.scale) - self.loc


class InvConv1x1(nn.Module):
    """
    Todo:
        - add LU decomposition
    """
    def __init__(self, in_channel):
        super().__init__()
        q, _ = torch.qr(torch.randn(in_channel, in_channel))  # why initialize orthogonal?
        w = q.unsqueeze(2).unsqueeze(3)  # why not unsqueeze(0).unsqueeze(1)?
        self.weight = nn.Parameter(w)  # the weight matrix

    def forward(self, inp):
        _, _, height, width = inp.shape
        out = F.conv2d(inp, self.weight)  # applying 1x1 convolution - why not nn.Conv2D?

        log_w = torch.slogdet(self.weight.squeeze().double())[1].float()  # use of double() and float()?
        log_det = height * width * log_w
        return out, log_det

    def reverse(self, output):
        return F.conv2d(  # why not nn.Conv2D?
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)  # how to make sure W is invertible?
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

    def forward(self, input):
        # print('In [InvConv1x1LU] forward')
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        # print('In [InvConv1x1LU] forward: done')
        return out, logdet

    def calc_weight(self):
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
    initially performs an identity function (I have difficulty understanding this at the moment)

    Todo:
        - is the constructor defined in the best way?
        - addressing the questions in the form of comments in the code
    """
    def __init__(self, in_channel, out_channel, padding=1):
        # padding: how many padding layes should be added to the sides of the input
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3)  # why kernels_size = 3?
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))  # why incorporating learnable scale?

    def forward(self, inp):
        # padding with additional 1 in each side to keep the spatial dimension unchanged after the convolution operation
        out = F.pad(input=inp, pad=[1, 1, 1, 1], value=1)  # why value=1?
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)  # what is this for?
        return out


class AffineCoupling(nn.Module):
    """
    Todo:
        - document the code
        - can the NN() be written as a more complex network such as ResNet?
    """
    def __init__(self, in_channel, n_filters=512, do_affine=True):
        super().__init__()

        self.do_affine = do_affine
        self.net = nn.Sequential(  # NN() in affine coupling
            nn.Conv2d(in_channels=in_channel // 2, out_channels=n_filters, kernel_size=3, padding=1),  # why such params?
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=1),
            nn.ReLU(inplace=True),
            ZeroInitConv2d(in_channel=n_filters, out_channel=in_channel)  # output channels size = input channels size
        )

        # Initializing the params - ZeroConv2d has its won way of initialization and is initialized once created
        self.net[0].weight.data.normal_(0, 0.05)  # other ways of initialization??
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, inp):
        # print('In [AffineCoupling] forward')
        inp_a, inp_b = inp.chunk(chunks=2, dim=1)  # chunk along the channel dimension
        if self.do_affine:  # make out_b a non-linear function of inp_a
            # passing inp_a through non-linearity (dimensions unchanged)
            # when initializing AffineCoupling in_channel should be (in_b channels * 2)??
            log_s, t = self.net(inp_a).chunk(chunks=2, dim=1)
            s = F.sigmoid(log_s + 2)  # why not exp(.)? why + 2?

            out_b = (inp_b + t) * s  # why first + then *??  # are the channels equal after chunk??
            log_det = torch.sum(torch.log(s).view(inp.shape[0], -1), 1)  # CHECK THIS!

        else:  # note: ZeroConv2d(in_channel=n_filters, out_channel=in_channel) should also be changed in this case
            print('Not implemented... Use --affine')
            out_b, log_det = None, None

        # print('In [AffineCoupling] forward: done')
        return torch.cat(tensors=[inp_a, out_b], dim=1), log_det

    def reverse(self, output):
        out_a, out_b = output.chunk(chunks=2, dim=1)  # here we know that out_a = inp_a (see the forward fn)
        if self.do_affine:
            log_s, t = self.net(out_a).chunk(chunks=2, dim=1)
            s = F.sigmoid(log_s + 2)
            inp_b = (out_b / s) - t

        else:
            print('Not implemented... Use --affine')
            inp_b = None

        return torch.cat(tensors=[out_a, inp_b], dim=1)


class Flow(nn.Module):
    """
    Todo:
        - change conv_lu parameter to True and implement it
    The Flow module does not change the dimensionality of its input.
    """
    def __init__(self, in_channel, do_affine=True, conv_lu=True):
        super().__init__()

        self.act_norm = ActNorm(in_channel=in_channel)
        self.inv_conv = InvConv1x1LU(in_channel) if conv_lu else InvConv1x1(in_channel)
        self.coupling = AffineCoupling(in_channel=in_channel, do_affine=do_affine)

    def forward(self, inp):
        # print('In [Flow] forward')
        out, act_logdet = self.act_norm(inp)
        out, conv_logdet = self.inv_conv(out)
        out, affine_logdet = self.coupling(out)

        log_det = act_logdet + conv_logdet + affine_logdet

        # print('In [Flow] forward: done')
        return out, log_det

    def reverse(self, output):
        inp = self.coupling.reverse(output)
        inp = self.inv_conv.reverse(inp)
        inp = self.act_norm.reverse(inp)
        return inp


def gaussian_log_p(x, mean, log_sd): # gives the log density of a point according to the mean and sd of the Gaussian
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):  # not sure what it does
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    """
    Todo:
        - is the constructor defined in the best way?
        - squeeze should be a function which takes the squeeze ratio
            - lines to change:
                - squeeze_dim = in_channel * 4
                - self.gaussian = ZeroInitConv2d(in_channel=in_channel * 2, out_channel=in_channel * 4)
                - squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)

    Note on the dimensionality: the input channel of each ZeroInitConv2d (which learns the mean and std of the
    Gaussian) is in_channel * 2 because of the split operation before that, and the output channels is
    in_channel * 4 because it should output the mean and std which are obtained by separating the ZeroInitConv2d output
    along the channel.
    """
    def __init__(self, in_channel, n_flow, do_split=True, do_affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4  # channels after the Squeeze operation - why * 4?
        self.do_split = do_split  # if the Block should split its output

        self.flows = nn.ModuleList()
        for i in range(n_flow):  # note: each Flow is applied after Squeeze
            self.flows.append(Flow(in_channel=squeeze_dim, do_affine=do_affine, conv_lu=conv_lu))

        # gaussian: the density according to which zi (z corresponding to this Block) is distributed
        if self.do_split:
            self.gaussian = ZeroInitConv2d(in_channel=in_channel * 2, out_channel=in_channel * 4)
        else:
            self.gaussian = ZeroInitConv2d(in_channel=in_channel * 4, out_channel=in_channel * 8)

        # the mean and log_sd of gaussian is computed through the ZeroInitConv2d networks

    def forward(self, inp):
        # print('In [Block] forward')
        b_size, in_channel, height, width = inp.shape
        # print(...)
        # squeeze operation
        squeezed = inp.view(b_size, in_channel, height // 2, 2, width // 2, 2)  # squeezing height and width
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)  # putting 3, 5 at first to index the height and width easily
        out = squeezed.contiguous().view(b_size, in_channel * 4, height // 2, width // 2)  # squeeze into extra channels

        # flow operations
        total_log_det = 0
        for flow in self.flows:
            out, log_det = flow(out)  # each flow step keeps the dimension unchanged
            total_log_det = total_log_det + log_det

        if self.do_split:  # output shape [b_size, n_channel * 4, height // 2, width // 2]
            out, z_new = out.chunk(chunks=2, dim=1)  # split along the channel dimension
            mean, log_sd = self.gaussian(out).chunk(chunks=2, dim=1)  # mean, log_sd the same size as z_new, out
            log_p = gaussian_log_p(z_new, mean, log_sd)  # still cannot understand the goal of this and previous line
            log_p = log_p.view(b_size, -1).sum(1)  # why view (bsize, -1)? sum(1)? what is dimension of log_p?

        else:
            zeros = torch.zeros_like(out)  # making zero output - why??
            mean, log_sd = self.gaussian(zeros).chunk(chunks=2, dim=1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        # print('In [Block] forward: done')
        return out, total_log_det, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):  # what are eps=None, reconstruct=False
        inp = output
        if reconstruct:
            if self.split:
                inp = torch.cat([output, eps], 1)
            else:
                inp = eps
        else:
            if self.do_split:
                # does it mean for each input, z is distributed with a different gaussinan?
                mean, log_sd = self.gaussian(inp).chunk(chunks=2, dim=1)
                z = gaussian_sample(eps, mean, log_sd)  # what is eps??  # esp is None??
                inp = torch.cat(tensors=[output, z], dim=1)
            else:
                zeros = torch.zeros_like(inp)
                mean, log_sd = self.gaussian(zeros).chunk(chunks=2, dim=1)
                z = gaussian_sample(eps, mean, log_sd)  # esp is None??
                inp = z

        for flow in self.flows[::-1]:
            inp = flow.reverse(inp)

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

    def forward(self, inp):
        # print('In [Glow] forward')
        """
        The forward functions take as input an image and performs all the operations to obtain z's: x->z
        :param inp:
        :return:
        """
        log_p_sum = 0
        log_det = 0
        out = inp
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            log_det = log_det + det
            log_p_sum = log_p_sum + log_p  # I am avoiding the if None condition as I do not know why it may be None

        # print('In [Glow] forward: done')
        return log_p_sum, log_det, z_outs

    def reverse(self, z_list, reconstruct=False):
        """
        The reverse function performs the operations in the path z->x.
        :param z_list:
        :param reconstruct:
        :return:
        """
        inp = None
        for i, block in enumerate(self.blocks[::-1]):  # NOT SURE WHAT IT DOES
            if i == 0:
                inp = block.reverse(output=z_list[-1], eps=z_list[-1], reconstruct=reconstruct)
            else:
                inp = block.reverse(output=inp, eps=z_list[-(i + 1)], reconstruct=reconstruct)
        return inp
