import torch
from torch import nn
from torch.nn import functional as F

from helper import label_to_tensor
from .actnorm import ActNorm, ActNormConditional
from .conv1x1 import InvConv1x1Unconditional, InvConv1x1LU, InvConv1x1Conditional, InvConv1x1LU
# from ..cond_net import CouplingCondNet
from .cond_net import CouplingCondNet
from globals import device


class ZeroInitConv2d(nn.Module):
    """
    To be used in the Affine Coupling step:
    The last convolution of each NN(), according to the paper is initialized with zeros, such that the each affine layer
    initially performs an identity function.
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, inp):
        # padding with additional 1 in each side to keep the spatial dimension unchanged after the convolution operation
        out = F.pad(input=inp, pad=[1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    def __init__(self, cond_shape, inp_shape, n_filters=512, use_cond_net=False):
        super().__init__()

        # currently cond net outputs have the same channels as input_channels
        in_channels = inp_shape[0]  # input from its own Glow - shape (C, H, W)
        extra_channels = in_channels if cond_shape is not None else 0  # no condition if con_shape is None
        conv_channels = in_channels // 2 + extra_channels  # channels: half of input tensor + extra channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=conv_channels, out_channels=n_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=1),
            nn.ReLU(inplace=True),
            ZeroInitConv2d(in_channel=n_filters, out_channel=in_channels)  # channels dimension same as input
        )

        # Initializing the params
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

        if use_cond_net:  # uses inp shape only if cond net is used
            self.use_cond_net = True
            self.cond_net = CouplingCondNet(cond_shape, inp_shape)  # without considering batch size dimension
        else:
            self.use_cond_net = False

    def compute_coupling_params(self, tensor, cond):
        if cond is not None:  # conditional
            cond_tensor = self.cond_net(cond) if self.use_cond_net else cond
            inp_a_conditional = torch.cat(tensors=[tensor, cond_tensor], dim=1)  # concat channel-wise
            log_s, t = self.net(inp_a_conditional).chunk(chunks=2, dim=1)
        else:
            log_s, t = self.net(tensor).chunk(chunks=2, dim=1)
        s = torch.sigmoid(log_s + 2)
        return s, t

    def forward(self, inp, cond=None):
        inp_a, inp_b = inp.chunk(chunks=2, dim=1)  # chunk along the channel dimension
        s, t = self.compute_coupling_params(inp_a, cond)
        out_b = (inp_b + t) * s
        log_det = torch.sum(torch.log(s).view(inp.shape[0], -1), 1)
        return torch.cat(tensors=[inp_a, out_b], dim=1), log_det

    def reverse(self, output, cond=None):
        out_a, out_b = output.chunk(chunks=2, dim=1)  # here we know that out_a = inp_a (see the forward fn)
        s, t = self.compute_coupling_params(out_a, cond)
        inp_b = (out_b / s) - t
        return torch.cat(tensors=[out_a, inp_b], dim=1)


class Flow(nn.Module):
    """
    The Flow module does not change the dimensionality of its input.
    """
    def __init__(self, inp_shape, cond_shape, configs):
        super().__init__()
        # now the output of cond nets has the same dimensions as inp_shape
        self.actnorm_has_cond_net, self.w_has_cond_net, \
            self.coupling_has_cond_net = [True, True, True] if configs['all_conditional'] else [False, False, False]

        self.act_norm = ActNormConditional(cond_shape, inp_shape) \
            if self.actnorm_has_cond_net else ActNorm(in_channel=inp_shape[0])

        if configs['do_lu']:
            self.inv_conv = InvConv1x1LU(in_channel=inp_shape[0], mode='conditional', cond_shape=cond_shape, inp_shape=inp_shape) \
                if self.w_has_cond_net else InvConv1x1LU(in_channel=inp_shape[0], mode='unconditional')
        else:
            self.inv_conv = InvConv1x1Conditional(cond_shape, inp_shape) if self.w_has_cond_net else InvConv1x1Unconditional(in_channel=inp_shape[0])

        self.coupling = AffineCoupling(cond_shape=cond_shape, inp_shape=inp_shape, use_cond_net=True) \
            if self.coupling_has_cond_net else AffineCoupling(cond_shape=cond_shape, inp_shape=inp_shape, use_cond_net=False)

    def forward(self, inp, act_cond, w_cond, coupling_cond, dummy_tensor=None):
        actnorm_out, act_logdet = self.act_norm(inp, act_cond) if self.actnorm_has_cond_net else self.act_norm(inp)
        w_out, w_logdet = self.inv_conv(actnorm_out, w_cond) if self.w_has_cond_net else self.inv_conv(actnorm_out)
        out, coupling_logdet = self.coupling(w_out, coupling_cond)
        log_det = act_logdet + w_logdet + coupling_logdet

        return actnorm_out, w_out, out, log_det

    def reverse(self, output, conditions):
        coupling_inp = self.coupling.reverse(output, cond=conditions['coupling_cond'])
        w_inp = self.inv_conv.reverse(coupling_inp, conditions['w_cond']) if self.w_has_cond_net else self.inv_conv.reverse(coupling_inp)
        inp = self.act_norm.reverse(w_inp, conditions['act_cond']) if self.actnorm_has_cond_net else self.act_norm.reverse(w_inp)
        return inp
