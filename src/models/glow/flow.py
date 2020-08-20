import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad, gradcheck

from helper import label_to_tensor
from collections import OrderedDict
from .actnorm import *
from .conv1x1 import *
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

    def get_params(self):
        return OrderedDict(
            list(self.conv._parameters.items()) +
            list(self._parameters.items())  # this only gives the scale param since only scale is defined as nn.Parameter
        )

    def forward(self, inp):
        # padding with additional 1 in each side to keep the spatial dimension unchanged after the convolution operation
        out = F.pad(input=inp, pad=[1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return out


class Flow(nn.Module):
    def __init__(self, inp_shape, cond_shape, configs):
        super().__init__()
        self.all_conditional = configs['all_conditional']
        n_filters = configs['n_filter'] if 'n_filter' in configs.keys() else 512
        const_memory = configs['const_memory']

        if self.all_conditional:
            self.act_norm = ActNorm(mode='conditional', const_memory=const_memory, cond_shape=cond_shape, inp_shape=inp_shape)
            self.inv_conv = InvConv1x1LU(in_channel=inp_shape[0], mode='conditional', const_memory=const_memory, cond_shape=cond_shape, inp_shape=inp_shape)
            self.coupling = AffineCoupling(cond_shape, inp_shape, const_memory=const_memory, n_filters=n_filters, use_cond_net=True)
        else:
            self.act_norm = ActNorm(mode='unconditional', const_memory=const_memory, in_channel=inp_shape[0])
            self.inv_conv = InvConv1x1LU(in_channel=inp_shape[0], mode='unconditional', const_memory=const_memory)  # always LU
            self.coupling = AffineCoupling(cond_shape=None, inp_shape=inp_shape, const_memory=const_memory, n_filters=n_filters, use_cond_net=False)

        self.params = self.act_norm.params + self.inv_conv.params + self.coupling.params  # tuple containing all params

    def forward(self, inp, act_cond, w_cond, coupling_cond):
        actnorm_out, act_logdet = self.act_norm(inp, act_cond)
        w_out, w_logdet = self.inv_conv(actnorm_out, w_cond)
        coupling_out, coupling_logdet = self.coupling(w_out, coupling_cond)
        log_det = act_logdet + w_logdet + coupling_logdet
        return actnorm_out, w_out, coupling_out, log_det

    def reverse(self, out, act_cond, w_cond, coupling_cond):
        inp = self.coupling.reverse(out, coupling_cond)
        inp = self.inv_conv.reverse(inp, w_cond)
        inp = self.act_norm.reverse(inp, act_cond)
        return inp

    def check_grad(self, inp, act_cond=None, w_cond=None, coupling_cond=None):
        return gradcheck(func=self.forward, inputs=(inp, act_cond, w_cond, coupling_cond), eps=1e-6)


class AffineCoupling(nn.Module):
    def __init__(self, cond_shape, inp_shape, n_filters, const_memory, use_cond_net):
        super().__init__()
        assert len(inp_shape) == 3 and (cond_shape is None or len(cond_shape) == 3), 'Inp shape (and cond shape) should have len 3 of form: (C, H, W)'

        self.const_memory = const_memory
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

        self.params = tuple(self.net[0]._parameters.values()) + \
                      tuple(self.net[2]._parameters.values()) + \
                      tuple(self.net[4].get_params().values())

        if use_cond_net:
            self.use_cond_net = True
            self.cond_net = CouplingCondNet(cond_shape, inp_shape)  # without considering batch size dimension
            self.params += self.cond_net.get_params()
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

    def usual_forward(self, inp, cond=None):
        inp_a, inp_b = inp.chunk(chunks=2, dim=1)  # chunk along the channel dimension
        s, t = self.compute_coupling_params(inp_a, cond)
        out_b = (inp_b + t) * s
        log_det = torch.sum(torch.log(s).view(inp.shape[0], -1), 1)
        return torch.cat(tensors=[inp_a, out_b], dim=1), log_det

    def forward(self, inp, condition=None):
        if self.const_memory:
            return CouplingFunction.apply(inp, condition, *self.params)
        return self.usual_forward(inp, condition)

    def reverse(self, output, cond=None):
        out_a, out_b = output.chunk(chunks=2, dim=1)  # here we know that out_a = inp_a (see the forward fn)
        s, t = self.compute_coupling_params(out_a, cond)
        inp_b = (out_b / s) - t
        return torch.cat(tensors=[out_a, inp_b], dim=1)

    def check_grads(self, inp, condition=None):
        if self.const_memory:
            return gradcheck(func=CouplingFunction.apply, inputs=(inp, condition, *self.params), eps=1e-6)
        return gradcheck(func=self.usual_forward, inputs=(inp, condition), eps=1e-6)  # only wrt inp and condition


class CouplingFunction(torch.autograd.Function):
    @staticmethod
    def perform_convolutions(x1, conv1_params, conv2_params, zero_conv_params):
        # sequential net operations
        out = F.conv2d(x1, weight=conv1_params[0], bias=conv1_params[1], padding=1)
        out = F.relu(out, inplace=False)
        out = F.conv2d(out, weight=conv2_params[0], bias=conv2_params[1])
        out = F.relu(out, inplace=False)
        # zero init conv
        out = F.pad(out, pad=[1, 1, 1, 1], value=1)
        out = F.conv2d(out, weight=zero_conv_params[0], bias=zero_conv_params[1])
        out = out * torch.exp(zero_conv_params[2] * 3)
        return out

    @staticmethod
    def forward_func(x1, x2, fx1, gx1):
        y1 = x1
        s = torch.sigmoid(fx1 + 2)
        y2 = s * (x2 + gx1)
        logdet = torch.sum(torch.log(s).view(x1.shape[0], -1), 1)  # shape[0]: batch size
        return y1, y2, logdet

    @staticmethod
    def reverse_func(y1, y2, fx1, gx1):
        x1 = y1
        s = torch.sigmoid(fx1 + 2)
        x2 = (y2 / s) - gx1
        return x1, x2

    @staticmethod
    def apply_cond_net(cond_inp, *params):
        conv1_params, conv2_params = params[:2], params[2:]
        cond_out = F.conv2d(cond_inp, weight=conv1_params[0], bias=conv1_params[1], padding=1)
        cond_out = F.relu(cond_out, inplace=False)
        cond_out = F.conv2d(cond_out, weight=conv2_params[0], bias=conv2_params[1], padding=1)
        cond_out = F.relu(cond_out, inplace=False)
        return cond_out

    @staticmethod
    def forward(ctx, x, cond_inp, *params):
        conv1_params = params[:2]  # weight, bias
        conv2_params = params[2:4]  # weight, bias
        zero_conv_params = params[4:7]  # weight, bias, parameter

        conditional = True if cond_inp is not None else False
        if conditional:
            cond_net_params = params[7:]  # length 4: two weights and two biases for 2 convolutions
            cond_out = CouplingFunction.apply_cond_net(cond_inp, *cond_net_params)

        with torch.no_grad():
            x1, x2 = x.chunk(chunks=2, dim=1)
            net_input = torch.cat(tensors=[x1, cond_out], dim=1) if conditional else x1
            fx1, gx1 = CouplingFunction.perform_convolutions(net_input, conv1_params, conv2_params, zero_conv_params).chunk(chunks=2, dim=1)
            y1, y2, logdet = CouplingFunction.forward_func(x1, x2, fx1, gx1)
            y = torch.cat(tensors=[y1, y2], dim=1)

        ctx.conv1_params = conv1_params
        ctx.conv2_params = conv2_params
        ctx.zero_conv_params = zero_conv_params
        if conditional:
            ctx.cond_net_params = cond_net_params
        ctx.cond_inp = cond_inp
        ctx.output = y

        return y, logdet

    @staticmethod
    def backward(ctx, grad_y, grad_logdet):
        y = ctx.output
        cond_inp = ctx.cond_inp
        conv1_params, conv2_params, zero_conv_params = ctx.conv1_params, ctx.conv2_params, ctx.zero_conv_params
        conditional = True if cond_inp is not None else False

        if conditional:
            cond_net_params = ctx.cond_net_params

        y1, y2 = y.chunk(chunks=2, dim=1)
        dy1, dy2 = grad_y.chunk(chunks=2, dim=1)

        with torch.enable_grad():  # so it creates the graph for the NN() operation and possibly con net
            x1 = y1
            x1.requires_grad = True
            if conditional:
                cond_out = CouplingFunction.apply_cond_net(cond_inp, *cond_net_params)

            net_input = torch.cat(tensors=[x1, cond_out], dim=1) if conditional else x1
            fgx1 = CouplingFunction.perform_convolutions(net_input, conv1_params, conv2_params, zero_conv_params)
            fx1, gx1 = fgx1.chunk(chunks=2, dim=1)

        with torch.no_grad():  # no grad for reconstructing input
            _, x2 = CouplingFunction.reverse_func(y1, y2, fx1, gx1)  # reconstruct input
            x2.requires_grad = True
            s = torch.sigmoid(fx1 + 2)
            d_sig_fx1 = torch.sigmoid(fx1 + 2) * (1 - torch.sigmoid(fx1 + 2))  # derivative of sigmoid

        with torch.enable_grad():  # compute grads
            CouplingFunction.forward_func(x1, x2, fx1, gx1)  # re-create computational graph
            dg = s * dy2
            dx2 = s * dy2
            ds_part1 = (1 / s) * grad_logdet  # grad for s coming determinant
            ds_part2 = (x2 + gx1) * dy2  # grad for s coming from y2
            df = d_sig_fx1 * (ds_part1 + ds_part2)

            # concat f and g grads and compute grads for input and net params
            dfg = torch.cat([df, dg], dim=1)
            dx1 = dy1 + grad(outputs=fgx1, inputs=x1, grad_outputs=dfg, retain_graph=True)[0]
            dwfg = grad(outputs=fgx1, inputs=tuple(conv1_params + conv2_params + zero_conv_params), grad_outputs=dfg, retain_graph=True)

            # final gradients
            grad_x = torch.cat([dx1, dx2], dim=1)
            grad_params = dwfg

            if conditional:
                grad_cond_inp = grad(outputs=fgx1, inputs=cond_inp, grad_outputs=dfg, retain_graph=True)[0]
                grad_cond_net_params = grad(outputs=fgx1, inputs=cond_net_params, grad_outputs=dfg, retain_graph=True)
                grad_params += grad_cond_net_params
            else:
                grad_cond_inp = None

        return (grad_x, grad_cond_inp) + grad_params  # append to tuple
