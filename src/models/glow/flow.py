import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad

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


class AffineCouplingNoMemory(nn.Module):
    def __init__(self, cond_shape, inp_shape, n_filters=512, use_cond_net=False):
        super().__init__()
        # currently cond net outputs have the same channels as input_channels
        in_channels = inp_shape[0]  # input from its own Glow - shape (C, H, W)
        extra_channels = in_channels if cond_shape is not None else 0  # no condition if con_shape is None
        conv_channels = in_channels // 2 + extra_channels  # channels: half of input tensor + extra channels

        # define net params
        self.register_parameter('conv1_weight', nn.Parameter(torch.Tensor(n_filters, conv_channels, 3, 3)))  # conv2d has shape (out_channels, in_channels, kH, kW)
        self.register_parameter('conv1_bias', nn.Parameter(torch.Tensor(n_filters)))  # conv2d bias has shape (out_channels)
        self.register_parameter('conv2_weight', nn.Parameter(torch.Tensor(n_filters, n_filters, 1, 1)))
        self.register_parameter('conv2_bias', nn.Parameter(torch.Tensor(n_filters)))
        self.register_parameter('zero_conv_weight', nn.Parameter(torch.Tensor(in_channels, n_filters, 3, 3)))
        self.register_parameter('zero_conv_bias', nn.Parameter(torch.Tensor(in_channels)))
        self.register_parameter('zero_conv_scale', nn.Parameter(torch.Tensor(1, in_channels, 1, 1)))

        self.init_params()

    def init_params(self):
        self._parameters['conv1_weight'].data.normal_(0, 0.05)
        self._parameters['conv1_bias'].data.zero_()
        self._parameters['conv2_weight'].data.normal_(0, 0.05)
        self._parameters['conv2_bias'].data.zero_()
        self._parameters['zero_conv_weight'].data.zero_()
        self._parameters['zero_conv_bias'].data.zero_()
        self._parameters['zero_conv_scale'].data.zero_()

    def print_params(self):
        print('Parameters are:')
        for name, param in self.named_parameters():
            print('name:', name)
            print('shape:', param.data.shape, '\n')
        print('waiting for input')
        input()

    def forward(self, inp):
        # self.print_params()
        return CouplingFunction.apply(inp, *self._parameters.values())


class CouplingFunction(torch.autograd.Function):
    @staticmethod
    def perform_convolutions(x1, conv1_params, conv2_params, zero_conv_params):
        # sequential net operations
        out = F.conv2d(x1, weight=conv1_params[0], bias=conv1_params[1], padding=1)
        out = F.relu(out, inplace=True)
        out = F.conv2d(out, weight=conv2_params[0], bias=conv2_params[1])
        out = F.relu(out, inplace=True)
        # zero init conv
        out = F.pad(out, pad=[1, 1, 1, 1], value=1)
        out = F.conv2d(out, weight=zero_conv_params[0], bias=zero_conv_params[1])
        out = out * torch.exp(zero_conv_params[2] * 3)
        return out

    @staticmethod
    def forward_func(x1, x2, fx1, gx1):
        y1 = x1
        y2 = (torch.exp(fx1) * x2) + gx1
        # logdet = torch.sum(torch.log(fx1).view(x1.shape[0], -1), 1)  # shape[0]: batch size
        return y1, y2

    @staticmethod
    def reverse_func(y1, y2, fx1, gx1):
        x1 = y1
        x2 = (y2 - gx1) / torch.exp(fx1)
        return x1, x2

    @staticmethod
    def f(x):
        return x ** 2

    @staticmethod
    def g(x):
        # return 2 * x
        return 0 * x

    @staticmethod
    def forward(ctx, x, *args):
        # print('In [forward]l: len params:', len(args))  # should be 7
        net_params = list(args)
        conv1_params = net_params[:2]  # weight, bias
        conv2_params = net_params[2:4]  # weight, bias
        zero_conv_params = net_params[4:]  # weight, bias, parameter

        with torch.no_grad():
            x1, x2 = x.chunk(chunks=2, dim=1)
            fx1, _ = CouplingFunction.perform_convolutions(x1, conv1_params, conv2_params, zero_conv_params).chunk(chunks=2, dim=1)
            gx1 = CouplingFunction.g(x1)  # zero
            y1, y2 = CouplingFunction.forward_func(x1, x2, fx1, gx1)
            y = torch.cat(tensors=[y1, y2], dim=1)

        ctx.conv1_params = conv1_params
        ctx.conv2_params = conv2_params
        ctx.zero_conv_params = zero_conv_params
        ctx.output = y
        return y

    @staticmethod
    def backward(ctx, grad_y):
        y = ctx.output
        conv1_params, conv2_params, zero_conv_params = ctx.conv1_params, ctx.conv2_params, ctx.zero_conv_params

        # with torch.no_grad():
        y1, y2 = y.chunk(chunks=2, dim=1)
        dy1, dy2 = grad_y.chunk(chunks=2, dim=1)

        with torch.enable_grad():
            x1 = y1
            x1.requires_grad = True
            # fx1, gx1 = CouplingFunction.f(x1), CouplingFunction.g(x1)
            fx1, _ = CouplingFunction.perform_convolutions(x1, conv1_params, conv2_params, zero_conv_params).chunk(chunks=2, dim=1)
            gx1 = CouplingFunction.g(x1)  # zero

        with torch.no_grad():
            _, x2 = CouplingFunction.reverse_func(y1, y2, fx1, gx1)  # reconstruct input
            x2.requires_grad = True
            exp_fx1 = torch.exp(fx1)

        with torch.enable_grad():
            y1, y2 = CouplingFunction.forward_func(x1, x2, fx1, gx1)  # re-create computational graph
            # compute grads
            dg = dy2
            # df = (exp_fx1 * x2 * dy2) + 1
            df = (exp_fx1 * x2 * dy2)

            dx1 = dy1 + grad(outputs=gx1, inputs=x1, grad_outputs=dg, retain_graph=True)[0] \
                      + grad(outputs=fx1, inputs=x1, grad_outputs=df, retain_graph=True)[0]
            dx2 = exp_fx1 * dy2
            dwg = 0
            dwf = grad(outputs=fx1, inputs=tuple(conv1_params + conv2_params + zero_conv_params), grad_outputs=df, retain_graph=True)
            grad_x = torch.cat([dx1, dx2], dim=1)
        # return tuple([None] * 8)
        grad_params = dwf
        return (grad_x,) + grad_params  # append to tuple


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
