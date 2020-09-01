import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad, gradcheck

from helper import label_to_tensor
from collections import OrderedDict
from .actnorm import *
from .conv1x1 import *
from .cond_net import CouplingCondNet
from .utils import *
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
        # self._parameters only gives the scale param since only scale is defined as nn.Parameter
        return tuple(self.conv._parameters.values()) + tuple(self._parameters.values())
        # return OrderedDict(
        #     list(self.conv._parameters.items()) +
        #     list(self._parameters.items())  # this only gives the scale param since only scale is defined as nn.Parameter
        # )

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
        self.const_memory = configs['const_memory']
        n_filters = configs['n_filter'] if 'n_filter' in configs.keys() else 512
        const_memory = configs['const_memory']  # could also use self.const_memory

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
    #
    # def forward(self, inp, act_cond, w_cond, coupling_cond):
    #     # if self.const_memory:
    #     params_lengths = (len(self.act_norm.params), len(self.inv_conv.params), len(self.coupling.params))
    #     return FlowFunction.apply(inp, act_cond, w_cond, coupling_cond, self.inv_conv.buffers, params_lengths, *self.params)
    #     # return FlowFunction.forward_func(inp, act_cond, w_cond, coupling_cond, self.inv_conv.buffers,
    #     #                                  self.act_norm.params, self.inv_conv.params, self.coupling.params)

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

        self.conv1_params = tuple(self.net[0]._parameters.values())
        self.conv2_params = tuple(self.net[2]._parameters.values())
        self.zero_conv_params = tuple(self.net[4].get_params())
        self.params = tuple(self.conv1_params + self.conv2_params + self.zero_conv_params)

        if use_cond_net:
            self.use_cond_net = True
            self.cond_net = CouplingCondNet(cond_shape, inp_shape)  # without considering batch size dimension
            self.cond_net_params = self.cond_net.get_params()
            self.params += self.cond_net_params
        else:
            self.use_cond_net = False
            self.cond_net_params = None

    def forward(self, inp, activations, condition=None):
        if self.const_memory:
            return CouplingFunction.apply(inp, activations, condition, *self.params)
        return CouplingFunction.func('forward', inp, condition, self.conv1_params, self.conv2_params, self.zero_conv_params, self.cond_net_params)

    def reverse(self, output, cond=None):
        return CouplingFunction.func('reverse', output, cond, self.conv1_params, self.conv2_params, self.zero_conv_params, self.cond_net_params)

    def check_grads(self, inp, condition=None):
        if self.const_memory:
            return gradcheck(func=CouplingFunction.apply, inputs=(inp, condition, *self.params), eps=1e-6)
        return gradcheck(func=self.forward, inputs=(inp, condition), eps=1e-6)  # only wrt inp and condition


class PairedFlow(torch.nn.Module):
    def __init__(self, cond_shape, inp_shape, configs):
        super().__init__()
        self.paired_actnorm = PairedActNorm(cond_shape, inp_shape)
        self.paired_inv_conv = PairedInvConv1x1(cond_shape, inp_shape)
        self.paired_coupling = PairedCoupling(cond_shape, inp_shape, configs['n_filters'])
        self.params = self.paired_actnorm.params + self.paired_inv_conv.params + self.paired_coupling.params
        self.buffers = self.paired_inv_conv.buffers

    def simple_forward_not_used(self, activations, inp_left, inp_right):
        left_out, left_act_logdet, right_out, right_act_logdet = self.paired_actnorm(activations, inp_left, inp_right)
        left_out, left_w_logdet, right_out, right_w_logdet = self.paired_inv_conv(activations, left_out, right_out)
        left_out, left_coupling_logdet, right_out, right_coupling_logdet = self.paired_coupling(activations, left_out, right_out)
        left_logdet = left_act_logdet + left_w_logdet + left_coupling_logdet
        right_logdet = right_act_logdet + right_w_logdet + right_coupling_logdet
        return left_out, left_logdet, right_out, right_logdet


class PairedCoupling(torch.nn.Module):
    def __init__(self, cond_shape, inp_shape, n_filters):
        super().__init__()
        self.left_coupling = AffineCoupling(inp_shape=inp_shape, cond_shape=None, n_filters=n_filters, const_memory=True, use_cond_net=False)
        self.right_coupling = AffineCoupling(inp_shape=inp_shape, cond_shape=cond_shape, n_filters=n_filters, const_memory=True, use_cond_net=True)
        self.params = self.left_coupling.params + self.right_coupling.params

    def forward(self, activations, inp_left, inp_right):
        return PairedCouplingFunction.apply(activations, inp_left, inp_right, *(self.left_coupling.params + self.right_coupling.params))

    def check_grads(self, inp_left, inp_right):
        out_left, _, out_right, _ = self.forward({}, inp_left, inp_right)
        activations = {'left': [out_left.data], 'right': [out_right.data]}
        return gradcheck(func=PairedCouplingFunction.apply,
                         inputs=(activations, inp_left, inp_right, *(self.left_coupling.params + self.right_coupling.params)))


class PairedCouplingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, backprop_info, inp_left, inp_right, *params):
        with torch.no_grad():
            left_params, right_params, cond_net_params = PairedCouplingFunction._extract_params(params, mode='paired')
            left_conv1_params, left_conv2_params, left_zero_conv_params = PairedCouplingFunction._extract_params(left_params, mode='single')
            left_out, left_logdet = CouplingFunction.func(direction='forward',
                                                          inp_or_out=inp_left,
                                                          cond_inp=None,
                                                          conv1_params=left_conv1_params,
                                                          conv2_params=left_conv2_params,
                                                          zero_conv_params=left_zero_conv_params,
                                                          cond_net_params=None)

            right_conv1_params, right_conv2_params, right_zero_conv_params = PairedCouplingFunction._extract_params(right_params, mode='single')
            right_out, right_logdet = CouplingFunction.func(direction='forward',
                                                            inp_or_out=inp_right,
                                                            cond_inp=left_out,
                                                            conv1_params=right_conv1_params,
                                                            conv2_params=right_conv2_params,
                                                            zero_conv_params=right_zero_conv_params,
                                                            cond_net_params=cond_net_params)

        ctx.save_for_backward(*params)
        ctx.backprop_info = backprop_info
        # ctx.left_out, ctx.right_out = left_out.data, right_out.data  # to be replaced by activations
        return left_out, left_logdet, right_out, right_logdet

    @staticmethod
    def backward(ctx, grad_out_left, grad_logdet_left, grad_out_right, grad_logdet_right):
        params = ctx.saved_tensors
        # left_out, right_out = ctx.left_out, ctx.right_out  # to be replaced by activations
        backprop_info = ctx.backprop_info
        left_out, right_out = backprop_info['left_activations'].pop(), backprop_info['right_activations'].pop()

        # unsqueeze output and concat with z_out if this coupling lies at the end of Block
        if backprop_info['current_i_flow'] in backprop_info['marginal_flows_inds']:
            with torch.no_grad():
                i_block = backprop_info['marginal_flows_inds'].index(backprop_info['current_i_flow'])
                z_out_left = backprop_info['z_outs_left'][i_block]
                z_out_right = backprop_info['z_outs_right'][i_block]
                # unsqueeze and concat
                left_out, right_out = unsqueeze_tensor(left_out), unsqueeze_tensor(right_out)
                left_out = torch.cat([left_out, z_out_left], dim=1)
                right_out = torch.cat([right_out, z_out_right], dim=1)

        # decrease flow index
        ctx.backprop_info['current_i_flow'] -= 1

        left_params, right_params, cond_net_params = PairedCouplingFunction._extract_params(params, mode='paired')
        left_conv1_params, left_conv2_params, left_zero_conv_params = PairedCouplingFunction._extract_params(left_params, mode='single')
        right_conv1_params, right_conv2_params, right_zero_conv_params = PairedCouplingFunction._extract_params(right_params, mode='single')

        left_out_a, left_out_b = left_out.chunk(chunks=2, dim=1)
        left_inp_a = left_out_a

        with torch.enable_grad():
            left_net_out = CouplingFunction.perform_convolutions(left_inp_a, left_conv1_params, left_conv2_params, left_zero_conv_params)

        # reconstruct left input
        with torch.no_grad():
            inp_left = CouplingFunction.operation('reverse', left_out, left_net_out)
            inp_left.requires_grad = True
            backprop_info['left_activations'].append(inp_left.data)

        right_out_a, right_out_b = right_out.chunk(chunks=2, dim=1)
        right_inp_a = right_out_a
        with torch.enable_grad():
            left_out, left_logdet = CouplingFunction.operation('forward', inp_left, left_net_out)
            cond_out = CouplingFunction.apply_cond_net(left_out, *cond_net_params)
            right_net_out = CouplingFunction.perform_convolutions(torch.cat([right_inp_a, cond_out], dim=1),
                                                                  right_conv1_params, right_conv2_params, right_zero_conv_params)
        # reconstruct right input
        with torch.no_grad():
            inp_right = CouplingFunction.operation('reverse', right_out, right_net_out)
            inp_right.requires_grad = True
            backprop_info['right_activations'].append(inp_right.data)

        with torch.enable_grad():
            right_out, right_logdet = CouplingFunction.operation('forward', inp_right, right_net_out)

            left_grads = grad(outputs=(left_out, left_logdet, right_out, right_logdet),
                              inputs=(inp_left, *left_params),
                              grad_outputs=(grad_out_left, grad_logdet_left, grad_out_right, grad_logdet_right), retain_graph=True)
            grad_inp_left = left_grads[0]
            grad_left_params = left_grads[1:]

            right_grads = grad(outputs=(right_out, right_logdet),
                               inputs=(inp_right, *right_params, *cond_net_params),
                               grad_outputs=(grad_out_right, grad_logdet_right), retain_graph=True)
            grad_inp_right = right_grads[0]
            grad_right_params = right_grads[1:1 + len(right_params)]
            grad_cond_params = right_grads[1 + len(right_params):]
            return (None, grad_inp_left, grad_inp_right, *(grad_left_params + grad_right_params + grad_cond_params))

    @staticmethod
    def _extract_params(params, mode):
        if mode == 'paired':
            left_params = params[:7]
            right_params = params[7:14]
            cond_net_params = params[14:]
            return left_params, right_params, cond_net_params
        else:  # 'single'
            conv1_params = params[:2]
            conv2_params = params[2:4]
            zero_conv_params = params[4:7]
            return conv1_params, conv2_params, zero_conv_params


class CouplingFunction(torch.autograd.Function):
    @staticmethod
    def func(direction, inp_or_out, cond_inp, conv1_params, conv2_params, zero_conv_params, cond_net_params):
        inp_a, inp_b = inp_or_out.chunk(chunks=2, dim=1)
        is_conditional = True if cond_inp is not None else False

        if is_conditional:
            cond_out = CouplingFunction.apply_cond_net(cond_inp, *cond_net_params)
            net_input = torch.cat(tensors=[inp_a, cond_out], dim=1)
        else:
            net_input = inp_a

        net_out = CouplingFunction.perform_convolutions(net_input, conv1_params, conv2_params, zero_conv_params)
        return CouplingFunction.operation(direction, inp_or_out, net_out)

    @staticmethod
    def operation(direction, inp_or_out, net_out):
        part_a, part_b = inp_or_out.chunk(chunks=2, dim=1)  # chunk along the channel dimension
        log_s, t = net_out.chunk(chunks=2, dim=1)
        s = torch.sigmoid(log_s + 2)

        if direction == 'forward':
            out_b = (part_b + t) * s
            log_det = torch.sum(torch.log(s).view(inp_or_out.shape[0], -1), 1)
            return torch.cat(tensors=[part_a, out_b], dim=1), log_det
        else:
            inp_b = (part_b / s) - t
            return torch.cat(tensors=[part_a, inp_b], dim=1)

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
    def apply_cond_net(cond_inp, *params):
        conv1_params, conv2_params = params[:2], params[2:]
        cond_out = F.conv2d(cond_inp, weight=conv1_params[0], bias=conv1_params[1], padding=1)
        cond_out = F.relu(cond_out, inplace=False)
        cond_out = F.conv2d(cond_out, weight=conv2_params[0], bias=conv2_params[1], padding=1)
        cond_out = F.relu(cond_out, inplace=False)
        return cond_out

    @staticmethod
    def forward(ctx, inp, activations, cond_inp, *params):
        conv1_params = params[:2]  # weight, bias
        conv2_params = params[2:4]  # weight, bias
        zero_conv_params = params[4:7]  # weight, bias, parameter

        is_conditional = True if cond_inp is not None else False
        with torch.no_grad():  # no_grad is probably not needed
            cond_net_params = params[7:] if is_conditional else None
            out, logdet = CouplingFunction.func('forward', inp, cond_inp, conv1_params, conv2_params, zero_conv_params, cond_net_params)

        ctx.conv1_params = conv1_params
        ctx.conv2_params = conv2_params
        ctx.zero_conv_params = zero_conv_params
        # if is_conditional:
        ctx.cond_net_params = cond_net_params
        ctx.cond_inp = cond_inp
        # ctx.output = out
        ctx.activations = activations
        return out, logdet

    @staticmethod
    def backward(ctx, grad_y, grad_logdet):
        # y = ctx.output
        activations = ctx.activations
        cond_inp = ctx.cond_inp
        conv1_params, conv2_params, zero_conv_params = ctx.conv1_params, ctx.conv2_params, ctx.zero_conv_params
        cond_net_params = ctx.cond_net_params
        is_conditional = True if cond_inp is not None else False

        y = activations.pop()

        y1, y2 = y.chunk(chunks=2, dim=1)
        dy1, dy2 = grad_y.chunk(chunks=2, dim=1)

        with torch.enable_grad():  # so it creates the graph for the NN() operation and possibly con net
            inp_a = y1
            inp_a.requires_grad = True

            if is_conditional:
                cond_out = CouplingFunction.apply_cond_net(cond_inp, *cond_net_params)
                net_input = torch.cat(tensors=[inp_a, cond_out], dim=1)
            else:
                net_input = inp_a

            net_out = CouplingFunction.perform_convolutions(net_input, conv1_params, conv2_params, zero_conv_params)
            fx1, gx1 = net_out.chunk(chunks=2, dim=1)

        with torch.no_grad():  # no grad for reconstructing input
            _, inp_b = CouplingFunction.operation('reverse', y, net_out).chunk(chunks=2, dim=1)
            inp_b.requires_grad = True

            reconstructed = torch.cat([inp_a, inp_b], dim=1)
            activations.append(reconstructed.data)

            s = torch.sigmoid(fx1 + 2)
            d_sig_fx1 = torch.sigmoid(fx1 + 2) * (1 - torch.sigmoid(fx1 + 2))  # derivative of sigmoid

        with torch.enable_grad():  # compute grads
            CouplingFunction.operation('forward', torch.cat([inp_a, inp_b], dim=1), torch.cat([fx1, gx1], dim=1))  # re-create computational graph
            dg = s * dy2
            dx2 = s * dy2
            ds_part1 = (1 / s) * grad_logdet  # grad for s coming determinant
            ds_part2 = (inp_b + gx1) * dy2  # grad for s coming from y2
            df = d_sig_fx1 * (ds_part1 + ds_part2)

            # concat f and g grads and compute grads for input and net params
            dfg = torch.cat([df, dg], dim=1)
            dx1 = dy1 + grad(outputs=net_out, inputs=inp_a, grad_outputs=dfg, retain_graph=True)[0]
            dwfg = grad(outputs=net_out, inputs=tuple(conv1_params + conv2_params + zero_conv_params), grad_outputs=dfg, retain_graph=True)

            # final gradients
            grad_x = torch.cat([dx1, dx2], dim=1)
            grad_params = dwfg

            if is_conditional:
                grad_cond_inp = grad(outputs=net_out, inputs=cond_inp, grad_outputs=dfg, retain_graph=True)[0]
                grad_cond_net_params = grad(outputs=net_out, inputs=cond_net_params, grad_outputs=dfg, retain_graph=True)
                grad_params += grad_cond_net_params
            else:
                grad_cond_inp = None

        return (grad_x, None, grad_cond_inp) + grad_params  # append to tuple
