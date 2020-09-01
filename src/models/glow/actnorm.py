import torch
from torch import nn
from torch.autograd import grad, gradcheck
from torch.nn import functional as F

from .cond_net import ActCondNet, compute_batch_stats, apply_cond_net_generic

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, mode, const_memory, in_channel=None, cond_shape=None, inp_shape=None):
        super().__init__()
        assert (cond_shape is None or len(cond_shape) == 3) and (inp_shape is None or len(inp_shape) == 3), \
            'Inp shape (and cond shape) should have len 3 of form: (C, H, W)'
        self.mode = mode
        self.const_memory = const_memory

        if self.mode == 'conditional':
            self.cond_net = ActCondNet(cond_shape, inp_shape)
            self.params = self.cond_net.conv_net.get_params() + self.cond_net.linear_net.get_params()
        else:
            self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))  # this operation is done channel-wise
            self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))  # loc, scale: vectors applied to all channels
            self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
            self.params = (self.loc, self.scale)

    def possibly_initialize_params(self, inp):
        if self.mode == 'conditional':
            if self.cond_net.linear_net.net[-1].initialized.item() == 0:
                self.cond_net.linear_net.init_data_zero_bias(inp)
                # print('In [ActNormNoMemory]: (conditional) init bias with data zero for the first time forward called')
        else:
            if self.initialized.item() == 0:  # to be initialized the first time
                mean, std = compute_batch_stats(inp)
                self.loc.data.copy_(-mean)  # data dependent initialization
                self.scale.data.copy_(1 / (std + 1e-6))
                self.initialized.fill_(1)
                # print('In [ActNormNoMemory]: (unconditional) first time init of params done')

    def usual_forward(self, inp, condition):
        loc, scale = ActNormFunction.compute_loc_and_scale(condition, *self.params) if self.mode == 'conditional' else self.params
        output, logdet = ActNormFunction.forward_func(inp, loc, scale)
        return output, logdet

    def forward(self, inp, condition=None):
        self.possibly_initialize_params(inp)
        if self.const_memory:
            return ActNormFunction.apply(inp, condition, *self.params)
        return self.usual_forward(inp, condition)

    def check_grad(self, inp, condition=None):
        if self.const_memory:
            return gradcheck(func=ActNormFunction.apply, inputs=(inp, condition, *self.params), eps=1e-6)
        return gradcheck(func=self.usual_forward, inputs=(inp, condition), eps=1e-6)

    def reverse(self, output, condition):
        loc, scale = ActNormFunction.compute_loc_and_scale(condition, *self.params) if self.mode == 'conditional' else self.params
        return ActNormFunction.reverse_func(output, loc, scale)


class PairedActNorm(torch.nn.Module):
    def __init__(self, cond_shape, inp_shape):
        super().__init__()
        inp_channels = inp_shape[0]  # inp_shape is (C, H, W)
        self.left_actnorm = ActNorm(mode='unconditional', const_memory=True, in_channel=inp_channels, cond_shape=None, inp_shape=None)
        self.right_actnorm = ActNorm(mode='conditional', const_memory=True, in_channel=inp_channels, cond_shape=cond_shape, inp_shape=inp_shape)
        self.params = self.left_actnorm.params + self.right_actnorm.params

    def possibly_initialize_params(self, inp_left, inp_right):
        self.left_actnorm.possibly_initialize_params(inp_left)
        self.right_actnorm.possibly_initialize_params(inp_right)

    def forward(self, activations, inp_left, inp_right):
        self.possibly_initialize_params(inp_left, inp_right)
        return PairedActNormFunction.apply(activations, inp_left, inp_right, *self.params)

    def check_grads(self, inp_left, inp_right):
        out_left, _, out_right, _ = self.forward({}, inp_left, inp_right)
        activations = {'left': [out_left.data], 'right': [out_right.data]}
        return gradcheck(func=PairedActNormFunction.apply,
                         inputs=(activations, inp_left, inp_right, *self.params))


class PairedActNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, backprop_info, inp_left, inp_right, *params):
        with torch.no_grad():
            left_loc, left_scale, cond_net_params = PairedActNormFunction._extract_params(params)
            out_left, logdet_left = ActNormFunction.forward_func(inp_left, left_loc, left_scale)
            right_loc, right_scale = ActNormFunction.compute_loc_and_scale(out_left, *cond_net_params)
            out_right, logdet_right = ActNormFunction.forward_func(inp_right, right_loc, right_scale)

        # del left_loc, left_scale
        # del right_loc, right_scale

        ctx.save_for_backward(*params)
        ctx.backprop_info = backprop_info
        # ctx.out_left = out_left.data
        # ctx.out_right = out_right.data
        # return out_left, logdet_left, out_right, logdet_right
        return out_left, logdet_left, out_right, logdet_right

    @staticmethod
    def backward(ctx, grad_out_left, grad_logdet_left, grad_out_right, grad_logdet_right):
        params = ctx.saved_tensors
        backprop_info = ctx.backprop_info
        left_out, right_out = backprop_info['left_activations'].pop(), backprop_info['right_activations'].pop()
        # out_left, out_right = ctx.out_left, ctx.out_right
        left_loc, left_scale, cond_net_params = PairedActNormFunction._extract_params(params)

        # reconstruct left input
        with torch.no_grad():
            inp_left = ActNormFunction.reverse_func(left_out, left_loc, left_scale)
            inp_left.requires_grad = True
            backprop_info['left_activations'].append(inp_left.data)

        with torch.enable_grad():
            left_out, logdet_left = ActNormFunction.forward_func(inp_left, left_loc, left_scale)
            right_loc, right_scale = ActNormFunction.compute_loc_and_scale(left_out, *cond_net_params)

        # reconstruct right input
        with torch.no_grad():
            inp_right = ActNormFunction.reverse_func(right_out, right_loc, right_scale)
            inp_right.requires_grad = True
            backprop_info['right_activations'].append(inp_right.data)

        with torch.enable_grad():
            right_out, logdet_right = ActNormFunction.forward_func(inp_right, right_loc, right_scale)
            grad_inp_left = grad(outputs=left_out, inputs=inp_left, grad_outputs=grad_out_left, retain_graph=True)[0] + \
                            grad(outputs=right_out, inputs=inp_left, grad_outputs=grad_out_right, retain_graph=True)[0] + \
                            grad(outputs=logdet_right, inputs=inp_left, grad_outputs=grad_logdet_right, retain_graph=True)[0]
            grad_inp_right = grad(outputs=right_out, inputs=inp_right, grad_outputs=grad_out_right, retain_graph=True)[0]

            grad_loc_left = grad(outputs=left_out, inputs=left_loc, grad_outputs=grad_out_left, retain_graph=True)[0]
            grad_scale_left = grad(outputs=left_out, inputs=left_scale, grad_outputs=grad_out_left, retain_graph=True)[0] + \
                              grad(outputs=logdet_left, inputs=left_scale, grad_outputs=grad_logdet_left, retain_graph=True)[0]

            grad_cond_params_wrt_out_right = grad(outputs=right_out, inputs=cond_net_params, grad_outputs=grad_out_right, retain_graph=True)
            grad_cond_params_wrt_logdet_right = grad(outputs=logdet_right, inputs=cond_net_params, grad_outputs=grad_logdet_right, retain_graph=True)
            grad_cond_params = tuple([sum(x) for x in zip(grad_cond_params_wrt_out_right, grad_cond_params_wrt_logdet_right)])

            # left_out.detach_(), logdet_left.detach_()
            # right_out.detach_(), logdet_right.detach_(), right_loc.detach(), right_scale.detach()
            # del left_out, logdet_left
            # del right_out, logdet_right, right_loc, right_scale
            # del inp_left, inp_right

            return (None, grad_inp_left, grad_inp_right, grad_loc_left, grad_scale_left, *grad_cond_params)

    @staticmethod
    def _extract_params(params):
        left_loc, left_scale = params[:2]
        cond_net_params = params[2:]
        return left_loc, left_scale, cond_net_params


class ActNormFunction(torch.autograd.Function):
    @staticmethod
    def forward_func(inp, loc, scale):
        _, _, height, width = inp.shape  # input of shape [bsize, in_channel, h, w]
        logdet = height * width * torch.sum(logabs(scale))
        output = scale * (inp + loc)
        return output, logdet

    @staticmethod
    def reverse_func(output, loc, scale):
        return (output / scale) - loc

    @staticmethod
    def compute_loc_and_scale(cond_inp, *params):
        cond_out = apply_cond_net_generic(cond_inp, *params)
        out_channels = cond_out.shape[1]  # e.g. (1, 12)
        cond_out = cond_out.view(cond_out.shape[0], 2, out_channels // 2)  # 12 --> 6 x 2 - output shape: (B, 2, C)
        scale = cond_out[:, 0, :].unsqueeze(2).unsqueeze(3)  # s, t shape (B, C, 1, 1)
        loc = cond_out[:, 1, :].unsqueeze(2).unsqueeze(3)
        return loc, scale

    @staticmethod
    def forward(ctx, inp, cond_inp, *params):
        conditional = True if cond_inp is not None else False
        loc, scale = ActNormFunction.compute_loc_and_scale(cond_inp, *params) if conditional else params
        output, logdet = ActNormFunction.forward_func(inp, loc, scale)

        ctx.save_for_backward(*params)
        ctx.cond_inp = cond_inp
        ctx.output = output
        return output, logdet

    @staticmethod
    def backward(ctx, grad_output, grad_logdet):
        output = ctx.output
        cond_inp = ctx.cond_inp
        params = ctx.saved_tensors

        # retrieve scale and loc parameters
        conditional = True if cond_inp is not None else False
        if conditional:
            with torch.enable_grad():
                loc, scale = ActNormFunction.compute_loc_and_scale(cond_inp, *params)
                cond_out = torch.cat([scale, loc], dim=1)
        else:
            loc, scale = params

        # reconstruct input
        with torch.no_grad():
            reconstructed = ActNormFunction.reverse_func(output, loc, scale)
            reconstructed.requires_grad = True

        # creating computational graph and compute gradients
        with torch.enable_grad():
            output, logdet = ActNormFunction.forward_func(reconstructed, loc, scale)
            # compute grad for loc
            grad_loc = grad(outputs=output, inputs=loc, grad_outputs=grad_output, retain_graph=True)[0]
            # compute grad for scale
            grad_scale_output = grad(outputs=output, inputs=scale, grad_outputs=grad_output, retain_graph=True)[0]
            grad_scale_logdet = grad(outputs=logdet, inputs=scale, grad_outputs=grad_logdet, retain_graph=True)[0]
            grad_scale = grad_scale_output + grad_scale_logdet
            # grad wrt input
            grad_inp = grad(outputs=output, inputs=reconstructed, grad_outputs=grad_output, retain_graph=True)[0]

            if conditional:  # grad wrt cond_inp and params of cond net
                grad_cond_out = torch.cat([grad_scale, grad_loc], dim=1)
                grad_cond_inp = grad(outputs=cond_out, inputs=cond_inp, grad_outputs=grad_cond_out, retain_graph=True)[0]
                grad_params = grad(outputs=cond_out, inputs=params, grad_outputs=grad_cond_out, retain_graph=True)
            else:
                grad_cond_inp = None
                grad_params = (grad_loc, grad_scale)
            return (grad_inp, grad_cond_inp) + grad_params  # concat tuples
