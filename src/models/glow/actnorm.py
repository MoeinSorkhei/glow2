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
