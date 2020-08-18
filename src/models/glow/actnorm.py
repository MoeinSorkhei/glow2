import torch
from torch import nn
from torch.autograd import grad, gradcheck
from torch.nn import functional as F

from .cond_net import ActCondNet, compute_batch_stats, apply_cond_net_generic

logabs = lambda x: torch.log(torch.abs(x))


class ActNormConditional(nn.Module):
    def __init__(self, cond_shape, inp_shape):
        super().__init__()
        self.cond_net = ActCondNet(cond_shape, inp_shape)

        print_params = False
        if print_params:
            total_params = sum(p.numel() for p in self.cond_net.parameters())
            print('ActNormConditional CondNet params:', total_params)

    def forward(self, inp, condition):
        cond_out = self.cond_net(condition, inp)  # output shape (B, 2, C)
        s = cond_out[:, 0, :].unsqueeze(2).unsqueeze(3)  # s, t shape (B, C, 1, 1)
        t = cond_out[:, 1, :].unsqueeze(2).unsqueeze(3)

        # computing log determinant
        _, _, height, width = inp.shape  # input of shape [bsize, in_channel, h, w]
        scale_logabs = logabs(s).mean(dim=0, keepdim=True)  # mean over batch - shape: (1, C, 1, 1)
        log_det = height * width * torch.sum(scale_logabs)  # scalar value
        return s * (inp + t), log_det

    def reverse(self, out, condition):
        cond_out = self.cond_net(condition)  # output shape (B, 2, C)
        s = cond_out[:, 0, :].unsqueeze(2).unsqueeze(3)  # s, t shape (B, C, 1, 1)
        t = cond_out[:, 1, :].unsqueeze(2).unsqueeze(3)
        return (out / s) - t


class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))  # this operation is done channel-wise
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))  # loc, scale: vectors applied to all channels
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, inp):
        mean, std = compute_batch_stats(inp)
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


class ActNormNoMemory(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))  # this operation is done channel-wise
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))  # loc, scale: vectors applied to all channels
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, inp):
        mean, std = compute_batch_stats(inp)
        # data dependent initialization
        self.loc.data.copy_(-mean)
        self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, inp):
        # data-dependent initialization of scale and shift
        if self.initialized.item() == 0:  # to be initialized the first time
            self.initialize(inp)
            self.initialized.fill_(1)
        # forward without storing activations
        return ActNormFunction.apply(inp, self.loc, self.scale)

    def reverse(self, out):
        return (out / self.scale) - self.loc


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
    def forward(ctx, inp, loc, scale):
        with torch.no_grad():  # compute output without forming computational graph
            output, logdet = ActNormFunction.forward_func(inp, loc, scale)

        ctx.save_for_backward(loc, scale)
        ctx.output = output
        return output, logdet

    @staticmethod
    def backward(ctx, grad_output, grad_logdet):
        loc, scale = ctx.saved_tensors
        output = ctx.output

        with torch.no_grad():  # apply the inverse of the operation
            reconstructed = ActNormFunction.reverse_func(output, loc, scale)
            reconstructed.requires_grad = True  # so we can compute grad for this

        with torch.enable_grad():  # creating computational graph and compute gradients
            output, logdet = ActNormFunction.forward_func(reconstructed, loc, scale)
            # compute grad for loc
            grad_loc = grad(outputs=output, inputs=loc, grad_outputs=grad_output, retain_graph=True)[0]
            # compute grad for scale
            grad_scale_output = grad(outputs=output, inputs=scale, grad_outputs=grad_output, retain_graph=True)[0]
            grad_scale_logdet = grad(outputs=logdet, inputs=scale, grad_outputs=grad_logdet, retain_graph=True)[0]
            grad_scale = grad_scale_output + grad_scale_logdet
            # compute grad for inp
            grad_inp = grad(outputs=output, inputs=reconstructed, grad_outputs=grad_output, retain_graph=True)[0]
        return grad_inp, grad_loc, grad_scale


class ActNormConditionalNoMemory(nn.Module):
    def __init__(self, cond_shape, inp_shape):
        super().__init__()
        self.cond_net = ActCondNet(cond_shape, inp_shape)
        self.params = self.cond_net.conv_net.get_params() + self.cond_net.linear_net.get_params()

    def forward(self, inp, condition):
        # data-dependent init of bias
        if self.cond_net.linear_net.net[-1].initialized.item() == 0:
            self.cond_net.linear_net.init_data_zero_bias(inp)
            print('In [ActNormConditionalNoMemory]: init bias with data zero for the first time forward called')
        # forward operation
        ActnormConditionalFunction.apply(inp, condition, *self.params)

    def check_grad(self, inp, condition):
        return gradcheck(func=ActnormConditionalFunction.apply, inputs=(inp, condition, *self.params))


class ActnormConditionalFunction(torch.autograd.Function):
    @staticmethod
    def forward_func(inp, loc, scale):
        return ActNormFunction.forward_func(inp, loc, scale)

    @staticmethod
    def reverse_func(output, loc, scale):
        return ActNormFunction.reverse_func(output, loc, scale)

    @staticmethod
    def apply_cond_net(cond_inp, *params):
        cond_out = apply_cond_net_generic(cond_inp, *params)
        out_channels = cond_out.shape[1]  # e.g. (1, 12)
        cond_out = cond_out.view(cond_out.shape[0], 2, out_channels // 2)  # 12 --> 6 x 2 - output shape: (B, 2, C)
        scale = cond_out[:, 0, :].unsqueeze(2).unsqueeze(3)  # s, t shape (B, C, 1, 1)
        loc = cond_out[:, 1, :].unsqueeze(2).unsqueeze(3)
        return scale, loc

    @staticmethod
    def forward(ctx, inp, cond_inp, *params):
        with torch.no_grad():
            scale, loc = ActnormConditionalFunction.apply_cond_net(cond_inp, *params)
            output, logdet = ActnormConditionalFunction.forward_func(inp, loc, scale)

        # save items for backward
        ctx.save_for_backward(*params)
        ctx.output = output
        ctx.cond_inp = cond_inp
        return output, logdet

    @staticmethod
    def backward(ctx, grad_output, grad_logdet):
        params = ctx.saved_tensors
        output = ctx.output
        cond_inp = ctx.cond_inp
        with torch.enable_grad():
            scale, loc = ActnormConditionalFunction.apply_cond_net(cond_inp, *params)
            cond_out = torch.cat([scale, loc], dim=1)

        with torch.no_grad():
            reconstructed = ActnormConditionalFunction.reverse_func(output, loc, scale)
            reconstructed.requires_grad = True

        with torch.enable_grad():
            output, logdet = ActnormConditionalFunction.forward_func(reconstructed, loc, scale)
            # grad w.e.t input
            grad_inp = grad(outputs=output, inputs=reconstructed, grad_outputs=grad_output, retain_graph=True)[0]

            # grad wrt scale and loc
            grad_loc = grad(outputs=output, inputs=loc, grad_outputs=grad_output, retain_graph=True)[0]
            grad_scale_output = grad(outputs=output, inputs=scale, grad_outputs=grad_output, retain_graph=True)[0]
            grad_scale_logdet = grad(outputs=logdet, inputs=scale, grad_outputs=grad_logdet, retain_graph=True)[0]
            grad_scale = grad_scale_output + grad_scale_logdet

            # grad wrt cond_inp and params of cond net
            grad_cond_out = torch.cat([grad_scale, grad_loc], dim=1)
            grad_cond_inp = grad(outputs=cond_out, inputs=cond_inp, grad_outputs=grad_cond_out, retain_graph=True)[0]
            grad_params = grad(outputs=cond_out, inputs=params, grad_outputs=grad_cond_out, retain_graph=True)
            return (grad_inp, grad_cond_inp) + grad_params  # concat tuples
