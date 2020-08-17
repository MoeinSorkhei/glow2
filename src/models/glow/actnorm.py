import torch
from torch import nn
from torch.autograd import grad

from .cond_net import ActCondNet, compute_batch_stats

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
