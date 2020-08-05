import torch
from torch import nn

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

