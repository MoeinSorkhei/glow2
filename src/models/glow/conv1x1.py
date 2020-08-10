import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import linalg as la

from .cond_net import WCondNet


logabs = lambda x: torch.log(torch.abs(x))


# non-LU unconditional
class InvConv1x1Unconditional(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        q, _ = torch.qr(torch.randn(in_channel, in_channel))
        # making it 1x1 conv: conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1)
        w = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(w)  # the weight matrix

    def forward(self, inp):
        _, _, height, width = inp.shape
        out = F.conv2d(inp, self.weight)

        log_w = torch.slogdet(self.weight.squeeze().double())[1].float()
        log_det = height * width * log_w
        return out, log_det

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


# non-LU conditional
class InvConv1x1Conditional(nn.Module):
    def __init__(self, cond_shape, inp_shape):
        super().__init__()
        self.cond_net = WCondNet(cond_shape, inp_shape)  # initialized with QR decomposition

        print_params = False
        if print_params:
            total_params = sum(p.numel() for p in self.cond_net.parameters())
            print('ActNormConditional CondNet params:', total_params)

    def forward(self, inp, condition):
        """
        F.conv2d doc: https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.conv2d
        :param inp:
        :param condition:
        :return:
        """
        _, _, height, width = inp.shape
        cond_net_out = self.cond_net(condition)  # shape (B, C, C)
        batch_size = inp.shape[0]
        log_w = 0
        output = []

        # convolve every batch item with its corresponding W
        for i in range(batch_size):
            corresponding_inp = inp[i].unsqueeze(0)  # re-adding batch dim - shape (1, C, H, W)
            corresponding_w = cond_net_out[i].unsqueeze(2).unsqueeze(3)  # shape: (C, C) --> (C, C, 1, 1)
            corresponding_out = F.conv2d(corresponding_inp, corresponding_w)
            output.append(corresponding_out.squeeze(0))  # removing batch dimension - will be added with torch.stack

            corresponding_log_w = torch.slogdet(corresponding_w.squeeze().double())[1].float()
            log_w += corresponding_log_w

        output = torch.stack(output, dim=0)  # convert list to tensor
        log_w = log_w / batch_size  # taking average
        log_det = height * width * log_w
        return output, log_det

    def reverse(self, output, condition):
        cond_net_out = self.cond_net(condition)  # shape (B, C, C)
        batch_size = output.shape[0]
        inp = []

        # convolve every batch item with its corresponding W inverse
        for i in range(batch_size):
            corresponding_out = output[i].unsqueeze(0)  # shape (1, C, H, W)
            corresponding_w_inv = cond_net_out[i].inverse().unsqueeze(2).unsqueeze(3)  # shape: (C, C) --> (C, C, 1, 1)
            corresponding_inp = F.conv2d(corresponding_out, corresponding_w_inv)
            inp.append(corresponding_inp.squeeze(0))

        inp = torch.stack(inp, dim=0)
        return inp


# LU for both conditional and unconditional
class InvConv1x1LU(nn.Module):
    def __init__(self, in_channel, mode='unconditional', cond_shape=None, inp_shape=None):
        super().__init__()
        self.mode = mode

        # initialize with LU decomposition
        q = la.qr(np.random.randn(in_channel, in_channel))[0].astype(np.float32)
        w_p, w_l, w_u = la.lu(q)

        w_s = np.diag(w_u)  # extract diagonal elements of U into vector w_s
        w_u = np.triu(w_u, 1)  # set diagonal elements of U to 0

        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_u = torch.from_numpy(w_u)
        w_s = torch.from_numpy(w_s)

        # non-trainable parameters
        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))

        if self.mode == 'conditional':
            matrices_flattened = torch.cat([torch.flatten(w_l), torch.flatten(w_u), logabs(w_s)])
            self.cond_net = WCondNet(cond_shape, inp_shape, do_lu=True, initial_bias=matrices_flattened)

        else:
            # learnable parameters
            self.w_l = nn.Parameter(w_l)
            self.w_u = nn.Parameter(w_u)
            self.w_s = nn.Parameter(logabs(w_s))

    def forward(self, inp, condition=None):
        _, _, height, width = inp.shape
        weight, s_vector = self.calc_weight(condition)
        out = F.conv2d(inp, weight)
        logdet = height * width * torch.sum(s_vector)
        return out, logdet

    def calc_weight(self, condition=None):
        if self.mode == 'conditional':
            l_matrix, u_matrix, s_vector = self.cond_net(condition)
        else:
            l_matrix, u_matrix, s_vector = self.w_l, self.w_u, self.w_s

        weight = (
                self.w_p
                @ (l_matrix * self.l_mask + self.l_eye)  # explicitly make it lower-triangular with 1's on diagonal
                @ ((u_matrix * self.u_mask) + torch.diag(self.s_sign * torch.exp(s_vector)))
        )
        return weight.unsqueeze(2).unsqueeze(3), s_vector

    def reverse_single(self, output, condition=None):
        weight, _ = self.calc_weight(condition)
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

    def reverse(self, output, condition=None):
        batch_size = output.shape[0]
        if batch_size == 1:
            return self.reverse_single(output, condition)
        # reverse one by one for batch size greater than 1. Improving this is not a priority since batch size is usually 1.
        batch_reversed = []
        for i_batch, batch_item in enumerate(output):
            batch_reversed.append(self.reverse(output[i_batch].unsqueeze(0), condition[i_batch].unsqueeze(0)))
        return torch.cat(batch_reversed)
