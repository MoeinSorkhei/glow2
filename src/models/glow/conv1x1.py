import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import linalg as la

from .cond_net import WCondNet


logabs = lambda x: torch.log(torch.abs(x))


class InvConv1x1Conditional(nn.Module):
    def __init__(self, inp_shape):
        super().__init__()
        self.cond_net = WCondNet(inp_shape)  # initialized with QR decomposition

        print_params = False
        if print_params:
            total_params = sum(p.numel() for p in self.cond_net.parameters())
            print('ActNormConditional CondNet params:', total_params)

    def forward(self, inp, w_left_out):
        """
        F.conv2d doc: https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.conv2d
        :param inp:
        :param w_left_out:
        :return:
        """
        _, _, height, width = inp.shape
        cond_net_out = self.cond_net(w_left_out)  # shape (B, C, C)
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

    def reverse(self, output, w_left_out):
        cond_net_out = self.cond_net(w_left_out)  # shape (B, C, C)
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


# not tested - should use PyTorch LU (if it has)
class InvConv1x1ConditionalLU(nn.Module):
    def __init__(self, inp_shape, conv_stride):
        super().__init__()
        self.cond_net = WCondNet(inp_shape, conv_stride)
        # self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

        q = self.cond_net[-1].bias.data.cpu().numpy().reshape((inp_shape[0], inp_shape[0]))
        # self.initialize_lu_params(q_weight)
        # q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        # kept fixed
        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))

        # learnable parameters
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def calc_weight(self, corresponding_cond_net_out):
        _, w_l, w_u = la.lu(corresponding_cond_net_out.cpu().numpy().astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)

        # w_p = torch.from_numpy(w_p)
        self.w_l = torch.from_numpy(w_l)
        self.w_s = torch.from_numpy(w_s)
        self.w_u = torch.from_numpy(w_u)

        weight = (
                self.w_p
                @ (self.w_l * self.l_mask + self.l_eye)
                @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight

    def forward(self, inp, w_left_out):
        _, _, height, width = inp.shape
        cond_net_out = self.cond_net(w_left_out)  # shape (B, C, C)
        batch_size = inp.shape[0]
        log_w = 0
        output = []

        # convolve every batch item with its corresponding W
        for i in range(batch_size):
            corresponding_inp = inp[i].unsqueeze(0)  # re-adding batch dim - shape (1, C, H, W)
            corresponding_w = self.calc_weight(cond_net_out[i]).unsqueeze(2).unsqueeze(3)  # shape: (C, C, 1, 1)

            corresponding_out = F.conv2d(corresponding_inp, corresponding_w)
            output.append(corresponding_out.squeeze(0))  # removing batch dimension - will be added with torch.stack

            # corresponding_log_w = torch.slogdet(corresponding_w.squeeze().double())[1].float()
            corresponding_log_w = torch.sum(self.w_s)
            log_w += corresponding_log_w

        output = torch.stack(output, dim=0)  # convert list to tensor
        log_w = log_w / batch_size  # taking average
        log_det = height * width * log_w
        return output, log_det

    def reverse(self, output, w_left_out):
        cond_net_out = self.cond_net(w_left_out)  # shape (B, C, C)
        batch_size = output.shape[0]
        inp = []

        # convolve every batch item with its corresponding W inverse
        for i in range(batch_size):
            corresponding_out = output[i].unsqueeze(0)  # shape (1, C, H, W)
            corresponding_w_inv = self.calc_weight(cond_net_out[i]).inverse().unsqueeze(2).unsqueeze(3)  # (C, C, 1, 1)
            corresponding_inp = F.conv2d(corresponding_out, corresponding_w_inv)
            inp.append(corresponding_inp.squeeze(0))

        inp = torch.stack(inp, dim=0)
        return inp


class InvConv1x1(nn.Module):
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


class InvConv1x1LU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        # kept fixed
        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))

        # learnable parameters
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, inp):
        _, _, height, width = inp.shape

        weight = self.calc_weight()

        out = F.conv2d(inp, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        # ablation: doing the following
        # why torch.exp(self.w_s)?
        # w_s is a matrix not a vector here?
        # s_sign changes while s is optimized, but is assumed to be fixed by register buffer. Why?
        # What is the role of mask and sign and eye matrices here?
        # What are squeeze unsqueeze doing here?
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
