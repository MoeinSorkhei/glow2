import torch
from torch import nn
import numpy as np


def compute_batch_stats(inp):
    with torch.no_grad():
        flatten = inp.permute(1, 0, 2, 3).contiguous().view(inp.shape[1], -1)
        mean = (
            flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
        )
        std = (
            flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
        )
        return mean, std


class ZeroInitConv2d(nn.Module):  # no usage at the moment
    """
    This is a modified version of ZeroInitConv2d in glow.py.
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, inp):
        out = self.conv(inp)
        out = out * torch.exp(self.scale * 3)
        return out


class ZeroWeightLinear(nn.Module):
    def __init__(self, in_features, out_features, bias_mode='zero'):
        super().__init__()

        self.bias_mode = bias_mode
        self.out_features = out_features

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.linear.weight.data.zero_()  # wight zero initialization

        if bias_mode == 'qr':  # in this case out_features should be channels ** 2 -- used for w
            channels = int(np.sqrt(out_features))  # 36 --> 6 x 6
            q, _ = torch.qr(torch.randn((channels, channels)))
            self.linear.bias.data = torch.flatten(q)  # qr decomposition init of bias - for the last Linear layer

        # data dependent initialization
        elif bias_mode == 'data_zero':  # in this case out_features should be channels * 2 -- used for actnorm
            self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))  # init with the first forward pass

        elif bias_mode == 'zero':
            self.linear.bias.data.zero_()  # zero initialization of bias

    def forward(self, inp):
        return self.linear(inp)


def compute_conv_out_shape(inp_shape, n_convs, kernel_size, stride):
    """
    :param inp_shape:
    :param n_convs:
    :param kernel_size:
    :param stride:
    :return:

    Notes:
        - Assumption: all convolutions have equal kernel size and stride
        - Formula from: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
    """
    h, w = inp_shape[1], inp_shape[2]  # inp_shape (C, H, W) -- no batch size

    conv_out_h, conv_out_w = h, w
    for i in range(n_convs):
        conv_out_h = (conv_out_h - kernel_size) // stride + 1
        conv_out_w = (conv_out_w - kernel_size) // stride + 1
        # print(f'for i = {i}: conv_out_shape: ({conv_out_h}, {conv_out_w})')

    return conv_out_h, conv_out_w


class ConvNet(nn.Module):
    def __init__(self, inp_shape):
        super().__init__()
        inp_channels = inp_shape[0]  # inp_shape (C, H, W) -- no batch size
        n_convs = 2  # convolutions other than 1x1
        conv_stride = 1
        conv_net_out_h, conv_net_out_w = compute_conv_out_shape(inp_shape, n_convs=n_convs,
                                                                kernel_size=3, stride=conv_stride)
        down_sampling_channels = 8
        final_conv_channels = 2
        self.conv_net_out_shape = final_conv_channels * conv_net_out_h * conv_net_out_w

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=inp_channels, out_channels=down_sampling_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=down_sampling_channels, out_channels=4, kernel_size=3, stride=conv_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4, out_channels=final_conv_channels, kernel_size=3, stride=conv_stride),
            nn.ReLU(inplace=True),
        )

    def output_shape(self):
        return self.conv_net_out_shape

    def forward(self, inp):
        return self.conv_net(inp)


class LinearNet(nn.Module):
    def __init__(self, in_features, out_features, condition):
        super().__init__()
        self.out_features = out_features

        # determine how to initialize the bias of last linear layer
        if condition == 'w':
            self.bias_mode = 'qr'
        elif condition == 'actnorm':
            self.bias_mode = 'data_zero'
        elif condition == 'w - LU':
            self.bias_mode = 'zero'
        else:
            raise NotImplementedError('Condition not implemented')

        # init last layer
        if condition == 'coupling':  # random init of last layer
            last_layer = nn.Linear(in_features=48, out_features=self.out_features)
        else:
            last_layer = ZeroWeightLinear(in_features=48, out_features=self.out_features, bias_mode=self.bias_mode)

        self.linear_net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=48),
            nn.ReLU(inplace=True),
            last_layer
        )

    def init_data_zero_bias(self, data_batch):  # used for conditional actnorm
        batch_mean, batch_std = compute_batch_stats(data_batch)
        batch_mean, batch_std = torch.flatten(batch_mean), torch.flatten(batch_std)

        self.linear_net[-1].linear.bias.data[:self.out_features // 2].copy_(1 / (batch_std + 1e-6))  # scale of actnorm
        self.linear_net[-1].linear.bias.data[self.out_features // 2:].copy_(-batch_mean)  # shift of actnorm
        self.linear_net[-1].initialized.fill_(1)

    def forward(self, conv_out, data_batch=None):
        # if 'data_zero' ==> data-dependent initialization of last linear layer
        if self.bias_mode == 'data_zero' and self.linear_net[-1].initialized.item() == 0:
            self.init_data_zero_bias(data_batch)

        return self.linear_net(conv_out)


class ActCondNet(nn.Module):
    def __init__(self, cond_shape, inp_shape):
        super().__init__()
        self.conv_net = ConvNet(cond_shape)
        conv_out_flat_length = self.conv_net.output_shape()

        inp_channels = inp_shape[0]  # inp_shape (C, H, W) -- no batch size
        self.linear_net = LinearNet(conv_out_flat_length, inp_channels * 2, condition='actnorm')

        self.print_params = False
        if self.print_params:
            p1 = sum(p.numel() for p in self.conv_net.parameters())
            p2 = sum(p.numel() for p in self.linear_net.parameters())
            print(f'conv_net params: {p1} - linear_net params: {p2} - '
                  f'Total: {p1 + p2}')

    def forward(self, cond_input, data_batch=None):
        # passing through the ConvNet
        conv_out = self.conv_net(cond_input)
        conv_out = conv_out.view(conv_out.shape[0], -1)

        # data_batch only used for data-dependent initialization, otherwise ignored
        out = self.linear_net(conv_out, data_batch)
        out_channels = out.shape[1]  # e.g. (1, 12)
        out = out.view(out.shape[0], 2, out_channels // 2)  # 12 --> 6 x 2 - output shape: (B, 2, C)
        return out


class WCondNet(nn.Module):
    def __init__(self, cond_shape, inp_shape, do_lu=False, initial_bias=None):
        super().__init__()
        self.inp_channels = inp_shape[0]  # inp_shape (C, H, W) -- no batch size
        self.do_lu = do_lu

        self.conv_net = ConvNet(cond_shape)
        conv_out_flat_length = self.conv_net.output_shape()

        if self.do_lu:
            linear_out_features = 2 * (self.inp_channels ** 2) + self.inp_channels  # for L, U, and s
            self.linear_net = LinearNet(in_features=conv_out_flat_length,
                                        out_features=linear_out_features,
                                        condition='w - LU')
            self.linear_net.linear_net[-1].linear.bias.data.copy_(initial_bias)  # init with flattened LU and s elements
        else:
            self.linear_net = LinearNet(in_features=conv_out_flat_length,
                                        out_features=self.inp_channels ** 2,
                                        condition='w')

        self.print_params = False
        if self.print_params:
            p1 = sum(p.numel() for p in self.conv_net.parameters())
            p2 = sum(p.numel() for p in self.linear_net.parameters())
            print(f'conv_net params: {p1} - linear_net params: {p2} - '
                  f'Total: {p1 + p2}')

    def forward(self, cond_input):
        conv_out = self.conv_net(cond_input)
        conv_out = conv_out.view(conv_out.shape[0], -1)
        out = self.linear_net(conv_out)  # shape (batch_size, out_features)

        if self.do_lu:
            out = out.squeeze(0)  # batch size 1
            channels_sqrt = self.inp_channels ** 2
            w_l_flattened = out[:channels_sqrt]
            w_u_flattened = out[channels_sqrt:channels_sqrt * 2]
            w_s = out[channels_sqrt * 2:]  # 1d tensor

            matrix_shape = (self.inp_channels, self.inp_channels)
            w_l = torch.reshape(w_l_flattened, matrix_shape)  # 2d tensor
            w_u = torch.reshape(w_u_flattened, matrix_shape)
            return w_l, w_u, w_s
        else:
            out = out.view(out.shape[0], self.inp_channels, self.inp_channels)  # 36 --> 6 x 6
            return out  # shape (B, C, C)


class CouplingCondNet(nn.Module):
    def __init__(self, cond_shape, inp_shape):
        super().__init__()
        cond_channels = cond_shape[0]  # might be segment + boundary - input shape of the cond net
        inp_channels = inp_shape[0]  # the actual channels of z - used as output shape of the cond net

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=cond_channels, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=inp_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.print_params = False
        if self.print_params:
            total_params = sum(p.numel() for p in self.conv_net.parameters())
            print(f'CouplingCondNet params: {total_params}')

    def forward(self, cond_input):
        conv_out = self.conv_net(cond_input)
        return conv_out

