import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad, gradcheck
import numpy as np
from scipy import linalg as la

from .cond_net import WCondNet, apply_cond_net_generic


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


class InvConv1x1LUNoMemory(nn.Module):
    def __init__(self, in_channel, mode='unconditional', cond_shape=None, inp_shape=None, input_dtype=torch.float32):
        super().__init__()
        assert len(inp_shape) == 3 and len(cond_shape) == 3, 'Cond shape and inp shape should have len 3 of form: (C, H, W)'
        self.mode = mode

        # initialize with LU decomposition
        # dtype = np.float32 if input_dtype == torch.float32 else np.float64
        q = la.qr(np.random.randn(in_channel, in_channel))[0].astype(np.float32 if input_dtype == torch.float32 else np.float64)
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
            self.params = self.cond_net.conv_net.get_params() + self.cond_net.linear_net.get_params()
            self.buffers = (self.w_p, self.u_mask, self.l_mask, self.s_sign, self.l_eye)
        else:
            # learnable parameters
            self.w_l = nn.Parameter(w_l)
            self.w_u = nn.Parameter(w_u)
            self.w_s = nn.Parameter(logabs(w_s))

    def forward(self, inp, condition):
        assert self.mode == 'conditional'
        WConditionalLUFunction.apply(inp, condition, self.buffers, *self.params)

    def check_grad(self, inp, condition):
        return gradcheck(func=WConditionalLUFunction.apply, inputs=(inp, condition, self.buffers, *self.params), eps=1e-6)


class WConditionalLUFunction(torch.autograd.Function):
    @staticmethod
    def forward_func(inp, weight, s_vector):
        _, _, height, width = inp.shape
        out = F.conv2d(inp, weight)
        logdet = height * width * torch.sum(s_vector)
        return out, logdet

    @staticmethod
    def reverse_func(output, weight):
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

    @staticmethod
    def apply_cond_net(inp_channels, cond_inp, buffers, *params):
        out = apply_cond_net_generic(cond_inp, *params)

        # get LU and S out of the cond output
        channels_sqrt = inp_channels ** 2
        w_l_flattened = out[:, :channels_sqrt]  # keep the batch dimension in order to compute gradients correctly
        w_u_flattened = out[:, channels_sqrt:channels_sqrt * 2]
        s_vector = out[:, channels_sqrt * 2:]

        matrix_shape = (inp_channels, inp_channels)
        l_matrix = torch.reshape(w_l_flattened.squeeze(0), matrix_shape)  # 2d tensor
        u_matrix = torch.reshape(w_u_flattened.squeeze(0), matrix_shape)
        w_p, u_mask, l_mask, s_sign, l_eye = buffers

        weight = (
                w_p
                @ (l_matrix * l_mask + l_eye)  # explicitly make it lower-triangular with 1's on diagonal
                @ ((u_matrix * u_mask) + torch.diag(s_sign * torch.exp(s_vector.squeeze(0))))
        )
        return weight.unsqueeze(2).unsqueeze(3), out, w_l_flattened, w_u_flattened, s_vector

    @staticmethod
    def forward(ctx, inp, cond_inp, buffers, *params):
        weight, _, _, _, s_vector = WConditionalLUFunction.apply_cond_net(inp.shape[1], cond_inp, buffers, *params)
        output, logdet = WConditionalLUFunction.forward_func(inp, weight, s_vector)

        # save items for backward
        ctx.save_for_backward(*params)
        ctx.buffers = buffers
        ctx.output = output
        ctx.cond_inp = cond_inp
        ctx.inp_channels = inp.shape[1]
        return output, logdet

    @staticmethod
    def backward(ctx, grad_output, grad_logdet):
        params = ctx.saved_tensors
        buffers = ctx.buffers
        output = ctx.output
        cond_inp = ctx.cond_inp
        inp_channels = ctx.inp_channels

        with torch.enable_grad():
            weight, cond_net_out, w_l_flattened, w_u_flattened, s_vector = \
                WConditionalLUFunction.apply_cond_net(inp_channels, cond_inp, buffers, *params)

        with torch.no_grad():
            reconstructed = WConditionalLUFunction.reverse_func(output, weight)
            reconstructed.requires_grad = True

        with torch.enable_grad():
            output, logdet = WConditionalLUFunction.forward_func(reconstructed, weight, s_vector)
            grad_inp = grad(outputs=output, inputs=reconstructed, grad_outputs=grad_output, retain_graph=True)[0]

            # compute intermediary grad for cond net out
            grad_wl_flattened = grad(outputs=output, inputs=w_l_flattened, grad_outputs=grad_output, retain_graph=True)[0]
            grad_wu_flattened = grad(outputs=output, inputs=w_u_flattened, grad_outputs=grad_output, retain_graph=True)[0]
            grad_s_vector = grad(outputs=output, inputs=s_vector, grad_outputs=grad_output, retain_graph=True)[0] + \
                            grad(outputs=logdet, inputs=s_vector, grad_outputs=grad_logdet, retain_graph=True)[0]
            grad_cond_out = torch.cat([grad_wl_flattened, grad_wu_flattened, grad_s_vector], dim=1)

            # grad wrt condition and params
            grad_cond_inp = grad(outputs=cond_net_out, inputs=cond_inp, grad_outputs=grad_cond_out, retain_graph=True)[0]
            grad_params = grad(outputs=cond_net_out, inputs=params, grad_outputs=grad_cond_out, retain_graph=True)
        return (grad_inp, grad_cond_inp, None) + grad_params  # concat tuples


class InvConv1x1NoMemory(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        q, _ = torch.qr(torch.randn(in_channel, in_channel))
        # making it 1x1 conv: conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1)
        w = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(w)  # the weight matrix

    def forward(self, inp):
        return WFunction.apply(inp, self.weight)

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class WFunction(torch.autograd.Function):
    @staticmethod
    def forward_func(inp, weight):
        _, _, height, width = inp.shape
        out = F.conv2d(inp, weight)
        log_w = torch.slogdet(weight.squeeze().double())[1].float()
        log_det = height * width * log_w
        return out, log_det

    @staticmethod
    def reverse_func(output, weight):
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

    @staticmethod
    def forward(ctx, inp, weight):
        with torch.no_grad():
            out, log_det = WFunction.forward_func(inp, weight)
        ctx.save_for_backward(weight)
        ctx.output = out
        return out, log_det

    @staticmethod
    def backward(ctx, grad_out, grad_logdet):
        weight, = ctx.saved_tensors
        output = ctx.output

        with torch.no_grad():  # reconstruct input
            reconstructed = WFunction.reverse_func(output, weight)
            reconstructed.requires_grad = True

        with torch.enable_grad():  # re-create computational graph
            output, logdet = WFunction.forward_func(reconstructed, weight)
            grad_w_out = grad(outputs=output, inputs=weight, grad_outputs=grad_out, retain_graph=True)[0]
            grad_w_logdet = grad(outputs=logdet, inputs=weight, grad_outputs=grad_logdet, retain_graph=True)[0]
            grad_weight = grad_w_out + grad_w_logdet
            grad_inp = grad(outputs=output, inputs=reconstructed, grad_outputs=grad_out, retain_graph=True)[0]
        return grad_inp, grad_weight
