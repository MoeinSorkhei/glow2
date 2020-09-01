import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad, gradcheck
import numpy as np
from scipy import linalg as la

from .cond_net import WCondNet, apply_cond_net_generic


logabs = lambda x: torch.log(torch.abs(x))


# Not used - non-LU unconditional
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


# Not used - non-LU conditional
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


# Not used
class InvConv1x1NoMemory(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        q, _ = torch.qr(torch.randn(in_channel, in_channel))
        # making it 1x1 conv: conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1)
        w = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(w)  # the weight matrix

    def forward(self, inp):
        return InvConvFunction.apply(inp, self.weight)

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


# Not used
class InvConvFunction(torch.autograd.Function):
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
            out, log_det = InvConvFunction.forward_func(inp, weight)
        ctx.save_for_backward(weight)
        ctx.output = out
        return out, log_det

    @staticmethod
    def backward(ctx, grad_out, grad_logdet):
        weight, = ctx.saved_tensors
        output = ctx.output

        with torch.no_grad():  # reconstruct input
            reconstructed = InvConvFunction.reverse_func(output, weight)
            reconstructed.requires_grad = True

        with torch.enable_grad():  # re-create computational graph
            output, logdet = InvConvFunction.forward_func(reconstructed, weight)
            grad_w_out = grad(outputs=output, inputs=weight, grad_outputs=grad_out, retain_graph=True)[0]
            grad_w_logdet = grad(outputs=logdet, inputs=weight, grad_outputs=grad_logdet, retain_graph=True)[0]
            grad_weight = grad_w_out + grad_w_logdet
            grad_inp = grad(outputs=output, inputs=reconstructed, grad_outputs=grad_out, retain_graph=True)[0]
        return grad_inp, grad_weight


# LU for both conditional and unconditional - also supports constant memory
class InvConv1x1LU(nn.Module):
    def __init__(self, in_channel, mode, const_memory, cond_shape=None, inp_shape=None):
        super().__init__()
        assert cond_shape is None or (len(inp_shape) == 3 and len(cond_shape) == 3), 'Cond shape and inp shape should have len 3 of form: (C, H, W)'

        self.mode = mode
        self.const_memory = const_memory

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
        self.buffers = (self.w_p, self.u_mask, self.l_mask, self.s_sign, self.l_eye)

        if self.mode == 'conditional':
            matrices_flattened = torch.cat([torch.flatten(w_l), torch.flatten(w_u), logabs(w_s)])
            self.cond_net = WCondNet(cond_shape, inp_shape, do_lu=True, initial_bias=matrices_flattened)
            self.params = self.cond_net.conv_net.get_params() + self.cond_net.linear_net.get_params()  # cond_net params
        else:
            # learnable parameters
            self.w_l = nn.Parameter(w_l)
            self.w_u = nn.Parameter(w_u)
            self.w_s = nn.Parameter(logabs(w_s))
            self.params = (self.w_l, self.w_u, self.w_s)

    def compute_weight(self, condition=None, inp_channels=None):
        if self.mode == 'conditional':
            weight, _, _, _, s_vector = \
                InvConvLUFunction.calc_conditional_weight(inp_channels, condition, self.buffers, *self.params)
        else:
            weight = InvConvLUFunction.calc_weight(self.w_l, self.w_u, self.w_s, self.buffers)
            s_vector = self.w_s
        return weight, s_vector

    def usual_forward(self, inp, condition=None):
        weight, s_vector = self.compute_weight(condition, inp_channels=inp.shape[1])
        out, logdet = InvConvLUFunction.forward_func(inp, weight, s_vector)
        return out, logdet

    def forward(self, inp, condition=None):
        if self.const_memory:
            return InvConvLUFunction.apply(inp, condition, self.buffers, *self.params)  # backprop with const memory
        return self.usual_forward(inp, condition)  # usual backprop

    def reverse_single(self, output, condition=None):
        weight, _ = self.compute_weight(condition, inp_channels=output.shape[1])
        return InvConvLUFunction.reverse_func(output, weight)

    def reverse(self, output, condition=None):
        batch_size = output.shape[0]
        if batch_size == 1:
            return self.reverse_single(output, condition)
        # reverse one by one for batch size greater than 1. Improving this is not a priority since batch size is usually 1.
        batch_reversed = []
        for i_batch, batch_item in enumerate(output):
            batch_reversed.append(self.reverse(output[i_batch].unsqueeze(0), condition[i_batch].unsqueeze(0)))
        return torch.cat(batch_reversed)

    def check_grad(self, inp, condition=None):
        if self.const_memory:
            return gradcheck(func=InvConvLUFunction.apply, inputs=(inp, condition, self.buffers, *self.params), eps=1e-6)
        return gradcheck(func=self.usual_forward, inputs=(inp, condition), eps=1e-6)


class PairedInvConv1x1(torch.nn.Module):
    def __init__(self, cond_shape, inp_shape):
        super().__init__()
        inp_channels = inp_shape[0]
        self.left_inv_conv = InvConv1x1LU(mode='unconditional', in_channel=inp_channels, const_memory=True)
        self.right_inv_conv = InvConv1x1LU(mode='conditional', in_channel=inp_channels, const_memory=True, cond_shape=cond_shape, inp_shape=inp_shape)
        self.params = self.left_inv_conv.params + self.right_inv_conv.params
        self.buffers = (self.left_inv_conv.buffers, self.right_inv_conv.buffers)  # tuple of tuples

    def forward(self, activations, inp_left, inp_right):
        return PairedInvConv1x1Function.apply(activations, inp_left, inp_right,
                                              self.left_inv_conv.buffers, self.right_inv_conv.buffers,
                                              *(self.left_inv_conv.params + self.right_inv_conv.params))

    def check_grads(self, inp_left, inp_right):
        out_left, _, out_right, _ = self.forward({}, inp_left, inp_right)
        activations = {'left': [out_left.data], 'right': [out_right.data]}
        return gradcheck(func=PairedInvConv1x1Function.apply,
                         inputs=(activations, inp_left, inp_right,
                                 self.left_inv_conv.buffers, self.right_inv_conv.buffers,
                                 *(self.left_inv_conv.params + self.right_inv_conv.params)))


class PairedInvConv1x1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activations, inp_left, inp_right, left_buffers, right_buffers, *params):
        inp_channels = inp_right.shape[1]
        left_w_l, left_w_u, left_s_vector, cond_net_params = PairedInvConv1x1Function._extract_params(params)
        left_weight = InvConvLUFunction.calc_weight(left_w_l, left_w_u, left_s_vector, left_buffers)
        left_out, left_logdet = InvConvLUFunction.forward_func(inp_left, left_weight, left_s_vector)

        right_weight, _, _, _, right_s_vector = InvConvLUFunction.calc_conditional_weight(inp_channels, left_out, right_buffers, *cond_net_params)
        right_out, right_logdet = InvConvLUFunction.forward_func(inp_right, right_weight, right_s_vector)

        ctx.save_for_backward(*params)
        ctx.left_buffers, ctx.right_buffers = left_buffers, right_buffers
        # ctx.activations = activations
        ctx.left_out = left_out.data  # to be replaced by activations
        ctx.right_out = right_out.data
        ctx.inp_channels = inp_channels
        return left_out, left_logdet, right_out, right_logdet

    @staticmethod
    def backward(ctx, grad_left_out, grad_left_logdet, grad_right_out, grad_right_logdet):
        params = ctx.saved_tensors
        left_buffers, right_buffers = ctx.left_buffers, ctx.right_buffers
        left_out, right_out = ctx.left_out, ctx.right_out  # to be replaced by activations
        # activations = ctx.activations
        inp_channels = ctx.inp_channels

        left_w_l, left_w_u, left_s_vector, cond_net_params = PairedInvConv1x1Function._extract_params(params)
        with torch.enable_grad():
            left_weight = InvConvLUFunction.calc_weight(left_w_l, left_w_u, left_s_vector, left_buffers)

        with torch.no_grad():
            inp_left = InvConvLUFunction.reverse_func(left_out, left_weight)
            inp_left.requires_grad = True

        with torch.enable_grad():
            left_out, left_logdet = InvConvLUFunction.forward_func(inp_left, left_weight, left_s_vector)
            right_weight, _, _, _, right_s_vector = InvConvLUFunction.calc_conditional_weight(inp_channels, left_out, right_buffers, *cond_net_params)
        with torch.no_grad():
            inp_right = InvConvLUFunction.reverse_func(right_out, right_weight)
            inp_right.requires_grad = True

        with torch.enable_grad():
            right_out, right_logdet = InvConvLUFunction.forward_func(inp_right, right_weight, right_s_vector)
            grad_inp_left = grad(outputs=left_out, inputs=inp_left, grad_outputs=grad_left_out, retain_graph=True)[0] + \
                            grad(outputs=right_out, inputs=inp_left, grad_outputs=grad_right_out, retain_graph=True)[0] + \
                            grad(outputs=right_logdet, inputs=inp_left, grad_outputs=grad_right_logdet, retain_graph=True)[0]

            grad_left_w_l = grad(outputs=left_out, inputs=left_w_l, grad_outputs=grad_left_out, retain_graph=True)[0]
            grad_left_w_u = grad(outputs=left_out, inputs=left_w_u, grad_outputs=grad_left_out, retain_graph=True)[0]
            grad_left_w_s = grad(outputs=left_out, inputs=left_s_vector, grad_outputs=grad_left_out, retain_graph=True)[0] + \
                            grad(outputs=left_logdet, inputs=left_s_vector, grad_outputs=grad_left_logdet, retain_graph=True)[0]

            grad_inp_right = grad(outputs=right_out, inputs=inp_right, grad_outputs=grad_right_out, retain_graph=True)[0]

            grad_cond_params_wrt_right_out = grad(outputs=right_out, inputs=cond_net_params, grad_outputs=grad_right_out, retain_graph=True)
            grad_cond_params_wrt_right_logdet = grad(outputs=right_logdet, inputs=cond_net_params, grad_outputs=grad_right_logdet, retain_graph=True)
            grad_cond_params = tuple([sum(x) for x in zip(grad_cond_params_wrt_right_out, grad_cond_params_wrt_right_logdet)])
            return (None, grad_inp_left, grad_inp_right, None, None, *(grad_left_w_l, grad_left_w_u, grad_left_w_s, *grad_cond_params))

    @staticmethod
    def _extract_params(params):
        left_w_l, left_w_u, left_w_s = params[:3]
        cond_net_params = params[3:]
        return left_w_l, left_w_u, left_w_s, cond_net_params


class InvConvLUFunction(torch.autograd.Function):
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
    def calc_weight(l_matrix, u_matrix, s_vector, buffers):
        w_p, u_mask, l_mask, s_sign, l_eye = buffers
        weight = (
                w_p
                @ (l_matrix * l_mask + l_eye)  # explicitly make it lower-triangular with 1's on diagonal
                @ ((u_matrix * u_mask) + torch.diag(s_sign * torch.exp(s_vector)))  # s_vector should be of shape (C,)
        )
        return weight.unsqueeze(2).unsqueeze(3)

    @staticmethod
    def calc_conditional_weight(inp_channels, cond_inp, buffers, *params):
        out = apply_cond_net_generic(cond_inp, *params)
        # get LU and S out of the cond output
        channels_sqrt = inp_channels ** 2
        w_l_flattened = out[:, :channels_sqrt]  # important: keep the batch dimension in order to compute gradients correctly
        w_u_flattened = out[:, channels_sqrt:channels_sqrt * 2]  # shape (1, channels_sqrt) for batch size 1
        s_vector = out[:, channels_sqrt * 2:]

        matrix_shape = (inp_channels, inp_channels)
        l_matrix = torch.reshape(w_l_flattened.squeeze(0), matrix_shape)  # 2d tensor (used squeeze to make it 2d)
        u_matrix = torch.reshape(w_u_flattened.squeeze(0), matrix_shape)
        weight = InvConvLUFunction.calc_weight(l_matrix, u_matrix, s_vector.squeeze(0), buffers)
        return weight, out, w_l_flattened, w_u_flattened, s_vector

    @staticmethod
    def forward(ctx, inp, cond_inp, buffers, *params):
        is_conditional = True if cond_inp is not None else False
        if is_conditional:  # conditional case - in this case s_vector has shape (1, C)
            weight, _, _, _, s_vector = InvConvLUFunction.calc_conditional_weight(inp.shape[1], cond_inp, buffers, *params)
        else:
            l_matrix, u_matrix, s_vector = params  # in this case l_matrix, u_matrix have shape: (C, C) and s_vector has shape (C) and
            weight = InvConvLUFunction.calc_weight(l_matrix, u_matrix, s_vector, buffers)

        # forward operation
        output, logdet = InvConvLUFunction.forward_func(inp, weight, s_vector)
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

        is_conditional = True if cond_inp is not None else False
        with torch.enable_grad():
            if is_conditional:  # conditional case
                weight, cond_net_out, w_l_flattened, w_u_flattened, s_vector = \
                    InvConvLUFunction.calc_conditional_weight(inp_channels, cond_inp, buffers, *params)
            else:  # unconditional case
                l_matrix, u_matrix, s_vector = params
                weight = InvConvLUFunction.calc_weight(l_matrix, u_matrix, s_vector, buffers)

        with torch.no_grad():
            reconstructed = InvConvLUFunction.reverse_func(output, weight)
            reconstructed.requires_grad = True

        with torch.enable_grad():
            output, logdet = InvConvLUFunction.forward_func(reconstructed, weight, s_vector)
            grad_inp = grad(outputs=output, inputs=reconstructed, grad_outputs=grad_output, retain_graph=True)[0]

            if is_conditional:  # conditional case, ie with cond net
                # compute intermediary grad for cond net out
                grad_wl_flattened = grad(outputs=output, inputs=w_l_flattened, grad_outputs=grad_output, retain_graph=True)[0]
                grad_wu_flattened = grad(outputs=output, inputs=w_u_flattened, grad_outputs=grad_output, retain_graph=True)[0]
                grad_s_vector = grad(outputs=output, inputs=s_vector, grad_outputs=grad_output, retain_graph=True)[0] + \
                                grad(outputs=logdet, inputs=s_vector, grad_outputs=grad_logdet, retain_graph=True)[0]
                grad_cond_out = torch.cat([grad_wl_flattened, grad_wu_flattened, grad_s_vector], dim=1)
                # grad wrt condition and params
                grad_cond_inp = grad(outputs=cond_net_out, inputs=cond_inp, grad_outputs=grad_cond_out, retain_graph=True)[0]
                grad_params = grad(outputs=cond_net_out, inputs=params, grad_outputs=grad_cond_out, retain_graph=True)

            else:  # unconditional case
                grad_cond_inp = None
                grad_l_matrix = grad(outputs=output, inputs=l_matrix, grad_outputs=grad_output, retain_graph=True)[0]
                grad_u_matrix = grad(outputs=output, inputs=u_matrix, grad_outputs=grad_output, retain_graph=True)[0]
                grad_s_vector = grad(outputs=output, inputs=s_vector, grad_outputs=grad_output, retain_graph=True)[0] + \
                                grad(outputs=logdet, inputs=s_vector, grad_outputs=grad_logdet, retain_graph=True)[0]
                grad_params = (grad_l_matrix, grad_u_matrix, grad_s_vector)
        return (grad_inp, grad_cond_inp, None) + grad_params  # concat tuples
