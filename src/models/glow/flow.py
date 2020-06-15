import torch
from torch import nn
from torch.nn import functional as F

from helper import label_to_tensor
from .actnorm import ActNorm, ActNormConditional
from .conv1x1 import InvConv1x1, InvConv1x1LU, InvConv1x1Conditional
from ..cond_net import CouplingCondNet
from globals import device


class ZeroInitConv2d(nn.Module):
    """
    To be used in the Affine Coupling step:
    The last convolution of each NN(), according to the paper is initialized with zeros, such that the each affine layer
    initially performs an identity function.
    """
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, inp):
        # padding with additional 1 in each side to keep the spatial dimension unchanged after the convolution operation
        out = F.pad(input=inp, pad=[1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    """
    This transforms part of the input tensor in a way that half of the output tensor in a way that half of the output
    tensor is a non-linear function of the other half. This non-linearity is obtained through the stacking some CNNs.

    Notes:
        - About the conditioning network: Cond shape is the shape of the condition, which might have larger channels if
          the condition has boundary maps etc. Inp shape is the actual shape of the corresponding z.
          Example: cond shape (C + 12, H, W) - inp shape (C, H, W), where the extra 12 is for boundary map.
          The output of the cond net will be the same shape as the corresponding z.

        - If no conditioning net is used, the inp shape would not be needed and the cond shape will directly be used
          for creating the CNNs (conv_channels parameter in the __init__ function).
    """
    def __init__(self, in_channel, n_filters=512, do_affine=True, cond_shape=None, use_cond_net=False, inp_shape=None):
        super().__init__()
        # adding extra channels for the condition (e.g., 10 for MNIST)
        # conv_channels = in_channel // 2 if cond_shape is None else (in_channel // 2) + cond_shape[0]

        if cond_shape is None:
            conv_channels = in_channel // 2  # e.g. z shape: (12, 128, 256) --> conv_shape: (6, 128, 256)

        elif not use_cond_net:  # e.g. z shape: (12, 128, 256) --> conv_shape: (12 + 18, 128, 256),  18 for bmaps
            conv_channels = (in_channel // 2) + cond_shape[0]

        else:  # with cond net. e.g. z shape: (12, 128, 256) --> conv_shape: (12, 128, 256),  12 for inp shape
            '''import helper
            helper.print_and_wait(f'inp shape: {inp_shape}')'''

            conv_channels = (in_channel // 2) + inp_shape[0]

        self.do_affine = do_affine
        self.net = nn.Sequential(  # NN() in affine coupling: neither channels shape nor spatial shape change after this
            # padding=1 is equivalent to padding=(1, 1), adding extra zeros to both h and w dimensions
            nn.Conv2d(in_channels=conv_channels, out_channels=n_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=1),
            nn.ReLU(inplace=True),
            ZeroInitConv2d(in_channel=n_filters, out_channel=in_channel)  # channels dimension same as input
        )

        # Initializing the params
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

        if use_cond_net:  # uses inp shape only if cond net is used
            self.use_cond_net = True
            self.cond_net = CouplingCondNet(inp_shape, cond_shape)  # without considering batch size dimension
        else:
            self.use_cond_net = False

    def forward(self, inp, cond=None):
        inp_a, inp_b = inp.chunk(chunks=2, dim=1)  # chunk along the channel dimension
        if self.do_affine:
            if cond is not None:  # conditional
                if cond['name'] == 'mnist':  # needs to be re-implemented
                    # expects the cond to be of shape (B, 10, H, W). Concatenate condition along channel: C -> C+10
                    # truncate spatial dimension so it spatially fits the actual tensor
                    cond_tensor = cond[1][:, :, :inp_a.shape[2], :inp_b.shape[3]]

                elif cond['name'] == 'transient':
                    cond_tensor = self.cond_net(cond['transient_cond']) if self.use_cond_net else cond['transient_cond']

                elif cond['name'] == 'real_cond':
                    cond_tensor = self.cond_net(cond['real_cond']) if self.use_cond_net else cond['real_cond']

                elif cond['name'] == 'segment':  # the whole xA with all the channels
                    cond_tensor = self.cond_net(cond['segment']) if self.use_cond_net else cond['segment']

                elif cond['name'] == 'segment_boundary':  # channel-wise concat segment with boundary
                    cond_concat = torch.cat([cond['segment'], cond['boundary']], dim=1)
                    cond_tensor = self.cond_net(cond_concat) if self.use_cond_net else cond_concat

                else:
                    raise NotImplementedError('In [Block] forward: Condition not implemented...')

                inp_a_conditional = torch.cat(tensors=[inp_a, cond_tensor], dim=1)  # concat channel-wise
                log_s, t = self.net(inp_a_conditional).chunk(chunks=2, dim=1)

            else:
                log_s, t = self.net(inp_a).chunk(chunks=2, dim=1)

            s = torch.sigmoid(log_s + 2)

            out_b = (inp_b + t) * s
            log_det = torch.sum(torch.log(s).view(inp.shape[0], -1), 1)

        else:
            # note: ZeroConv2d(in_channel=n_filters, out_channel=in_channel) should also be changed for additive
            print('Not implemented... Use --affine')
            out_b, log_det = None, None

        return torch.cat(tensors=[inp_a, out_b], dim=1), log_det

    def reverse(self, output, cond=None):
        out_a, out_b = output.chunk(chunks=2, dim=1)  # here we know that out_a = inp_a (see the forward fn)
        if self.do_affine:
            if cond is not None:
                if cond['name'] == 'mnist':
                    # concatenate with the same condition as in the forward pass
                    label, n_samples = cond[1], cond[2]
                    cond_tensor = label_to_tensor(label, out_a.shape[2], out_a.shape[3], n_samples).to(device)

                elif cond['name'] == 'transient':
                    cond_tensor = self.cond_net(cond['transient_cond']) if self.use_cond_net else cond['transient_cond']

                elif cond['name'] == 'real_cond':
                    cond_tensor = self.cond_net(cond['real_cond']) if self.use_cond_net else cond['real_cond']

                elif cond['name'] == 'segment':  # the whole xA with all the channels
                    cond_tensor = self.cond_net(cond['segment']) if self.use_cond_net else cond['segment']

                elif cond['name'] == 'segment_boundary':  # channel-wise concat segment with boundary
                    cond_concat = torch.cat([cond['segment'], cond['boundary']], dim=1)
                    cond_tensor = self.cond_net(cond_concat) if self.use_cond_net else cond_concat

                else:
                    raise NotImplementedError('In [Block] reverse: Condition not implemented...')

                out_a_conditional = torch.cat(tensors=[out_a, cond_tensor], dim=1)
                log_s, t = self.net(out_a_conditional).chunk(chunks=2, dim=1)

            else:
                log_s, t = self.net(out_a).chunk(chunks=2, dim=1)

            s = torch.sigmoid(log_s + 2)
            inp_b = (out_b / s) - t

        else:
            print('Not implemented... Use --affine')
            inp_b = None

        return torch.cat(tensors=[out_a, inp_b], dim=1)


class Flow(nn.Module):
    """
    The Flow module does not change the dimensionality of its input.
    """
    def __init__(self, in_channel, do_affine=True, conv_lu=True, coupling_cond_shape=None,
                 w_conditional=False, act_conditional=False, use_coupling_cond_net=False,
                 inp_shape=None, conv_stride=None):
        super().__init__()

        # initializing actnorm
        if act_conditional:
            self.act_norm = ActNormConditional(inp_shape, conv_stride)  # inp_shape, conv_stride the same as W cond net
            self.act_conditional = True
        else:
            self.act_norm = ActNorm(in_channel=in_channel)
            self.act_conditional = False

        # initializing W
        if w_conditional:
            self.inv_conv = InvConv1x1Conditional(inp_shape, conv_stride)
            self.w_conditional = True
        else:
            self.inv_conv = InvConv1x1LU(in_channel) if conv_lu else InvConv1x1(in_channel)
            self.w_conditional = False

        # initializing coupling
        if use_coupling_cond_net:
            self.use_coupling_cond_net = True
            self.coupling = AffineCoupling(in_channel=in_channel, do_affine=do_affine, cond_shape=coupling_cond_shape,
                                           use_cond_net=True, inp_shape=inp_shape)
        else:
            self.use_coupling_cond_net = False
            self.coupling = AffineCoupling(in_channel=in_channel, do_affine=do_affine, cond_shape=coupling_cond_shape,
                                           use_cond_net=False)

    def forward(self, inp, coupling_cond=None, w_left_out=None, act_left_out=None,
                return_w_out=False, return_act_out=False):
        # actnorm forward
        if self.act_conditional:
            '''import helper
            helper.print_and_wait(f'act_left shape: {act_left_out.shape}')'''

            actnorm_out, act_logdet = self.act_norm(inp, act_left_out)  # conditioned on left actnorm
        else:
            actnorm_out, act_logdet = self.act_norm(inp)

        # W forward
        if self.w_conditional:
            w_out, conv_logdet = self.inv_conv(actnorm_out, w_left_out)
        else:
            w_out, conv_logdet = self.inv_conv(actnorm_out)

        # coupling forward
        out, affine_logdet = self.coupling(w_out, cond=coupling_cond)
        log_det = act_logdet + conv_logdet + affine_logdet

        output_dict = {'out': out, 'log_det': log_det}
        if return_w_out:
            output_dict['w_out'] = w_out
        if return_act_out:
            output_dict['act_out'] = actnorm_out
        return output_dict

    def reverse(self, output, coupling_cond=None, w_left_out=None, act_left_out=None):
        # coupling reverse
        coupling_inp = self.coupling.reverse(output, cond=coupling_cond)

        # W reverse
        if self.w_conditional:
            w_inp = self.inv_conv.reverse(coupling_inp, w_left_out)
        else:
            w_inp = self.inv_conv.reverse(coupling_inp)

        # actnorm reverse
        if self.act_conditional:
            inp = self.act_norm.reverse(w_inp, act_left_out)
        else:
            inp = self.act_norm.reverse(w_inp)
        return inp
