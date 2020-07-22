from . import tfops as Z
from . import optim
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

import os.path
import json


# class ResultLogger(object):
#     def __init__(self, path, *args, **kwargs):
#         if os.path.exists(path) and kwargs["restore_path"] != '':
#              self.f_log = open(path, 'a+')
#         else:
#              self.f_log = open(path, 'w')
#         self.f_log.write(json.dumps(kwargs) + '\n')
#
#     def log(self, **kwargs):
#         self.f_log.write(json.dumps(kwargs) + '\n')
#         self.f_log.flush()
#
#     def close(self):
#         self.f_log.close()


def checkpoint(z, logdet):
    zshape = Z.int_shape(z)
    z = tf.reshape(z, [-1, zshape[1]*zshape[2]*zshape[3]*zshape[4]])
    logdet = tf.reshape(logdet, [-1, 1])
    combined = tf.concat([z, logdet], axis=1)
    tf.add_to_collection('checkpoints', combined)
    logdet = combined[:, -1]
    z = tf.reshape(combined[:, :-1], [-1, zshape[1], zshape[2], zshape[3], zshape[4]])
    return z, logdet

@add_arg_scope
def revnet3d(name, z, logdet, level, hps, reverse=False):   # this is a Block of Glow
    with tf.variable_scope(name):
        if not reverse:
            for i in range(hps.depth[level]):  # adding Flows to the Blocks
                z, logdet = checkpoint(z, logdet)
                z, logdet = revnet3d_step(str(i), z, logdet, hps, reverse)
            z, logdet = checkpoint(z, logdet)
        else:
            for i in reversed(range(hps.depth[level])):
                z, logdet = revnet3d_step(str(i), z, logdet, hps, reverse)
    return z, logdet


# Simpler, new version
@add_arg_scope
def revnet3d_step(name, z, logdet, hps, reverse):
    with tf.variable_scope(name):

        shape = Z.int_shape(z)
        n_z = shape[4]
        assert n_z % 2 == 0

        if not reverse:

            z, logdet = Z.actnorm("actnorm", z, logdet=logdet)

            if hps.flow_permutation == 0:
                z = Z.reverse_features("reverse", z)
            elif hps.flow_permutation == 1:
                z = Z.shuffle_features("shuffle", z)
            elif hps.flow_permutation == 2:
                z, logdet = invertible_1x1_conv("invconv", z, logdet)
            else:
                raise Exception()

            z1 = z[:, :, :, :, :n_z // 2]
            z2 = z[:, :, :, :, n_z // 2:]

            if hps.flow_coupling == 0:  # additive coupling
                z2 += f("f1", z1, hps.width)

            elif hps.flow_coupling == 1:  # affine
                h = f("f1", z1, hps.width, n_z)  # the NN(.) in the Glow paper
                shift = h[:, :, :, :, 0::2]
                # scale = tf.exp(h[:, :, :, 1::2])
                scale = tf.nn.sigmoid(h[:, :, :, :, 1::2] + 2.)
                z2 += shift
                z2 *= scale
                logdet += tf.reduce_sum(tf.log(scale), axis=[1, 2, 3, 4])
            else:
                raise Exception()

            z = tf.concat([z1, z2], 4)

        else:

            z1 = z[:, :, :, :, :n_z // 2]
            z2 = z[:, :, :, :, n_z // 2:]

            if hps.flow_coupling == 0:
                z2 -= f("f1", z1, hps.width)
            elif hps.flow_coupling == 1:
                h = f("f1", z1, hps.width, n_z)
                shift = h[:, :, :, :, 0::2]
                # scale = tf.exp(h[:, :, :, 1::2])
                scale = tf.nn.sigmoid(h[:, :, :, :, 1::2] + 2.)
                z2 /= scale
                z2 -= shift
                logdet -= tf.reduce_sum(tf.log(scale), axis=[1, 2, 3, 4])
            else:
                raise Exception()

            z = tf.concat([z1, z2], 4)

            if hps.flow_permutation == 0:
                z = Z.reverse_features("reverse", z, reverse=True)
            elif hps.flow_permutation == 1:
                z = Z.shuffle_features("shuffle", z, reverse=True)
            elif hps.flow_permutation == 2:
                z, logdet = invertible_1x1_conv(
                    "invconv", z, logdet, reverse=True)
            else:
                raise Exception()

            z, logdet = Z.actnorm("actnorm", z, logdet=logdet, reverse=True)

    return z, logdet


def f(name, h, width, n_out=None):  # the NN(.) in the Glow paper
    n_out = n_out or int(h.get_shape()[4])
    with tf.variable_scope(name):
        h = tf.nn.relu(Z.conv3d("l_1", h, width))
        h = tf.nn.relu(Z.conv3d("l_2", h, width, filter_size=[1, 1, 1]))
        h = Z.conv3d_zeros("l_last", h, n_out)
    return h


# Invertible 1x1 conv
@add_arg_scope
def invertible_1x1_conv(name, z, logdet, reverse=False):
    if True:  # Set to "False" to use the LU-decomposed version
        with tf.variable_scope(name):
            shape = Z.int_shape(z)
            w_shape = [shape[4], shape[4]]

            # Sample a random orthogonal matrix:
            w_init = np.linalg.qr(np.random.randn(
                *w_shape))[0].astype('float32')

            w = tf.get_variable("W", dtype=tf.float32, initializer=w_init)

            # dlogdet = tf.linalg.LinearOperator(w).log_abs_determinant() * shape[1]*shape[2]
            dlogdet = tf.cast(tf.log(abs(tf.matrix_determinant(
                tf.cast(w, 'float64')))), 'float32') * shape[1]*shape[2]*shape[3]

            if not reverse:

                _w = tf.reshape(w, [1, 1, 1] + w_shape)
                z = tf.nn.conv3d(z, _w, [1, 1, 1, 1, 1],
                                 'SAME', data_format='NDHWC')
                logdet += dlogdet

                return z, logdet
            else:

                _w = tf.matrix_inverse(w)
                _w = tf.reshape(_w, [1, 1, 1]+w_shape)
                z = tf.nn.conv3d(z, _w, [1, 1, 1, 1, 1],
                                 'SAME', data_format='NDHWC')
                logdet -= dlogdet

                return z, logdet

    else:

        # LU-decomposed version
        shape = Z.int_shape(z)
        with tf.variable_scope(name):

            dtype = 'float64'

            # Random orthogonal matrix:
            import scipy
            np_w = scipy.linalg.qr(np.random.randn(shape[4], shape[4]))[
                0].astype('float32')

            np_p, np_l, np_u = scipy.linalg.lu(np_w)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(abs(np_s))
            np_u = np.triu(np_u, k=1)

            p = tf.get_variable("P", initializer=np_p, trainable=False)
            l = tf.get_variable("L", initializer=np_l)
            sign_s = tf.get_variable(
                "sign_S", initializer=np_sign_s, trainable=False)
            log_s = tf.get_variable("log_S", initializer=np_log_s)
            # S = tf.get_variable("S", initializer=np_s)
            u = tf.get_variable("U", initializer=np_u)

            p = tf.cast(p, dtype)
            l = tf.cast(l, dtype)
            sign_s = tf.cast(sign_s, dtype)
            log_s = tf.cast(log_s, dtype)
            u = tf.cast(u, dtype)

            w_shape = [shape[4], shape[4]]

            l_mask = np.tril(np.ones(w_shape, dtype=dtype), -1)
            l = l * l_mask + tf.eye(*w_shape, dtype=dtype)
            u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
            w = tf.matmul(p, tf.matmul(l, u))

            if True:
                u_inv = tf.matrix_inverse(u)
                l_inv = tf.matrix_inverse(l)
                p_inv = tf.matrix_inverse(p)
                w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))
            else:
                w_inv = tf.matrix_inverse(w)

            w = tf.cast(w, tf.float32)
            w_inv = tf.cast(w_inv, tf.float32)
            log_s = tf.cast(log_s, tf.float32)

            if not reverse:

                w = tf.reshape(w, [1, 1, 1] + w_shape)
                z = tf.nn.conv3d(z, w, [1, 1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet += tf.reduce_sum(log_s) * (shape[1]*shape[2]*shape[3])

                return z, logdet
            else:

                w_inv = tf.reshape(w_inv, [1, 1, 1]+w_shape)
                z = tf.nn.conv3d(
                    z, w_inv, [1, 1, 1, 1, 1], 'SAME', data_format='NHWC')
                logdet -= tf.reduce_sum(log_s) * (shape[1]*shape[2]*shape[3])

                return z, logdet


@add_arg_scope
def split3d(name, level, z, y_onehot, z_prior=None, objective=0.):
    with tf.variable_scope(name + str(level)):
        n_z = Z.int_shape(z)[4]
        z1 = z[:, :, :, :, :n_z // 2]
        z2 = z[:, :, :, :, n_z // 2:]
        shape = [tf.shape(z1)[0]] + Z.int_shape(z1)[1:]
        #############################
        # z_p = z1
        # if z_prior is not None:
        #     n_z_prior = Z.int_shape(z_prior)[3]
        #     n_z_p = Z.int_shape(z_p)[3]
        #     # w = tf.get_variable("W_split", [1, 1, n_z_prior, n_z_p], tf.float32,
        #     #                     initializer=tf.zeros_initializer())
        #     # z_p -= tf.nn.conv2d(z_prior, w, strides=[1, 1, 1, 1], padding='SAME')###########!!!!!!!!!!####### +  or - ##
        #     # z_p -= Z.conv2d_zeros('p_o', z_prior, n_z_prior, n_z_p)
        #     z_p += Z.myMLP(3, z_prior, n_z_prior, n_z_p)
        #############################
        pz = split3d_prior(y_onehot, shape,  z_prior, level)
        objective += pz.logp(z2)

        z1 = Z.imitate_squeeze_3d(z1)

        # z1 = Z.squeeze3d(z1)
        eps = pz.get_eps(z2)
        return z1, z2, objective, eps,


@add_arg_scope
def split3d_reverse(name, level, z,  y_onehot, z_provided, eps, eps_std, z_prior=None):
    with tf.variable_scope(name + str(level)):

        # z1 = Z.unsqueeze3d(z)
        z1 = Z.imitate_unsqueeze_3d(z)

        # n_z = Z.int_shape(z1)[3]
        shape = [tf.shape(z1)[0]] + Z.int_shape(z1)[1:]

        # z_p = z1
        #############################
        # if z_prior is not None:
        #     #z_prior = Z.unsqueeze2d(z_prior)
        #     n_z_prior = Z.int_shape(z_prior)[3]
        #     # w = tf.get_variable("W_split", [1, 1, n_z_prior, n_z], tf.float32,
        #     #                     initializer=tf.zeros_initializer())
        #     # z_p -= tf.nn.conv2d(z_prior, w, strides=[1, 1, 1, 1], padding='SAME') ###########!!!!!!!!!!####### +  or - ##
        #
        #     z_p += Z.myMLP(3, z_prior, n_z_prior, n_z)
        # #############################

        pz = split3d_prior(y_onehot, shape, z_prior, level)

        if z_provided is not None:
            y_onehot2 = (y_onehot - 0.5) * (-1) + 0.5
            # y_onehot = tf.zeros_like(y_onehot)
            # y_onehot2 = tf.ones_like(y_onehot)
            pz2_ = split3d_prior(y_onehot2, shape, z_prior, level)
            # z2 = z_provided +  pz.mean - pz2_.mean
            z2 = z_provided  - pz.mean + pz2_.mean #+  0.5 * (pz.logsd - pz2_.logsd)
            # z2 = pz2_.sample2(pz.get_eps(z_provided * 0.5))
                #pz2_.mean  + 0.6 * tf.exp(pz2_.logsd)
        else:
            if eps is not None:
                # Already sampled eps
                z2 = pz.sample2(eps)
            elif eps_std is not None:
                # Sample with given eps_std
                z2 = pz.sample2(pz.eps * tf.reshape(eps_std, [-1, 1, 1, 1, 1]))
            else:
                # Sample normally
                z2 = pz.sample

        z = tf.concat([z1, z2], 4)
        return z


@add_arg_scope
def split3d_prior(y, shape, z_prior, level):
    n_z = shape[-1]
    h = tf.zeros([shape[0]] + shape[1:4] + [2 * n_z])

    mean = h[:, :, :, :, :n_z]
    logsd = h[:, :, :, :, n_z:]

    if y is not None:
        temp_v = Z.linear_zeros("y_emb", y, n_z)
        mean += tf.reshape(temp_v, [-1, 1, 1, 1, n_z])


    if z_prior is not None:
        mean, logsd = Z.condFun(mean, logsd, z_prior, level)
    


    # n_z2 = int(z.get_shape()[3])
    # n_z1 = n_z2
    # h = Z.conv2d_zeros("conv", z, 2 * n_z1)
    #
    # mean = h[:, :, :, 0::2]
    # logs = h[:, :, :, 1::2]
    return Z.gaussian_diag(mean, logsd)
