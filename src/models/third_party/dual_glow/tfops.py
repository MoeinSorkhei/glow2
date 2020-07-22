import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np
# import horovod.tensorflow as hvd

# Debugging function
do_print_act_stats = True


# def print_act_stats(x, _str=""):
#     if not do_print_act_stats:
#         return x
#
#     # if hvd.rank() != 0:
#     #     return x
#
#     if len(x.get_shape()) == 1:
#         x_mean, x_var = tf.nn.moments(x, [0], keep_dims=True)
#     if len(x.get_shape()) == 2:
#         x_mean, x_var = tf.nn.moments(x, [0], keep_dims=True)
#     if len(x.get_shape()) == 4:
#         x_mean, x_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
#     stats = [tf.reduce_min(x_mean), tf.reduce_mean(x_mean), tf.reduce_max(x_mean),
#              tf.reduce_min(tf.sqrt(x_var)), tf.reduce_mean(tf.sqrt(x_var)), tf.reduce_max(tf.sqrt(x_var))]
#     return tf.Print(x, stats, "["+_str+"] "+x.name)

# Allreduce methods


def allreduce_sum(x):
    return x  # hvd.size() always 1 in our runs
    # if hvd.size() == 1:
    #     return x
    # return hvd.mpi_ops._allreduce(x)


def allreduce_mean(x):  # this returns the tensor itself since we have only 1 GPU
    # x = allreduce_sum(x) / hvd.size()
    x = allreduce_sum(x)  # we have one GPU, divide by 1
    return x


def default_initial_value(shape, std=0.05):
    return tf.random_normal(shape, 0., std)


def default_initializer(std=0.05):
    return tf.random_normal_initializer(0., std)


def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))

# wrapper tf.get_variable, augmented with 'init' functionality
# Get variable with data dependent init


@add_arg_scope
def get_variable_ddi(name, shape, initial_value, dtype=tf.float32, init=False, trainable=True):
    w = tf.get_variable(name, shape, dtype, None, trainable=trainable)
    if init:
        w = w.assign(initial_value)
        with tf.control_dependencies([w]):
            return w
    return w

# Activation normalization
# Convenience function that does centering+scaling


@add_arg_scope
def actnorm(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, init=False, trainable=True):
    if arg_scope([get_variable_ddi], trainable=trainable):
        if not reverse:
            x = actnorm_center(name+"_center", x, reverse)
            x = actnorm_scale(name+"_scale", x, scale, logdet,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
        else:
            x = actnorm_scale(name + "_scale", x, scale, logdet,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
            x = actnorm_center(name+"_center", x, reverse)
        if logdet != None:
            return x, logdet
        return x

# Activation normalization


@add_arg_scope
def actnorm_center(name, x, reverse=False):
    shape = x.get_shape()
    with tf.variable_scope(name):
        assert len(shape) == 2 or len(shape) == 5
        if len(shape) == 2:
            x_mean = tf.reduce_mean(x, [0], keepdims=True)
            b = get_variable_ddi(
                "b", (1, int_shape(x)[1]), initial_value=-x_mean)
        elif len(shape) == 5:
            x_mean = tf.reduce_mean(x, [0, 1, 2, 3], keepdims=True)
            b = get_variable_ddi(
                "b", (1, 1, 1, 1, int_shape(x)[4]), initial_value=-x_mean)

        if not reverse:
            x += b
        else:
            x -= b

        return x

# Activation normalization


@add_arg_scope
def actnorm_scale(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, init=False, trainable=True):
    shape = x.get_shape()
    with tf.variable_scope(name), arg_scope([get_variable_ddi], trainable=trainable):
        assert len(shape) == 2 or len(shape) == 5
        if len(shape) == 2:
            x_var = tf.reduce_mean(x**2, [0], keepdims=True)
            logdet_factor = 1
            _shape = (1, int_shape(x)[1])

        elif len(shape) == 5:
            x_var = tf.reduce_mean(x**2, [0, 1, 2, 3], keepdims=True)
            logdet_factor = int(shape[1])*int(shape[2])*int(shape[3])
            _shape = (1, 1, 1, 1, int_shape(x)[4])

        if batch_variance:
            x_var = tf.reduce_mean(x**2, keepdims=True)

        if init and False:
            # MPI all-reduce
            x_var = allreduce_mean(x_var)
            # Somehow this also slows down graph when not initializing
            # (it's not optimized away?)

        if True:
            logs = get_variable_ddi("logs", _shape, initial_value=tf.log(
                scale/(tf.sqrt(x_var)+1e-6))/logscale_factor)*logscale_factor
            if not reverse:
                x = x * tf.exp(logs)
            else:
                x = x * tf.exp(-logs)
        else:
            # Alternative, doesn't seem to do significantly worse or better than the logarithmic version above
            s = get_variable_ddi("s", _shape, initial_value=scale /
                                 (tf.sqrt(x_var) + 1e-6) / logscale_factor)*logscale_factor
            logs = tf.log(tf.abs(s))
            if not reverse:
                x *= s
            else:
                x /= s

        if logdet != None:
            dlogdet = tf.reduce_sum(logs) * logdet_factor
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet

        return x

# Linear layer with layer norm


@add_arg_scope
def linear(name, x, width, do_weightnorm=True, do_actnorm=True, initializer=None, scale=1.):
    initializer = initializer or default_initializer()
    with tf.variable_scope(name):
        n_in = int(x.get_shape()[1])
        w = tf.get_variable("W", [n_in, width],
                            tf.float32, initializer=initializer)
        if do_weightnorm:
            w = tf.nn.l2_normalize(w, [0])
        x = tf.matmul(x, w)
        x += tf.get_variable("b", [1, width],
                             initializer=tf.zeros_initializer())
        if do_actnorm:
            x = actnorm("actnorm", x, scale)
        return x

# Linear layer with zero init

@add_arg_scope
def linear_zeros(name, x, width, logscale_factor=3):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        n_in = int(x.get_shape()[1])
        w = tf.get_variable("W", [n_in, width], tf.float32,
                            initializer=tf.zeros_initializer())
        x = tf.matmul(x, w)
        x += tf.get_variable("b", [1, width],
                             initializer=tf.zeros_initializer())
        x *= tf.exp(tf.get_variable("logs",
                                    [1, width], initializer=tf.zeros_initializer()) * logscale_factor)
        return x

# Slow way to add edge padding
def add_edge_padding(x, filter_size):
    assert filter_size[0] % 2 == 1
    if filter_size[0] == 1 and filter_size[1] == 1 and filter_size[2] == 1:
        return x
    a = (filter_size[0] - 1) // 2  # anteroposterior padding size (depth)
    b = (filter_size[1] - 1) // 2  # vertical padding size (height)
    c = (filter_size[2] - 1) // 2  # horizontal padding size (width)
    if True:
        x = tf.pad(x, [[0, 0], [a, a], [b, b], [c, c], [0, 0]])
        name = "_".join([str(dim) for dim in [a, b, c, *int_shape(x)[1:4]]])
        pads = tf.get_collection(name)
        if not pads:
            # if hvd.rank() == 0:
            #     print("Creating pad", name)
            pad = np.zeros([1] + int_shape(x)[1:4] + [1], dtype='float32')
            pad[:, :a, :, :, 0] = 1.
            pad[:, -a:, :, :, 0] = 1.
            pad[:, :, :b, :, 0] = 1.
            pad[:, :, -b:, :, 0] = 1.
            pad[:, :, :, :c, 0] = 1.
            pad[:, :, :, -c:, 0] = 1.
            pad = tf.convert_to_tensor(pad)
            tf.add_to_collection(name, pad)
        else:
            pad = pads[0]
        pad = tf.tile(pad, [tf.shape(x)[0], 1, 1, 1, 1])
        x = tf.concat([x, pad], axis=4)
    else:
        pad = tf.pad(tf.zeros_like(x[:, :, :, :, :1]) - 1,
                     [[0, 0], [a, a], [b, b], [c, c], [0, 0]]) + 1
        x = tf.pad(x, [[0, 0], [a, a], [b, b], [c, c], [0, 0]])
        x = tf.concat([x, pad], axis=4)
    return x


@add_arg_scope
def conv3d(name, x, width, filter_size=[3, 3, 3], stride=[1, 1, 1], pad="SAME", do_weightnorm=False,
           do_actnorm=True, context1d=None, skip=1, edge_bias=True):
    with tf.variable_scope(name):
        if edge_bias and pad == "SAME":
            x = add_edge_padding(x, filter_size)
            pad = 'VALID'

        n_in = int(x.get_shape()[4])

        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        w = tf.get_variable("W", filter_shape, tf.float32,
                            initializer=default_initializer())
        if do_weightnorm:
            w = tf.nn.l2_normalize(w, [0, 1, 2, 3])
        if skip == 1:
            x = tf.nn.conv3d(x, w, stride_shape, pad, data_format='NDHWC')
        else:
            assert stride[0] == 1 and stride[1] == 1 and stride[2] == 1
            x = tf.nn.conv3d(x, w, stride_shape, pad, data_format='NDHWC', dilations=[1, 1, skip, skip, 1])

        if do_actnorm:
            x = actnorm("actnorm", x)
        else:
            x += tf.get_variable("b", [1, 1, 1, 1, width],
                                 initializer=tf.zeros_initializer())

        if context1d != None:
            x += tf.reshape(linear("context", context1d,
                                   width), [-1, 1, 1, 1, width])
    return x


# @add_arg_scope
# def separable_conv2d(name, x, width, filter_size=[3, 3], stride=[1, 1], padding="SAME", do_actnorm=True, std=0.05):
#     n_in = int(x.get_shape()[3])
#     with tf.variable_scope(name):
#         assert filter_size[0] % 2 == 1 and filter_size[1] % 2 == 1
#         strides = [1] + stride + [1]
#         w1_shape = filter_size + [n_in, 1]
#         w1_init = np.zeros(w1_shape, dtype='float32')
#         w1_init[(filter_size[0]-1)//2, (filter_size[1]-1)//2, :,
#                 :] = 1.  # initialize depthwise conv as identity
#         w1 = tf.get_variable("W1", dtype=tf.float32, initializer=w1_init)
#         w2_shape = [1, 1, n_in, width]
#         w2 = tf.get_variable("W2", w2_shape, tf.float32,
#                              initializer=default_initializer(std))
#         x = tf.nn.separable_conv2d(
#             x, w1, w2, strides, padding, data_format='NHWC')
#         if do_actnorm:
#             x = actnorm("actnorm", x)
#         else:
#             x += tf.get_variable("b", [1, 1, 1, width],
#                                  initializer=tf.zeros_initializer(std))
#
#     return x


@add_arg_scope
def linear_MLP(name, x, downsample_factor=4, out_final=0, trainable=True):
    n_in = int(x.get_shape()[4])
    ###############################
    ################ depends on the images_size
    ###############################
    n_l = int(np.log2(int(x.get_shape()[2]))/2)
    #print(name + ' layer of linear_MLP for condition: ' + str(n_l))
    with tf.variable_scope(name):
        width = n_in
        for i in range(0, n_l):
            n_out = width * downsample_factor
            w = tf.get_variable("filter" + str(i), [3, 3, 3, width, n_out], tf.float32, trainable=trainable,
                                initializer=tf.initializers.random_uniform(minval=-0.01, maxval=0.01))
            x = tf.nn.conv3d(x, w, strides=[1, 2, 2, 2, 1], padding='SAME')
            b = tf.get_variable("b" + str(i), [n_out], initializer=tf.zeros_initializer())
            x = tf.nn.bias_add(x, b)
            x = tf.nn.pool(x, window_shape=[2, 2, 2], pooling_type = 'AVG', strides=[2, 2, 2], padding='SAME')
#            x = tf.nn.leaky_relu(x)
            width = n_out
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, out_final,  name='fully_conn')
    return x


@add_arg_scope
def myconv3d(name, x, n_in, n_out, strides=[1, 1, 1, 1, 1], trainable=True, filter_size=[3,3,3]):
    w = tf.get_variable("filter" + name, filter_size+[n_in, n_out], tf.float32, trainable=trainable,
                        initializer=tf.initializers.random_uniform(minval=-0.05, maxval=0.05))
    x = tf.nn.conv3d(x, w, strides=strides, padding='SAME')
    # x += tf.get_variable("b" + name, [1, 1, 1, n_out],
    #                      initializer=tf.zeros_initializer())
    # x *= tf.exp(tf.get_variable("logs" + name,
    #                             [1, n_out], initializer=tf.zeros_initializer()) * logscale_factor)

    x = tf.nn.leaky_relu(x)
    return x

def downsample(x, dif, factor):

    # depth
    if dif[0] == 0:
        x = x
    elif dif[0] > 2 * factor[0]:
        x = x[:, factor[0]:-factor[0], :, :, :]
        dif[0] = dif[0] - 2 * factor[0]
    else:
        top = int(np.floor(dif[0] / 2))
        buttom = dif[0] - top
        x = x[:, top:-buttom, :, :, :]
        dif[0] = 0

    # height
    if dif[1] == 0:
        x = x
    elif dif[1] > 2 * factor[1]:
        x = x[:, :, factor[1]:-factor[1], :, :]
        dif[1] = dif[1] - 2 * factor[1]
    else:
        top = int(np.floor(dif[1] / 2))
        buttom = dif[1] - top
        x = x[:, :, top:-buttom, :, :]
        dif[1] = 0

    # width
    if dif[2] == 0:
        x = x
    elif dif[2] > 2 * factor[2]:
        x = x[:, :, :, factor[2]:-factor[2], :]
        dif[2] = dif[2] - 2 * factor[2]
    else:
        top = int(np.floor(dif[2] / 2))
        buttom = dif[2] - top
        x = x[:, :, :, top:-buttom, :]
        dif[2] = 0

    return x, dif


@add_arg_scope
def myMLP(layers, x, n_out,  width=256,  dif=[0,0,0], trainable=True):
    downsample_factor = [int(np.ceil(i / (layers * 2))) for i in dif]

    n_in = x.get_shape()[4]
    x = myconv3d('0', x, n_in, width, trainable=trainable, filter_size=[5,5,5])
    x, dif = downsample(x, dif, downsample_factor)

    for i in range(1, layers):
        if i < layers - 1:
            x = myconv3d(str(i), x, width, width, strides=[1, 1, 1, 1, 1],
                         trainable=trainable)
            x, dif = downsample(x, dif, downsample_factor)
        else:
            x = myconv3d(str(i), x, width, n_out, trainable=trainable)
            x, dif = downsample(x, dif, downsample_factor)
    return x


@add_arg_scope
def condFun(mean, logsd, z_prior, n_layer=2, trainable=True):
    n_z = int(mean.get_shape()[4])

    dif = [i-j for i, j in zip(z_prior.get_shape().as_list()[1:4],
                                 mean.get_shape().as_list()[1:4])]

    if n_layer == 0:
        w = tf.get_variable("W_prior", mean.get_shape().as_list()[1:], tf.float32, trainable=trainable,
                            initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1))
        mean += tf.multiply(w, z_prior)
        logsd = logsd#0.5 * tf.log(tf.subtract(tf.exp(2 * logsd), w ** 2))

    elif n_layer == 1:
        n_in = z_prior.get_shape()[4]
        z_prior = myconv3d('0', z_prior, n_in, n_z, trainable=trainable, filter_size=[5, 5, 5])

        mean += z_prior[:, :, :, :, :n_z]
        logsd += 0  # z_prior[:, :, :, n_z:]
    else:
        z_prior = myMLP(n_layer, z_prior, n_z, dif=dif, trainable=trainable)
        mean += z_prior[:, :, :, :, :n_z]
        logsd += 0#z_prior[:, :, :, n_z:]

    return mean, logsd




@add_arg_scope
def myMLP_2x_downsample(layers, x, n_out,  width=256,  downsample_factor=1, trainable=True, skip=1):
    n_in = x.get_shape()[4]
    x = myconv3d('0', x, n_in, width, trainable=trainable, filter_size=[5,5,5])
    for i in range(1, layers):
        if i < layers - 1:
            if downsample_factor > 1:
                x = myconv3d(str(i), x, width, width, strides=[1, 2, 2, 2, 1],
                             trainable=trainable)
                downsample_factor /= 2
            else:
                x = myconv3d(str(i), x, width, width, strides=[1, 1, 1, 1, 1],
                             trainable=trainable)

    else:
        x = myconv3d(str(i), x, width, n_out, trainable=trainable)
    return x

# @add_arg_scope
# def condFun(mean, logsd, z_prior, n_layer=2):
#     n_z = int(z_prior.get_shape()[3])
#
#     if n_layer == 0:
#         w = tf.get_variable("W_prior", mean.get_shape().as_list()[1:], tf.float32,
#                             initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1))
#         mean += tf.multiply(w, z_prior)
#     elif n_layer == 1:
#         z_prior = myconv2d('l1_Net', z_prior, n_z, n_z, logscale_factor=3)
#         mean += z_prior
#     elif n_layer > 1:
#         z_prior = myMLP(n_layer, z_prior, n_z, n_z)
#         mean += z_prior
#
#     #logsd = 0.5 * tf.log(tf.subtract(tf.exp(2*logsd), w**2))
#     logsd = tf.get_variable("Sigma_cond", logsd.get_shape().as_list()[1:], tf.float32,
#                             initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1))
#
#     return mean, logsd

@add_arg_scope
def condFun_2x_downsample(mean, logsd, z_prior, n_layer=2, trainable=True):
    n_z = int(mean.get_shape()[4])
    downsample_factor = int(z_prior.get_shape().as_list()[1] / mean.get_shape().as_list()[1])

    if n_layer == 0:
        w = tf.get_variable("W_prior", mean.get_shape().as_list()[1:], tf.float32, trainable=trainable,
                            initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1))
        mean += tf.multiply(w, z_prior)
        logsd = logsd#0.5 * tf.log(tf.subtract(tf.exp(2 * logsd), w ** 2))

    elif n_layer == 1:
        if downsample_factor > 2:
            raise ValueError('One layer network for a large downsample_factor.')

        z_prior = myconv3d('l1_Net', z_prior, n_z, n_z ,
                           strides=[1, downsample_factor, downsample_factor, downsample_factor,1],
                           trainable=trainable)
        mean += z_prior[:,:,:,:,n_z]
        logsd += 0#z_prior[:,:,:,n_z:]

    elif n_layer > 1:
        z_prior = myMLP_2x_downsample(n_layer, z_prior, n_z,
                        downsample_factor=downsample_factor, trainable=trainable)
        mean += z_prior[:, :, :, :, n_z]
        logsd += 0#z_prior[:, :, :, n_z:]

    #logsd = 0.5 * tf.log(tf.subtract(tf.exp(2*logsd), w**2))
    # logsd = tf.get_variable("Sigma_cond", logsd.get_shape().as_list()[1:], tf.float32,
    #                         initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1))

    return mean, logsd



@add_arg_scope
def conv3d_zeros(name, x, width, filter_size=[3, 3, 3], stride=[1, 1, 1], pad="SAME",
                 logscale_factor=3, skip=1, edge_bias=True):
    with tf.variable_scope(name):
        if edge_bias and pad == "SAME":
            x = add_edge_padding(x, filter_size)
            pad = 'VALID'

        n_in = int(x.get_shape()[4])
        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        w = tf.get_variable("W", filter_shape, tf.float32,
                            initializer=tf.zeros_initializer())

        if skip == 1:
            x = tf.nn.conv3d(x, w, stride_shape, pad, data_format='NDHWC')
        else:
            assert stride[0] == 1 and stride[1] == 1 and stride[2] == 1
            x = tf.nn.conv3d(x, w, stride_shape, pad, data_format='NDHWC', dilations=[1, 1, skip, skip, 1])
        x += tf.get_variable("b", [1, 1, 1, 1, width],
                             initializer=tf.zeros_initializer())
        x *= tf.exp(tf.get_variable("logs",
                                    [1, width], initializer=tf.zeros_initializer()) * logscale_factor)
    return x


# 2X nearest-neighbour upsampling, also inspired by Jascha Sohl-Dickstein's code
# def upsample2d_nearest_neighbour(x):
#     shape = x.get_shape()
#     n_batch = int(shape[0])
#     height = int(shape[1])
#     width = int(shape[2])
#     n_channels = int(shape[3])
#     x = tf.reshape(x, (n_batch, height, 1, width, 1, n_channels))
#     x = tf.concat(2, [x, x])
#     x = tf.concat(4, [x, x])
#     x = tf.reshape(x, (n_batch, height*2, width*2, n_channels))
#     return x


# def upsample(x, factor=2):
#     shape = x.get_shape()
#     height = int(shape[1])
#     width = int(shape[2])
#     x = tf.image.resize_nearest_neighbor(x, [height * factor, width * factor])
#     return x


def imitate_squeeze_3d(x, factor=2):
    x = tf.squeeze(x, axis=1)
    x = squeeze2d(x, factor=factor)
    x = tf.expand_dims(x, axis=1)
    return x


def squeeze3d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    depth = int(shape[1])  # D
    height = int(shape[2])  # H
    width = int(shape[3])  # W
    n_channels = int(shape[4])  # C

    assert depth % factor == 0 and height % factor == 0 and width % factor == 0
    x = tf.reshape(x, [-1, depth//factor, factor,
                           height//factor, factor,
                           width//factor, factor, n_channels])
    x = tf.transpose(x, [0, 1, 3, 5, 7, 2, 4, 6])
    x = tf.reshape(x, [-1, depth//factor,
                           height//factor,
                           width//factor,
                           n_channels*factor*factor*factor])
    return x


def imitate_unsqueeze_3d(x, factor=2):
    x = tf.squeeze(x, axis=1)
    x = unsqueeze2d(x, factor=factor)
    x = tf.expand_dims(x, axis=1)
    return x


def unsqueeze3d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    depth = int(shape[1])
    height = int(shape[2])
    width = int(shape[3])
    n_channels = int(shape[4])
    assert n_channels >= factor**3 and n_channels % (factor**3) == 0
    x = tf.reshape(
        x, (-1, depth, height, width, int(n_channels/factor**3), factor, factor, factor))
    x = tf.transpose(x, [0, 1, 5, 2, 6, 3, 7, 4])
    x = tf.reshape(x, (-1, int(depth*factor),
                           int(height*factor),
                           int(width*factor),
                           int(n_channels/factor**3)))
    return x


def squeeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert height % factor == 0 and width % factor == 0
    x = tf.reshape(x, [-1, height//factor, factor,
                       width//factor, factor, n_channels])
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
    x = tf.reshape(x, [-1, height//factor, width //
                       factor, n_channels*factor*factor])
    return x


def unsqueeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert n_channels >= 4 and n_channels % 4 == 0
    x = tf.reshape(
        x, (-1, height, width, int(n_channels/factor**2), factor, factor))
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(x, (-1, int(height*factor),
                       int(width*factor), int(n_channels/factor**2)))
    return x


def list_unsqueeze3d(xs):
    _x = unsqueeze3d(xs[-1])
    for i in reversed(range(len(xs) - 1)):
        _x = tf.concat([_x, xs[i]], axis=4)
        _x = unsqueeze3d(_x)
    return _x

# Reverse features across channel dimension


def reverse_features(name, h, reverse=False):
    return h[:, :, :, :, ::-1]

# Shuffle across the channel dimension


def shuffle_features(name, h, indices=None, return_indices=False, reverse=False):
    with tf.variable_scope(name):

        rng = np.random.RandomState(
            (abs(hash(tf.get_variable_scope().name))) % 10000000)

        if indices == None:
            # Create numpy and tensorflow variables with indices
            n_channels = int(h.get_shape()[-1])
            indices = list(range(n_channels))
            rng.shuffle(indices)
            # Reverse it
            indices_inverse = [0]*n_channels
            for i in range(n_channels):
                indices_inverse[indices[i]] = i

        tf_indices = tf.get_variable("indices", dtype=tf.int32, initializer=np.asarray(
            indices, dtype='int32'), trainable=False)
        tf_indices_reverse = tf.get_variable("indices_inverse", dtype=tf.int32, initializer=np.asarray(
            indices_inverse, dtype='int32'), trainable=False)

        _indices = tf_indices
        if reverse:
            _indices = tf_indices_reverse

        if len(h.get_shape()) == 2:
            # Slice
            h = tf.transpose(h)
            h = tf.gather(h, _indices)
            h = tf.transpose(h)
        elif len(h.get_shape()) == 5:
            # Slice
            h = tf.transpose(h, [4, 1, 2, 3, 0])
            h = tf.gather(h, _indices)
            h = tf.transpose(h, [4, 1, 2, 3, 0])
        if return_indices:
            return h, indices
        return h


# def embedding(name, y, n_y, width):
#     with tf.variable_scope(name):
#         params = tf.get_variable(
#             "embedding", [n_y, width], initializer=default_initializer())
#         embeddings = tf.gather(params, y)
#         return embeddings

# Random variables


def flatten_sum(logps):
    if len(logps.get_shape()) == 2:
        return tf.reduce_sum(logps, [1])
    elif len(logps.get_shape()) == 5:
        return tf.reduce_sum(logps, [1, 2, 3, 4])
    else:
        raise Exception()


# def standard_gaussian(shape):
#     return gaussian_diag(tf.zeros(shape), tf.zeros(shape))


def gaussian_diag(mean, logsd):
    class o(object):
        pass
    o.mean = mean
    o.logsd = logsd
    o.eps = tf.random_normal(tf.shape(mean))
    o.sample = mean + tf.exp(logsd) * o.eps
    o.sample2 = lambda eps: mean + tf.exp(logsd) * eps
    o.logps = lambda x: -0.5 * \
        (np.log(2 * np.pi) + 2. * logsd + (x - mean) ** 2 / tf.exp(2. * logsd))
    o.logp = lambda x: flatten_sum(o.logps(x))
    o.get_eps = lambda x: (x - mean) / tf.exp(logsd)
    return o


# def discretized_logistic_old(mean, logscale, binsize=1 / 256.0, sample=None):
#    scale = tf.exp(logscale)
#    sample = (tf.floor(sample / binsize) * binsize - mean) / scale
#    logp = tf.log(tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + 1e-7)
#    return tf.reduce_sum(logp, [1, 2, 3])

# def discretized_logistic(mean, logscale, binsize=1. / 256):
#     class o(object):
#         pass
#     o.mean = mean
#     o.logscale = logscale
#     scale = tf.exp(logscale)
#
#     def logps(x):
#         x = (x - mean) / scale
#         return tf.log(tf.sigmoid(x + binsize / scale) - tf.sigmoid(x) + 1e-7)
#     o.logps = logps
#     o.logp = lambda x: flatten_sum(logps(x))
#     return o


# def _symmetric_matrix_square_root(mat, eps=1e-10):
#     """Compute square root of a symmetric matrix.
#     Note that this is different from an elementwise square root. We want to
#     compute M' where M' = sqrt(mat) such that M' * M' = mat.
#     Also note that this method **only** works for symmetric matrices.
#     Args:
#       mat: Matrix to take the square root of.
#       eps: Small epsilon such that any element less than eps will not be square
#         rooted to guard against numerical instability.
#     Returns:
#       Matrix square root of mat.
#     """
#     # Unlike numpy, tensorflow's return order is (s, u, v)
#     s, u, v = tf.svd(mat)
#     # sqrt is unstable around 0, just use 0 in such case
#     si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
#     # Note that the v returned by Tensorflow is v = V
#     # (when referencing the equation A = U S V^T)
#     # This is unlike Numpy which returns v = V^T
#     return tf.matmul(
#         tf.matmul(u, tf.diag(si)), v, transpose_b=True)
