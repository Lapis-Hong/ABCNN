#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/14
import tensorflow as tf

from model import BaseModel


class BCNN(BaseModel):
    """This class implements BCNN model (https://arxiv.org/pdf/1512.05193.pdf)."""

    def __init__(self, iterator, params, mode):
        self.s1 = params.s0_max_len
        self.s2 = params.s1_max_len
        self.n = params.num_filters
        self.k = params.num_layers
        self.regularizer = tf.contrib.layers.l2_regularizer(params.l2_reg)
        super(BCNN, self).__init__(iterator, params, mode)

    @staticmethod
    def _wide_conv_padding(x, w):
        """Zero padding to inputs for wide convolution,
        padding w-1 for both sides  (s -> s+w-1)
        Args:
            x: input tensor (b, s, d, c)
            w: filter size
        Returns:
            padded input (b, s+w-1, d, c)
        """
        return tf.pad(x, [[0, 0], [w - 1, w - 1], [0, 0], [0, 0]], name="wide_conv_pad")

    def _conv(self, x, w):
        """Convolution layers
        Args:
            x: input tensor (b, s, d, c)
            w: filter size
        Returns:
            conv output tensor (b, s+w-1, n, 1)
        """
        d = x.get_shape().as_list()[2]
        conv = tf.layers.conv2d(
            inputs=x,
            filters=self.n,
            kernel_size=(w, d),
            padding="VALID",
            activation=tf.nn.tanh,  # origin paper use tanh
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            kernel_regularizer=self.regularizer,
            bias_initializer=tf.constant_initializer(0.01),
        )  # (b, s+w-1, 1, n)
        conv = tf.transpose(conv, [0, 1, 3, 2], name="conv_trans")  # (b, s+w-1, n, 1)
        print("conv_trans", conv.get_shape().as_list())
        return conv

    @staticmethod
    def _pooling(x, pool_size, method="avg", name=None):
        """Pooling layer, include both w_pool and all_pool
        Args:
            x: Input tensor (b, s+w-1, d, c)
            pool_size: pool_size, determine w_pool or all_pool
            method: pooling method, `avg` or `max`, default to avg, as used in paper.
        """
        if method.lower() == "avg":
            pool = tf.layers.average_pooling2d(inputs=x, pool_size=(pool_size, 1), strides=1, name=name)
        elif method.lower() == "max":
            pool = tf.layers.max_pooling2d(inputs=x, pool_size=(pool_size, 1), strides=1, name=name)
        else:
            raise ValueError("Invalid pooling type, expected `avg` or `max`, found {}".format(method))

        return pool

    def _cnn_block(self, x1, x2, s1, s2, w, k, name):
        """Each block contains input -> wide-conv -> w-pool
        Args:
            x1: Input Tensor (b, s1, d, 1)
            x2: Input Tensor (b, s2, d, 1)
            s1: Input Tensor sequence length
            s2: Input Tensor sequence length
            w: Filter size
            k: Num layers
        Returns:
            tuple of Tensor list, each element has same shape with input x1, x2

        """
        out1, out2 = [], []  # Representation vector list of input x1, x2
        for i in range(k):
            with tf.variable_scope("{}/cnn-block-{}".format(name, i + 1)):
                conv1 = self._conv(x=self._wide_conv_padding(x1, w), w=w)  # (b, s0+w-1, d)
                conv2 = self._conv(x=self._wide_conv_padding(x2, w), w=w)  # (b, s1+w-1, d)

                pool1 = self._pooling(x=conv1, pool_size=w, name="w_pool1")  # (b, s0, d)
                pool2 = self._pooling(x=conv2, pool_size=w, name="w_pool2")  # (b, s1, d)

                all_pool1 = self._pooling(x=conv1, pool_size=s1 + w - 1, name="all_pool1")  # (b, 1, d)
                all_pool2 = self._pooling(x=conv2, pool_size=s2 + w - 1, name="all_pool2")  # (b, 1, d)
                out1.append(tf.squeeze(all_pool1, axis=[1, 3]))  # flatten to (b, d)
                out2.append(tf.squeeze(all_pool2, axis=[1, 3]))
                x1 = pool1  # w-pool output as input for next block
                x2 = pool2

        return out1, out2

    def _build_logits(self):
        params = self.params
        sim_vec = []  # k * n similarity score, n is num_filters, k is num_layers
        for i, w in enumerate(map(int, params.filter_sizes.split(','))):
            self.out1, self.out2 = self._cnn_block(
                self.x1, self.x2, self.s1, self.s2, w, self.k, name="window-{}".format(w))

            for r1, r2 in zip(self.out1, self.out2):
                sim = self._sim(r1, r2, params.sim_method)  # (b,)
                sim_vec.append(tf.expand_dims(sim, 1))  # [(b,1), (b,1)...]
        sims = tf.concat(sim_vec, axis=1)  # (b, k*n)

        with tf.name_scope("features"):
            if self.features:
                self.features = tf.concat([self.features, sims], axis=1, name="features")
            else:
                self.features = sims

        logits = tf.layers.dense(
            self.features, 2,
            bias_initializer=tf.constant_initializer(0.1),
            name="softmax",
        )

        return logits  # (k*n, 2)







