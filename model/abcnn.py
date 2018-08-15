#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/14
import tensorflow as tf

from bcnn import BCNN


class ABCNN(BCNN):
    """This class implements ABCNN models (https://arxiv.org/pdf/1512.05193.pdf)."""

    def __init__(self, iterator, params, mode, model_type):
        """model_type: Int 1 for ABCNN-1, 2 for ABCNN-2, 3 for ABCNN-3"""
        self.model_type = model_type
        super(ABCNN, self).__init__(iterator, params, mode)

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
        """
        d = x.get_shape().as_list()[2]
        conv = tf.layers.conv2d(
            inputs=x,
            filters=self.d,
            kernel_size=(w, d),
            padding="VALID",
            activation=tf.nn.tanh,  # origin paper use tanh
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            kernel_regularizer=self.regularizer,
            bias_initializer=tf.constant_initializer(0.01),
        )  # (b, s+w-1, 1, d)
        conv = tf.transpose(conv, [0, 1, 3, 2], name="conv_trans")  # (b, s+w-1, d, 1)
        print("conv_trans", conv.get_shape().as_list())
        return conv

    @staticmethod
    def _pooling(x, pool_size, method="avg"):
        """Pooling layer, include both w_pool and all_pool
        Args:
            x: Input tensor (b, s+w-1, d, c)
            pool_size: pool_size, determine w_pool or all_pool
            method: pooling method, `avg` or `max`, default to avg, as used in paper.
        """
        if method.lower() == "avg":
            pool_func = tf.layers.average_pooling2d
        elif method.lower() == "max":
            pool_func = tf.layers.max_pooling2d
        else:
            raise ValueError("Invalid pooling type, expected `avg` or `max`, found {}".format(method))

        pool = pool_func(
            inputs=x,
            pool_size=(pool_size, 1),
            strides=1,
            padding="VALID",
            name="pool",
        )
        print("pool", pool.get_shape().as_list())
        return pool

    @staticmethod
    def _euclidean_distance_matrix(a, b):
        """Calculate two matrix vectors euclidean distance.
        Args:
            a: Tensor(b, s1, d, 1)
            b: Tensor(b, s2, d, 1)
        """
        a = tf.squeeze(a, axis=3)  # remove dim 4
        b = tf.squeeze(b, axis=3)
        ab = tf.matmul(a, b, transpose_b=True)  # b*m*n
        a_sq = tf.reduce_sum(a*a, axis=2, keepdims=True)
        b_sq = tf.reduce_sum(b*b, axis=2, keepdims=True)
        b_sq = tf.matrix_transpose(b_sq)  # b*1*n
        a_sq_tile = tf.tile(a_sq, multiples=[1, 1, tf.shape(ab)[2]])  # b*m*n
        b_sq_tile = tf.tile(b_sq, multiples=[1, tf.shape(ab)[1], 1])  # b*n*m
        return tf.sqrt(a_sq_tile + b_sq_tile - 2*ab)

    def gen_attention_matrix(self, f1, f2):
        """Generate attention matrix for feature map f1 and f2."""
        with tf.name_scope("attention_matrix"):
            euclidean = self._euclidean_distance_matrix(f1, f2)
            # Attention Matrix A: Tensor(m, n), A_ij = 1/(1+|f0_i - f1_j|)
            a = tf.div(1.0, 1 + euclidean, name="Attention_matrix")
        return a

    def gen_attention_feature_map(self, f1, f2):
        """generate attention feature map for feature map f1, f2
        Args:
            f1: feature map for sentence 0, (b, s1, d, 1)
            f2: feature map for sentence 0, (b, s2, d, 1)
        Returns:
            attention feature map (f1, f2) same shape with input
        """
        d = int(f1.get_shape().as_list()[2])  # representation dim
        a = self.gen_attention_matrix(f1, f2)  # (b, s1, s2)
        w1 = tf.get_variable(
            name="W1",
            shape=(self.s2, d),
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=self.regularizer,
        )
        w2 = tf.get_variable(
            name="W2",
            shape=(self.s1, d),
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=self.regularizer,
        )
        w1 = tf.tile(tf.expand_dims(w1, axis=0), [self.params.batch_size, 1, 1])  # (s2, d) -> (b, s2, d)
        w2 = tf.tile(tf.expand_dims(w2, axis=0), [self.params.batch_size, 1, 1])
        # Attention feature map F1 = A * W0  F2 = t(A) * W1
        f1 = tf.expand_dims(tf.matmul(a, w1), axis=-1, name="attention_feat_map1")
        f2 = tf.expand_dims(tf.matmul(a, w2, transpose_a=True), axis=-1, name="attention_feat_map2")
        print("attention feature map: ", f1.get_shape().as_list())
        print("attention feature map: ", f2.get_shape().as_list())
        return f1, f2

    @staticmethod
    def _attention_base_pooling(x, w, s, attention_weights, method="avg"):
        """Attention based window pooling.
        Args:
            x: Input tensor (b, s+w-1, d, 1)
            w: Filter size
            s: Sequence length 
            attention_weights: Tensor (b, s+w-1)
            method: Pooling method, `avg` or `max`, default to avg, as used in paper.
        Returns:
            Tensor (b, w, d, 1)
        """
        if method.lower() == "avg":
            pool_func = tf.reduce_mean
        elif method.lower() == "max":
            pool_func = tf.reduce_max
        else:
            raise ValueError("Invalid pooling type, expected `avg` or `max`, found {}".format(method))

        pools = []
        # s = x.get_shape().as_list[1]
        weight = tf.expand_dims(tf.expand_dims(attention_weights, axis=-1), axis=-1)  # (b, s+w-1, 1, 1)
        for i in range(s):  # (b, 1, d)
            pools.append(pool_func(
                x[:, i:i + w, :, :] * weight[:, i:i + w, :, :], axis=1, keepdims=True))
        pool = tf.concat(pools, axis=1, name="attention_pool")
        print("attention pool", pool.get_shape().as_list())

        return pool

    def _cnn_block(self, x1, x2, s1, s2, w, n, name):  # overwrite
        """Each block contains input(attention) -> wide-conv -> w-pool(attention)
        Args:
            x1: Input Tensor (b, s1, d, 1)
            x2: Input Tensor (b, s2, d, 1)
            s1: Input Tensor sequence length
            s2: Input Tensor sequence length
            w: Filter size
            n: Num filters
        Returns:
            tuple of Tensor list, each element has shape (b, d)
        """
        out1, out2 = [], []  # Representation vector list of input x1, x2
        for i in range(n):
            with tf.variable_scope("{}/cnn-block-{}".format(name, i+1)):
                if self.model_type == 1 or self.model_type == 3:
                    with tf.name_scope("attention_feauture_map"):
                        # Note in TensorFlow, the dimension is different from paper in order.
                        self.f1, self.f2 = self.gen_attention_feature_map(x1, x2)  # (b, s1, d, 1)
                        # Stack attention feature map with original embbeding input, add dim 4 chanel for stack
                        x1 = tf.concat([x1, self.f1], axis=3)  # (b, s1, d, 2)
                        x2 = tf.concat([x2, self.f2], axis=3)  # (b, s2, d, 2)

                conv1 = self._conv(x=self._wide_conv_padding(x1, w), w=w)  # (b, s0+w-1, d, 1)
                conv2 = self._conv(x=self._wide_conv_padding(x2, w), w=w)  # (b, s1+w-1, d, 1)

                if self.model_type == 2 or self.model_type == 3:
                    self.a = self.gen_attention_matrix(conv1, conv2)  # (s0+w-1, s1+w-1)
                    attention_w1, attention_w2 = tf.reduce_sum(self.a, axis=2), tf.reduce_sum(self.a, axis=1)
                    pool1 = self._attention_base_pooling(x=conv1, w=w, s=s1, attention_weights=attention_w1)
                    pool2 = self._attention_base_pooling(x=conv2, w=w, s=s2, attention_weights=attention_w2)
                else:
                    pool1 = self._pooling(x=conv1, pool_size=w)  # (b, s1, d, 1)
                    pool2 = self._pooling(x=conv2, pool_size=w)  # (b, s1, d, 1)
                all_pool1 = self._pooling(x=conv1, pool_size=s1+w-1)  # (b, 1, d, 1)
                all_pool2 = self._pooling(x=conv2, pool_size=s2+w-1)  # (b, 1, d, 1)
                out1.append(tf.squeeze(all_pool1, axis=[1, 3]))  # flatten to (b, d)
                out2.append(tf.squeeze(all_pool2, axis=[1, 3]))
                x1 = pool1  # w-pool output as input for next block
                x2 = pool2

        return out1, out2

    def _build_logits(self):
        self.x1 = tf.expand_dims(self.x1, axis=-1)
        self.x2 = tf.expand_dims(self.x2, axis=-1)
        params = self.params
        sim_vec = []  # k * n similarity score, n is num_filters, k is num_layers
        for i, w in enumerate(map(int, params.filter_sizes.split(','))):
            self.out1, self.out2 = self._cnn_block(
                self.x1, self.x2, self.s1, self.s2, w, self.n, name="window-{}".format(w))
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





