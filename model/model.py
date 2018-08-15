#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/13
"""This module implement AP-NN abstract model class.
References:
    `Attentive Pooling Networks`, 2017
"""

import abc

import tensorflow as tf

from model_utils import create_or_load_embed


class BaseModel(object):
    """AB abstract base class."""

    def __init__(self, iterator, params, mode):
        """Initialize model, build graph.
        Args:
          iterator: instance of class BatchedInput, defined in dataset.  
          params: parameters.
          mode: train | eval | predict mode defined with tf.estimator.ModeKeys.
        """
        self.iterator = iterator
        self.params = params
        self.mode = mode
        self.target = iterator.target
        self.features = iterator.features
        self.scope = self.__class__.__name__  # instance class name

        self._build_graph()
        self._model_stats()  # print model statistics info

    def _build_graph(self):
        params = self.params
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            embeddings = create_or_load_embed(
                params.vocab_file, params.embed_file, params.vocab_size, params.embedding_dim)
            self.x1 = tf.nn.embedding_lookup(embeddings, self.iterator.s0)  # [batch_size, seq_length, embedding_size]
            self.x2 = tf.nn.embedding_lookup(embeddings, self.iterator.s1)

            self.logits = self._build_logits()  # (batch_size, num_classes)
            self.scores = tf.nn.softmax(self.logits, name="score")
            self.pred = tf.argmax(self.scores, 1, output_type=tf.int32, name="predict")  # batch_size

            if self.mode != tf.estimator.ModeKeys.PREDICT:
                with tf.name_scope("loss"):
                    # regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                    # self.target = tf.one_hot(iterator.target, params.num_classes, dtype=tf.float32)
                    # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    #     labels=self.target, logits=self.logits)) + tf.losses.get_regularization_loss()
                    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.target, logits=self.logits)) + tf.losses.get_regularization_loss()

                    if params.optimizer == "rmsprop":
                        opt = tf.train.RMSPropOptimizer(params.lr)
                    elif params.optimizer == "adam":
                        opt = tf.train.AdamOptimizer(params.lr)
                    elif params.optimizer == "sgd":
                        opt = tf.train.MomentumOptimizer(params.lr, 0.9)
                    else:
                        raise ValueError("Unsupported optimizer %s" % params.optimizer)
                    train_vars = tf.trainable_variables()
                    gradients = tf.gradients(self.loss, train_vars)
                    if params.use_grad_clip:
                        gradients, grad_norm = tf.clip_by_global_norm(
                            gradients, params.grad_clip_norm)

                    self.global_step = tf.Variable(0, trainable=False)
                    self.update = opt.apply_gradients(
                        zip(gradients, train_vars), global_step=self.global_step)

    @staticmethod
    def _sim(x, y, method="cosine"):
        """x, y shape (batch_size, vector_size)
        Args:
            method: 
                `cosine`: Defined as <x, y>/ |x|*|y|
                `euclidean`: Defined as 1/(1+|x-y|), |x-y|is Euclidean distance.
        Returns:
            sim Tensor (batch_size,)
        """
        if method.lower() == "cosine":
            # normalize_x = tf.nn.l2_normalize(x, 0)
            # normalize_y = tf.nn.l2_normalize(y, 0)
            # cosine = tf.reduce_sum(tf.multiply(normalize_x, normalize_y), 1)
            sim = tf.div(
                tf.reduce_sum(x*y, 1),
                tf.sqrt(tf.reduce_sum(x*x, 1)) * tf.sqrt(tf.reduce_sum(y*y, 1)) + 1e-6,
                name="cosine_sim")
        elif method.lower() == "euclidean":
            sim = tf.sqrt(tf.reduce_sum(tf.square(x - y), 1), name="euclidean_sim")
        else:
            raise ValueError("Invalid method, expected `cosine` or `euclidean`, found{}".format(method))

        return sim

    @abc.abstractmethod
    def _build_logits(self):
        """Subclass must implement this method.
            1. Generate representation of input x1, x2
            2. Calculate similarity between x1, x2
            3. Concat sims for classify layers
        Returns: 
            Output Tensor (x1, x2) shape(batch_size, dim, 2).
        """
        pass

    @staticmethod
    def _model_stats():
        """Print trainable variables and total model size."""

        def size(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())
        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))

    def train(self, sess):
        return sess.run([self.update, self.loss])

    def eval(self, sess):
        return sess.run([self.loss, self.pred])

    def predict(self, sess):
        return sess.run([self.pred])

    def save(self, sess, path):
        saver = tf.train.Saver(max_to_keep=self.params.num_keep_ckpts)
        saver.save(sess, path, global_step=self.global_step.eval())





