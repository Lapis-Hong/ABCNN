#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/13
"""This module contains hyperparameters using tf.flags."""
import tensorflow as tf

# data params
tf.flags.DEFINE_string("train_file", 'data/MSRP/test', "train file")
tf.flags.DEFINE_string("dev_file", 'data/MSRP/dev', "dev file")
tf.flags.DEFINE_string("test_file", 'data/MSRP/test', "test file")
tf.flags.DEFINE_string("vocab_file", 'data/MSRP/vocab', "vocab file")
tf.flags.DEFINE_string("embed_file", None, "embed file")
tf.flags.DEFINE_string("predict_file", 'data/WikiQA/pred', "predict file")
tf.flags.DEFINE_string("output_file", 'result.txt', "output file")

tf.flags.DEFINE_integer("s0_max_len", 40, "max sentence length [40]")
tf.flags.DEFINE_integer("s1_max_len", 45, "max sentence length [40]")
tf.flags.DEFINE_integer("num_buckets", 1, "buckets of sequence length [1]")
tf.flags.DEFINE_integer("embedding_dim", 300, "embedding dim [300]")
tf.flags.DEFINE_integer("shuffle_buffer_size", 10000, "Shuffle buffer size")

# model params
tf.flags.DEFINE_string("model_name", "bcnn", "mdoel name, `bcnn` or `abcnn`.")
tf.flags.DEFINE_integer("model_type", 2, "model type, 1 for APCNN-1, 2 APCNN-2, 3 for APCNN-3")
tf.flags.DEFINE_string("model_dir", "save", "model path")

tf.flags.DEFINE_float("dropout", 0.8, "dropout keep prob [0.8]")
tf.flags.DEFINE_integer("num_layers", 2, "num of hidden layers [2]")
tf.flags.DEFINE_float("l2_reg", 0.004, "l2 regularization weight [0.004]")
tf.flags.DEFINE_string("pooling_method", "max", "pooling methods, `avg` or `max`.")
tf.flags.DEFINE_string("sim_method", "cosine", "similarity metrics, `cosine` or `euclidean`.")
tf.flags.DEFINE_integer("num_filters", 50, "num of conv filters [50]")
tf.flags.DEFINE_string("filter_sizes", '2,3,4', "filter sizes [2,3,4]")


# training params
tf.flags.DEFINE_integer("batch_size", 32, "train batch size [64]")
tf.flags.DEFINE_integer("max_epoch", 50, "max epoch [50]")
tf.flags.DEFINE_float("lr", 0.002, "init learning rate [adam: 0.002, sgd: 1.1]")
tf.flags.DEFINE_integer("lr_decay_epoch", 3, "learning rate decay interval [3]")
tf.flags.DEFINE_float("lr_decay_rate", 0.5, "learning rate decay rate [0.5]")
tf.flags.DEFINE_string("optimizer", "adam", "optimizer, `adam` | `rmsprop` | `sgd` [adam]")
tf.flags.DEFINE_integer("stats_per_steps", 10, "show train info steps [100]")
tf.flags.DEFINE_integer("save_per_epochs", 1, "every epochs to save model [1]")
tf.flags.DEFINE_boolean("use_learning_decay", True, "use learning decay or not [True]")
tf.flags.DEFINE_boolean("use_grad_clip", True, "whether to clip grads [False]")
tf.flags.DEFINE_integer("grad_clip_norm", 5, "max grad norm if use grad clip [5]")
tf.flags.DEFINE_integer("num_keep_ckpts", 5, "max num ckpts [5]")
tf.flags.DEFINE_integer("random_seed", 123, "random seed [123]")

# auto params, do not need to set
tf.flags.DEFINE_integer("vocab_size", None, "vocabulary size")


FLAGS = tf.flags.FLAGS
