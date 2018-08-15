#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/14
"""This module contains some model utility functions."""
import codecs

import numpy as np
import tensorflow as tf

from utils import load_vocab


def _load_embed_txt(embed_file):
    """Load embed_file into a python dictionary.

    Note: the embed_file should be a Glove formated txt file.
        Assuming embed_size=5, for example:
        the -0.071549 0.093459 0.023738 -0.090339 0.056123
        to 0.57346 0.5417 -0.23477 -0.3624 0.4037
        and 0.20327 0.47348 0.050877 0.002103 0.060547

    Args:
      embed_file: file path to the embedding file.
    Returns:
      a dictionary that maps word to vector, and the size of embedding dimensions.
    """
    emb_dict = dict()
    emb_size = None
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
        for line in f:
            tokens = line.strip().split(" ")
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(vec), "All embedding size should be same."
            else:
                emb_size = len(vec)
    return emb_dict, emb_size


def _create_pretrained_emb_from_txt(
        vocab_file, embed_file, num_trainable_tokens=3, dtype=tf.float32, scope=None):
    """Load pretrain embeding from embed file, and return an embedding matrix.
    Args:
        embed_file: embed file path.
        num_trainable_tokens: Make the first n tokens in the vocab file as trainable
            variables. Default is 3, which is "<unk>", "<s>" and "</s>".
    """
    _, vocab, _ = load_vocab(vocab_file)
    trainable_tokens = vocab[:num_trainable_tokens]
    print("Using pretrained embedding: %s." % embed_file)
    print("  with trainable tokens:")
    emb_dict, emb_size = _load_embed_txt(embed_file)
    for token in trainable_tokens:
        print("    %s" % token)
    for token in vocab:
        if token not in emb_dict:
            emb_dict[token] = [0.0] * emb_size
    emb_mat = np.array(
        [emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())
    emb_mat = tf.constant(emb_mat)
    emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
    with tf.variable_scope(scope or "pretrain_embeddings", dtype=dtype):
        emb_mat_var = tf.get_variable(
                "emb_mat_var", [num_trainable_tokens, emb_size])  # TODO
    return tf.concat([emb_mat_var, emb_mat_const], 0)


def create_or_load_embed(vocab_file, embed_file, vocab_size, embed_size, scope=None):
    """Create a new or load an existing embedding matrix."""
    with tf.variable_scope(scope or "embedding", reuse=tf.AUTO_REUSE):
        if vocab_file and embed_file:
            embedding = _create_pretrained_emb_from_txt(vocab_file, embed_file)
        else:
            embedding = tf.get_variable(
                "W", [vocab_size, embed_size],
                initializer=tf.random_uniform_initializer(-1, 1))
            # tf.random_normal_initializer(0., embed_size ** -0.5)
            # or tf.keras.initializers.he_uniform() or
    return embedding
