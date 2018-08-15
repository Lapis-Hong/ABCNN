#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/6
"""This module contains some model utility functions."""
import codecs
from collections import namedtuple

import tensorflow as tf


def print_args(flags):
    """Print arguments."""
    print("\nParameters:")
    for attr in flags:
        value = flags[attr].value
        print("{}={}".format(attr, value))
    print("")


def load_vocab(vocab_file):
    """load vocab from vocab file.
    Args:
        vocab_file: vocab file path
    Returns:
        vocab_table, vocab, vocab_size
    """
    vocab_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=vocab_file, default_value=0)
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
    return vocab_table, vocab, vocab_size


def load_model(sess, ckpt):
    with sess.as_default():
        with sess.graph.as_default():
            init_ops = [tf.global_variables_initializer(),
                        tf.local_variables_initializer(), tf.tables_initializer()]
            sess.run(init_ops)
            # load saved model
            ckpt_path = tf.train.latest_checkpoint(ckpt)
            print("Loading saved model: " + ckpt_path)

            # reader = tf.train.NewCheckpointReader(ckpt+'model.ckpt_0.876-580500')
            # variables = reader.get_variable_to_shape_map()
            # for v in variables:
            #     print(v)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_path)


def metrics_for_ie(qa_pairs):
    """Compute MAP and MRR for ie.
    Args:
        qa_pairs: question answers dict key: question, value: (answer, label, prob).
    Returns:
        A namedtuple(map, mrr)
    """
    map_, mrr = 0, 0
    for q in qa_pairs:
        p, ap = 0, 0  # p positive num
        mrr_check = False

        qa_pairs[q] = sorted(qa_pairs[q], key=lambda x: x[-1], reverse=True)

        for idx, (s2, label, prob) in enumerate(qa_pairs[q]):
            if label == 1:
                if not mrr_check:
                    mrr += 1 / (idx + 1)
                    mrr_check = True
                p += 1
                ap += p / (idx + 1)
        ap /= p
        map_ += ap

    map_ /= len(qa_pairs)
    mrr /= len(qa_pairs)

    return map_, mrr


def metrics_for_binary_classification(y_true, y_pred):
    """Compute metircs of acc, precision, recall, f1 and confusion matrix.
    Args:
        y_true: real label, zero for negative, non-zero for positive.
        y_pred: predicted label, zero for negative, non-zero for positive.
    Returns:
        A namedtuple(acc, p, r, f1, cm)
    """
    tp = fp = tn = fn = 0
    for real, pred in zip(y_true, y_pred):
        if real and pred:
            tp += 1
        elif not real and pred:
            fp += 1
        elif real and not pred:
            fn += 1
        else:
            tn += 1
    acc = (tp+tn) / float(tp+fp+tn+fn)
    p = tp / float(tp+fp) if tp+fp > 0 else 0
    r = tp / float(tp+fn) if tp+fn > 0 else 0
    f1 = 2*p*r / float(p+r) if p+r > 0 else 0
    cm = [[tn, fp], [fn, tp]]
    m = namedtuple('Metrics', 'acc p r f1 cm')
    return m(acc, p, r, f1, cm)

