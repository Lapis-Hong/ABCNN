#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/13
"""This module for model prediction."""

import time
import codecs

import tensorflow as tf

from config import FLAGS
from dataset import get_infer_iterator
from model import *
from utils import print_args, load_vocab, load_model


def predict():
    writer = codecs.getwriter("utf-8")(tf.gfile.GFile(FLAGS.output_file, "wb"))
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        # load model
        load_model(sess, FLAGS.model_dir)

        print('Start Predicting...')
        step = 0
        sess.run(iterator.initializer)
        while True:
            try:
                scores = model.predict(sess)
                for score in scores[0]:
                    writer.write(str(score)+'\n')
            except tf.errors.OutOfRangeError:
                break
            step += 1
            if step % 100 == 0:
                now_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
                print('{} predict {:2d} lines'.format(now_time, step*FLAGS.batch_size))
        print("Done. Write output into {}".format(FLAGS.output_file))
    writer.close()

if __name__ == '__main__':
    # Params Preparation
    print_args(FLAGS)
    vocab_table, _, vocab_size = load_vocab(FLAGS.vocab_file)
    FLAGS.vocab_size = vocab_size
    FLAGS.batch_size = 1

    # Model Preparation
    mode = tf.estimator.ModeKeys.PREDICT

    iterator = get_infer_iterator(
        FLAGS.predict_file, vocab_table, 1,
        s0_max_len=FLAGS.s0_max_len,
        s1_max_len=FLAGS.s1_max_len,
        padding=True,
    )
    if FLAGS.model_name.lower() == "bcnn":
        model = BCNN(iterator, FLAGS, mode)
    else:
        model = ABCNN(iterator, FLAGS, mode, FLAGS.model_type)

    predict()
