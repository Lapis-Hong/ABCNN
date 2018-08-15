#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/14
"""This module for model training."""
import os
import datetime

import tensorflow as tf

from config import FLAGS
from model import *
from dataset import get_iterator
from utils import print_args, load_vocab


def train():
    # Training
    tf.set_random_seed(FLAGS.random_seed)
    with tf.Session() as sess:
        init_ops = [tf.global_variables_initializer(),
                    tf.local_variables_initializer(), tf.tables_initializer()]
        sess.run(init_ops)

        for epoch in range(FLAGS.max_epoch):
            step = 0
            if FLAGS.use_learning_decay and (epoch+1) % FLAGS.lr_decay_epoch == 0:
                FLAGS.lr *= FLAGS.lr_decay_rate
            print('\nepoch: {}\tlearning rate: {}'.format(epoch+1, FLAGS.lr))

            sess.run(iterator.initializer)
            while True:
                try:
                    # x1, x2 = sess.run([model.x1, model.x2])
                    # print(x1)
                    _, loss = model.train(sess)

                    # logits = sess.run(model.logits)
                    # print(logits)
                    # out1, out2, sim = sess.run([model.out1, model.out2, model.features])
                    # # print(out1[0])
                    # # print(out2[0])
                    # f1, f2, a = sess.run([model.f1, model.f2, model.a])
                    # print(f1)
                    # print(f2)
                    # print(sim)
                    # print(a)
                    step += 1
                    # show train batch metrics
                    if step % FLAGS.stats_per_steps == 0:
                        time_str = datetime.datetime.now().isoformat()
                        print('{}\tepoch {:2d}\tstep {:3d}\ttrain loss={:.4f}'.format(
                            time_str, epoch+1, step, loss))
                except tf.errors.OutOfRangeError:
                    print("\n"+"="*25+" Finish train {} epoch ".format(epoch+1)+"="*25+"\n")
                    break

            if (epoch+1) % FLAGS.save_per_epochs == 0:
                if not os.path.exists(FLAGS.model_dir):
                    os.mkdir(FLAGS.model_dir)
                save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
                model.save(sess, save_path)
                print("Epoch {}, saved checkpoint to {}".format(epoch+1, save_path))


if __name__ == '__main__':
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Params Preparation
    print_args(FLAGS)
    vocab_table, _, vocab_size = load_vocab(FLAGS.vocab_file)
    FLAGS.vocab_size = vocab_size

    # Model Preparation
    mode = tf.estimator.ModeKeys.TRAIN
    iterator = get_iterator(
        FLAGS.train_file, vocab_table, FLAGS.batch_size,
        s0_max_len=FLAGS.s0_max_len,
        s1_max_len=FLAGS.s1_max_len,
        num_buckets=FLAGS.num_buckets,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size,
        padding=True,
    )
    if FLAGS.model_name.lower() == "bcnn":
        model = BCNN(iterator, FLAGS, mode)
    else:
        model = ABCNN(iterator, FLAGS, mode, FLAGS.model_type)

    train()