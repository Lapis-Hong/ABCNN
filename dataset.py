#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/13
"""This module contains efficient data read and transform using tf.data API.

Data iterator for modeling sentence pairs.
Related NLP Tasks:
    AS: answer selection
    PI: paraphrase identification
    TE:  textual entailment

Provide Two Mode: 
1) dataset with additional comma separated field features, set use_features=True
    Train data: 
        label \t s0 \t s1 \t features  
    Prediction data: 
        s0 \t s1 \t features

2) dataset without additional features, set use_features=False
    Train data: 
        label \t s0 \t s1
    Prediction data: 
        s0 \t s1
"""
import collections

import tensorflow as tf


class BatchedInput(
    collections.namedtuple(
        "BatchedInput", ("initializer", "s0", "s1", "target", "s0_len", "s1_len", "features"))):
    """
    s0, s1 for sentence pairs, target for label, s0_len, s1_len for each sequence length, features for additional features.
    for inference data, target is None.
    """
    pass


def _parse_csv(use_features):
    """Parse train data."""
    cols_types = [[0], [''], [''], ['']] if use_features else [[0], [''], ['']]

    def parse_func(line):
        return tf.decode_csv(line, record_defaults=cols_types, field_delim='\t')

    return parse_func


def _parse_infer_csv(use_features):
    """Parse inference data."""
    cols_types = [['']] * 3 if use_features else [['']] * 2

    def parse_func(line):
        return tf.decode_csv(line, record_defaults=cols_types, field_delim='\t')

    return parse_func


def get_infer_iterator(data_file, vocab_table, batch_size,
                       s0_max_len=None, s1_max_len=None, padding=False, use_features=False):
    """Iterator for inference.
    Args:
        data_file: data file, each line contains question, answer
        vocab_table: tf look-up table
        s0_max_len: sentence 0 max length
        s1_max_len: sentence 1 max length
        padding: Bool
            set True for cnn model to pad all samples into same length, must set s_max_len
            set False for rnn model 
        use_features: Bool, whether to use additional features.
    Returns:
        BatchedInput instance
    """
    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(_parse_infer_csv(use_features)).prefetch(batch_size)

    if not use_features:
        features = None
        dataset = dataset.map(
            lambda s0, s1: (tf.string_split([s0]).values, tf.string_split([s1]).values))
        if s0_max_len:
            dataset = dataset.map(lambda s0, s1: (s0[:s0_max_len], s1))
        if s1_max_len:
            dataset = dataset.map(lambda s0, s1: (s0, s1[:s1_max_len]))
        # Convert the word strings to ids
        dataset = dataset.map(
            lambda s0, s1: (
                tf.cast(vocab_table.lookup(s0), tf.int32),
                tf.cast(vocab_table.lookup(s1), tf.int32),
                tf.size(s0), tf.size(s1)))

        s0_pad_size = s0_max_len if padding else None
        s1_pad_size = s1_max_len if padding else None
        batched_dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([s0_pad_size]),
                tf.TensorShape([s1_pad_size]),
                tf.TensorShape([]), tf.TensorShape([])),
            padding_values=(0, 0, 0, 0))
        batched_iter = batched_dataset.make_initializable_iterator()
        s0_ids, s1_ids, s0_len, s1_len = batched_iter.get_next()

    else:
        dataset = dataset.map(
            lambda s0, s1, f: (
                tf.string_split([s0]).values, tf.string_split([s1]).values, tf.string_split([f, ","]).value))
        if s0_max_len:
            dataset = dataset.map(lambda s0, s1, f: (s0[:s0_max_len], s1, f))
        if s1_max_len:
            dataset = dataset.map(lambda s0, s1, f: (s0, s1[:s1_max_len], f))
        # Convert the word strings to ids
        dataset = dataset.map(
            lambda s0, s1, f: (
                tf.cast(vocab_table.lookup(s0), tf.int32),
                tf.cast(vocab_table.lookup(s1), tf.int32),
                tf.cast(f, tf.float32),
                tf.size(s0), tf.size(s1)))

        s0_pad_size = s0_max_len if padding else None
        s1_pad_size = s1_max_len if padding else None
        batched_dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([s0_pad_size]),
                tf.TensorShape([s1_pad_size]),
                tf.TensorShape([None]),
                tf.TensorShape([]), tf.TensorShape([])),
            padding_values=(0, 0, 0, 0, 0))
        batched_iter = batched_dataset.make_initializable_iterator()
        s0_ids, s1_ids, features, s0_len, s1_len = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        s0=s0_ids, s1=s1_ids, target=None, s0_len=s0_len, s1_len=s1_len, features=features)


def get_iterator(data_file, vocab_table, batch_size,
                 s0_max_len=None, s1_max_len=None, padding=False, use_features=False,
                 num_buckets=1, num_parallel_calls=4, shuffle_buffer_size=None):
    """Iterator for train and eval.
    Args:
        data_file: data file, each line contains question, answer_pos, answer_neg 
        vocab_table: tf look-up table
        s0_max_len: sentence 0 max length
        s1_max_len: sentence 1 max length
        padding: Bool
            set True for cnn or attention based model to pad all samples into same length, must set seq_max_len
            set False for rnn model 
        use_features: Bool, whether to use additional features.
        num_buckets: bucket according to sequence length
        shuffle_buffer_size: buffer size for shuffle
    Returns:
        BatchedInput instance
    """
    shuffle_buffer_size = shuffle_buffer_size or batch_size * 1000

    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(_parse_csv(use_features))
    if not use_features:
        features = None
        dataset = dataset.map(
            lambda target, s0, s1: (
               target, tf.string_split([s0]).values, tf.string_split([s1]).values),
            num_parallel_calls=num_parallel_calls)
        if s0_max_len:
            dataset = dataset.map(
                lambda target, s0, s1: (target, s0[:s0_max_len], s1),
                num_parallel_calls=num_parallel_calls)
        if s1_max_len:
            dataset = dataset.map(
                lambda target, s0, s1: (target, s0, s1[:s1_max_len]),
                num_parallel_calls=num_parallel_calls)

        # Convert the word strings to ids.  Word strings that are not in the
        # vocab get the lookup table's default_value integer.
        dataset = dataset.map(
            lambda target, s0, s1: (
                target,
                tf.cast(vocab_table.lookup(s0), tf.int32),
                tf.cast(vocab_table.lookup(s1), tf.int32),
                tf.size(s0), tf.size(s1)),
            num_parallel_calls=num_parallel_calls)

        s0_pad_size = s0_max_len if padding else None
        s1_pad_size = s1_max_len if padding else None
        if num_buckets > 1:  # Bucket by sequence length (buckets for lengths 0-9, 10-19, ...)
            buckets_length = s0_max_len // num_buckets
            buckets_boundaries = [buckets_length * (i+1) for i in range(num_buckets)]
            buckets_batch_sizes = [batch_size] * (len(buckets_boundaries) + 1)

            batching_func = tf.contrib.data.bucket_by_sequence_length(
                element_length_func=lambda target, s0, s1, s0_len, s1_len: (s0_len + s1_len) // 2,
                bucket_boundaries=buckets_boundaries,
                bucket_batch_sizes=buckets_batch_sizes,
                padded_shapes=(
                    tf.TensorShape([]),
                    tf.TensorShape([s0_pad_size]),
                    tf.TensorShape([s1_pad_size]),
                    tf.TensorShape([]), tf.TensorShape([])),
                padding_values=(0, 0, 0, 0, 0),
                pad_to_bucket_boundary=False
            )
            batched_dataset = dataset.apply(batching_func).prefetch(2*batch_size)
        else:
            batching_func = tf.contrib.data.padded_batch_and_drop_remainder(
                batch_size,
                padded_shapes=(
                    tf.TensorShape([]),
                    tf.TensorShape([s0_pad_size]),
                    tf.TensorShape([s1_pad_size]),
                    tf.TensorShape([]), tf.TensorShape([])),
                padding_values=(0, 0, 0, 0, 0)
            )
            batched_dataset = dataset.apply(batching_func).prefetch(2 * batch_size)

            # Note tf.data default to include last smaller batch, it cause error.
            # From tf version >= 1.10, we can use drop_remainder options
            # batched_dataset = dataset.padded_batch(
            #     batch_size,
            #     padded_shapes=(
            #         tf.TensorShape([s0_pad_size]),
            #         tf.TensorShape([s1_pad_size]),
            #         tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])),
            #         padding_values=(0, 0, 0, 0, 0)
            #     drop_remainder=True
            # ).prefetch(2*batch_size)

        batched_iter = batched_dataset.make_initializable_iterator()
        target, s0_ids, s1_ids, s0_len, s1_len = batched_iter.get_next()

    else:
        dataset = dataset.map(
            lambda target, s0, s1, f: (
                target, tf.string_split([s0]).values, tf.string_split([s1]).values, tf.string_split([f, ","]).values),
            num_parallel_calls=num_parallel_calls)
        if s0_max_len:
            dataset = dataset.map(
                lambda target, s0, s1, f: (target, s0[:s0_max_len], s1, f),
                num_parallel_calls=num_parallel_calls)
        if s1_max_len:
            dataset = dataset.map(
                lambda target, s0, s1, f: (target, s0, s1[:s1_max_len], f),
                num_parallel_calls=num_parallel_calls)

        # Convert the word strings to ids.  Word strings that are not in the
        # vocab get the lookup table's default_value integer.
        dataset = dataset.map(
            lambda target, s0, s1, f: (
                target,
                tf.cast(vocab_table.lookup(s0), tf.int32),
                tf.cast(vocab_table.lookup(s1), tf.int32),
                tf.cast(f, tf.float32),
                tf.size(s0), tf.size(s1)),
            num_parallel_calls=num_parallel_calls)

        s0_pad_size = s0_max_len if padding else None
        s1_pad_size = s1_max_len if padding else None
        if num_buckets > 1:  # Bucket by sequence length (buckets for lengths 0-9, 10-19, ...)
            buckets_length = s0_max_len // num_buckets
            buckets_boundaries = [buckets_length * (i + 1) for i in range(num_buckets)]
            buckets_batch_sizes = [batch_size] * (len(buckets_boundaries) + 1)

            batching_func = tf.contrib.data.bucket_by_sequence_length(
                element_length_func=lambda target, s0, s1, f, s0_len, s1_len: (s0_len + s1_len) // 2,
                bucket_boundaries=buckets_boundaries,
                bucket_batch_sizes=buckets_batch_sizes,
                padded_shapes=(
                    tf.TensorShape([]),
                    tf.TensorShape([s0_pad_size]),
                    tf.TensorShape([s1_pad_size]),
                    tf.TensorShape(None),
                    tf.TensorShape([]), tf.TensorShape([])),
                padding_values=(0, 0, 0, 0, 0, 0),
                pad_to_bucket_boundary=False
            )
            batched_dataset = dataset.apply(batching_func).prefetch(2 * batch_size)
        else:
            batching_func = tf.contrib.data.padded_batch_and_drop_remainder(
                batch_size,
                padded_shapes=(
                    tf.TensorShape([]),
                    tf.TensorShape([s0_pad_size]),
                    tf.TensorShape([s1_pad_size]),
                    tf.TensorShape(None),
                    tf.TensorShape([]), tf.TensorShape([])),
                padding_values=(0, 0, 0, 0, 0, 0)
            )
            batched_dataset = dataset.apply(batching_func).prefetch(2 * batch_size)
        batched_iter = batched_dataset.make_initializable_iterator()
        target, s0_ids, s1_ids, features, s0_len, s1_len = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        s0=s0_ids, s1=s1_ids, target=target, s0_len=s0_len, s1_len=s1_len, features=features)

