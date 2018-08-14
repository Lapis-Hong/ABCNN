#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/13
"""This module to generate additional text features for origin data."""
import re
from collections import Counter


def add_features():
    """Add aditional features."""
    pass


def clean_str(string):
    """Text cleaner."""
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def text_filter(infile, outfile):
    with open(outfile, 'w') as fo:
        for line in open(infile):
            label, s0, s1 = line.strip().split('\t')
            fo.write(label+'\t'+clean_str(s0)+'\t'+clean_str(s1)+'\n')


def build_vocab(infile, outfile, max_vocab_size=100000, min_word_count=3):
    """Build vocablury file."""
    tokens = []
    for line in open(infile):
        _, s0, s1 = line.strip().split('\t')
        s0 = s0.split(' ')
        s1 = s1.split(' ')
        for word in s0+s1:
            tokens.append(word)

    counter = Counter(tokens)
    word_count = counter.most_common(max_vocab_size - 1)  # sort by word freq.
    vocab = ['UNK', '<PAD>']  # for oov words and padding
    vocab += [w[0] for w in word_count if w[0] and w[1] >= min_word_count]
    print("Vocabulary size: {}".format(len(vocab)))
    with open(outfile, 'w') as fo:
        fo.write('\n'.join(vocab))

if __name__ == '__main__':
    # add_features()
    # text_filter("MSRP/train.txt", "MSRP/train")
    text_filter("MSRP/test.txt", "MSRP/test")
    text_filter("MSRP/dev.txt", "MSRP/dev")
    # build_vocab("MSRP/train", "MSRP/vocab")