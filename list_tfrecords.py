#!/usr/bin/env python3

import sys

import numpy as np
import tensorflow as tf

from argparse import ArgumentParser


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--vocab', default=None, help='BERT vocabulary')
    ap.add_argument('input_file', nargs='+', help='Input TFRecord file(s)')
    return ap


def load_vocab(fn):
    vocab = []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            vocab.append(l)
    return vocab


def list_tfrecord(fn, options):
    if options.vocab is None:
        vocab_map = None
    else:
        vocab_map = { i: t for i, t in enumerate(options.vocab) }
    # deprecated
    # for example in tf.compat.v1.io.tf_record_iterator(fn):
    #     print(tf.train.Example.FromString(example))
    dataset = tf.data.TFRecordDataset(fn)
    for record in iter(dataset):
        example = tf.train.Example.FromString(record.numpy())
        edict = dict(example.features.feature)
        for key, value in sorted(edict.items()):
            # NOTE: assumes int64
            npvalue = np.array(value.int64_list.value)
            if vocab_map is None or key != 'Input-Token':
                print('{} {}:\n{}'.format(key, npvalue.shape, npvalue))
            else:
                tokens = [vocab_map[i] for i in npvalue]
                print('{} {}:\n{}'.format(key, npvalue.shape, ' '.join(tokens)))


def main(argv):
    args = argparser().parse_args(argv[1:])
    if args.vocab is not None:
        args.vocab = load_vocab(args.vocab)
    for fn in args.input_file:
        list_tfrecord(fn, args)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
