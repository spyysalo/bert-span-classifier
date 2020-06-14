#!/usr/bin/env python3

import sys

import tensorflow as tf

from argparse import ArgumentParser

from common import decode_tfrecord


def argparser():
    ap = ArgumentParser()
    ap.add_argument(
        '--input_file', required=True,
        help='Input TF example file (or comma-separated list of files)'
    )
    return ap


def list_tfrecord(fn, options):
    # deprecated
    # for example in tf.compat.v1.io.tf_record_iterator(fn):
    #     print(tf.train.Example.FromString(example))
    dataset = tf.data.TFRecordDataset(fn)
    dataset = dataset.map(decode_tfrecord)
    for example in iter(dataset):
        print(example)

    
def main(argv):
    args = argparser().parse_args(argv[1:])
    for fn in args.input_file.split(','):
        list_tfrecord(fn, args)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
