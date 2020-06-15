#!/usr/bin/env python3

import sys

import numpy as np
import tensorflow as tf

from argparse import ArgumentParser


def argparser():
    ap = ArgumentParser()
    ap.add_argument('input_file', nargs='+', help='Input TFRecord file(s)')
    return ap


def list_tfrecord(fn, options):
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
            print('{}:\n{}'.format(key, npvalue))


def main(argv):
    args = argparser().parse_args(argv[1:])
    for fn in args.input_file:
        list_tfrecord(fn, args)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
