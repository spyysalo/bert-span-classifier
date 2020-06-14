#!/usr/bin/env python3

import sys

import tensorflow as tf

import bert_tokenization as tokenization

from collections import OrderedDict
from argparse import ArgumentParser

from common import load_labels, tsv_generator
from config import DEFAULT_SEQ_LEN


def argparser():
    ap = ArgumentParser()
    ap.add_argument(
        '--input_file', required=True,
        help='Input data in TSV format'
    )
    ap.add_argument(
        '--output_file', required=True,
       help='Output TF example file'
    )
    ap.add_argument(
        '--labels', required=True,
        help='File containing list of labels'
    )
    ap.add_argument(
        '--vocab_file', required=True,
        help='Vocabulary file that BERT model was trained on'
    )
    ap.add_argument(
        '--max_seq_length', type=int, default=DEFAULT_SEQ_LEN,
        help='Maximum input sequence length in WordPieces'
    )
    ap.add_argument(
        '--do_lower_case', default=False, action='store_true',
        help='Lower case input text (for uncased models)'
    )
    ap.add_argument(
        '--replace_span', default=None,
        help='Replace span text with given special token'
    )
    ap.add_argument(
        '--label_field', type=int, default=-4,
        help='Index of label in TSV data (1-based)'
    )
    ap.add_argument(
        '--text_fields', type=int, default=-3,
        help='Index of first text field in TSV data (1-based)'
    )
    return ap


class Example(object):
    def __init__(self, x, y):
        assert len(x) == 2
        self.token_ids = x[0]
        self.segment_ids = x[1]
        self.label = y

    def to_tf_example(self):
        features = OrderedDict()
        features['Input-Token'] = create_int_feature(self.token_ids)
        features['Input-Segment'] = create_int_feature(self.segment_ids)
        features['label'] = create_int_feature([self.label])
        return tf.train.Example(features=tf.train.Features(feature=features))

    def __str__(self):
        return 'token_ids: {}\nsegment_ids: {}\nlabel: {}'.format(
            ' '.join(str(t) for t in self.token_ids),
            ' '.join(str(s) for s in self.segment_ids),
            self.label
        )


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def write_examples(examples, output_file):
    count = 0
    with tf.io.TFRecordWriter(output_file) as writer:
        for example in examples:
            tf_example = example.to_tf_example()
            writer.write(tf_example.SerializeToString())
            count += 1
    print('wrote {} examples to {}'.format(count, output_file), file=sys.stderr)


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file,
        do_lower_case=args.do_lower_case
    )
    label_list = load_labels(args.labels)
    label_map = { l: i for i, l in enumerate(label_list) }

    examples = []
    for x, y in tsv_generator(args.input_file, tokenizer, label_map, args):
        examples.append(Example(x, y))

    write_examples(examples, args.output_file)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
