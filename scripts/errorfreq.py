#!/usr/bin/env python3

import sys

from collections import Counter


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--min-count', default=2, type=int)
    ap.add_argument('gold')
    ap.add_argument('errors')
    return ap


def load_tsv(fn, field_num):
    data = []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split('\t')
            if len(fields) != field_num:
                raise ValueError('Expected {} TAB-separated fields, got {}'
                                 ' on line {} of file {}: {}'.format(
                                     field_num, len(fields), ln, fn, l))
            data.append(fields)
    print('Read {} lines from {}'.format(len(data), fn), file=sys.stderr)
    return data


def target_counts(data):
    counts = Counter()
    for fields in data:
        type_, text = fields[2], fields[4]
        counts[(type_, text)] += 1
    return counts


def main(argv):
    args = argparser().parse_args(argv[1:])
    gold = load_tsv(args.gold, 6)
    errors = load_tsv(args.errors, 7)

    gold_count = target_counts(gold)
    error_count = target_counts(errors)

    error_freq = { k: error_count[k] / gold_count[k] for k in error_count }

    for (type_, text), ef in sorted(error_freq.items(), key=lambda i: -i[1]):
        gc, ec = gold_count[(type_, text)], error_count[(type_, text)]
        if ec >= args.min_count:
            print('{:.5f}\t{}\t{}\t{}\t{}'.format(ef, gc, ec, type_, text))
    
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

