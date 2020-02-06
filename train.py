#!/usr/bin/env python3

import sys

import numpy as np

from common import argument_parser
from common import load_pretrained, load_tsv_data
from common import tokenize_texts, encode_tokenized
from common import create_model, create_optimizer, save_model


def main(argv):
    args = argument_parser('train').parse_args(argv[1:])
    train_labels, train_texts = load_tsv_data(args.train_data, args)
    if args.dev_data is not None:
        dev_labels, dev_texts = load_tsv_data(args.dev_data, args)
    else:
        dev_labels, dev_texts = None, None
    pretrained_model, tokenizer = load_pretrained(args)

    max_seq_len = args.max_seq_length
    replace_span = args.replace_span

    label_list = sorted(list(set(train_labels)))
    label_map = { l: i for i, l in enumerate(label_list) }
    inv_label_map = { v: k for k, v in label_map.items() }

    train_tok = tokenize_texts(train_texts, tokenizer)
    train_x = encode_tokenized(train_tok, tokenizer, max_seq_len, replace_span)
    train_y = [label_map[l] for l in train_labels]

    if dev_labels is not None and dev_texts is not None:
        dev_tok = tokenize_texts(dev_texts, tokenizer)
        dev_x = encode_tokenized(dev_tok, tokenizer, max_seq_len, replace_span)
        dev_y = [label_map[l] for l in dev_labels]
        validation_data = (dev_x, dev_y)
    else:
        validation_data = None

    model = create_model(pretrained_model, len(label_list), int(max_seq_len/2))
    model.summary(print_fn=print)

    optimizer = create_optimizer(len(train_x[0]), args)
    
    model.compile(
        optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    model.fit(
        train_x,
        train_y,
        epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        validation_data=validation_data
    )

    if validation_data is not None:
        probs = model.predict(dev_x, batch_size=args.batch_size)
        preds = np.argmax(probs, axis=-1)
        correct, total = sum(g==p for g, p in zip(dev_y, preds)), len(dev_y)
        print('Final dev accuracy: {:.1%} ({}/{})'.format(
            correct/total, correct, total))

    if args.model_dir is not None:
        print('Saving model in {}'.format(args.model_dir))
        save_model(model, tokenizer, label_list, args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
