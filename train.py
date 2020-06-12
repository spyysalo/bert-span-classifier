#!/usr/bin/env python3

import sys

import numpy as np

from common import argument_parser
from common import load_pretrained, load_labels, load_dataset, TsvSequence
from common import tokenize_texts, encode_tokenized
from common import create_model, create_optimizer, save_model


def main(argv):
    args = argument_parser('train').parse_args(argv[1:])
    pretrained_model, tokenizer = load_pretrained(args)

    label_list = load_labels(args.labels)
    label_map = { l: i for i, l in enumerate(label_list) }
    inv_label_map = { v: k for k, v in label_map.items() }

    train_generator = TsvSequence(args.train_data, tokenizer, label_map, args)

    # train_x, train_y = load_dataset(args.train_data, tokenizer,
    #                                 args.max_seq_length, args.replace_span,
    #                                 label_map, args)

    if args.dev_data is None:
        dev_x, dev_y = None, None
        validation_data = None
    else:
        dev_x, dev_y = load_dataset(args.dev_data, tokenizer,
                                    args.max_seq_length, args.replace_span,
                                    label_map, args)
        validation_data = (dev_x, dev_y)

    output_offset = int(args.max_seq_length/2)
    model = create_model(pretrained_model, len(label_list), output_offset,
                         args.output_layer)
    model.summary(print_fn=print)

    #optimizer = create_optimizer(len(train_x[0]), args)
    optimizer = create_optimizer(train_generator.num_examples, args)
    
    model.compile(
        optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    model.fit(
        train_generator,
        epochs=args.num_train_epochs,
        validation_data=validation_data,
        workers=10
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
