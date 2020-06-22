#!/usr/bin/env python3

import sys
import os

import numpy as np

from logging import warning

from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import ModelCheckpoint

from common import argument_parser, print_versions
from common import load_pretrained, load_model, get_tokenizer, load_labels
from common import load_dataset, train_tfrecord_input, TsvSequence
from common import tokenize_texts, encode_tokenized, num_examples
from common import create_model, create_optimizer, save_model_etc

from config import CHECKPOINT_PATH


def restore_or_create_model(num_train_examples, num_labels, global_batch_size,
                            options):
    checkpoints = [
        os.path.join(options.checkpoint_dir, fn)
        for fn in os.listdir(options.checkpoint_dir)
    ]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Loading checkpoint from', latest_checkpoint, file=sys.stderr,
              flush=True)
        return load_model(latest_checkpoint)
    else:
        print('Creating new model', file=sys.stderr, flush=True)
        pretrained_model = load_pretrained(options)
        output_offset = int(options.max_seq_length/2)
        model = create_model(pretrained_model, num_labels, output_offset,
                             options.output_layer)
        optimizer = create_optimizer(num_train_examples, global_batch_size,
                                     options)
        model.compile(
            optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
        return model


def main(argv):
    print_versions()
    args = argument_parser('train').parse_args(argv[1:])

    args.train_data = args.train_data.split(',')
    if args.checkpoint_steps is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    strategy = MirroredStrategy()
    num_devices = strategy.num_replicas_in_sync
    # Batch datasets with global batch size (local * GPUs)
    global_batch_size = args.batch_size * num_devices

    tokenizer = get_tokenizer(args)

    label_list = load_labels(args.labels)
    label_map = { l: i for i, l in enumerate(label_list) }
    inv_label_map = { v: k for k, v in label_map.items() }

    if args.train_data[0].endswith('.tsv'):
        if len(args.train_data) > 1:
            raise NotImplementedError('Multiple TSV inputs')
        train_data = TsvSequence(args.train_data[0], tokenizer, label_map,
                                 global_batch_size, args)
        input_format = 'tsv'
    elif args.train_data[0].endswith('.tfrecord'):
        train_data = train_tfrecord_input(args.train_data, args.max_seq_length,
                                          global_batch_size)
        input_format = 'tfrecord'
    else:
        raise ValueError('--train_data must be .tsv or .tfrecord')

    if args.dev_data is None:
        dev_x, dev_y = None, None
        validation_data = None
    else:
        dev_x, dev_y = load_dataset(args.dev_data, tokenizer,
                                    args.max_seq_length, args.replace_span,
                                    label_map, args)
        validation_data = (dev_x, dev_y)

    print('Number of devices: {}'.format(num_devices), file=sys.stderr, 
          flush=True)
    if num_devices > 1 and input_format != 'tfrecord':
        warning('TFRecord input recommended for multi-device training')

    num_train_examples = num_examples(args.train_data)
    num_labels = len(label_list)
    print('num_train_examples: {}'.format(num_train_examples),
          file=sys.stderr, flush=True)

    with strategy.scope():
        model = restore_or_create_model(num_train_examples, num_labels, 
                                        global_batch_size, args)
    model.summary(print_fn=print)

    callbacks = []
    if args.checkpoint_steps is not None:
        callbacks.append(ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_dir, CHECKPOINT_PATH),
            save_freq=args.checkpoint_steps
        ))

    if input_format == 'tsv':
        other_args = {
            'workers': 10,    # TODO
        }
    else:
        assert input_format == 'tfrecord', 'internal error'
        steps_per_epoch = int(np.ceil(num_train_examples/global_batch_size))
        other_args = {
            'steps_per_epoch': steps_per_epoch
        }

    model.fit(
        train_data,
        epochs=args.num_train_epochs,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_batch_size=global_batch_size,
        **other_args
    )

    if validation_data is not None:
        probs = model.predict(dev_x, batch_size=global_batch_size)
        preds = np.argmax(probs, axis=-1)
        correct, total = sum(g==p for g, p in zip(dev_y, preds)), len(dev_y)
        print('Final dev accuracy: {:.1%} ({}/{})'.format(
            correct/total, correct, total))

    if args.model_dir is not None:
        print('Saving model in {}'.format(args.model_dir))
        save_model_etc(model, tokenizer, label_list, args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
