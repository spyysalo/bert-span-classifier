#!/usr/bin/env python3

import sys
import os

os.environ['TF_KERAS'] = '1'

from functools import wraps
from time import time
from argparse import ArgumentParser
from logging import warning

from tensorflow import keras
from bert import tokenization
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import calc_train_steps, AdamWarmup

from tensorflow.keras.layers import Lambda

from config import DEFAULT_SEQ_LEN, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
from config import DEFAULT_LR, DEFAULT_WARMUP_PROPORTION


def timed(f, out=sys.stderr):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        print('{} completed in {:.1f} sec'.format(f.__name__, time()-start),
              file=out)
        return result
    return wrapper


def argument_parser(mode):
    argparser = ArgumentParser()
    if mode == 'train':
        argparser.add_argument(
            '--train_data', required=True,
            help='Training data'
        )
        argparser.add_argument(
            '--dev_data', default=None,
            help='Development data'
        )
        argparser.add_argument(
            '--vocab_file', required=True,
            help='Vocabulary file that BERT model was trained on'
        )
        argparser.add_argument(
            '--bert_config_file', required=True,
            help='Configuration for pre-trained BERT model'
        )
        argparser.add_argument(
            '--init_checkpoint', required=True,
            help='Initial checkpoint for pre-trained BERT model'
        )
        argparser.add_argument(
            '--max_seq_length', type=int, default=DEFAULT_SEQ_LEN,
            help='Maximum input sequence length in WordPieces'
        )
        argparser.add_argument(
            '--do_lower_case', default=False, action='store_true',
            help='Lower case input text (for uncased models)'
        )
        argparser.add_argument(
        '--learning_rate', type=float, default=DEFAULT_LR,
            help='Initial learning rate'
        )
        argparser.add_argument(
            '--num_train_epochs', type=int, default=DEFAULT_EPOCHS,
            help='Number of training epochs'
        )
        argparser.add_argument(
            '--warmup_proportion', type=float, default=DEFAULT_WARMUP_PROPORTION,
            help='Proportion of training to perform LR warmup for'
        )
        argparser.add_argument(
            '--replace_span', default=None,
            help='Replace span text with given special token'
        )
    argparser.add_argument(
        '--label_field', type=int, default=-4,
        help='Index of label in TSV data (1-based)'
    )
    argparser.add_argument(
        '--text_fields', type=int, default=-3,
        help='Index of first text field in TSV data (1-based)'
    )
    argparser.add_argument(
        '--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
        help='Batch size for training'
    )
    argparser.add_argument(
        '--model_dir', default=None,
        help='Trained model directory'
    )
    return argparser


@timed
def load_pretrained(options):
    model = load_trained_model_from_checkpoint(
        options.bert_config_file,
        options.init_checkpoint,
        training=False,
        trainable=True,
        seq_len=options.max_seq_length,
    )
    tokenizer = tokenization.FullTokenizer(
        vocab_file=options.vocab_file,
        do_lower_case=options.do_lower_case
    )
    return model, tokenizer


def create_model(pretrained_model, num_labels):
    model_inputs = pretrained_model.inputs[:2]
    cls_out = Lambda(lambda x: x[:, 0])(pretrained_model.output)
    model_output = keras.layers.Dense(
        num_labels,
        activation='softmax'
    )(cls_out)
    model = keras.models.Model(inputs=model_inputs, outputs=model_output)
    return model


def create_optimizer(num_example, options):
    total_steps, warmup_steps = calc_train_steps(
        num_example=num_example,
        batch_size=options.batch_size,
        epochs=options.num_train_epochs,
        warmup_proportion=options.warmup_proportion,
    )
    optimizer = AdamWarmup(
        total_steps,
        warmup_steps,
        lr=options.learning_rate,
        epsilon=1e-6,
        weight_decay=0.01,
        weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo']
    )
    return optimizer


def tokenize_texts(texts, tokenizer):
    tokenized = []
    for left, span, right in texts:
        left_tok = tokenizer.tokenize(left)
        span_tok = tokenizer.tokenize(span)
        right_tok = tokenizer.tokenize(right)
        tokenized.append([left_tok, span_tok, right_tok])
    return tokenized


def encode_tokenized(tokenized_texts, tokenizer, options):
    seq_len = options.max_seq_length
    tids, sids = [], []
    for left, span, right in tokenized_texts:
        tokens = ['[CLS]']
        tokens.extend(left)
        if options.replace_span is None:
            tokens.extend(span)
        else:
            tokens.append(options.replace_span)
        tokens.extend(right)
        if len(tokens) >= seq_len-1:    # -1 for [SEP]
            tokens, chopped = tokens[:seq_len-1], tokens[seq_len-1:]
            warning('chopping tokens to {}: {} ///// {}'.format(
                seq_len-1, ' '.join(tokens), ' '.join(chopped)))
        tokens.append('[SEP]')
        tokens.extend(['[PAD]'] * (seq_len-len(tokens)))
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * seq_len
        tids.append(token_ids)
        sids.append(segment_ids)
    # Sanity check
    assert all(len(t) == seq_len for t in tids)
    assert all(len(s) == seq_len for s in sids)
    return tids, sids


def load_tsv_data(fn, options):
    def positive_index(i, fields):
        return i if i >= 0 else len(fields)+i
    labels, texts = [], []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split('\t')
            if len(fields) < 4:
                raise ValueError(
                    'Expected at least 4 tab-separated fields, got '
                    '{} on {} line {}: {}'.format(len(fields), fn, ln, l)
                )
            label = fields[options.label_field]
            text_end = positive_index(options.text_fields, fields) + 3
            text = fields[options.text_fields:text_end]
            labels.append(label)
            texts.append(text)
    return labels, texts
