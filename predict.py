import os
import sys

import numpy as np

from common import argument_parser
from common import load_model_etc, load_tsv_data
from common import tokenize_texts, encode_tokenized


def main(argv):
    args = argument_parser('predict').parse_args(argv[1:])

    model, tokenizer, labels, config = load_model_etc(args.model_dir)
    _, test_texts = load_tsv_data(args.test_data, args)

    max_seq_len = config['max_seq_length']
    replace_span = config['replace_span']

    label_map = { t: i for i, t in enumerate(labels) }
    inv_label_map = { v: k for k, v in label_map.items() }

    test_tok = tokenize_texts(test_texts, tokenizer)
    test_x = encode_tokenized(test_tok, tokenizer, max_seq_len, replace_span)

    probs = model.predict(test_x, batch_size=args.batch_size)
    preds = np.argmax(probs, axis=-1)
    for p in preds:
        print(inv_label_map[p])
    
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
