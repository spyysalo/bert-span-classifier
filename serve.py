import os
import sys

import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS

from common import argument_parser
from common import load_model
from common import tokenize_texts, encode_tokenized


app = Flask(__name__)
CORS(app)


@app.route('/')
def predict():
    left = request.values['left']
    span = request.values.get('span', '')
    right = request.values['right']

    model, tokenizer, labels, config = (
        app.model, app.tokenizer, app.labels, app.model_config
    )
    max_seq_len = config['max_seq_length']
    replace_span = config['replace_span']
    tokenized = tokenize_texts([[left, span, right]], tokenizer)
    test_x = encode_tokenized(tokenized, tokenizer, max_seq_len, replace_span)
    with app.graph.as_default():
        with app.session.as_default():
            probs = model.predict(test_x)
    response = { l: float(p) for l, p in zip(labels, list(probs[0])) }
    for i, k in enumerate(('left', 'span', 'right')):
        response[k] = tokenized[0][i]
    return jsonify(response)


def main(argv):
    args = argument_parser('serve').parse_args(argv[1:])
    session = tf.Session()
    graph = tf.get_default_graph()
    with graph.as_default():
        with session.as_default():
            app.model, app.tokenizer, app.labels, app.model_config = load_model(
                args.model_dir)
            app.session = session
            app.graph = graph
    app.run(port=args.port, debug=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
