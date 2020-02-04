#!/bin/bash

# Download pretrained BERT models

# https://stackoverflow.com/a/246128
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -euo pipefail

DATADIR="$SCRIPTDIR/../models"

mkdir -p "$DATADIR"

GOOGLE_BASE_URL="https://storage.googleapis.com/bert_models"

for url in "$GOOGLE_BASE_URL/2018_10_18/cased_L-12_H-768_A-12.zip"; do
    b=$(basename "$url" .zip)
    if [ -e "$DATADIR/$b" ]; then
	echo "$b exists, skipping ..." >&2
    else
	wget "$url" -O "$DATADIR/$b.zip"
	unzip "$DATADIR/$b.zip" -d "$DATADIR"
	rm "$DATADIR/$b.zip"
    fi
done
