#!/bin/bash

# Download pretrained BERT models

# https://stackoverflow.com/a/246128
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -euo pipefail

DATADIR="$SCRIPTDIR/../models"

mkdir -p "$DATADIR"

GOOGLE_BASE_URL="https://storage.googleapis.com/bert_models"
NCBI_BASE_URL="https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT"

for url in "$GOOGLE_BASE_URL/2018_10_18/cased_L-12_H-768_A-12.zip" \
	   "$NCBI_BASE_URL/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12.zip"; do
    b=$(basename "$url" .zip)
    if [ -e "$DATADIR/$b" ]; then
	echo "$b exists, skipping ..." >&2
    else
	wget "$url" -O "$DATADIR/$b.zip"
	if [[ $url == "$NCBI_BASE_URL"* ]]; then
	    unzip "$DATADIR/$b.zip" -d "$DATADIR/$b"
	else
	    unzip "$DATADIR/$b.zip" -d "$DATADIR"
	fi
	rm "$DATADIR/$b.zip"
    fi
done
