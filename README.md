# bert-span-classifier

Text span classifier using BERT

## Quickstart

```
./scripts/get-models.sh

python train.py \
    --init_checkpoint models/cased_L-12_H-768_A-12/bert_model.ckpt \
    --vocab_file models/cased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file models/cased_L-12_H-768_A-12/bert_config.json \
    --train_data example-data/train.tsv --dev_data example-data/dev.tsv
```
