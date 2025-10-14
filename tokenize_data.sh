#!/bin/bash

set -ex

echo "Preprocessing data step"

unset PYTHONPATH
module load pytorch/2.6.0

pip install clearml

MEGATRON_DIR=${MEGATRON_DIR}
DATA_DIR=${DATA_DIR}
TOKENIZER_DIR=${TOKENIZER_DIR}

echo "Megatron directory: ${MEGATRON_DIR}"
echo "Data directory: ${DATA_DIR}"
echo "Tokenizer directory: ${TOKENIZER_DIR}"

python ${MEGATRON_DIR}/tools/preprocess_data.py \
    --input ${DATA_DIR}/wikitext103_train.json \
    --output-prefix ${DATA_DIR}/wikitext103_gpt2 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_DIR} \
    --append-eod \
    --workers 4

echo "Preprocessing complete"
