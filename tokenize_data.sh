#!/bin/bash

set -ex

echo "Preprocessing data step"

unset PYTHONPATH
module load pytorch/2.6.0

pip install clearml

# Use Python to extract ClearML task parameters and export them
eval $(python - <<'EOF'
from clearml import Task

task = Task.init(project_name="Megatron", task_name="tokenize-data-step", reuse_last_task_id=True)
params = task.get_parameters_as_dict()

megatron_dir = params.get('General', {}).get('MEGATRON_DIR', '')
data_dir = params.get('General', {}).get('DATA_DIR', '')
tokenizer_dir = params.get('General', {}).get('TOKENIZER_DIR', '')

print(f"export MEGATRON_DIR='{megatron_dir}'")
print(f"export DATA_DIR='{data_dir}'")
print(f"export TOKENIZER_DIR='{tokenizer_dir}'")
EOF
)

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
