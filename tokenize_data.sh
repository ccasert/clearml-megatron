#!/bin/bash

set -ex

echo "Tokenizing data step"

unset PYTHONPATH
module load pytorch/2.6.0

pip install clearml

# Use Python to extract ClearML task parameters and export them
# Redirect Task.init output to stderr, only print exports to stdout
eval $(python - <<'EOF' 2>&1 >/dev/null
import sys
from clearml import Task

# Redirect Task.init output
task = Task.init(project_name="Megatron", task_name="tokenize-data-step", reuse_last_task_id=True)
params = task.get_parameters_as_dict()

megatron_dir = params.get('General', {}).get('MEGATRON_DIR', '')
data_dir = params.get('General', {}).get('DATA_DIR', '')
tokenizer_dir = params.get('General', {}).get('TOKENIZER_DIR', '')

# Print exports to stdout (which is redirected back)
print(f"export MEGATRON_DIR='{megatron_dir}'", file=sys.stderr)
print(f"export DATA_DIR='{data_dir}'", file=sys.stderr)
print(f"export TOKENIZER_DIR='{tokenizer_dir}'", file=sys.stderr)
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

echo "Tokenization complete"