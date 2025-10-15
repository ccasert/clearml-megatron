#!/bin/bash

set -ex

echo "Training step"


# Container configuration
SHIFTER_IMAGE=nersc/pytorch:25.02.01
SHIFTER_MODULES=gpu,nccl-plugin

# Unset ClearML agent's python setup
unset PYTHONPATH

export PYTHONUSERBASE=".local/clearml-megatron"
shifter --image=$SHIFTER_IMAGE --module=$SHIFTER_MODULES \
    bash -c "pip install clearml"


export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export HF_HOME=${SCRATCH}/cache/huggingface

CHECKPOINT_PATH=${SCRATCH}/megatron/checkpoints/gpt3_175b
TENSORBOARD_LOGS_PATH=${SCRATCH}/megatron/logs/tensorboard
VOCAB_FILE=${SCRATCH}/data/wikitext/tokenizer_gpt2/vocab.json
MERGE_FILE=${SCRATCH}/data/wikitext/tokenizer_gpt2/merges.txt
DATA_PATH=${SCRATCH}/data/wikitext/wikitext103_gpt2_text_document


# mkdir -p ${SCRATCH_PATH}/logs
mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${TENSORBOARD_LOGS_PATH}

DISTRIBUTED_ARGS=(
    # --nproc_per_node=${SLURM_GPUS_PER_TASK}
    --nproc_per_node=4
    --nnodes=${SLURM_JOB_NUM_NODES}
    --rdzv-backend=c10d
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT}
)


GPT_MODEL_ARGS=(
    --num-layers 24
    --hidden-size 1024
    --num-attention-heads 16
    --seq-length 2048
    --max-position-embeddings 2048
    --attention-backend fused
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 64
    --train-iters 10000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --fp16
    --lr 2.5e-4
    --lr-decay-style cosine
    --min-lr 2.5e-5
    --lr-warmup-fraction 0.001
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 1000
    --eval-interval 100
    --save $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)


cmd="torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}"

echo "${cmd}"

shifter --image=$SHIFTER_IMAGE --module=$SHIFTER_MODULES \
    bash -c "$cmd"