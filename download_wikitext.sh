#!/bin/bash

set -ex

echo "Downloading data step"

# Module-based environment setup
unset PYTHONPATH
module load pytorch/2.6.0

# Install required packages
pip install clearml

# Set HF cache
export HF_HOME=$SCRATCH/cache/huggingface

# Run the Python download script
python download_wikitext.py

echo "Download complete"
