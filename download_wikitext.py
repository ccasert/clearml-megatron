import os
from datasets import load_dataset
from transformers import GPT2TokenizerFast

HF_HOME = "$SCRATCH/cache/huggingface"
DATA_DIR = "$SCRATCH/data/wikitext"

# set hf home and make data dir
os.environ["HF_HOME"] = HF_HOME
os.makedirs(DATA_DIR, exist_ok=True)

# tokenization rules
TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizer_gpt2")
os.makedirs(TOKENIZER_DIR, exist_ok=True)

# download
print("Downloading WikiText-103...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

# save as json lines (megatron takes json by default)
print("Saving dataset as JSON lines...")
dataset['train'].to_json(os.path.join(DATA_DIR, "wikitext103_train.json"), orient="records", lines=True)
dataset['validation'].to_json(os.path.join(DATA_DIR, "wikitext103_valid.json"), orient="records", lines=True)
dataset['test'].to_json(os.path.join(DATA_DIR, "wikitext103_test.json"), orient="records", lines=True)

# download tokenizer
print("Downloading GPT-2 tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.save_pretrained(TOKENIZER_DIR)
