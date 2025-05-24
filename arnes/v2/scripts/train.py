import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import os

# Configuration
MODEL_ID = "cjvt/GaMS-9B-Instruct"
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "/models/finetuned")
DATASET_PATH = os.getenv("TRAIN_DATA", "/data/train.json")

# 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Added

# Load full dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Tokenization function with padding & truncation, batched=True
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",   # or "longest" or True for dynamic padding
        truncation=True,
        max_length=512
    )

# Tokenize dataset with batching to avoid dimension errors
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Split tokenized dataset (note using tokenized_dataset here!)
split_1 = tokenized_dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
train_set = split_1['train']
test_set = split_1['test']

# split_2 = temp_set.train_test_split(test_size=0.5, seed=42, shuffle=False)
# test_set = split_2['train']
# val_set = split_2['test']  # For validation if needed

# Debug: check keys and a sample
print(train_set[0].keys())  # Should include 'input_ids', 'attention_mask'
print(train_set[0])

# LoRA Config
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# Training Setup
training_args = TrainingArguments(
    output_dir=ADAPTER_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    max_grad_norm=0.3,
    max_steps=1000, # Adjust as needed, 500 default
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    report_to="none",
    remove_unused_columns=False
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_set,
    eval_dataset=test_set,
    peft_config=peft_config,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    packing=False
)

# Execute Training
try:
    trainer.train()
finally:
    trainer.save_model(ADAPTER_DIR)
    print(f"Saved adapter weights to {ADAPTER_DIR}")
