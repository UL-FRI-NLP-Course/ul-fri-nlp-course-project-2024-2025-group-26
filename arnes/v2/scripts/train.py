import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
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
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",  # Gemma recommends it
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def preprocess(example, max_length=2048):
    prompt = example["Input"]
    response = example["GroundTruth"]

    formatted = (
        "<start_of_turn>user\n" + prompt + "<end_of_turn>\n"
        "<start_of_turn>model\n" + response + "<end_of_turn>"
    )

    max_length = max_length or tokenizer.model_max_length

    # Reserve 1 token for the eos
    tokens = tokenizer(
        formatted,
        truncation=True,
        max_length=max_length - 1,
        padding=False,
        add_special_tokens=False,
    )

    # Manually append EOS token
    tokens["input_ids"].append(tokenizer.eos_token_id)
    tokens["attention_mask"].append(1)

    return tokens


# Debug
print(tokenizer.eos_token)  # Should return something like "</s>" or "<|endoftext|>"
print(tokenizer.eos_token_id)  # Should be an integer

# Load full dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Tokenize dataset
# tokenized_dataset = dataset.map(preproces s, remove_columns=["Input", "GroundTruth"])
tokenized_dataset = dataset.map(
    preprocess, remove_columns=["Input", "GroundTruth", "Date"]
)

# Split tokenized dataset (note using tokenized_dataset here!)
split_1 = tokenized_dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
train_set = split_1["train"]
test_set = split_1["test"]

# split_2 = temp_set.train_test_split(test_size=0.5, seed=42, shuffle=False)
# test_set = split_2['train']
# val_set = split_2['test']  # For validation if needed

# Debug: check keys and a sample
print(train_set[0].keys())  # Should include 'input_ids', 'attention_mask'
print(train_set[0])

# LoRA Config
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)

# Training Setup
tokenizer.pad_token = tokenizer.eos_token
# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_set,
    eval_dataset=test_set,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir=ADAPTER_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        max_grad_norm=0.3,
        max_steps=5000,  # Adjust as needed, 500 default
        warmup_ratio=0.03,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        report_to="none",
        remove_unused_columns=False,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    packing=False,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# Execute Training
try:
    trainer.train()
finally:
    model_to_save = (
        trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(ADAPTER_DIR)
    print(f"Saved adapter weights to {ADAPTER_DIR}")
