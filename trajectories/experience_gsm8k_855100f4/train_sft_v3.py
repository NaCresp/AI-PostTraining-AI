"""SFT v3: Large augmented dataset (37k examples) with filtered OpenMathInstruct-2."""
import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

MODEL_PATH = "/root/models/Qwen/Qwen3-1.7B-Base"
OUTPUT_DIR = "checkpoints/sft_v3"
DATA_PATH = "data/sft_v3_augmented.jsonl"
MAX_SEQ_LEN = 1024

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.pad_token = "<|im_end|>"

# Load pre-prepared augmented dataset
texts = []
with open(DATA_PATH) as f:
    for line in f:
        texts.append(json.loads(line))
ds = Dataset.from_list(texts)
print(f"Dataset size: {len(ds)}")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    bf16=True,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=3,
    max_length=MAX_SEQ_LEN,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,
    report_to="none",
    seed=42,
    packing=True,
    ddp_find_unused_parameters=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
print(f"Training complete. Model saved to {OUTPUT_DIR}/final")
