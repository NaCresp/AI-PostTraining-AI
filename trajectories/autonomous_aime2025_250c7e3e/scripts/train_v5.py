#!/usr/bin/env python3
"""
SFT v5 training - longer reasoning chains:
- max_seq_length=8192 to allow longer reasoning traces
- 3 epochs (reduced from 5 to avoid overfitting)
- Lower LR (3e-5) for stability with longer sequences
- Batch size 1 per device due to longer sequences, grad_accum=8
- Save best checkpoint based on eval_loss
- Auto-copy best model to final_model/
"""
import os
import json
import shutil
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL_PATH = "/home/user/models/Qwen/Qwen3-1.7B-Base"
TRAIN_DATA = "/home/ben/task/train_data/train_v5.jsonl"
VAL_DATA = "/home/ben/task/train_data/val_v5.jsonl"
OUTPUT_DIR = "/home/ben/task/sft_output_v5"
FINAL_MODEL_DIR = "/home/ben/task/final_model"
MAX_SEQ_LENGTH = 8192

CHAT_TEMPLATE = """{%- for message in messages %}
    {%- if message.role == 'system' %}
        {{- '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == 'user' %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == 'assistant' %}
        {{- '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"""


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    print(f"<|im_end|> token ID: {im_end_id}")

    tokenizer.chat_template = CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    dataset = load_dataset("json", data_files={
        "train": TRAIN_DATA,
        "validation": VAL_DATA,
    })

    print(f"Train size: {len(dataset['train'])}")
    print(f"Val size: {len(dataset['validation'])}")

    # Verify format
    sample = dataset["train"][0]
    test_text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
    print(f"\nSample formatted text (first 500 chars):\n{test_text[:500]}")
    print(f"\nLast 200 chars:\n{test_text[-200:]}")

    test_tokens = tokenizer.apply_chat_template(sample["messages"], tokenize=True)
    im_end_count = test_tokens.count(im_end_id)
    print(f"\n<|im_end|> appears {im_end_count} times in tokenized sample")
    print(f"Total tokens: {len(test_tokens)}")

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=MAX_SEQ_LENGTH,
        packing=True,
        dataloader_num_workers=4,
        remove_unused_columns=True,
        report_to="none",
        seed=42,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # Save best model (load_best_model_at_end=True ensures best checkpoint is loaded)
    print("Saving best model...")
    best_dir = os.path.join(OUTPUT_DIR, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Also save final model
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Fix EOS config for both saved models
    for save_dir in [best_dir, final_dir]:
        gen_config_path = os.path.join(save_dir, "generation_config.json")
        with open(gen_config_path, 'r') as f:
            gen_config = json.load(f)
        gen_config["eos_token_id"] = [151643, im_end_id]
        gen_config["max_new_tokens"] = 16000
        with open(gen_config_path, 'w') as f:
            json.dump(gen_config, f, indent=2)

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        config["eos_token_id"] = [151643, im_end_id]
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    # Copy best model to final_model directory
    print(f"Copying best model to {FINAL_MODEL_DIR}...")
    if os.path.exists(FINAL_MODEL_DIR):
        # Backup old final model
        backup_dir = FINAL_MODEL_DIR + "_v4_backup"
        if not os.path.exists(backup_dir):
            shutil.copytree(FINAL_MODEL_DIR, backup_dir)
        shutil.rmtree(FINAL_MODEL_DIR)
    shutil.copytree(best_dir, FINAL_MODEL_DIR)

    print("Training complete! Best model copied to final_model/")

if __name__ == "__main__":
    main()
