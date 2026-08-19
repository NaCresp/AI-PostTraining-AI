#!/usr/bin/env python3
"""
SFT v3 training with format matching AIME evaluation:
- No system messages in training data
- User prompt matches exact evaluation format
- Proper <think>...</think> + ANSWER: N format
- Using qwen3.jinja template style
"""
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL_PATH = "/home/user/models/Qwen/Qwen3-1.7B-Base"
TRAIN_DATA = "/home/ben/task/train_data/train_v3.jsonl"
VAL_DATA = "/home/ben/task/train_data/val_v3.jsonl"
OUTPUT_DIR = "/home/ben/task/sft_output_v3"
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
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
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

    # Save final model
    print("Saving final model...")
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Update generation config and model config for proper EOS
    gen_config_path = os.path.join(final_dir, "generation_config.json")
    with open(gen_config_path, 'r') as f:
        gen_config = json.load(f)
    gen_config["eos_token_id"] = [151643, im_end_id]
    gen_config["max_new_tokens"] = 16000
    with open(gen_config_path, 'w') as f:
        json.dump(gen_config, f, indent=2)

    config_path = os.path.join(final_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    config["eos_token_id"] = [151643, im_end_id]
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("Training complete!")

if __name__ == "__main__":
    main()
