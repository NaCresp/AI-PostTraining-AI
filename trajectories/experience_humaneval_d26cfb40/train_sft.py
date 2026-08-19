"""
SFT training script for Qwen3-1.7B-Base on Python function completion.
Uses TRL SFTTrainer with LoRA on 4 GPUs.
"""
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

# Config
MODEL_PATH = "/root/models/Qwen/Qwen3-1.7B-Base"
DATA_PATH = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data.jsonl"
OUTPUT_DIR = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/checkpoints/sft_v1"
RUN_NAME = "sft_v1"

# Training hyperparams
NUM_EPOCHS = 5
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 1024
LORA_RANK = 64
LORA_ALPHA = 128

def load_data():
    examples = []
    with open(DATA_PATH) as f:
        for line in f:
            item = json.loads(line)
            text = item['prompt'] + item['completion']
            examples.append({"text": text})
    return Dataset.from_list(examples)

def main():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading data...")
    dataset = load_data()
    print(f"Training examples: {len(dataset)}")

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Save steps - save at end of each epoch and mid-epoch
    steps_per_epoch = len(dataset) // (BATCH_SIZE * GRAD_ACCUM * 4)  # 4 GPUs
    total_steps = steps_per_epoch * NUM_EPOCHS
    save_steps = max(steps_per_epoch, 1)

    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}, Save every: {save_steps}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=MAX_SEQ_LENGTH,
        bf16=True,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        remove_unused_columns=True,
        report_to="none",
        seed=42,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # Save final model (merged)
    print("Saving final merged model...")
    final_dir = f"{OUTPUT_DIR}/final_merged"
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final merged model saved to {final_dir}")

    # Also save the LoRA adapter separately
    lora_dir = f"{OUTPUT_DIR}/lora_adapter"
    model.save_pretrained(lora_dir)
    print(f"LoRA adapter saved to {lora_dir}")

if __name__ == '__main__':
    main()
