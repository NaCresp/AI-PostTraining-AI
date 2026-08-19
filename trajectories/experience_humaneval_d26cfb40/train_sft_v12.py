"""
SFT v12 - Continue from v11 checkpoint with all augmented data including v6_extra.
Stage 3 of the training chain: base→v7→v11→v12.
"""
import json
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

BASE_MODEL_PATH = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/checkpoints/sft_v11/final_merged"
DATA_PATH_V2 = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data_v2.jsonl"
DATA_PATH_EXTRA_V3 = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data_v3_extra.jsonl"
DATA_PATH_EXTRA_V4 = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data_v4_extra.jsonl"
DATA_PATH_EXTRA_V5 = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data_v5_extra.jsonl"
DATA_PATH_EXTRA_V6 = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data_v6_extra.jsonl"
OUTPUT_DIR = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/checkpoints/sft_v12"

NUM_EPOCHS = 2
LEARNING_RATE = 3e-5
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 1024
LORA_RANK = 32
LORA_ALPHA = 64

USER_INSTRUCTION = "Complete the following Python function. Only output the function body."


def extract_prompt_and_body(text: str):
    lines = text.strip().split('\n')
    def_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            def_idx = i
            break
    if def_idx is None:
        return None, None

    in_docstring = False
    docstring_end = def_idx
    quote_char = None
    for i in range(def_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote_char = stripped[:3]
                if stripped.count(quote_char) >= 2 and len(stripped) > 5:
                    docstring_end = i
                    break
                else:
                    in_docstring = True
            elif stripped == '':
                continue
            else:
                docstring_end = def_idx
                break
        else:
            if quote_char in stripped:
                docstring_end = i
                in_docstring = False
                break
    if in_docstring:
        return None, None

    prompt = '\n'.join(lines[def_idx:docstring_end + 1])
    body_lines = lines[docstring_end + 1:]
    while body_lines and body_lines[0].strip() == '':
        body_lines.pop(0)
    while body_lines and body_lines[-1].strip() == '':
        body_lines.pop()
    body = '\n'.join(body_lines)
    if not body.strip():
        return None, None
    return prompt, body


def load_all_data():
    examples = []
    skipped = 0

    with open(DATA_PATH_V2) as f:
        for line in f:
            item = json.loads(line)
            text = item.get('text', '')
            if not text:
                text = item.get('prompt', '') + item.get('completion', '')
            func_prompt, body = extract_prompt_and_body(text)
            if func_prompt and body:
                chat_prompt = f"<|im_start|>user\n{USER_INSTRUCTION}\n\n{func_prompt}<|im_end|>\n<|im_start|>assistant\n"
                completion = f"{body}<|endoftext|>"
                examples.append({"prompt": chat_prompt, "completion": completion})
            else:
                skipped += 1

    extra_count = 0
    for extra_path in [DATA_PATH_EXTRA_V3, DATA_PATH_EXTRA_V4, DATA_PATH_EXTRA_V5, DATA_PATH_EXTRA_V6]:
        if os.path.exists(extra_path):
            with open(extra_path) as f:
                for line in f:
                    item = json.loads(line)
                    func_prompt = item['prompt']
                    body = item['body']
                    chat_prompt = f"<|im_start|>user\n{USER_INSTRUCTION}\n\n{func_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    completion = f"{body}<|endoftext|>"
                    examples.append({"prompt": chat_prompt, "completion": completion})
                    extra_count += 1

    print(f"Total: {len(examples)} ({len(examples) - extra_count} original + {extra_count} extra), skipped {skipped}")
    return Dataset.from_list(examples)


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_all_data()
    print(f"Training examples: {len(dataset)}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

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

    steps_per_epoch = len(dataset) // (BATCH_SIZE * GRAD_ACCUM * 4)
    save_steps = max(steps_per_epoch, 1)
    print(f"Steps per epoch: {steps_per_epoch}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        run_name="sft_v12_from_v11",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,
        bf16=True,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=2,
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

    print("Saving final merged model...")
    final_dir = f"{OUTPUT_DIR}/final_merged"
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    for fname in ["generation_config.json", "config.json"]:
        fpath = os.path.join(final_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                cfg = json.load(f)
            cfg["eos_token_id"] = 151643
            with open(fpath, 'w') as f:
                json.dump(cfg, f, indent=2)

    print(f"Model saved to {final_dir}")


if __name__ == '__main__':
    main()
