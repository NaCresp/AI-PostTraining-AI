"""
SFT v4 training script - fixes the stop token issue from v3.
Key changes:
1. Uses <|endoftext|> (the actual EOS token, id 151643) instead of <|im_end|> as sequence terminator
   so vLLM naturally stops generation.
2. Uses prompt-completion format with completion_only_loss=True for focused learning.
3. After training, patches generation_config.json to include both stop tokens.
"""
import json
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

MODEL_PATH = "/root/models/Qwen/Qwen3-1.7B-Base"
DATA_PATH = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data_v2.jsonl"
OUTPUT_DIR = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/checkpoints/sft_v4"

NUM_EPOCHS = 6
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 1024
LORA_RANK = 64
LORA_ALPHA = 128

USER_INSTRUCTION = "Complete the following Python function. Only output the function body."


def extract_prompt_and_body(text: str):
    """Extract function signature+docstring (prompt) and body from a complete function."""
    lines = text.strip().split('\n')

    def_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def '):
            def_idx = i
            break

    if def_idx is None:
        return None, None

    import_lines = []
    for i in range(def_idx):
        stripped = lines[i].strip()
        if stripped and (stripped.startswith('from ') or stripped.startswith('import ')):
            import_lines.append(lines[i])

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
                    continue
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

    header_lines = []
    if import_lines:
        header_lines.extend(import_lines)
        header_lines.append('')
        header_lines.append('')
    header_lines.extend(lines[def_idx:docstring_end + 1])
    prompt = '\n'.join(header_lines)

    body_lines = lines[docstring_end + 1:]
    while body_lines and body_lines[0].strip() == '':
        body_lines.pop(0)
    while body_lines and body_lines[-1].strip() == '':
        body_lines.pop()
    body = '\n'.join(body_lines)

    if not body.strip():
        return None, None

    return prompt, body


def load_and_format_data():
    examples = []
    skipped = 0

    with open(DATA_PATH) as f:
        for line in f:
            item = json.loads(line)
            text = item.get('text', '')
            if not text:
                text = item.get('prompt', '') + item.get('completion', '')

            func_prompt, body = extract_prompt_and_body(text)
            if func_prompt and body:
                # prompt: chat format up to assistant turn
                chat_prompt = f"<|im_start|>user\n{USER_INSTRUCTION}\n\n{func_prompt}<|im_end|>\n<|im_start|>assistant\n"
                # completion: function body + EOS token
                completion = f"{body}<|endoftext|>"
                examples.append({"prompt": chat_prompt, "completion": completion})
            else:
                skipped += 1

    print(f"Loaded {len(examples)} examples, skipped {skipped}")
    return Dataset.from_list(examples)


def main():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify format and tokenization
    test_prompt = f"<|im_start|>user\n{USER_INSTRUCTION}\n\ndef add(a, b):\n    \"\"\"Add two numbers.\"\"\"<|im_end|>\n<|im_start|>assistant\n"
    test_completion = "    return a + b<|endoftext|>"
    test_full = test_prompt + test_completion
    test_tokens = tokenizer.encode(test_full, add_special_tokens=False)
    print(f"Format test ({len(test_tokens)} tokens):")
    print(f"  Prompt: {repr(test_prompt[:80])}...")
    print(f"  Completion: {repr(test_completion)}")
    print(f"  Last token id: {test_tokens[-1]} (should be 151643)")
    assert test_tokens[-1] == 151643, f"Last token should be <|endoftext|> (151643), got {test_tokens[-1]}"
    print("  OK: last token is <|endoftext|>")
    print()

    print("Loading data...")
    dataset = load_and_format_data()
    print(f"Training examples: {len(dataset)}")

    print("\n--- Sample prompt ---")
    print(dataset[0]['prompt'][:300])
    print("--- Sample completion ---")
    print(dataset[0]['completion'][:200])
    print("---\n")

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
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
    total_steps = steps_per_epoch * NUM_EPOCHS
    save_steps = max(steps_per_epoch, 1)
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        run_name="sft_v4_eos_fix",
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

    # Patch generation_config.json to include both stop tokens
    gen_config_path = os.path.join(final_dir, "generation_config.json")
    if os.path.exists(gen_config_path):
        with open(gen_config_path) as f:
            gen_config = json.load(f)
        gen_config["eos_token_id"] = [151643, 151645]
        with open(gen_config_path, 'w') as f:
            json.dump(gen_config, f, indent=2)
        print("Patched generation_config.json with both EOS tokens")

    # Also patch config.json
    config_path = os.path.join(final_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config["eos_token_id"] = [151643, 151645]
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("Patched config.json with both EOS tokens")

    print(f"Final merged model saved to {final_dir}")


if __name__ == '__main__':
    main()
