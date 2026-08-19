"""
SFT v2 training script - uses chat template format matching the evaluator.
Key fix: Training data formatted with apply_chat_template to match eval-time format.
"""
import json
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

MODEL_PATH = "/root/models/Qwen/Qwen3-1.7B-Base"
DATA_PATH = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data_v2.jsonl"
OUTPUT_DIR = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/checkpoints/sft_v2"

NUM_EPOCHS = 5
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 1024
LORA_RANK = 64
LORA_ALPHA = 128

SYSTEM_PROMPT = "You are a helpful assistant."
USER_INSTRUCTION = "Complete the following Python function. Only output the function body."


def extract_prompt_and_body(text: str):
    """Extract function signature+docstring (prompt) and body from a complete function."""
    lines = text.strip().split('\n')

    # Find the def line
    def_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def ') or stripped.startswith('from ') or stripped.startswith('import '):
            if stripped.startswith('def '):
                def_idx = i
                break
            continue
        elif def_idx is None and stripped.startswith('def '):
            def_idx = i
            break

    # Find any imports before the def
    import_lines = []
    if def_idx is not None:
        for i in range(def_idx):
            stripped = lines[i].strip()
            if stripped and (stripped.startswith('from ') or stripped.startswith('import ')):
                import_lines.append(lines[i])

    if def_idx is None:
        # Try to split on the first non-docstring line
        return None, None

    # Find docstring end
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

    # Build prompt (imports + function header + docstring)
    header_lines = []
    if import_lines:
        header_lines.extend(import_lines)
        header_lines.append('')
        header_lines.append('')
    header_lines.extend(lines[def_idx:docstring_end + 1])
    prompt = '\n'.join(header_lines)

    # Build body (everything after docstring)
    body_lines = lines[docstring_end + 1:]
    while body_lines and body_lines[0].strip() == '':
        body_lines.pop(0)
    while body_lines and body_lines[-1].strip() == '':
        body_lines.pop()
    body = '\n'.join(body_lines)

    if not body.strip():
        return None, None

    return prompt, body


def format_chat_message(prompt: str, body: str, tokenizer) -> str:
    """Format as a chat message using the tokenizer's chat template."""
    user_content = f"{USER_INSTRUCTION}\n\n{prompt}"
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": body}
    ]
    # Use apply_chat_template to get the exact format the evaluator uses
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return text


def load_and_format_data(tokenizer):
    examples = []
    skipped = 0

    with open(DATA_PATH) as f:
        for line in f:
            item = json.loads(line)
            text = item.get('text', '')
            if not text:
                # Old format with prompt/completion
                text = item.get('prompt', '') + item.get('completion', '')

            prompt, body = extract_prompt_and_body(text)
            if prompt and body:
                chat_text = format_chat_message(prompt, body, tokenizer)
                examples.append({"text": chat_text})
            else:
                skipped += 1

    print(f"Loaded {len(examples)} examples, skipped {skipped}")
    return Dataset.from_list(examples)


def main():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify chat template works
    test_messages = [
        {"role": "user", "content": "Complete the following Python function. Only output the function body.\n\ndef add(a, b):\n    \"\"\"Add two numbers.\"\"\""},
        {"role": "assistant", "content": "    return a + b"}
    ]
    test_output = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    print("Chat template test:")
    print(repr(test_output[:300]))
    print()

    print("Loading data...")
    dataset = load_and_format_data(tokenizer)
    print(f"Training examples: {len(dataset)}")

    # Show a sample
    print("\n--- Sample ---")
    print(dataset[0]['text'][:500])
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

    steps_per_epoch = len(dataset) // (BATCH_SIZE * GRAD_ACCUM * 4)  # 4 GPUs
    total_steps = steps_per_epoch * NUM_EPOCHS
    save_steps = max(steps_per_epoch, 1)

    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}, Save every: {save_steps}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        run_name="sft_v2_chat",
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

    # Also save LoRA adapter
    lora_dir = f"{OUTPUT_DIR}/lora_adapter"
    model.save_pretrained(lora_dir)
    print(f"LoRA adapter saved to {lora_dir}")


if __name__ == '__main__':
    main()
