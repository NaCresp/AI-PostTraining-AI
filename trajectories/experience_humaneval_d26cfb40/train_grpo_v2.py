"""
GRPO v2 training script - starts from SFT v4 checkpoint.
Uses code execution reward to improve code correctness.
Key: uses <|endoftext|> as EOS to match the v4 training format.
"""
import json
import re
import os
import multiprocessing
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

SFT_MODEL_PATH = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/checkpoints/sft_v4/final_merged"
OUTPUT_DIR = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/checkpoints/grpo_v2"

NUM_EPOCHS = 2
LEARNING_RATE = 5e-6
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_COMPLETION_LENGTH = 512
NUM_GENERATIONS = 4
LORA_RANK = 32
LORA_ALPHA = 64

USER_INSTRUCTION = "Complete the following Python function. Only output the function body."


def execute_code_safe(code_str: str, timeout: int = 5) -> dict:
    def _run(code, q):
        try:
            exec_globals = {}
            exec(code, exec_globals)
            q.put({"passed": True, "error": None})
        except Exception as e:
            q.put({"passed": False, "error": str(e)[:200]})

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run, args=(code_str, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join(1)
        return {"passed": False, "error": "timeout"}

    return q.get() if not q.empty() else {"passed": False, "error": "no result"}


def create_problems_dataset():
    from grpo_problems import create_grpo_problems
    problems = create_grpo_problems()

    simple_problems = [
        {"prompt": 'def add(a: int, b: int) -> int:\n    """Add two integers.\n    >>> add(2, 3)\n    5\n    """',
         "test": 'assert add(2, 3) == 5\nassert add(-1, 1) == 0'},
        {"prompt": 'def multiply(a: int, b: int) -> int:\n    """Multiply two integers.\n    >>> multiply(3, 4)\n    12\n    """',
         "test": 'assert multiply(3, 4) == 12\nassert multiply(0, 5) == 0'},
        {"prompt": 'def absolute_value(x: int) -> int:\n    """Return absolute value.\n    >>> absolute_value(-5)\n    5\n    """',
         "test": 'assert absolute_value(-5) == 5\nassert absolute_value(5) == 5\nassert absolute_value(0) == 0'},
        {"prompt": 'def max_of_three(a: int, b: int, c: int) -> int:\n    """Return the maximum of three integers.\n    >>> max_of_three(1, 2, 3)\n    3\n    """',
         "test": 'assert max_of_three(1, 2, 3) == 3\nassert max_of_three(3, 2, 1) == 3'},
        {"prompt": 'def is_even(n: int) -> bool:\n    """Check if a number is even.\n    >>> is_even(4)\n    True\n    """',
         "test": 'assert is_even(4) == True\nassert is_even(3) == False\nassert is_even(0) == True'},
        {"prompt": 'def list_sum(lst: list) -> int:\n    """Sum all elements in a list.\n    >>> list_sum([1, 2, 3])\n    6\n    """',
         "test": 'assert list_sum([1, 2, 3]) == 6\nassert list_sum([]) == 0'},
        {"prompt": 'def reverse_string(s: str) -> str:\n    """Reverse a string.\n    >>> reverse_string("hello")\n    \'olleh\'\n    """',
         "test": 'assert reverse_string("hello") == "olleh"\nassert reverse_string("") == ""'},
        {"prompt": 'def factorial(n: int) -> int:\n    """Compute factorial of n.\n    >>> factorial(5)\n    120\n    """',
         "test": 'assert factorial(5) == 120\nassert factorial(0) == 1\nassert factorial(1) == 1'},
        {"prompt": 'def count_char(s: str, c: str) -> int:\n    """Count occurrences of character c in string s.\n    >>> count_char("hello", "l")\n    2\n    """',
         "test": 'assert count_char("hello", "l") == 2\nassert count_char("hello", "z") == 0'},
        {"prompt": 'def remove_spaces(s: str) -> str:\n    """Remove all spaces from a string.\n    >>> remove_spaces("hello world")\n    \'helloworld\'\n    """',
         "test": 'assert remove_spaces("hello world") == "helloworld"'},
        {"prompt": 'def string_length(s: str) -> int:\n    """Return the length of a string.\n    >>> string_length("abc")\n    3\n    """',
         "test": 'assert string_length("abc") == 3\nassert string_length("") == 0'},
        {"prompt": 'def first_element(lst: list):\n    """Return the first element of a list, or None if empty.\n    >>> first_element([1, 2, 3])\n    1\n    """',
         "test": 'assert first_element([1, 2, 3]) == 1\nassert first_element([]) is None'},
        {"prompt": 'def square(n: int) -> int:\n    """Return the square of a number.\n    >>> square(4)\n    16\n    """',
         "test": 'assert square(4) == 16\nassert square(-3) == 9'},
        {"prompt": 'def is_positive(n: int) -> bool:\n    """Check if a number is positive.\n    >>> is_positive(5)\n    True\n    """',
         "test": 'assert is_positive(5) == True\nassert is_positive(-1) == False\nassert is_positive(0) == False'},
        {"prompt": 'def clamp(val: int, min_val: int, max_val: int) -> int:\n    """Clamp value between min and max.\n    >>> clamp(5, 1, 10)\n    5\n    >>> clamp(15, 1, 10)\n    10\n    """',
         "test": 'assert clamp(5, 1, 10) == 5\nassert clamp(15, 1, 10) == 10\nassert clamp(-5, 1, 10) == 1'},
    ]

    all_problems = problems + simple_problems

    rows = []
    for _ in range(10):
        for prob in all_problems:
            rows.append({
                "prompt": [
                    {"role": "user", "content": f"{USER_INSTRUCTION}\n\n{prob['prompt']}"}
                ],
                "test_code": prob['test'],
                "func_prompt": prob['prompt'],
            })

    return Dataset.from_list(rows)


def make_reward_function(tokenizer):
    def reward_fn(completions, **kwargs):
        prompts = kwargs.get("prompts", [])
        rewards = []
        for i, completion in enumerate(completions):
            if isinstance(completion, list):
                gen_text = completion[0].get("content", "") if completion else ""
            else:
                gen_text = completion

            gen_text = re.sub(r'<think>.*?</think>', '', gen_text, flags=re.DOTALL).strip()

            func_prompt = kwargs.get("func_prompt", [""])[i] if "func_prompt" in kwargs else ""
            test_code = kwargs.get("test_code", [""])[i] if "test_code" in kwargs else ""

            if not test_code or not func_prompt:
                rewards.append(0.0)
                continue

            full_code = func_prompt + "\n" + gen_text + "\n\n" + test_code

            result = execute_code_safe(full_code, timeout=5)

            if result["passed"]:
                rewards.append(1.0)
            else:
                try:
                    compile(func_prompt + "\n" + gen_text, '<string>', 'exec')
                    rewards.append(0.1)
                except:
                    rewards.append(0.0)

        return rewards

    return reward_fn


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Creating dataset...")
    dataset = create_problems_dataset()
    print(f"Dataset size: {len(dataset)}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH,
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

    reward_fn = make_reward_function(tokenizer)

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="grpo_v2",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
        bf16=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting GRPO training...")
    trainer.train()

    print("Saving final model...")
    final_dir = f"{OUTPUT_DIR}/final_merged"
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Ensure EOS token is just <|endoftext|> (151643)
    for fname in ["generation_config.json", "config.json"]:
        fpath = os.path.join(final_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                cfg = json.load(f)
            cfg["eos_token_id"] = 151643
            with open(fpath, 'w') as f:
                json.dump(cfg, f, indent=2)

    print(f"Final merged model saved to {final_dir}")


if __name__ == '__main__':
    main()
