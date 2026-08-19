"""Prepare math training data v4 - improved system prompt to prevent template regurgitation."""
import pandas as pd
from datasets import Dataset
import os

TRAIN_ARROW = "/root/.cache/huggingface/datasets/parquet/default-a81414a50954c389/0.0.0/5ee32a436a477cb54f12417e18b7e7cf41a234af73ca4fd86cc7c57108a4cbca/parquet-train.arrow"
VAL_ARROW = "/root/.cache/huggingface/datasets/parquet/default-709814f08ad5760f/0.0.0/5ee32a436a477cb54f12417e18b7e7cf41a234af73ca4fd86cc7c57108a4cbca/parquet-train.arrow"

SYSTEM_PROMPT = "Solve the following math problem step by step. Show your work, then give your final answer on a new line in the format:\nAnswer: [your numerical answer]"

SUFFIXES_TO_REMOVE = [
    '\n\nRemember to put your answer on its own line after "Answer:".',
    '\nRemember to put your answer on its own line after "Answer:".',
    'Remember to put your answer on its own line after "Answer:".',
]

PREFIXES_TO_REMOVE = [
    "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n",
    "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n",
]

def clean_user_content(text):
    for prefix in PREFIXES_TO_REMOVE:
        if text.startswith(prefix):
            text = text[len(prefix):]
    for suffix in SUFFIXES_TO_REMOVE:
        if text.endswith(suffix):
            text = text[:-len(suffix)]
    return text.strip()

def convert_dataset(arrow_path, output_path, name="train"):
    ds = Dataset.from_file(arrow_path)
    print(f"Loaded {len(ds)} examples from {name}")

    rows = []
    for i in range(len(ds)):
        sample = ds[i]
        user_content = sample['prompt'][-1]['content']
        user_content = clean_user_content(user_content)
        ground_truth = str(sample['reward_model']['ground_truth'])

        rows.append({
            'prompt': [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            'reward_model': {"ground_truth": ground_truth},
            'data_source': 'math'
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    print(f"Saved {len(df)} examples to {output_path}")

    check = pd.read_parquet(output_path)
    print(f"Verification: {len(check)} rows")
    s = check.iloc[0]
    print(f"  system: {s['prompt'][0]['content']}")
    print(f"  user: {s['prompt'][1]['content'][:150]}")

if __name__ == "__main__":
    out_dir = "/workspace/AI4AI/experiments/claude-code-verl/workspace/data"
    convert_dataset(TRAIN_ARROW, f"{out_dir}/train_v4.parquet", "train")
    convert_dataset(VAL_ARROW, f"{out_dir}/val_v4.parquet", "val")
