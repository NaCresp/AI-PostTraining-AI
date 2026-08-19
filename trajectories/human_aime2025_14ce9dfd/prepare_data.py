"""Prepare training data for GRPO from dapo_math_17k."""
import pandas as pd
import numpy as np
import os

OUT_DIR = "/workspace/AI4AI/experiments/claude-code-human/workspace/data"

def prepare():
    df = pd.read_parquet("/workspace/AI4AI/benchmark/dapo_math_17k/train.parquet")
    print(f"Loaded {len(df)} examples from dapo_math_17k")

    rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()

        gt = str(row["reward_model"]["ground_truth"]) if isinstance(row["reward_model"], dict) else str(row["reward_model"])

        rows.append({
            "prompt": prompt,
            "reward_model": {"ground_truth": gt},
            "data_source": "math_dapo",
        })

    out_df = pd.DataFrame(rows)

    np.random.seed(42)
    idx = np.random.permutation(len(out_df))
    val_size = min(500, len(out_df) // 20)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    train_df = out_df.iloc[train_idx].reset_index(drop=True)
    val_df = out_df.iloc[val_idx].reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    train_df.to_parquet(f"{OUT_DIR}/train.parquet")
    val_df.to_parquet(f"{OUT_DIR}/val.parquet")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Sample prompt: {train_df.iloc[0]['prompt'][0]['content'][:200]}")
    print(f"Sample ground_truth: {train_df.iloc[0]['reward_model']}")

if __name__ == "__main__":
    prepare()
