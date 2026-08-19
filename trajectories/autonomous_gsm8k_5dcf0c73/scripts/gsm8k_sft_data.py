#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


GSM8K_DATASET = ("openai/gsm8k", "main")


@dataclass(frozen=True)
class Gsm8kExample:
    prompt: str
    completion: str
    target: str


MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def gsm8k_record_to_example(record: dict[str, Any]) -> Gsm8kExample:
    delim = "####"
    question = record["question"].strip()
    answer_parts = record["answer"].split(delim)
    target = answer_parts.pop().strip()
    reasoning = delim.join(answer_parts).strip()

    prompt = MATH_PROMPT_TEMPLATE.format(prompt=question)
    # Append the model's EOS token so vLLM stops generation cleanly; otherwise
    # it can continue past the answer line until `max_tokens`.
    completion = f"{reasoning}\n\nANSWER: {target}<|endoftext|>".strip()
    return Gsm8kExample(prompt=prompt, completion=completion, target=target)
