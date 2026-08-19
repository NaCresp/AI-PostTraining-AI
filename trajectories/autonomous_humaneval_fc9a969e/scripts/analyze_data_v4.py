#!/usr/bin/env python3
import json
import random

# Read all samples
with open('/home/ben/task/training_data_v4.jsonl', 'r') as f:
    samples = [json.loads(line) for line in f]

print(f'Total samples: {len(samples)}')

# Count samples with markdown blocks
markdown_count = sum(1 for s in samples if '```python' in s['text'])
print(f'Samples with markdown blocks: {markdown_count} ({100*markdown_count/len(samples):.1f}%)')

# Check system prompt variety
system_prompts = {}
for s in samples:
    text = s['text']
    start = text.find('<|im_start|>system\n') + len('<|im_start|>system\n')
    end = text.find('<|im_end|>', start)
    sys_msg = text[start:end]
    system_prompts[sys_msg] = system_prompts.get(sys_msg, 0) + 1

print(f'\nSystem prompt variety: {len(system_prompts)} unique prompts')
for prompt, count in sorted(system_prompts.items(), key=lambda x: x[1], reverse=True):
    print(f'  {count:5d} samples: {prompt[:60]}...')

# Show a few random samples
print('\n' + '='*80)
print('Random sample with PLAIN format:')
print('='*80)
plain_samples = [s for s in samples if '```python' not in s['text']]
if plain_samples:
    print(random.choice(plain_samples)['text'][:1000])

print('\n' + '='*80)
print('Random sample with MARKDOWN format:')
print('='*80)
md_samples = [s for s in samples if '```python' in s['text']]
if md_samples:
    print(random.choice(md_samples)['text'][:1000])

# Check that all have the thinking prefix
missing_think = [i for i, s in enumerate(samples) if '<think>\n\n</think>\n\n' not in s['text']]
if missing_think:
    print(f'\nWARNING: {len(missing_think)} samples missing proper thinking prefix!')
    print(f'First few indices: {missing_think[:5]}')
else:
    print('\n✓ All samples have proper <think>\\n\\n</think>\\n\\n prefix')
