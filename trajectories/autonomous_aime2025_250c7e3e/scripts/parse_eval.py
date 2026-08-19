#!/usr/bin/env python3
import json

# Read the JSON file
with open('/home/ben/task/logs/2026-06-09T18-36-59+00-00_aime2025_KvqH4D7TYvLmzEgpRq95ae.json', 'r') as f:
    data = json.load(f)

# Extract overall accuracy
accuracy = data['results']['scores'][0]['metrics']['accuracy']['value']
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"This means {int(accuracy * 30)}/30 problems correct\n")

# Extract token usage statistics
total_input_tokens = data['stats']['model_usage']['vllm//home/ben/task/final_model']['input_tokens']
total_output_tokens = data['stats']['model_usage']['vllm//home/ben/task/final_model']['output_tokens']
total_tokens = data['stats']['model_usage']['vllm//home/ben/task/final_model']['total_tokens']

print(f"Total Token Usage:")
print(f"  Input tokens: {total_input_tokens:,}")
print(f"  Output tokens: {total_output_tokens:,}")
print(f"  Total tokens: {total_tokens:,}")
print(f"  Average tokens per problem: {total_tokens/30:.1f}")
print(f"  Average output tokens per problem: {total_output_tokens/30:.1f}\n")

# Extract per-problem details
print("=" * 100)
print("DETAILED PROBLEM-BY-PROBLEM ANALYSIS")
print("=" * 100)

correct_problems = []
incorrect_problems = []
degenerate_outputs = []

for sample in data['samples']:
    problem_id = sample['id']
    target = sample['target']

    # Extract model answer and correctness
    score_info = sample['scores']['aime_scorer']
    model_answer = score_info.get('answer', 'N/A')
    correct = (score_info['value'] == 'C')

    # Extract token usage
    output_tokens = sample['output']['usage']['output_tokens']
    input_tokens = sample['output']['usage']['input_tokens']
    total_tokens_problem = sample['output']['usage']['total_tokens']

    # Check for degenerate output
    completion = sample['output']['completion']
    is_degenerate = False
    degenerate_reason = ""

    if len(completion) < 10:
        is_degenerate = True
        degenerate_reason = "Very short output"
    elif "ANSWER:" not in completion:
        is_degenerate = True
        degenerate_reason = "Missing ANSWER format"
    elif output_tokens > 5000:
        is_degenerate = True
        degenerate_reason = f"Excessive tokens ({output_tokens})"

    # Categorize
    if correct:
        correct_problems.append(int(problem_id))
    else:
        incorrect_problems.append(int(problem_id))

    if is_degenerate:
        degenerate_outputs.append({
            'id': problem_id,
            'reason': degenerate_reason,
            'tokens': output_tokens,
            'completion_preview': completion[:100]
        })

    # Get problem statement (first 80 chars)
    problem_text = sample['input'][:80] + "..." if len(sample['input']) > 80 else sample['input']

    status_symbol = "✓" if correct else "✗"
    print(f"\nProblem {int(problem_id):2d}: {status_symbol} {'CORRECT' if correct else 'INCORRECT'}")
    print(f"  Problem: {problem_text}")
    print(f"  Target answer: {target}")
    print(f"  Model answer:  {model_answer}")
    print(f"  Tokens: {output_tokens} output, {input_tokens} input, {total_tokens_problem} total")
    if is_degenerate:
        print(f"  ⚠️  DEGENERATE OUTPUT: {degenerate_reason}")

print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

print(f"\nCorrect Problems ({len(correct_problems)}/30):")
if correct_problems:
    print(f"  Problem IDs: {sorted(correct_problems)}")
else:
    print("  None")

print(f"\nIncorrect Problems ({len(incorrect_problems)}/30):")
if incorrect_problems:
    print(f"  Problem IDs: {sorted(incorrect_problems)}")

print(f"\nProblems with Degenerate Output ({len(degenerate_outputs)}):")
if degenerate_outputs:
    for deg in degenerate_outputs:
        print(f"  Problem {deg['id']}: {deg['reason']} - {deg['tokens']} tokens")
else:
    print("  None detected")

print("\n" + "=" * 100)
print("TOKEN USAGE ANALYSIS")
print("=" * 100)

# Calculate token statistics per problem
token_counts = []
for sample in data['samples']:
    token_counts.append(sample['output']['usage']['output_tokens'])

token_counts.sort()
avg_tokens = sum(token_counts) / len(token_counts)
median_tokens = token_counts[len(token_counts)//2]
min_tokens = min(token_counts)
max_tokens = max(token_counts)

print(f"\nOutput Tokens per Problem:")
print(f"  Average: {avg_tokens:.1f}")
print(f"  Median:  {median_tokens}")
print(f"  Min:     {min_tokens}")
print(f"  Max:     {max_tokens}")
print(f"  Range:   {max_tokens - min_tokens}")
