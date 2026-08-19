"""
Prepare v3 training data: fixes import handling.
In v2, imports before 'def' were separated into the prompt but NOT included in the completion.
When the model generates code that uses re/math/functools etc., it fails because the eval
prepends only the function signature (from HumanEval prompt), not the imports.

Fix: include necessary imports AS PART OF the function body in training completions.
"""
import json
import re

INPUT_PATH = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data_v2.jsonl"
OUTPUT_PATH = "/workspace/AI4AI/experiments/claude-code-humaneval/workspace/train_data_v3.jsonl"

# Common imports that should be included in completions
COMMON_IMPORTS = {'re', 'math', 'functools', 'itertools', 'collections', 'hashlib',
                  'heapq', 'bisect', 'random', 'string', 'operator', 'typing'}


def detect_needed_imports(body: str, available_imports: list) -> list:
    """Detect which imports from the available list are actually used in the body."""
    needed = []
    for imp_line in available_imports:
        # Parse: 'import X' or 'from X import Y'
        imp_line = imp_line.strip()
        if imp_line.startswith('from '):
            match = re.match(r'from\s+(\S+)\s+import\s+(.*)', imp_line)
            if match:
                module = match.group(1)
                names = [n.strip() for n in match.group(2).split(',')]
                for name in names:
                    name = name.strip()
                    if name and re.search(r'\b' + re.escape(name) + r'\b', body):
                        needed.append(imp_line)
                        break
        elif imp_line.startswith('import '):
            match = re.match(r'import\s+(\S+)(?:\s+as\s+(\S+))?', imp_line)
            if match:
                module = match.group(1)
                alias = match.group(2) or module
                if re.search(r'\b' + re.escape(alias) + r'\b', body):
                    needed.append(imp_line)
    return list(set(needed))


def extract_and_fix(text: str):
    """Extract function with imports properly included in the body."""
    lines = text.strip().split('\n')

    def_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def '):
            def_idx = i
            break

    if def_idx is None:
        return None, None

    # Collect imports before def
    import_lines = []
    for i in range(def_idx):
        stripped = lines[i].strip()
        if stripped and (stripped.startswith('from ') or stripped.startswith('import ')):
            import_lines.append(stripped)

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

    # Build prompt (function signature + docstring only, NO imports)
    prompt = '\n'.join(lines[def_idx:docstring_end + 1])

    # Build body
    body_lines = lines[docstring_end + 1:]
    while body_lines and body_lines[0].strip() == '':
        body_lines.pop(0)
    while body_lines and body_lines[-1].strip() == '':
        body_lines.pop()
    body = '\n'.join(body_lines)

    if not body.strip():
        return None, None

    # Detect which imports are actually needed by the body
    if import_lines:
        needed = detect_needed_imports(body, import_lines)
        if needed:
            # Prepend imports to body with proper indentation
            # Since body is inside a function, imports need to go BEFORE the body
            # But actually, in Python, imports can be at the top level
            # The evaluator prepends the function signature, so we need imports before the function
            # But we can't control that... unless we add them as inline imports inside the function body

            # Convert to inline imports inside the function
            inline_imports = []
            for imp in needed:
                inline_imports.append('    ' + imp)  # Indent inside function
            body = '\n'.join(inline_imports) + '\n' + body

    return prompt, body


def main():
    examples = []
    skipped = 0

    with open(INPUT_PATH) as f:
        for line in f:
            item = json.loads(line)
            text = item.get('text', '')
            if not text:
                text = item.get('prompt', '') + item.get('completion', '')

            prompt, body = extract_and_fix(text)
            if prompt and body:
                examples.append({"prompt": prompt, "body": body})
            else:
                skipped += 1

    print(f"Loaded {len(examples)} examples, skipped {skipped}")

    # Write output
    with open(OUTPUT_PATH, 'w') as f:
        for ex in examples:
            f.write(json.dumps({"prompt": ex["prompt"], "body": ex["body"]}) + '\n')

    # Show some examples with imports
    import_count = 0
    for ex in examples:
        if ex["body"].strip().startswith("    import ") or ex["body"].strip().startswith("    from "):
            if import_count < 3:
                print(f"\n--- Example with inline imports ---")
                print(f"Prompt: {ex['prompt'][:100]}...")
                print(f"Body:\n{ex['body'][:200]}")
            import_count += 1
    print(f"\nExamples with inline imports: {import_count}")


if __name__ == '__main__':
    main()
