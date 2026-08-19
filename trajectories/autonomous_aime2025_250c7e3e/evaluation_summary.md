# AIME 2025 Evaluation Summary

**Evaluation Date:** 2026-06-09
**Model:** vllm//home/ben/task/final_model
**Dataset:** math-ai/aime25
**Total Problems:** 30

---

## 1. Overall Performance

- **Accuracy:** 0.0333 (3.33%)
- **Problems Correct:** 1 out of 30
- **Problems Incorrect:** 29 out of 30

---

## 2. Token Usage Statistics

| Metric | Value |
|--------|-------|
| Total Input Tokens | 8,302 |
| Total Output Tokens | 53,828 |
| Total Tokens | 62,130 |
| **Average Tokens per Problem** | **2,071.0** |
| **Average Output Tokens per Problem** | **1,794.3** |

### Output Token Distribution
- **Average:** 1,794.3 tokens
- **Median:** 814 tokens
- **Min:** 477 tokens
- **Max:** 10,672 tokens
- **Range:** 10,195 tokens

---

## 3. Problem-by-Problem Results

| Problem | Target | Model Answer | Correct | Output Tokens | Notes |
|---------|--------|--------------|---------|---------------|-------|
| 0 | 70 | 49 | ✗ | 741 | |
| 1 | 588 | 894 | ✗ | 543 | |
| 2 | 16 | 226 | ✗ | 635 | |
| 3 | 117 | 51 | ✗ | 604 | |
| 4 | 279 | 550 | ✗ | 784 | |
| 5 | 504 | 144 | ✗ | 4,836 | |
| 6 | 821 | 883 | ✗ | 1,063 | |
| 7 | 77 | 4 | ✗ | 747 | ⚠️ Very short output |
| 8 | 62 | 10 | ✗ | 1,021 | |
| 9 | 81 | 295 | ✗ | 4,880 | |
| 10 | 259 | 371 | ✗ | 789 | |
| 11 | 510 | 142 | ✗ | 786 | |
| 12 | 204 | 1211 | ✗ | 3,981 | |
| 13 | 60 | 36 | ✗ | 907 | |
| 14 | 735 | 6 | ✗ | 477 | ⚠️ Very short output |
| 15 | 468 | 3200 | ✗ | 576 | |
| **16** | **49** | **49** | **✓** | **3,265** | **ONLY CORRECT ANSWER** |
| 17 | 82 | 6 | ✗ | 581 | ⚠️ Very short output |
| 18 | 106 | 9 | ✗ | 880 | ⚠️ Very short output |
| 19 | 336 | 1680 | ✗ | 4,687 | |
| 20 | 293 | 7 | ✗ | 640 | ⚠️ Very short output |
| 21 | 237 | 8191 | ✗ | 711 | |
| 22 | 610 | 91 | ✗ | 623 | |
| 23 | 149 | 51 | ✗ | 10,672 | ⚠️ EXCESSIVE TOKENS |
| 24 | 907 | 153 | ✗ | 559 | |
| 25 | 113 | 9831 | ✗ | 818 | |
| 26 | 19 | 154 | ✗ | 814 | |
| 27 | 248 | 235 | ✗ | 953 | |
| 28 | 104 | 98 | ✗ | 4,294 | |
| 29 | 240 | 73 | ✗ | 961 | |

---

## 4. Patterns and Analysis

### 4.1 Correct Problems
- **Only Problem 16 was solved correctly**
  - Problem: "Find the sum of all positive integers $n$ such that $n + 2$ divides the product..."
  - Target: 49
  - Model Answer: 49
  - Tokens Used: 3,265 (above average, suggesting more thorough reasoning)

### 4.2 Degenerate/Problematic Outputs

**Six problems showed degenerate output patterns:**

1. **Very Short Outputs (5 problems):** Problems where the model likely gave up or failed to reason properly
   - Problem 7: Answer 4 (target 77) - 747 tokens
   - Problem 14: Answer 6 (target 735) - 477 tokens (SHORTEST)
   - Problem 17: Answer 6 (target 82) - 581 tokens
   - Problem 18: Answer 9 (target 106) - 880 tokens
   - Problem 20: Answer 7 (target 293) - 640 tokens

2. **Excessive Tokens (1 problem):** Problem where the model got stuck in repetitive reasoning
   - Problem 23: Answer 51 (target 149) - 10,672 tokens (LONGEST, ~6x average)

### 4.3 Near Misses
Some problems where the model was close to the correct answer:
- Problem 27: Model answered 235, target was 248 (off by 13)
- Problem 28: Model answered 98, target was 104 (off by 6)
- Problem 6: Model answered 883, target was 821 (off by 62)

### 4.4 Wildly Incorrect Answers
Problems where the model was very far off:
- Problem 21: Answer 8191 vs target 237 (34.5x too large)
- Problem 25: Answer 9831 vs target 113 (87x too large)
- Problem 15: Answer 3200 vs target 468 (6.8x too large)
- Problem 19: Answer 1680 vs target 336 (5x too large)

### 4.5 Problem Difficulty Pattern
- No clear pattern emerges based on problem number (the one correct answer was problem 16, roughly in the middle)
- The model struggled uniformly across all problem types
- Degenerate outputs appear randomly distributed (problems 7, 14, 17, 18, 20, 23)

---

## 5. Key Findings

1. **Very Poor Performance:** Only 3.33% accuracy (1/30) indicates the model is not effective at solving AIME-level mathematics problems.

2. **Inconsistent Reasoning:** The wide variance in token usage (477 to 10,672) suggests unstable reasoning patterns.

3. **Degenerate Outputs:** 20% of problems (6/30) showed signs of degenerate behavior, either giving up too early or getting stuck in loops.

4. **No Pattern to Success:** The single correct answer (problem 16) doesn't reveal any clear pattern about what types of problems the model can solve.

5. **Token Efficiency:** The model used an average of 1,794 output tokens per problem, but this didn't correlate with correctness (the correct answer used 3,265 tokens, well above average).

6. **Answer Distribution Issues:** Several answers were orders of magnitude off from the target, suggesting fundamental reasoning errors rather than minor calculation mistakes.

---

## 6. Conclusion

The model demonstrates severe difficulties with AIME-level mathematical reasoning, achieving only 1 correct answer out of 30. The presence of degenerate outputs and wildly incorrect answers suggests the model lacks the mathematical reasoning capabilities needed for competition-level mathematics problems. The high token usage without corresponding accuracy indicates the model is capable of generating lengthy reasoning chains but struggles to arrive at correct conclusions.
