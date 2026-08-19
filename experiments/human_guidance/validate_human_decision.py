#!/usr/bin/env python3
"""Validate fixed-format human guidance replies for claude-code-human."""

from __future__ import annotations

import json
import sys


def main() -> int:
    raw = sys.stdin.read().strip()
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"invalid JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(decision, dict):
        print("decision must be a JSON object", file=sys.stderr)
        return 1

    allowed_keys = {"option", "reason"}
    extra_keys = set(decision) - allowed_keys
    missing_keys = allowed_keys - set(decision)
    if extra_keys or missing_keys:
        if missing_keys:
            print(f"missing keys: {sorted(missing_keys)}", file=sys.stderr)
        if extra_keys:
            print(f"unexpected keys: {sorted(extra_keys)}", file=sys.stderr)
        return 1

    option = decision["option"]
    reason = decision["reason"]
    if option not in {"keep", "revert"}:
        print('option must be "keep" or "revert"', file=sys.stderr)
        return 1
    if not isinstance(reason, str) or not reason.strip():
        print("reason must be a non-empty string", file=sys.stderr)
        return 1

    print(json.dumps({"option": option, "reason": reason.strip()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
