#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from typing import Any


def _pick_random_jsonl(path: str, rng: random.Random) -> dict[str, Any]:
    pick = None
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            if rng.randrange(seen) == 0:
                pick = json.loads(line)
    if pick is None:
        raise ValueError(f"empty jsonl: {path}")
    return pick


def _choose_file(run_dir: str, step: int, split: str, rng: random.Random) -> tuple[str, str]:
    train_path = os.path.join(run_dir, "train_generations", f"{step}.jsonl")
    val_path = os.path.join(run_dir, "val_generations", f"{step}.jsonl")

    if split == "train":
        if not os.path.exists(train_path):
            raise FileNotFoundError(train_path)
        return "train", train_path

    if split == "val":
        if not os.path.exists(val_path):
            raise FileNotFoundError(val_path)
        return "val", val_path

    if split == "both":
        candidates = []
        if os.path.exists(train_path):
            candidates.append(("train", train_path))
        if os.path.exists(val_path):
            candidates.append(("val", val_path))
        if not candidates:
            raise FileNotFoundError(f"no files for step={step} in train_generations/val_generations")
        return rng.choice(candidates)

    if os.path.exists(val_path):
        return "val", val_path
    if os.path.exists(train_path):
        return "train", train_path
    raise FileNotFoundError(f"no files for step={step} in train_generations/val_generations")


def _extract_r0(output: str) -> str | None:
    m = re.search(r"<answer>(.*?)(</answer>|$)", output, flags=re.IGNORECASE | re.DOTALL)
    block = m.group(1) if m else None
    if block is None:
        t = re.split(r"</think>", output, maxsplit=1, flags=re.IGNORECASE)
        if len(t) != 2:
            return None
        block = t[1]
    mm = list(re.finditer(r"R0\s*=\s*(-?\d+)", block, flags=re.IGNORECASE))
    if not mm:
        return None
    return f"R0={int(mm[-1].group(1))}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--split", choices=["auto", "train", "val", "both"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    split_used, path = _choose_file(args.run_dir, args.step, args.split, rng)
    row = _pick_random_jsonl(path, rng)

    output = row.get("output", "")
    parsed = _extract_r0(output)

    meta = {k: v for k, v in row.items() if k not in {"input", "output"}}
    meta["parsed_r0"] = parsed

    print("=" * 100)
    print(f"FILE: {path}")
    print(f"SPLIT: {split_used}")
    print(f"STEP: {args.step}")
    print("=" * 100)
    print("TASK")
    print("-" * 100)
    print(row.get("input", ""))
    print("-" * 100)
    print("QWEN RESPONSE")
    print("-" * 100)
    print(output)
    print("-" * 100)
    print("METADATA")
    print("-" * 100)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
