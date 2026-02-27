#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path


def pick_random_jsonl(path: Path, rng: random.Random):
    pick = None
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            if rng.randrange(seen) == 0:
                pick = json.loads(line)
    if pick is None:
        raise ValueError(f"empty jsonl: {path}")
    return pick, seen


def parse_step(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return 10**18


def extract_r0(output: str):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show-input", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    val_dir = run_dir / "val_generations"
    if not val_dir.exists():
        raise FileNotFoundError(f"missing directory: {val_dir}")

    files = sorted(val_dir.glob("*.jsonl"), key=parse_step)
    if not files:
        raise FileNotFoundError(f"no jsonl files in {val_dir}")

    rng = random.Random(args.seed)

    print("=" * 120)
    print(f"RUN_DIR: {run_dir}")
    print(f"VAL_FILES: {len(files)}")
    print("=" * 120)

    for path in files:
        row, n_rows = pick_random_jsonl(path, rng)
        step = row.get("step")
        if step is None:
            step = parse_step(path)

        output = row.get("output", "")
        gts = row.get("gts")
        if isinstance(gts, list):
            gts = gts[0] if gts else None

        parsed = extract_r0(output)

        print(f"STEP {step} | file={path.name} | sampled_from={n_rows} rows")
        print(f"score={row.get('score')} | acc={row.get('acc')} | gts={gts} | parsed={parsed}")
        if args.show_input:
            print("-" * 120)
            print("INPUT")
            print("-" * 120)
            print(row.get("input", ""))
        print("-" * 120)
        print("OUTPUT")
        print("-" * 120)
        print(output)
        print("=" * 120)


if __name__ == "__main__":
    main()
