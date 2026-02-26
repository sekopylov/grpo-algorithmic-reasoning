#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset

SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
R0=INTEGER
</answer>
""".strip()

RULES = """
You are solving register-machine tasks.
Rules are fixed and always the same:
- Registers are integers modulo M.
- Instruction pointer starts at ip=0.
- Step limit protects from infinite loops.
- Execution stops at HALT or TIMEOUT (if step limit is reached first).
- Instructions:
  SET Ri x   : Ri = x mod M
  ADD Ri x   : Ri = (Ri + x) mod M
  XOR Ri x   : Ri = Ri XOR x
  MOD Ri x   : Ri = Ri % x (x > 0)
  SWP Ri Rj  : swap Ri and Rj
  JNZ Ri off : if Ri != 0 then ip = ip + off else ip = ip + 1
  NOP        : do nothing
  HALT       : stop execution
Output exactly one final answer line inside <answer> in this format:
R0=INTEGER
""".strip()


@dataclass
class Data:
    question: str
    answer: str
    difficulty: int = 1
    metadata: Optional[dict[str, Any]] = None
    gpt_response: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response,
        }


class Verifier(ABC):
    @abstractmethod
    def verify(self, data: Data, test_answer: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract_answer(self, test_solution: str) -> Optional[str]:
        raise NotImplementedError


class Env(ABC):
    def __init__(self, name: str, verifier: type[Verifier]):
        self.name = name
        self.verifier = verifier()

    @abstractmethod
    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs,
    ) -> list[Data]:
        raise NotImplementedError

    def verify(self, data: Data, test_solution: str) -> bool:
        return self.verifier.verify(data, test_solution)

    @abstractmethod
    def extract_answer(self, test_solution: str) -> Optional[str]:
        raise NotImplementedError


class RegisterVerifier(Verifier):
    ANSWER_RE = re.compile(r"R0\s*=\s*(-?\d+)", re.IGNORECASE)

    def _last_r0(self, text: str) -> Optional[str]:
        matches = list(self.ANSWER_RE.finditer(text))
        if not matches:
            return None
        return f"R0={int(matches[-1].group(1))}"

    def extract_answer(self, test_solution: str) -> Optional[str]:
        tag_match = re.search(r"<answer>(.*?)(</answer>|$)", test_solution, flags=re.IGNORECASE | re.DOTALL)
        if tag_match:
            parsed = self._last_r0(tag_match.group(1).strip())
            if parsed is not None:
                return parsed
        after_think = re.split(r"</think>", test_solution, maxsplit=1, flags=re.IGNORECASE)
        if len(after_think) != 2:
            return None
        return self._last_r0(after_think[1].strip())

    def verify(self, data: Data, test_answer: str) -> bool:
        return self.extract_answer(test_answer) == data.answer


class TaskEnv(Env):
    def __init__(self, modulus: int = 1000, max_steps: int = 220):
        super().__init__("register_task", RegisterVerifier)
        self.modulus = modulus
        self.max_steps = max_steps

    def extract_answer(self, test_solution: str) -> Optional[str]:
        return self.verifier.extract_answer(test_solution)

    def _params_from_difficulty(self, difficulty: int) -> dict[str, int]:
        d = max(1, min(10, difficulty))
        if d <= 3:
            return {"n_registers": 3, "pre_ops": 2 + d, "loop_ops": 1 + d // 2, "loop_iters": 2 + d}
        if d <= 6:
            return {"n_registers": 4, "pre_ops": 4 + d, "loop_ops": 2 + d // 2, "loop_iters": 3 + d}
        return {"n_registers": 5, "pre_ops": 6 + d, "loop_ops": 3 + d // 2, "loop_iters": 4 + d}

    def _rand_const(self, rng: random.Random) -> int:
        return rng.randint(-30, 60)

    def _rand_pos(self, rng: random.Random) -> int:
        return rng.randint(2, 11)

    def _rand_reg(self, rng: random.Random, n: int, exclude: Optional[set[int]] = None) -> int:
        ex = exclude or set()
        return rng.choice([i for i in range(n) if i not in ex])

    def _build_program(self, rng: random.Random, p: dict[str, int]) -> list[str]:
        n = p["n_registers"]
        pre_ops = p["pre_ops"]
        loop_ops = p["loop_ops"]
        loop_iters = p["loop_iters"]

        program: list[str] = []
        for i in range(n):
            program.append(f"SET R{i} {rng.randint(0, 25)}")

        loop_reg = n - 1

        for _ in range(pre_ops):
            op = rng.choice(["ADD", "XOR", "MOD", "SWP", "NOP"])
            if op == "ADD":
                r = self._rand_reg(rng, n)
                program.append(f"ADD R{r} {self._rand_const(rng)}")
            elif op == "XOR":
                r = self._rand_reg(rng, n)
                program.append(f"XOR R{r} {rng.randint(0, 127)}")
            elif op == "MOD":
                r = self._rand_reg(rng, n)
                program.append(f"MOD R{r} {self._rand_pos(rng)}")
            elif op == "SWP":
                r1 = self._rand_reg(rng, n)
                r2 = self._rand_reg(rng, n, exclude={r1})
                program.append(f"SWP R{r1} R{r2}")
            else:
                program.append("NOP")

        program.append(f"SET R{loop_reg} {loop_iters}")
        loop_start = len(program)

        for _ in range(loop_ops):
            op = rng.choice(["ADD", "XOR", "MOD", "SWP", "NOP"])
            if op == "ADD":
                r = self._rand_reg(rng, n, exclude={loop_reg})
                program.append(f"ADD R{r} {self._rand_const(rng)}")
            elif op == "XOR":
                r = self._rand_reg(rng, n, exclude={loop_reg})
                program.append(f"XOR R{r} {rng.randint(0, 127)}")
            elif op == "MOD":
                r = self._rand_reg(rng, n, exclude={loop_reg})
                program.append(f"MOD R{r} {self._rand_pos(rng)}")
            elif op == "SWP":
                r1 = self._rand_reg(rng, n, exclude={loop_reg})
                r2 = self._rand_reg(rng, n, exclude={loop_reg, r1})
                program.append(f"SWP R{r1} R{r2}")
            else:
                program.append("NOP")

        program.append(f"ADD R{loop_reg} -1")
        jump_off = loop_start - len(program)
        program.append(f"JNZ R{loop_reg} {jump_off}")

        for _ in range(rng.randint(1, 3)):
            op = rng.choice(["ADD", "XOR", "MOD", "SWP", "NOP"])
            if op == "ADD":
                r = self._rand_reg(rng, n)
                program.append(f"ADD R{r} {self._rand_const(rng)}")
            elif op == "XOR":
                r = self._rand_reg(rng, n)
                program.append(f"XOR R{r} {rng.randint(0, 127)}")
            elif op == "MOD":
                r = self._rand_reg(rng, n)
                program.append(f"MOD R{r} {self._rand_pos(rng)}")
            elif op == "SWP":
                r1 = self._rand_reg(rng, n)
                r2 = self._rand_reg(rng, n, exclude={r1})
                program.append(f"SWP R{r1} R{r2}")
            else:
                program.append("NOP")

        program.append("HALT")
        return program

    def _simulate(self, program: list[str], n: int) -> tuple[list[int], int]:
        regs = [0 for _ in range(n)]
        ip = 0
        steps = 0

        def norm(x: int) -> int:
            return x % self.modulus

        while 0 <= ip < len(program) and steps < self.max_steps:
            steps += 1
            parts = program[ip].split()
            op = parts[0]
            if op == "HALT":
                break
            if op == "NOP":
                ip += 1
                continue
            if op == "SET":
                r = int(parts[1][1:])
                x = int(parts[2])
                regs[r] = norm(x)
                ip += 1
                continue
            if op == "ADD":
                r = int(parts[1][1:])
                x = int(parts[2])
                regs[r] = norm(regs[r] + x)
                ip += 1
                continue
            if op == "XOR":
                r = int(parts[1][1:])
                x = int(parts[2])
                regs[r] = regs[r] ^ x
                ip += 1
                continue
            if op == "MOD":
                r = int(parts[1][1:])
                x = int(parts[2])
                regs[r] = regs[r] % x
                ip += 1
                continue
            if op == "SWP":
                r1 = int(parts[1][1:])
                r2 = int(parts[2][1:])
                regs[r1], regs[r2] = regs[r2], regs[r1]
                ip += 1
                continue
            if op == "JNZ":
                r = int(parts[1][1:])
                off = int(parts[2])
                if regs[r] != 0:
                    ip = ip + off
                else:
                    ip = ip + 1
                continue
            ip += 1

        return regs, steps

    def _question(self, n: int, program: list[str]) -> str:
        lines = [f"{i}: {ins}" for i, ins in enumerate(program)]
        return (
            f"{RULES}\n\n"
            f"Current task:\n"
            f"- Number of registers: {n}\n"
            f"- Registers are initialized to 0 unless set by the program\n"
            f"- Modulus M: {self.modulus}\n"
            f"- Step limit: {self.max_steps}\n\n"
            f"Program:\n" + "\n".join(lines)
        )

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs,
    ) -> list[Data]:
        out: list[Data] = []
        attempts = 0
        d = int(difficulty if difficulty is not None else 1)
        p = self._params_from_difficulty(d)
        for k in ["n_registers", "pre_ops", "loop_ops", "loop_iters"]:
            if k in kwargs and kwargs[k] is not None:
                p[k] = int(kwargs[k])

        while len(out) < num_of_questions and attempts < max_attempts * num_of_questions:
            attempts += 1
            seed = kwargs.get("seed")
            if seed is None:
                seed = random.randint(0, 10**9)
            rng = random.Random(seed + attempts)

            program = self._build_program(rng, p)
            regs, steps = self._simulate(program, p["n_registers"])

            out.append(
                Data(
                    question=self._question(p["n_registers"], program),
                    answer=f"R0={regs[0]}",
                    difficulty=d,
                    metadata={
                        "n_registers": p["n_registers"],
                        "program_len": len(program),
                        "steps": steps,
                        "seed": seed + attempts,
                    },
                )
            )

        return out


def to_verl_row(item: Data, split: str, index: int) -> dict[str, Any]:
    return {
        "data_source": "taskenv/register",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.question},
        ],
        "ability": "reasoning",
        "reward_model": {"style": "rule", "ground_truth": item.answer},
        "extra_info": {
            "split": split,
            "index": index,
            "difficulty": item.difficulty,
            "answer": item.answer,
            **(item.metadata or {}),
        },
    }


def build_train(env: TaskEnv, train_size: int, difficulties: list[int], seed: int, max_attempts: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for i in range(train_size):
        d = difficulties[i % len(difficulties)]
        item = env.generate(
            num_of_questions=1,
            max_attempts=max_attempts,
            difficulty=d,
            seed=rng.randint(0, 10**9),
        )[0]
        rows.append(to_verl_row(item, split="train", index=i))
    return rows


def build_val(
    env: TaskEnv,
    val_per_difficulty: int,
    difficulties: list[int],
    seed: int,
    max_attempts: int,
) -> dict[int, list[dict[str, Any]]]:
    rng = random.Random(seed)
    out: dict[int, list[dict[str, Any]]] = {}
    for d in difficulties:
        rows: list[dict[str, Any]] = []
        for i in range(val_per_difficulty):
            item = env.generate(
                num_of_questions=1,
                max_attempts=max_attempts,
                difficulty=d,
                seed=rng.randint(0, 10**9),
            )[0]
            rows.append(to_verl_row(item, split=f"val_d{d}", index=i))
        out[d] = rows
    return out


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    Dataset.from_list(rows).to_parquet(str(path))


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local_save_dir", default="data/taskenv")
    p.add_argument("--train_size", type=int, default=8192)
    p.add_argument("--val_per_difficulty", type=int, default=128)
    p.add_argument("--train_difficulties", default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--val_difficulties", default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--train_seed", type=int, default=42)
    p.add_argument("--val_seed", type=int, default=31415)
    p.add_argument("--modulus", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=220)
    p.add_argument("--max_attempts", type=int, default=50)
    args = p.parse_args()

    save_dir = Path(args.local_save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    env = TaskEnv(modulus=args.modulus, max_steps=args.max_steps)
    train_diffs = parse_int_list(args.train_difficulties)
    val_diffs = parse_int_list(args.val_difficulties)

    train_rows = build_train(env, args.train_size, train_diffs, args.train_seed, args.max_attempts)
    val_by_diff = build_val(env, args.val_per_difficulty, val_diffs, args.val_seed, args.max_attempts)

    val_rows: list[dict[str, Any]] = []
    for d in val_diffs:
        rows = val_by_diff[d]
        write_parquet(rows, save_dir / f"val_d{d}.parquet")
        val_rows.extend(rows)

    write_parquet(train_rows, save_dir / "train.parquet")
    write_parquet(val_rows, save_dir / "test.parquet")

    meta = {
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "val_per_difficulty": args.val_per_difficulty,
        "train_difficulties": train_diffs,
        "val_difficulties": val_diffs,
        "modulus": args.modulus,
        "max_steps": args.max_steps,
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"save_dir": str(save_dir), **meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
