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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    @classmethod
    def from_json_dict(cls, json_dict: dict[str, Any]) -> "Data":
        item = cls(
            question=json_dict["question"],
            answer=json_dict["answer"],
            difficulty=json_dict.get("difficulty", 1),
            metadata=json_dict.get("metadata"),
        )
        item.gpt_response = json_dict.get("gpt_response", "")
        return item


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


def save_jsonl(items: list[Data], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x.to_json(), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[Data]:
    out: list[Data] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Data.from_json_dict(json.loads(line)))
    return out


class Runner:
    def __init__(self, model_name: str, max_new_tokens: int):
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def _prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{question}\n\nAssistant:\n"

    @torch.inference_mode()
    def solve_batch(self, questions: list[str]) -> list[str]:
        prompts = [self._prompt(q) for q in questions]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        lens = inputs["attention_mask"].sum(dim=1).tolist()
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        preds = []
        for i, ln in enumerate(lens):
            gen = out[i, int(ln) :]
            preds.append(self.tokenizer.decode(gen, skip_special_tokens=True).strip())
        return preds


def cmd_prepare(args: argparse.Namespace) -> None:
    env = TaskEnv(modulus=args.modulus, max_steps=args.max_steps)
    out_file = Path(args.out_file)
    diffs = [int(x.strip()) for x in args.difficulties.split(",") if x.strip()]
    if not diffs:
        raise ValueError("empty difficulties")

    tasks: list[Data] = []
    rng = random.Random(args.seed)
    for i in range(args.num_questions):
        d = diffs[i % len(diffs)]
        item = env.generate(
            num_of_questions=1,
            max_attempts=10,
            difficulty=d,
            seed=rng.randint(0, 10**9),
            n_registers=args.n_registers,
            pre_ops=args.pre_ops,
            loop_ops=args.loop_ops,
            loop_iters=args.loop_iters,
        )[0]
        tasks.append(item)

    save_jsonl(tasks, out_file)
    print(f"saved {len(tasks)} tasks to {out_file}")


def cmd_run(args: argparse.Namespace) -> None:
    env = TaskEnv(modulus=args.modulus, max_steps=args.max_steps)
    tasks_file = Path(args.tasks_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_jsonl(tasks_file)
    runner = Runner(args.model, args.max_new_tokens)

    rows = []
    bs = max(1, int(args.batch_size))
    for start in range(0, len(tasks), bs):
        batch = tasks[start : start + bs]
        preds = runner.solve_batch([x.question for x in batch])
        for i, (item, pred) in enumerate(zip(batch, preds)):
            idx = start + i
            parsed = env.extract_answer(pred)
            correct = parsed == item.answer
            rows.append(
                {
                    "task_file": str(tasks_file),
                    "task_id": idx,
                    "difficulty": item.difficulty,
                    "gold_answer": item.answer,
                    "pred_raw": pred,
                    "pred_parsed": parsed,
                    "reward": 1.0 if correct else 0.0,
                    "correct": correct,
                    "metadata": item.metadata,
                }
            )

    results_file = out_dir / "results.jsonl"
    summary_file = out_dir / "summary.json"

    with results_file.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    diffs = sorted({int(r["difficulty"]) for r in rows})
    by_diff: dict[int, dict[str, float]] = {}
    for d in diffs:
        ss = [r for r in rows if int(r["difficulty"]) == d]
        by_diff[d] = {"acc": sum(x["correct"] for x in ss) / len(ss), "count": len(ss)}

    summary = {
        "task_file": str(tasks_file),
        "results_file": str(results_file),
        "num_tasks": len(rows),
        "overall_acc": sum(r["correct"] for r in rows) / len(rows) if rows else 0.0,
        "by_difficulty": by_diff,
        "num_correct": sum(r["correct"] for r in rows),
        "num_format_fail": sum(1 for r in rows if r["pred_parsed"] is None),
    }

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare")
    pp.add_argument("--out-file", default="/home/seankopylov/projects/GRPO/hw_2/tasks/tasks.jsonl")
    pp.add_argument("--num-questions", type=int, default=24)
    pp.add_argument("--difficulties", default="1,2,3,4,6,8")
    pp.add_argument("--seed", type=int, default=42)
    pp.add_argument("--modulus", type=int, default=1000)
    pp.add_argument("--max-steps", type=int, default=220)
    pp.add_argument("--n-registers", type=int, default=None)
    pp.add_argument("--pre-ops", type=int, default=None)
    pp.add_argument("--loop-ops", type=int, default=None)
    pp.add_argument("--loop-iters", type=int, default=None)
    pp.set_defaults(func=cmd_prepare)

    pr = sub.add_parser("run")
    pr.add_argument("--tasks-file", default="/home/seankopylov/projects/GRPO/hw_2/tasks/tasks.jsonl")
    pr.add_argument("--output-dir", default="/home/seankopylov/projects/GRPO/hw_2/code_and_results/results")
    pr.add_argument("--model", default="Qwen/Qwen3-1.7B")
    pr.add_argument("--max-new-tokens", type=int, default=1024)
    pr.add_argument("--batch-size", type=int, default=16)
    pr.add_argument("--modulus", type=int, default=1000)
    pr.add_argument("--max-steps", type=int, default=220)
    pr.set_defaults(func=cmd_run)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
