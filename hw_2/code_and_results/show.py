#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_question(path: Path, task_id: int) -> str:
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == task_id:
                return json.loads(line).get('question', '')
    return ''


def main() -> None:
    p = argparse.ArgumentParser(description='show')
    p.add_argument('--task-id', type=int, required=True)
    p.add_argument('--results-file', default='/home/seankopylov/projects/GRPO/hw_2/code_and_results/results/results.jsonl')
    args = p.parse_args()

    results_file = Path(args.results_file)
    rows = load_jsonl(results_file)
    row = [r for r in rows if int(r.get('task_id', -1)) == args.task_id][0]
    task_file = Path(row['task_file'])
    q = load_question(task_file, args.task_id)

    print('=' * 100)
    print(f'Results file: {results_file}')
    print(f'Task file:    {task_file}')
    print(f'Task ID:      {args.task_id}')
    print('=' * 100)
    print('TASK')
    print('-' * 100)
    print(q.strip())
    print('-' * 100)
    print('TASK METADATA')
    print('-' * 100)
    print(json.dumps(row.get('metadata', {}), ensure_ascii=False, indent=2))
    print('-' * 100)
    print('QWEN RESPONSE')
    print('-' * 100)
    print(row.get('pred_raw', '').strip())
    print('-' * 100)
    print('REWARD')
    print('-' * 100)
    print(f"reward={row.get('reward')} | correct={row.get('correct')}")
    print(f"gold={row.get('gold_answer')} | parsed={row.get('pred_parsed')}")
    print('=' * 100)


if __name__ == '__main__':
    main()
