#!/usr/bin/env python3
"""Parallel SciIF API evaluation runner.

This script keeps SciIF's prompts and evaluation functions unchanged. It only
parallelizes independent problem/model evaluation tasks and writes the same
JSONL record shape as evaluate_api_models.py.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sciif.evaluator import (  # noqa: E402
    evaluate_problem,
    generate_statistics_report,
    load_answer_cache,
    load_problems,
    setup_logging,
)


Task = Tuple[Dict[str, Any], str, str, str, str, int, bool, Optional[List[str]]]


def problem_id_for(problem: Dict[str, Any], original_idx: int) -> str:
    problem_id = problem.get("problem_id")
    if problem_id:
        return problem_id
    return f"{problem.get('subject', 'unknown')}_{original_idx + 1}"


def load_existing(output_file: str) -> set[tuple[str, str]]:
    existing: set[tuple[str, str]] = set()
    if not os.path.exists(output_file):
        return existing
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem_id = record.get("problem_id")
            model_name = record.get("evaluation", {}).get("model")
            if problem_id and model_name:
                existing.add((problem_id, model_name))
    return existing


def build_tasks(
    problems: List[Dict[str, Any]],
    models: Iterable[str],
    start_index: int,
    num_problems: Optional[int],
    cache_dir: str,
    validator_model: str,
    timeout: int,
    use_multi_judge: bool,
    judge_models: Optional[List[str]],
    existing: set[tuple[str, str]],
) -> list[Task]:
    selected = problems[start_index : start_index + num_problems if num_problems is not None else None]
    tasks: list[Task] = []
    for local_idx, problem in enumerate(selected):
        original_idx = start_index + local_idx
        problem_id = problem_id_for(problem, original_idx)
        for model_name in models:
            if (problem_id, model_name) in existing:
                continue
            cache_file = os.path.join(cache_dir, f"answer_cache_{model_name}.jsonl")
            tasks.append((problem, problem_id, model_name, cache_file, validator_model, timeout, use_multi_judge, judge_models))
    return tasks


def evaluate_task(task: Task) -> Dict[str, Any]:
    problem, problem_id, model_name, cache_file, validator_model, timeout, use_multi_judge, judge_models = task
    setup_logging("warning", None)
    answer_cache = load_answer_cache(cache_file)
    evaluation = evaluate_problem(
        problem=problem,
        model_name=model_name,
        validator_model=validator_model,
        timeout=timeout,
        answer_cache=answer_cache,
        cache_file=cache_file,
        problem_id=problem_id,
        use_multi_judge=use_multi_judge,
        judge_models=judge_models,
    )
    return {
        "problem_id": problem_id,
        "problem": problem,
        "evaluation": evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel SciIF API evaluation runner.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--validator_model", default="gpt-5")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_problems", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use_multi_judge", action="store_true")
    parser.add_argument("--judge_models", nargs="+", default=None)
    parser.add_argument("--log_level", default="info")
    parser.add_argument("--log_file", default=None)
    args = parser.parse_args()

    setup_logging(args.log_level, args.log_file)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    problems = load_problems(args.input_file)
    existing = load_existing(args.output_file) if args.resume else set()
    tasks = build_tasks(
        problems=problems,
        models=args.models,
        start_index=args.start_index,
        num_problems=args.num_problems,
        cache_dir=args.cache_dir,
        validator_model=args.validator_model,
        timeout=args.timeout,
        use_multi_judge=args.use_multi_judge,
        judge_models=args.judge_models,
        existing=existing,
    )

    print(f"existing={len(existing)} pending={len(tasks)} workers={args.max_workers}", flush=True)
    mode = "a" if args.resume and os.path.exists(args.output_file) else "w"
    with open(args.output_file, mode, encoding="utf-8") as out:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(evaluate_task, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Parallel evaluation", unit="task"):
                record = fut.result()
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

    generate_statistics_report(args.output_file, __import__("logging").getLogger("sciif.parallel"), save_to_file=True)


if __name__ == "__main__":
    main()
