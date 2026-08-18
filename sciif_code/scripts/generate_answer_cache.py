#!/usr/bin/env python3
"""Generate SciIF answer caches without running judge evaluation."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sciif.api_client import run_responses_api
from sciif.evaluator import (
    build_answer_prompt,
    get_answer_cache_key,
    load_answer_cache,
    load_problems,
    parse_text_field,
    save_answer_to_cache,
)


def raw_answer_text(problem: Dict[str, Any]) -> tuple[str, str]:
    raw_qwc = problem.get("question_with_constraints", "")
    question = problem.get("question", "")
    reference_answer = problem.get("answer", "")

    if raw_qwc and isinstance(raw_qwc, str):
        parsed = parse_text_field(raw_qwc)
        if parsed.get("question") and not question:
            question = parsed["question"]
        if parsed.get("answer") and not reference_answer:
            reference_answer = parsed["answer"]

    if not question:
        text = problem.get("text", "")
        if text:
            parsed = parse_text_field(text)
            question = parsed.get("question") or text

    raw_for_answer = raw_qwc if isinstance(raw_qwc, str) and raw_qwc.strip() else problem.get("text", "") or question
    constraints_str = problem.get("constraints", "")
    if not isinstance(constraints_str, str):
        constraints_str = ""
    return raw_for_answer, constraints_str


async def generate_one(
    problem: Dict[str, Any],
    problem_id: str,
    model_name: str,
    cache_file: str,
    timeout: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str, str]:
    raw_for_answer, constraints_str = raw_answer_text(problem)
    cache_key = get_answer_cache_key(raw_for_answer, constraints_str, model_name, problem_id=problem_id)

    cache = load_answer_cache(cache_file)
    if cache_key in cache:
        return problem_id, model_name, "cached"

    answer_prompt = build_answer_prompt(raw_for_answer, constraints_str)

    async with semaphore:
        try:
            answer = await asyncio.to_thread(
                run_responses_api,
                model=model_name,
                input_text=answer_prompt,
                timeout=timeout,
            )
            save_answer_to_cache(
                cache_file,
                cache_key,
                answer,
                problem_id=problem_id,
                model_name=model_name,
            )
            return problem_id, model_name, "ok"
        except Exception as exc:
            return problem_id, model_name, f"error: {type(exc).__name__}: {exc}"


async def main_async(args: argparse.Namespace) -> int:
    problems = load_problems(args.input_file)
    selected = problems[args.start_index : args.start_index + args.num_problems if args.num_problems else None]
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    for local_idx, problem in enumerate(selected):
        problem_id = problem.get("problem_id")
        if not problem_id:
            original_idx = args.start_index + local_idx
            problem_id = f"{problem.get('subject', 'unknown')}_{original_idx + 1}"

        for model_name in args.models:
            cache_file = os.path.join(args.cache_dir, f"answer_cache_{model_name}.jsonl")
            tasks.append(generate_one(problem, problem_id, model_name, cache_file, args.timeout, semaphore))

    ok = cached = errors = 0
    for fut in asyncio.as_completed(tasks):
        problem_id, model_name, status = await fut
        print(f"{problem_id}\t{model_name}\t{status}", flush=True)
        if status == "ok":
            ok += 1
        elif status == "cached":
            cached += 1
        else:
            errors += 1

    print(f"summary\tok={ok}\tcached={cached}\terrors={errors}", flush=True)
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SciIF answer cache only.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_problems", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
