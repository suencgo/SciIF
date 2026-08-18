#!/usr/bin/env python3
"""
Analyze evaluation results and generate comprehensive statistics report
"""
import json
import sys
from collections import defaultdict
from typing import Dict, Any, List
from datetime import datetime

def analyze_results(result_file: str):
    """Analyze evaluation results and generate comprehensive statistics report"""
    results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not results:
        print("No results found in file")
        return
    
    total = len(results)
    
    # Statistics
    single_constraint_pass = 0
    single_constraint_total = 0
    multi_constraint_pass = 0
    multi_constraint_total = 0
    answer_correctness_pass = 0
    answer_correctness_total = 0
    errors = 0
    
    # Per-constraint statistics
    constraint_stats = defaultdict(lambda: {"pass": 0, "total": 0, "fail": 0})
    
    # Model information
    model_name = None
    judge_models = []
    
    # Problem-level statistics
    problem_stats = {
        "total": total,
        "passed_all": 0,
        "failed_constraints": 0,
        "failed_answer": 0,
        "errors": 0
    }
    
    for result in results:
        # Get model and judge information
        evaluation = result.get("evaluation", {})
        if not model_name:
            model_name = evaluation.get("model", "unknown")
        
        # Check for multi-judge
        constraint_validation = result.get("constraint_validation", {})
        if constraint_validation.get("judge_votes"):
            judge_info = constraint_validation.get("judge_votes", {})
            if judge_info.get("models"):
                judge_models = judge_info.get("models", [])
        
        # Check for errors
        if result.get("status") == "ERROR":
            errors += 1
            problem_stats["errors"] += 1
                continue
            
        # Single constraint validation
        individual = constraint_validation.get("individual", {})
        
        if individual:
            for constraint_name, constraint_result in individual.items():
                single_constraint_total += 1
                constraint_stats[constraint_name]["total"] += 1
                status = constraint_result.get("status", "UNKNOWN")
                if status == "PASS":
                    single_constraint_pass += 1
                    constraint_stats[constraint_name]["pass"] += 1
                else:
                    constraint_stats[constraint_name]["fail"] += 1
        
        # Multi-constraint validation (overall)
                    overall = constraint_validation.get("overall", {})
        if overall:
                        multi_constraint_total += 1
            status = overall.get("status", "UNKNOWN")
            if status == "PASS":
                            multi_constraint_pass += 1
                        else:
                problem_stats["failed_constraints"] += 1
                    
        # Answer correctness
        answer_correctness = result.get("answer_correctness", {})
        if answer_correctness:
            answer_correctness_total += 1
            status = answer_correctness.get("status", "UNKNOWN")
            if status == "PASS":
                answer_correctness_pass += 1
            else:
                problem_stats["failed_answer"] += 1
        
        # Check if problem passed all checks
        if (overall and overall.get("status") == "PASS" and 
            answer_correctness and answer_correctness.get("status") == "PASS"):
            problem_stats["passed_all"] += 1
    
    # Calculate rates
    single_constraint_rate = (single_constraint_pass / single_constraint_total * 100) if single_constraint_total > 0 else 0
    multi_constraint_rate = (multi_constraint_pass / multi_constraint_total * 100) if multi_constraint_total > 0 else 0
    answer_correctness_rate = (answer_correctness_pass / answer_correctness_total * 100) if answer_correctness_total > 0 else 0
    overall_pass_rate = (problem_stats["passed_all"] / total * 100) if total > 0 else 0
    
    # Generate comprehensive report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Evaluation Results Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated at: {timestamp}")
    report_lines.append(f"Result file: {result_file}")
    report_lines.append("")
    
    # Model information
    report_lines.append("Model Configuration:")
    report_lines.append("-" * 80)
    report_lines.append(f"Answer Model: {model_name}")
    if judge_models:
        report_lines.append(f"Judge Models: {', '.join(judge_models)}")
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("Overall Statistics:")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Problems Evaluated: {total}")
    report_lines.append(f"Problems Passed All Checks: {problem_stats['passed_all']} ({overall_pass_rate:.2f}%)")
    report_lines.append(f"Problems Failed Constraints: {problem_stats['failed_constraints']}")
    report_lines.append(f"Problems Failed Answer Correctness: {problem_stats['failed_answer']}")
    report_lines.append(f"Errors: {errors}")
    report_lines.append("")
    
    # Main metrics table
    report_lines.append("Main Metrics:")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Metric':<35} {'Pass':<10} {'Total':<10} {'Rate':<15}")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Single Constraint Pass Rate':<35} {single_constraint_pass:<10} {single_constraint_total:<10} {single_constraint_rate:>6.2f}%")
    report_lines.append(f"{'Multi-Constraint Pass Rate':<35} {multi_constraint_pass:<10} {multi_constraint_total:<10} {multi_constraint_rate:>6.2f}%")
    report_lines.append(f"{'Answer Correctness Rate':<35} {answer_correctness_pass:<10} {answer_correctness_total:<10} {answer_correctness_rate:>6.2f}%")
    report_lines.append(f"{'Overall Pass Rate (All Checks)':<35} {problem_stats['passed_all']:<10} {total:<10} {overall_pass_rate:>6.2f}%")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Per-constraint breakdown
    if constraint_stats:
        report_lines.append("Per-Constraint Statistics:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Constraint Name':<45} {'Pass':<10} {'Fail':<10} {'Total':<10} {'Rate':<15}")
        report_lines.append("-" * 80)
        for constraint_name in sorted(constraint_stats.keys()):
            stats = constraint_stats[constraint_name]
            rate = (stats["pass"] / stats["total"] * 100) if stats["total"] > 0 else 0
            report_lines.append(f"{constraint_name:<45} {stats['pass']:<10} {stats['fail']:<10} {stats['total']:<10} {rate:>6.2f}%")
        report_lines.append("=" * 80)
        report_lines.append("")
    
    # Summary
    report_lines.append("Summary:")
    report_lines.append("-" * 80)
    report_lines.append(f"• Single Constraint Compliance: {single_constraint_rate:.2f}% ({single_constraint_pass}/{single_constraint_total})")
    report_lines.append(f"• Multi-Constraint Compliance: {multi_constraint_rate:.2f}% ({multi_constraint_pass}/{multi_constraint_total})")
    report_lines.append(f"• Answer Correctness: {answer_correctness_rate:.2f}% ({answer_correctness_pass}/{answer_correctness_total})")
    report_lines.append(f"• Overall Pass Rate: {overall_pass_rate:.2f}% ({problem_stats['passed_all']}/{total})")
    report_lines.append("=" * 80)
    
    # Print to console
    for line in report_lines:
        print(line)
    
    # Save to file
    output_file = result_file.replace('.jsonl', '_report.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
        f.write('\n')
    
    # Also save as markdown format
    md_file = result_file.replace('.jsonl', '_report.md')
    md_lines = []
    md_lines.append("# Evaluation Results Report")
    md_lines.append("")
    md_lines.append(f"**Generated at:** {timestamp}  ")
    md_lines.append(f"**Result file:** `{result_file}`  ")
    md_lines.append("")
    md_lines.append("## Model Configuration")
    md_lines.append("")
    md_lines.append(f"- **Answer Model:** {model_name}")
    if judge_models:
        md_lines.append(f"- **Judge Models:** {', '.join(judge_models)}")
    md_lines.append("")
    md_lines.append("## Overall Statistics")
    md_lines.append("")
    md_lines.append(f"- **Total Problems Evaluated:** {total}")
    md_lines.append(f"- **Problems Passed All Checks:** {problem_stats['passed_all']} ({overall_pass_rate:.2f}%)")
    md_lines.append(f"- **Problems Failed Constraints:** {problem_stats['failed_constraints']}")
    md_lines.append(f"- **Problems Failed Answer Correctness:** {problem_stats['failed_answer']}")
    md_lines.append(f"- **Errors:** {errors}")
    md_lines.append("")
    md_lines.append("## Main Metrics")
    md_lines.append("")
    md_lines.append("| Metric | Pass | Total | Rate |")
    md_lines.append("|--------|------|-------|------|")
    md_lines.append(f"| Single Constraint Pass Rate | {single_constraint_pass} | {single_constraint_total} | {single_constraint_rate:.2f}% |")
    md_lines.append(f"| Multi-Constraint Pass Rate | {multi_constraint_pass} | {multi_constraint_total} | {multi_constraint_rate:.2f}% |")
    md_lines.append(f"| Answer Correctness Rate | {answer_correctness_pass} | {answer_correctness_total} | {answer_correctness_rate:.2f}% |")
    md_lines.append(f"| Overall Pass Rate (All Checks) | {problem_stats['passed_all']} | {total} | {overall_pass_rate:.2f}% |")
    md_lines.append("")
    
    if constraint_stats:
        md_lines.append("## Per-Constraint Statistics")
        md_lines.append("")
        md_lines.append("| Constraint Name | Pass | Fail | Total | Rate |")
        md_lines.append("|----------------|------|------|-------|------|")
        for constraint_name in sorted(constraint_stats.keys()):
            stats = constraint_stats[constraint_name]
            rate = (stats["pass"] / stats["total"] * 100) if stats["total"] > 0 else 0
            md_lines.append(f"| {constraint_name} | {stats['pass']} | {stats['fail']} | {stats['total']} | {rate:.2f}% |")
        md_lines.append("")
    
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append(f"- Single Constraint Compliance: **{single_constraint_rate:.2f}%** ({single_constraint_pass}/{single_constraint_total})")
    md_lines.append(f"- Multi-Constraint Compliance: **{multi_constraint_rate:.2f}%** ({multi_constraint_pass}/{multi_constraint_total})")
    md_lines.append(f"- Answer Correctness: **{answer_correctness_rate:.2f}%** ({answer_correctness_pass}/{answer_correctness_total})")
    md_lines.append(f"- Overall Pass Rate: **{overall_pass_rate:.2f}%** ({problem_stats['passed_all']}/{total})")
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
        f.write('\n')
    
    print(f"\nReports generated:")
    print(f"   - Text report: {output_file}")
    print(f"   - Markdown report: {md_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <result_file.jsonl>")
        sys.exit(1)
    
    result_file = sys.argv[1]
    analyze_results(result_file)
