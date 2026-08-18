#!/usr/bin/env python3
"""
SciIF: Scientific Instruction Following Evaluation Framework

A comprehensive evaluation framework for assessing LLM performance on
scientific instruction following tasks with constraint validation.
"""

import os
import json
import argparse
# MASK4SIG BUILD: exactly-4-significant-figures judge requirement is disabled in validator prompts.
import logging
import sys
import hashlib
import fcntl
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
import re
from tqdm import tqdm
import asyncio

from .api_client import run_responses_api

# Import parse_text_field from local_model if available, otherwise define it here
def parse_text_field(text: str) -> Dict[str, str]:
    """Parse question, constraints, answer from text field"""
    result = {
        "question": "",
        "constraints": "",
        "answer": ""
    }
    
    if not text:
        return result
    
    lines = text.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        question_match = re.match(r'(?i)^question\s*[:：]\s*(.+)', line)
        constraints_match = re.match(r'(?i)^constraints\s*[:：]\s*(.+)', line)
        answer_match = re.match(r'(?i)^answer\s*[:：]\s*(.+)', line)
        
        if question_match:
            if current_section and current_content:
                result[current_section] = '\n'.join(current_content).strip()
            current_section = "question"
            current_content = [question_match.group(1)] if question_match.group(1).strip() else []
        elif constraints_match:
            if current_section and current_content:
                result[current_section] = '\n'.join(current_content).strip()
            current_section = "constraints"
            current_content = [constraints_match.group(1)] if constraints_match.group(1).strip() else []
        elif answer_match:
            if current_section and current_content:
                result[current_section] = '\n'.join(current_content).strip()
            current_section = "answer"
            current_content = [answer_match.group(1)] if answer_match.group(1).strip() else []
        elif current_section:
            current_content.append(line)
    
    if current_section and current_content:
        result[current_section] = '\n'.join(current_content).strip()
    
    return result

logger = logging.getLogger("sciif.evaluator")

# Control the strictness of validator prompt: strict / loose, can be set via environment variable
# strict: all points must pass
# loose: only main points must pass, secondary points can fail
VALIDATOR_PROMPT_MODE = os.environ.get("VALIDATOR_PROMPT_MODE", "strict").lower()
# Control the strictness of the answer prompt for the model under test. Default follows validator,
# to align output format and evidence requirements under strict judge.
ANSWER_PROMPT_MODE = os.environ.get("ANSWER_PROMPT_MODE", VALIDATOR_PROMPT_MODE).lower()

# Key point definitions for loose mode: main points must pass, secondary points can fail
# Note: "Applicability Range" in loose mode also requires all points to pass (because point_1 and point_2 are both important)
LOOSE_KEY_POINTS = {
    "Assumptions": {
        "main": ["point_1", "point_2", "point_3"],
        "secondary": ["point_4", "point_5", "point_6"],
    },
    "Boundary Conditions": {
        "main": ["point_1", "point_2"],
        "secondary": ["point_3"],
    },
    "Applicability Range": {
        # In loose mode, all points must also pass (because point_1 and point_2 are both important)
        "main": ["point_1", "point_2"],
        "secondary": [],
    },
    "Units Standard": {
        "main": ["point_1"],
        "secondary": ["point_2", "point_3", "point_4", "point_5"],
    },
    "Cross-disciplinary Term Disambiguation": {
        "main": ["point_1", "point_2"],
        "secondary": ["point_3", "point_4", "point_5", "point_6", "point_7"],
    },
    "Intra-discipline Term Definitions": {
        "main": ["point_1", "point_2", "point_5", "point_8"],
        "secondary": ["point_3", "point_4", "point_6", "point_7", "point_9"],
    },
    "Symbols & Constants Standardization": {
        "main": ["point_1", "point_4"],
        "secondary": ["point_2", "point_3"],
    },
    "Variable Naming Consistency": {
        "main": ["point_1", "point_2", "point_3"],
        "secondary": [],
    },
    "Numerical Methods": {
        "main": ["point_1", "point_2", "point_4"],
        "secondary": ["point_3", "point_5", "point_6"],
    },
    "Experimental Methods": {
        "main": ["point_1", "point_3", "point_5"],
        "secondary": ["point_2", "point_4", "point_6", "point_7"],
    },
}


def normalize_constraint_name(constraint_name: str) -> str:
    """Normalize constraint name, extract core constraint name and return English name
    
    Examples:
    - "Variable naming consistency, e.g., v[m/s], I[A]" -> "Variable Naming Consistency"
    - "Boundary Conditions, Units standard + scientific notation" -> "Boundary Conditions, Units Standard" (combined constraints)
    - "5. Terminology Constraints/Cross-disciplinary Term Disambiguation (Cross-disciplinary term disambiguation): ..." -> "Cross-disciplinary Term Disambiguation"
    """
    constraint_name = constraint_name.strip()
    
    # Legacy Chinese constraint name to English mapping (for backward compatibility only)
    # Note: All inputs should now be in English. This mapping is kept for edge cases.
    chinese_to_english = {
        # Chinese keys removed - system now uses English constraint names only
    }
    
    # English constraint name normalization (various forms to standard English)
    english_variants = {
        "assumptions": "Assumptions",
        "boundary conditions": "Boundary Conditions",
        "applicability range": "Applicability Range",
        "units and notation": "Units Standard",
        "units standard": "Units Standard",
        "units standard + scientific notation": "Units Standard",
        "cross-disciplinary term disambiguation": "Cross-disciplinary Term Disambiguation",
        "intra-discipline term definitions": "Intra-discipline Term Definitions",
        "symbols and constants standardization": "Symbols & Constants Standardization",
        "variable naming consistency": "Variable Naming Consistency",
        "numerical methods": "Numerical Methods",
        "experimental methods": "Experimental Methods"
    }
    
    # First check if it's a Chinese constraint name
    for chinese_name, english_name in chinese_to_english.items():
        if chinese_name in constraint_name:
            # Check if there are multiple constraints (combined constraints)
            found_count = sum(1 for name in chinese_to_english.keys() if name in constraint_name)
            if found_count > 1:
                # Combined constraints, extract all constraint names
                found_constraints = [chinese_to_english[name] for name in chinese_to_english.keys() if name in constraint_name]
                return ", ".join(sorted(set(found_constraints)))
            return english_name
    
    # Try to extract constraint name from detailed description (e.g., "5. Terminology Constraints/Cross-disciplinary Term Disambiguation")
    # Pattern: number. English constraint name/Chinese constraint name
    pattern = r'\d+\.\s*[^/]+/([^(:]+)'
    match = re.search(pattern, constraint_name)
    if match:
        chinese_part = match.group(1).strip()
        # Check if it's a known Chinese constraint name
        for chinese_name, english_name in chinese_to_english.items():
            if chinese_name in chinese_part:
                return english_name
        # If Chinese part is matched but not found in mapping, try to extract English part
        # Pattern: number. English/Chinese
        pattern2 = r'\d+\.\s*([^/]+)/'
        match2 = re.search(pattern2, constraint_name)
        if match2:
            english_part = match2.group(1).strip()
            constraint_lower = english_part.lower()
            for variant, standard in english_variants.items():
                if variant in constraint_lower:
                    return standard
    
    # Try to match English constraint names
    constraint_lower = constraint_name.lower()
    for variant, standard in english_variants.items():
        if variant in constraint_lower:
            return standard
    
    # Known constraint core names (English, sorted by length from long to short)
    core_constraints = [
        ("Units standard + scientific notation", "Units Standard"),
        ("Cross-disciplinary term disambiguation", "Cross-disciplinary Term Disambiguation"),
        ("Cross-disciplinary Term Disambiguation", "Cross-disciplinary Term Disambiguation"),
        ("Intra-discipline term definitions", "Intra-discipline Term Definitions"),
        ("Intra-discipline Term Definitions", "Intra-discipline Term Definitions"),
        ("Symbols and constants standardization", "Symbols & Constants Standardization"),
        ("Symbols and Constants Standardization", "Symbols & Constants Standardization"),
        ("Symbols & Constants Standardization", "Symbols & Constants Standardization"),
        ("Variable naming consistency", "Variable Naming Consistency"),
        ("Variable Naming Consistency", "Variable Naming Consistency"),
        ("Numerical Methods", "Numerical Methods"),
        ("Experimental Methods", "Experimental Methods"),
        ("Applicability Range", "Applicability Range"),
        ("Boundary Conditions", "Boundary Conditions"),
        ("Units and Notation", "Units Standard"),
        ("Units Standard", "Units Standard"),
        ("Assumptions", "Assumptions")
    ]
    
    # Check if it contains multiple independent constraints (combined constraints)
    found_cores = []
    for english_core, standard_name in core_constraints:
        pattern = r'(?:^|,\s*|\s+)' + re.escape(english_core.lower()) + r'(?:\s*,|\s*$|\s*\+|\s*\(|\s*:)'
        if re.search(pattern, constraint_lower):
            found_cores.append(standard_name)
    
    # If multiple core constraints are found, this is a combined constraint
    if len(found_cores) > 1:
        return ", ".join(sorted(set(found_cores)))
    
    # If only one core constraint is found, return the corresponding English name
    if len(found_cores) == 1:
        return found_cores[0]
    
    # If no core constraint is found, try to match prefix
    for english_core, standard_name in core_constraints:
        if constraint_lower.startswith(english_core.lower()):
            return standard_name
    
    # If no match is found, return the original name (may be combined constraint or other format)
    return constraint_name


def generate_statistics_report(result_file: str, logger: logging.Logger, save_to_file: bool = True):
    """Generate detailed statistics report, including single constraint pass rate, multi-constraint pass rate, and answer correctness rate
    
    This function reads the entire result file (including historical data) and only counts the latest record for each problem_id
    (supports resume functionality to avoid duplicate counting)
    
    Args:
        result_file: Path to the result file
        logger: Logger instance
        save_to_file: Whether to save report to file (default: True)
    """
    if not os.path.exists(result_file):
        logger.warning(f"Result file does not exist: {result_file}")
        return
    
    # First read all records, deduplicate by (problem_id, model_name), keep the latest record
    records_by_key = {}  # key: (problem_id, model_name), value: record
    total_lines = 0
    duplicate_count = 0
    
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_lines += 1
                try:
                    record = json.loads(line)
                    problem_id = record.get("problem_id", "")
                    evaluation = record.get("evaluation", {})
                    model_name = evaluation.get("model", "unknown")
                    
                    key = (problem_id, model_name)
                    # If already exists, it is a duplicate (caused by resume), keep the latest (later ones overwrite earlier ones)
                    if key in records_by_key:
                        duplicate_count += 1
                    records_by_key[key] = record
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON line (line {line_num}): {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Failed to read result file: {e}")
        return
    
    if duplicate_count > 0:
        logger.info(f"📊 Detected duplicate records (deduplicated, using latest results)")
    
    # Statistics grouped by model
    model_stats = defaultdict(lambda: {
        "total": 0,
        "single_constraint_count": 0,
        "single_constraint_pass": 0,
        "single_constraint_fail": 0,
        "multi_constraint_count": 0,
        "multi_constraint_overall_pass": 0,
        "multi_constraint_overall_fail": 0,
        "multi_constraint_individual_stats": {},
        "answer_correctness_pass": 0,
        "answer_correctness_fail": 0,
        "error_count": 0,
        "cache_hits": 0
    })
    
    # Statistics on deduplicated records
    for record in records_by_key.values():
        evaluation = record.get("evaluation", {})
        model_name = evaluation.get("model", "unknown")
        
        # Check if there are errors
        if evaluation.get("status") == "ERROR":
            model_stats[model_name]["error_count"] += 1
            continue
        
        model_stats[model_name]["total"] += 1
        
        # Constraint validation statistics
        constraint_validation = evaluation.get("constraint_validation", {})
        is_single_constraint = evaluation.get("is_single_constraint", False)
        
        if is_single_constraint:
            model_stats[model_name]["single_constraint_count"] += 1
            overall_status = constraint_validation.get("overall", {}).get("status", "UNKNOWN")
            if overall_status == "PASS":
                model_stats[model_name]["single_constraint_pass"] += 1
            elif overall_status == "FAIL":
                model_stats[model_name]["single_constraint_fail"] += 1
            
            # Statistics on constraints of single-constraint problems to individual constraint statistics
            problem = record.get("problem", {})
            constraint_name_str = problem.get("constraints", "")
            if constraint_name_str:
                # Normalize constraint name
                normalized_name = normalize_constraint_name(constraint_name_str)
                
                if normalized_name not in model_stats[model_name]["multi_constraint_individual_stats"]:
                    model_stats[model_name]["multi_constraint_individual_stats"][normalized_name] = {
                        "pass": 0,
                        "fail": 0,
                        "total": 0
                    }
                
                if overall_status == "PASS":
                    model_stats[model_name]["multi_constraint_individual_stats"][normalized_name]["pass"] += 1
                elif overall_status == "FAIL":
                    model_stats[model_name]["multi_constraint_individual_stats"][normalized_name]["fail"] += 1
                model_stats[model_name]["multi_constraint_individual_stats"][normalized_name]["total"] += 1
        else:
            model_stats[model_name]["multi_constraint_count"] += 1
            # Combined constraint statistics
            overall_status = constraint_validation.get("overall", {}).get("status", "UNKNOWN")
            if overall_status == "PASS":
                model_stats[model_name]["multi_constraint_overall_pass"] += 1
            elif overall_status == "FAIL":
                model_stats[model_name]["multi_constraint_overall_fail"] += 1
            
            # Individual constraint statistics
            individual = constraint_validation.get("individual", {})
            for constraint_name, result in individual.items():
                # Normalize constraint name (extract core constraint name, merge identical constraints)
                normalized_name = normalize_constraint_name(constraint_name)
                
                if normalized_name not in model_stats[model_name]["multi_constraint_individual_stats"]:
                    model_stats[model_name]["multi_constraint_individual_stats"][normalized_name] = {
                        "pass": 0,
                        "fail": 0,
                        "total": 0
                    }
                status = result.get("status", "UNKNOWN")
                if status == "PASS":
                    model_stats[model_name]["multi_constraint_individual_stats"][normalized_name]["pass"] += 1
                elif status == "FAIL":
                    model_stats[model_name]["multi_constraint_individual_stats"][normalized_name]["fail"] += 1
                model_stats[model_name]["multi_constraint_individual_stats"][normalized_name]["total"] += 1
        
        # Answer correctness statistics
        answer_correctness = evaluation.get("answer_correctness", {})
        correctness_status = answer_correctness.get("status", "UNKNOWN")
        if correctness_status == "PASS":
            model_stats[model_name]["answer_correctness_pass"] += 1
        elif correctness_status == "FAIL":
            model_stats[model_name]["answer_correctness_fail"] += 1
    
    # Prepare report content
    report_lines = []
    
    report_lines.append("="*100)
    report_lines.append("📊 Detailed Evaluation Statistics Report")
    report_lines.append("="*100)
    report_lines.append("")
    
    # Generate statistics by model
    for model_name in sorted(model_stats.keys()):
        stats = model_stats[model_name]
        
        if stats["total"] == 0:
            continue
        
        report_lines.append(f"Model: {model_name}")
        report_lines.append("-"*100)
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("📈 Overall Statistics:")
        report_lines.append(f"   Total problems: {stats['total']}")
        if stats["error_count"] > 0:
            report_lines.append(f"   Error count: {stats['error_count']}")
        report_lines.append("")
        
        # Single constraint statistics
        if stats["single_constraint_count"] > 0:
            single_pass_rate = (stats["single_constraint_pass"] / stats["single_constraint_count"] * 100) if stats["single_constraint_count"] > 0 else 0
            report_lines.append(f"🔹 Single Constraint Problem Statistics ({stats['single_constraint_count']} problems):")
            report_lines.append(f"   Pass: {stats['single_constraint_pass']} ({single_pass_rate:.1f}%)")
            report_lines.append(f"   Fail: {stats['single_constraint_fail']} ({100 - single_pass_rate:.1f}%)")
            report_lines.append("")
        
        # Multi-constraint statistics
        if stats["multi_constraint_count"] > 0:
            multi_overall_pass_rate = (stats["multi_constraint_overall_pass"] / stats["multi_constraint_count"] * 100) if stats["multi_constraint_count"] > 0 else 0
            report_lines.append(f"🔹 Multi-Constraint Problem Statistics ({stats['multi_constraint_count']} problems):")
            report_lines.append(f"   Overall Constraint Pass: {stats['multi_constraint_overall_pass']} ({multi_overall_pass_rate:.1f}%)")
            report_lines.append(f"   Overall Constraint Fail: {stats['multi_constraint_overall_fail']} ({100 - multi_overall_pass_rate:.1f}%)")
            report_lines.append("")
            
            if stats["multi_constraint_individual_stats"]:
                report_lines.append("   Individual constraint pass rate:")
                # Sort by pass rate
                sorted_constraints = sorted(
                    stats["multi_constraint_individual_stats"].items(),
                    key=lambda x: (x[1]["pass"] / x[1]["total"] if x[1]["total"] > 0 else 0),
                    reverse=True
                )
                for constraint_name, constraint_stats in sorted_constraints:
                    pass_rate = (constraint_stats["pass"] / constraint_stats["total"] * 100) if constraint_stats["total"] > 0 else 0
                    report_lines.append(f"      • {constraint_name[:70]:<70} {constraint_stats['pass']:>3}/{constraint_stats['total']:<3} ({pass_rate:>5.1f}%)")
                report_lines.append("")
        
        # Answer correctness statistics
        correctness_pass_rate = (stats["answer_correctness_pass"] / stats["total"] * 100) if stats["total"] > 0 else 0
        report_lines.append(f"🔹 Answer Correctness Statistics ({stats['total']} problems):")
        report_lines.append(f"   Correct: {stats['answer_correctness_pass']} ({correctness_pass_rate:.1f}%)")
        report_lines.append(f"   Incorrect: {stats['answer_correctness_fail']} ({100 - correctness_pass_rate:.1f}%)")
        report_lines.append("")
        
        # Generate summary table
        report_lines.append("Summary Table")
        report_lines.append("-"*100)
        report_lines.append(f"{'Metric':<40} {'Pass Count':<12} {'Fail Count':<12} {'Total':<12} {'Pass Rate':<12}")
        report_lines.append("-"*100)
        
        # Single constraint row
        if stats["single_constraint_count"] > 0:
            single_pass_rate = (stats["single_constraint_pass"] / stats["single_constraint_count"] * 100) if stats["single_constraint_count"] > 0 else 0
            report_lines.append(f"{'Single Constraint Pass Rate':<40} {stats['single_constraint_pass']:<12} {stats['single_constraint_fail']:<12} {stats['single_constraint_count']:<12} {single_pass_rate:>10.1f}%")
        
        # Multi-constraint combined row
        if stats["multi_constraint_count"] > 0:
            multi_overall_pass_rate = (stats["multi_constraint_overall_pass"] / stats["multi_constraint_count"] * 100) if stats["multi_constraint_count"] > 0 else 0
            report_lines.append(f"{'Multi-Constraint Overall Pass Rate':<40} {stats['multi_constraint_overall_pass']:<12} {stats['multi_constraint_overall_fail']:<12} {stats['multi_constraint_count']:<12} {multi_overall_pass_rate:>10.1f}%")
        
        # Answer correctness row
        correctness_pass_rate = (stats["answer_correctness_pass"] / stats["total"] * 100) if stats["total"] > 0 else 0
        report_lines.append(f"{'Answer Correctness Rate':<40} {stats['answer_correctness_pass']:<12} {stats['answer_correctness_fail']:<12} {stats['total']:<12} {correctness_pass_rate:>10.1f}%")
        
        report_lines.append("")
        report_lines.append("="*100)
        report_lines.append("")
    
    # If there are multiple models, generate comparison table
    if len(model_stats) > 1:
        report_lines.append("📊 Model Comparison Table")
        report_lines.append("="*100)
        report_lines.append("")
        
        # Table header
        header = f"{'Metric':<30}"
        for model_name in sorted(model_stats.keys()):
            header += f" {model_name:<25}"
        report_lines.append(header)
        report_lines.append("-"*len(header))
        
        # Total number of problems
        row = f"{'Total Problems':<30}"
        for model_name in sorted(model_stats.keys()):
            row += f" {model_stats[model_name]['total']:<25}"
        report_lines.append(row)
        
        # Single constraint pass rate
        row = f"{'Single Constraint Pass Rate':<30}"
        for model_name in sorted(model_stats.keys()):
            stats = model_stats[model_name]
            if stats["single_constraint_count"] > 0:
                rate = (stats["single_constraint_pass"] / stats["single_constraint_count"] * 100)
                row += f" {stats['single_constraint_pass']}/{stats['single_constraint_count']} ({rate:.1f}%){'':<10}"
            else:
                row += f" {'N/A':<25}"
        report_lines.append(row)
        
        # Multi-constraint combined pass rate
        row = f"{'Multi-Constraint Overall Pass Rate':<30}"
        for model_name in sorted(model_stats.keys()):
            stats = model_stats[model_name]
            if stats["multi_constraint_count"] > 0:
                rate = (stats["multi_constraint_overall_pass"] / stats["multi_constraint_count"] * 100)
                row += f" {stats['multi_constraint_overall_pass']}/{stats['multi_constraint_count']} ({rate:.1f}%){'':<10}"
            else:
                row += f" {'N/A':<25}"
        report_lines.append(row)
        
        # Answer correctness rate
        row = f"{'Answer Correctness Rate':<30}"
        for model_name in sorted(model_stats.keys()):
            stats = model_stats[model_name]
            rate = (stats["answer_correctness_pass"] / stats["total"] * 100) if stats["total"] > 0 else 0
            row += f" {stats['answer_correctness_pass']}/{stats['total']} ({rate:.1f}%){'':<10}"
        report_lines.append(row)
        
        report_lines.append("")
        report_lines.append("="*100)
        report_lines.append("")
    
    # Output to log
    for line in report_lines:
        logger.info(line)
    
    # Save to file
    if save_to_file:
        # Determine report file path (in the same directory as result file, filename with _statistics_report.txt)
        result_path = Path(result_file)
        report_file = result_path.parent / f"{result_path.stem}_statistics_report.txt"
        
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
                f.write("\n")
            logger.info(f"\nDetailed statistics report saved to: {report_file}")
        except Exception as e:
            logger.warning(f"Failed to save statistics report: {e}")


def setup_logging(log_level: str = "info", log_file: Optional[str] = None):
    """Initialize logging output"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True
    )
    logging.getLogger("sciif.evaluator").setLevel(level)
    
    # Disable logging output from third-party libraries to avoid interfering with tqdm progress bar
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_problems(file_path: str) -> List[Dict[str, Any]]:
    """Load problem file"""
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                logger.info(f"Successfully loaded problems")
                return data
            else:
                # Merge all available problem lists
                all_items = []
                found_keys = []
                
                # Check and merge in priority order
                for key in ['corpus', 'selected_300_no_numexp', 'extra_numexp_questions']:
                    if key in data and isinstance(data[key], list):
                        all_items.extend(data[key])
                        found_keys.append(key)
                
                if all_items:
                    logger.info(f"Successfully loaded problems (from keys)")
                    return all_items
                else:
                    logger.error(f"JSON file format error: should be a list format or a dictionary format containing corpus/selected_300_no_numexp/extra_numexp_questions keys")
                    return []
    except Exception as e:
        logger.error(f"Failed to read file ({file_path}): {e}")
        return []


def parse_constraints(constraints_str: Any) -> List[str]:
    """Parse constraints (compatible with string/list/dict), output standardized constraint list"""
    if not constraints_str:
        return []

    # New: prioritize list/dict input
    if isinstance(constraints_str, list):
        out: List[str] = []
        for c in constraints_str:
            if isinstance(c, dict):
                cand = c.get("title") or c.get("name") or c.get("content") or c.get("id") or c.get("no")
                if cand:
                    out.append(normalize_constraint_name(str(cand)))
            elif isinstance(c, str):
                out.append(normalize_constraint_name(c))
        return out

    if isinstance(constraints_str, dict):
        return [normalize_constraint_name(str(v)) for v in constraints_str.values() if v]

    # New: more robust string splitting (comma/dunhao/semicolon), try first
    if isinstance(constraints_str, str):
        rough_parts = [p.strip() for p in re.split(r"[、;,，；]+", constraints_str) if p.strip()]
        if len(rough_parts) > 1:
            return [normalize_constraint_name(p) for p in rough_parts]
    
    # List of known constraint names (used to identify real constraints)
    # Note: these are core names of constraints, may have descriptions after (e.g., "e.g., v[m/s], I[A]")
    # Full constraint names (must match exactly) should be placed first
    known_constraints_full = [
        "Units standard + scientific notation",  # Full constraint name, match first
    ]
    
    known_constraints = [
        "Assumptions",
        "Boundary Conditions",
        "Applicability Range",
        "Units Standard",
        "Units and Notation",  # Alternative name
        "Units standard",  # Prefix of "Units standard + scientific notation" (but full name has priority)
        "Cross-disciplinary Term Disambiguation",
        "Cross-disciplinary term disambiguation",  # Lowercase variant
        "Intra-discipline Term Definitions",
        "Intra-discipline term definitions",  # Lowercase variant
        "Symbols & Constants Standardization",
        "Symbols and Constants Standardization",  # Alternative name
        "Symbols and constants standardization",  # Lowercase variant
        "Variable Naming Consistency",
        "Variable naming consistency"  # Lowercase variant
    ]
    
    # Smart parsing: identify real constraint names
    # Strategy: first split by comma, then identify which are constraint names and which are descriptions
    constraints = []
    parts = [p.strip() for p in constraints_str.split(',') if p.strip()]
    
    i = 0
    while i < len(parts):
        part = parts[i]
        
        # First check if it is a full constraint name (e.g., "Units standard + scientific notation")
        # Need to check if current part and subsequent parts can form a complete constraint
        matched_constraint = None
        matched_full = False
        
        # Check full constraint name (may need to merge multiple parts)
        for full_constraint in known_constraints_full:
            full_lower = full_constraint.lower()
            # Check if current part matches the beginning of full constraint
            if part.lower().startswith(full_lower.split()[0].lower()):
                # Try to merge subsequent parts to match full constraint
                test_parts = [part]
                j = i + 1
                while j < len(parts) and len(', '.join(test_parts).lower()) < len(full_lower):
                    test_parts.append(parts[j])
                    test_str = ', '.join(test_parts).lower()
                    if test_str == full_lower or test_str.startswith(full_lower):
                        # Found full match
                        matched_constraint = full_constraint
                        matched_full = True
                        # Update i to skip matched parts
                        i = j
                        break
                    j += 1
                if matched_full:
                    break
        
        # If no full constraint is matched, check common constraint names
        if not matched_constraint:
            for known in known_constraints:
                part_lower = part.lower()
                known_lower = known.lower()
                # Exact match or prefix match (but require space, comma or end of string after matched part)
                if (part_lower == known_lower or 
                    (part_lower.startswith(known_lower) and 
                     (len(part) == len(known) or part[len(known):len(known)+1] in [' ', '+', ',']))):
                    matched_constraint = known
                    break
        
        if matched_constraint:
            if matched_full:
                # If it is a full constraint (e.g., "Units standard + scientific notation"), add directly
                constraints.append(matched_constraint)
                i += 1
            else:
                # Found matching constraint, collect subsequent description parts (until next known constraint)
                constraint_parts = [part]
                i += 1
                
                # Collect subsequent description parts (e.g., "e.g., v[m/s], I[A]")
                # But note: if another constraint name is encountered, should stop
                while i < len(parts):
                    next_part = parts[i]
                    
                    # First check if current part and subsequent parts can form a complete constraint (e.g., "Units standard + scientific notation")
                    is_full_constraint = False
                    full_constraint_match = None
                    for full_constraint in known_constraints_full:
                        # Check from current part, if it can form a complete constraint
                        test_parts = [next_part]
                        j = i + 1
                        # Try to merge subsequent parts
                        while j < len(parts):
                            test_parts.append(parts[j])
                            combined = ', '.join(test_parts).lower().strip()
                            full_lower = full_constraint.lower().strip()
                            if combined == full_lower:
                                # Found full match
                                is_full_constraint = True
                                full_constraint_match = full_constraint
                                break
                            # If combined string already exceeds full constraint length, stop trying
                            if len(combined) > len(full_lower):
                                break
                            j += 1
                        if is_full_constraint:
                            break
                    
                    if is_full_constraint:
                        # Encountered full constraint, stop collecting description parts of current constraint
                        break
                    
                    # Check if it is the next known constraint
                    is_next_constraint = False
                    for known2 in known_constraints:
                        next_part_lower = next_part.lower()
                        known2_lower = known2.lower()
                        if (next_part_lower == known2_lower or 
                            (next_part_lower.startswith(known2_lower) and 
                             (len(next_part) == len(known2) or next_part[len(known2):len(known2)+1] in [' ', '+', ',']))):
                            # But note: if matched is "Units standard", need to check if there is "+ scientific notation" after
                            if known2_lower == "units standard":
                                # Check if subsequent part contains "+ scientific notation"
                                if i + 1 < len(parts):
                                    next_next = parts[i + 1].lower()
                                    if '+' in next_next or 'scientific notation' in next_next:
                                        # This may be part of full constraint "Units standard + scientific notation"
                                        # Stop collecting description parts of current constraint, let next loop handle full constraint
                                        is_next_constraint = True
                                        break
                                # If no subsequent part or subsequent part does not contain "+ scientific notation", treat as next constraint
                                is_next_constraint = True
                                break
                            else:
                                is_next_constraint = True
                                break
                    
                    if is_next_constraint:
                        # Encountered next constraint, stop collecting
                        break
                    
                    # Check if it is an obvious descriptive marker (e.g., "e.g.")
                    if next_part.lower() in ['e.g.', 'eg.', 'example', 'e.g.']:
                        # This is the start of description, continue collecting
                        constraint_parts.append(next_part)
                        i += 1
                        continue
                    
                    # Check if it contains unit markers (e.g., "[m/s]", "[A]"), this is usually part of description
                    if '[' in next_part and ']' in next_part:
                        # This is likely a description part
                        constraint_parts.append(next_part)
                        i += 1
                        continue
                    
                    # Check if it contains "+", may be extension of constraint (e.g., "Units standard + scientific notation")
                    # But if "Units standard" is followed by "+ scientific notation", should identify as full constraint
                    if '+' in next_part:
                        # Check if current constraint is "Units standard"
                        current_constraint_lower = ', '.join(constraint_parts).lower()
                        if 'units standard' in current_constraint_lower:
                            # This may be full constraint "Units standard + scientific notation"
                            # Check if it matches full constraint
                            test_combined = (current_constraint_lower + ', ' + next_part.lower()).strip()
                            if test_combined == "units standard + scientific notation":
                                # This is full constraint, stop collecting current constraint, let next loop handle
                                break
                        # Otherwise, this may be extension part of current constraint
                        constraint_parts.append(next_part)
                        i += 1
                        continue
                    
                    # If next part does not look like constraint name, may be description
                    # But for safety, if it is short and does not contain special characters, may be new constraint
                    # Here adopt conservative strategy: if next part looks like constraint name (first letter capitalized, no special symbols), stop collecting
                    if (len(next_part) > 2 and 
                        next_part[0].isupper() and 
                        not any(c in next_part for c in ['[', ']', '(', ')', 'e.g.', 'eg.'])):
                        # May be new constraint, stop collecting
                        break
                    
                    # Otherwise, continue collecting as description
                    constraint_parts.append(next_part)
                    i += 1
                
                # Merge constraint name and description
                constraint = ', '.join(constraint_parts)
                constraints.append(constraint)
        else:
            # If not a known constraint, may be new constraint type, add separately
            constraints.append(part)
            i += 1
    
    # If no known constraint found, fall back to simple comma splitting (but filter out obvious descriptive content)
    if not constraints:
        all_parts = [c.strip() for c in constraints_str.split(',') if c.strip()]
        # Filter out obvious descriptive content (e.g., "e.g.")
        filtered = []
        for part in all_parts:
            # Skip obvious descriptive markers
            if part.lower() in ['e.g.', 'eg.', 'example', 'e.g.']:
                continue
            # If looks like example (contains brackets and units), may be description part of constraint
            if '[' in part and ']' in part and any(unit in part.lower() for unit in ['m/s', 'a]', 'v[', 'i[']):
                # This may be description of constraint, try to merge to previous constraint
                if filtered:
                    filtered[-1] = filtered[-1] + ', ' + part
                else:
                    filtered.append(part)
            else:
                filtered.append(part)
        constraints = filtered
    
    return constraints


# --- Override with stricter, noise-tolerant parser ---
def parse_constraints(constraints_str: Any) -> List[str]:
    """
    Parse constraints (compatible with string/list/dict), output normalized constraint list.
    Only keep standard constraint set, remove numeric labels, descriptive text and abnormal fragments.
    """
    allowed = {
        "Assumptions",
        "Boundary Conditions",
        "Applicability Range",
        "Units Standard",
        "Cross-disciplinary Term Disambiguation",
        "Intra-discipline Term Definitions",
        "Symbols & Constants Standardization",
        "Variable Naming Consistency",
        "Numerical Methods",
        "Experimental Methods",
    }

    def clean_token(tok: str) -> str:
        # Remove quotes, brackets, prefix numbers
        tok = tok.strip().strip('\"\'[](){}')
        tok = re.sub(r"^[0-9]+[.．、)]\\s*", "", tok)
        tok = tok.strip()
        return tok

    def add_tokens_from_obj(obj, acc: List[str]):
        if obj is None:
            return
        if isinstance(obj, str):
            tokens = re.split(r"[、;,，；]+", obj)
            for t in tokens:
                t = clean_token(t)
                if t:
                    acc.append(t)
        elif isinstance(obj, dict):
            # Check both key and value
            for k, v in obj.items():
                add_tokens_from_obj(k, acc)
                add_tokens_from_obj(v, acc)
        elif isinstance(obj, list):
            for it in obj:
                add_tokens_from_obj(it, acc)

    raw_tokens: List[str] = []
    add_tokens_from_obj(constraints_str, raw_tokens)

    normalized: List[str] = []
    for t in raw_tokens:
        norm = normalize_constraint_name(t)
        if norm in allowed:
            normalized.append(norm)

    # Deduplicate but preserve order
    seen = set()
    result = []
    for n in normalized:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result


# This function needs to be defined after all prompt function definitions
# Temporarily use a simple version identifier
_PROMPT_VERSION = "v1.0"  # When modifying prompt, manually update this version number

def get_prompt_version_hash() -> str:
    """Get hash value of prompt version (used to distinguish different prompt versions)"""
    import inspect
    import sys
    
    # Get current module
    current_module = sys.modules[__name__]
    
    # Get source code of all prompt building functions
    prompt_function_names = [
        "build_answer_prompt",
        "build_single_constraint_validation_prompt",
        "build_multi_constraint_validation_prompt",
        "build_answer_correctness_prompt"
    ]
    
    source_code = _PROMPT_VERSION  # Use version number as base
    for func_name in prompt_function_names:
        func = getattr(current_module, func_name, None)
        if func:
            try:
                source_code += inspect.getsource(func)
            except Exception:
                # If cannot get source code, use function name
                source_code += func.__name__
    
    # Calculate hash value (take first 8 digits)
    return hashlib.md5(source_code.encode()).hexdigest()[:8]


def get_answer_cache_key(question: str, constraints_str: str, model_name: str, problem_id: Optional[str] = None) -> str:
    """Generate answer cache key
    
    Prefer using problem_id (if provided) to ensure consistency with IDs in refined_corpus.json and evaluation_results.jsonl
    If no problem_id, use content hash as fallback
    """
    if problem_id:
        # Use problem_id and model_name as cache key, ensure strict correspondence with problem ID
        return f"{problem_id}|{model_name}"
    else:
        # Fallback: use content hash
        content = f"{question}|{constraints_str}|{model_name}"
        return hashlib.md5(content.encode()).hexdigest()


def load_answer_cache(cache_file: str) -> Dict[str, str]:
    """Load answer cache
    
    Return format: {cache_key: answer}
    Supports two formats:
    1. New format: cache containing problem_id field
    2. Old format: cache with only cache_key and answer (backward compatible)
    """
    if not os.path.exists(cache_file):
        return {}
    
    cache = {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    cache_key = item.get("cache_key")
                    answer = item.get("answer")
                    problem_id = item.get("problem_id")
                    
                    # Prefer using cache_key, if not exists then build from problem_id and model_name
                    if not cache_key and problem_id:
                        model_name = item.get("model_name")
                        if model_name:
                            cache_key = f"{problem_id}|{model_name}"
                    
                    if cache_key and answer:
                        cache[cache_key] = answer
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Failed to load answer cache: {e}")
    
    return cache


def save_answer_to_cache(cache_file: str, cache_key: str, answer: str, problem_id: Optional[str] = None, model_name: Optional[str] = None):
    """Save answer to cache (append if not exists)
    
    Use file lock to ensure thread safety in concurrent environments
    
    Args:
        cache_file: Cache file path
        cache_key: Cache key (format: problem_id|model_name or MD5 hash)
        answer: Answer content
        problem_id: Problem ID (optional, for ensuring ID consistency)
        model_name: Model name (optional, for ensuring ID consistency)
    """
    try:
        # Ensure cache directory exists
        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use file lock to ensure concurrency safety (double-check locking pattern)
        lock_file = cache_file + ".lock"
        # Ensure lock file directory exists
        Path(lock_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(lock_file, "w") as lock:
            try:
                # Get exclusive lock (blocking mode, wait for other processes to release)
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

                # Double check: reload cache within lock (avoid race condition)
                existing_cache = load_answer_cache(cache_file)
                if cache_key in existing_cache:
                    # If cache exists and answer is same, skip saving
                    if existing_cache[cache_key] == answer:
                        logger.debug(f"  Cache already exists, skipping save: {cache_key[:50]}...")
                        return
                    # If answer differs, log debug info (may be caused by concurrency), keep existing
                    logger.debug(f"  Cache key exists but answer differs (possible concurrent write): {cache_key[:50]}..., skipping save, keeping existing answer")
                    return

                # Write in append mode (only when cache does not exist)
                with open(cache_file, "a", encoding="utf-8") as f:
                    cache_item = {
                        "cache_key": cache_key,
                        "answer": answer
                    }
                    # If problem_id and model_name are provided, also save them to ensure consistency
                    if problem_id:
                        cache_item["problem_id"] = problem_id
                    if model_name:
                        cache_item["model_name"] = model_name
                    f.write(json.dumps(cache_item, ensure_ascii=False) + "\n")
                    f.flush()  # Ensure immediate write to disk
                    os.fsync(f.fileno())  # Force sync to disk
                logger.debug(f"  Saved to cache: {cache_key[:50]}...")
            finally:
                # Release lock (auto-release, but explicit release is clearer)
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                except:
                    pass
    except Exception as e:
        logger.warning(f"Failed to save answer cache: {e}")


def get_output_directory(
    base_dir: str,
    model_name: str,
    validator_model: str,
    prompt_version: str
) -> str:
    """Generate output directory based on model name, validator model and prompt version"""
    # Directory structure: base_dir/model_name/validator_model/prompt_version/
    output_dir = os.path.join(
        base_dir,
        model_name,
        validator_model,
        f"prompt_{prompt_version}"
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def build_answer_prompt(question: str, constraints: str) -> str:
    """Build prompt for model to answer question

    Design goals (Route A aligned with strict judge):
    - If the sample provides question_with_constraints (format: "question: ...\\nconstraints: ..."), present it "as-is in full" to the model under test;
    - In strict mode, additionally overlay required-evidence checklist and structured output requirements consistent with judge side;
    - Avoid duplicating constraints (do not add detailed constraints when original text already contains them).
    """
    mode = ANSWER_PROMPT_MODE
    constraints_text = (constraints or "").strip()
    # Detect if question field is already the whole text of "question_with_constraints"
    embedded_constraints = bool(re.search(r"(?im)^\s*constraints\s*[:：]", question or ""))

    # Some problems explicitly require "only use given symbols/species, no additional terms".
    # In such cases, strict output template must prioritize following problem boundary conditions: no additional symbols should appear in expression (e.g., P° / p°).
    q_norm = (question or "")
    forbid_extra_symbols_in_expression = bool(
        re.search(r"(?i)no additional terms|do not (?:add|introduce) any additional", q_norm)
        or re.search(r"(?i)each .* must appear exactly once", q_norm)
        or re.search(r"(?i)provide .* using\\s+P_", q_norm)
        or re.search(r"(?i)use\\s+P_[A-Za-z0-9]+.*P_[A-Za-z0-9]+", q_norm)
    )

    # Constraint name list (used to generate checklist and output structure in strict mode)
    # Prefer using passed constraints (usually simplified constraint name string); if empty and question contains embedded constraints, extract from question.
    if constraints_text:
        constraint_names = parse_constraints(constraints_text)
    elif embedded_constraints:
        constraint_names = parse_constraints(question)
    else:
        constraint_names = []

    if mode == "strict" and constraint_names:
        # Generate "required evidence points" for each constraint, align with strict judge side rubric
        # Note: some points are judge-only and should not appear in test model prompt
        judge_only_points = {
            "Units Standard": ["point_5"],  # Validator intent note - only for judge
        }
        
        evidence_blocks = []
        for c in constraint_names:
            req_data = _strict_required_evidence(c)
            req_points = req_data.get("points", []) if isinstance(req_data, dict) else []
            
            # Filter out judge-only points
            filtered_points = []
            for pid, text in req_points:
                if c in judge_only_points and pid in judge_only_points[c]:
                    continue  # Skip judge-only points
                filtered_points.append((pid, text))
            
            # Generate formatted text
            if filtered_points:
                formatted_lines = []
                for i, (pid, text) in enumerate(filtered_points, 1):
                    formatted_lines.append(f"{i}) {text}")
                req_text = "\n".join(formatted_lines)
                evidence_blocks.append(f"[{c}] Required evidence for compliance:\n{req_text}")
        
        evidence_text = "\n\n".join(evidence_blocks).strip()
        evidence_text = evidence_text if evidence_text else "(No strict evidence rules found.)"

        # Format: The main text should be "like normal problem solving", not forced to be divided by constraints, and should not output rubric template blocks (e.g., Evidence anchors / checklist / symbol tables).
        # The "auditable evidence" for constraints should be integrated into the narrative: use short sentences/formulas to directly state which conditions you satisfy, rather than wrapping them in template blocks.
        tag_map = {
            "Assumptions": "[Assumption]",
            "Boundary Conditions": "[Boundary]/[Boundary Check]",
            "Applicability Range": "[Applicability Range]",
            "Units Standard": "[Units]/[Unit Self-check]",
            "Cross-disciplinary Term Disambiguation": "[Disambiguation]/[Disambiguation Consequence]/[Consistency Evidence]",
            "Intra-discipline Term Definitions": "[Term Definition]/[Formal Criterion]/[Theory Note]/[Application to This Problem]",
            "Symbols & Constants Standardization": "[Symbol Definition]/[Constant Source]/[Symbol Consistency]",
            "Variable Naming Consistency": "[Naming Consistency]",
            "Numerical Methods": "[Numerical Methods]",
            "Experimental Methods": "[Experimental Design]",
        }
        tag_lines = []
        for c in constraint_names:
            if c in tag_map:
                tag_lines.append(f"- {c}: {tag_map[c]}")
        inline_tag_guide = "\n".join(tag_lines).strip()
        if not inline_tag_guide:
            inline_tag_guide = "- (No tag hints available for these constraints.)"

        # Provide problem text as-is: if question already contains constraints, do not repeat detailed constraints
        raw_problem_block = (
            f"RAW QUESTION_WITH_CONSTRAINTS (verbatim):\n{question}\n"
            if embedded_constraints
            else f"Problem:\n{question}\n"
        )
        extra_constraints_block = (
            "" if embedded_constraints or not constraints_text else f"\nConstraints (full text, follow them):\n{constraints_text}\n"
        )

        extra_symbol_guard = ""
        if forbid_extra_symbols_in_expression:
            extra_symbol_guard = """CRITICAL expression-format guard (highest priority):
- If the problem restricts the final expression to specific symbols/species and says “no additional terms”, you MUST NOT introduce any extra symbols or factors in the FINAL expression (e.g., do NOT include P°/p° or divide by standard-state pressure).
- You may mention the standard-state convention (1 bar) in plain text if needed, but keep the FINAL expression strictly in terms of the requested symbols only.
"""

        prompt = f"""Solve the following university-level problem. Be STRICT about satisfying the constraints.

{raw_problem_block}{extra_constraints_block}
Constraints (names for rubric sections):
{", ".join(constraint_names)}

Rubric-aligned required evidence (MUST appear in your answer; missing any item => non-compliant):
{evidence_text}

Format (mandatory, readability-first):
- Do NOT include rubric/template blocks or fixed labels such as: "Evidence anchors:", "Symbol table:", "Checklist:", "Rejected alternative:", "Accepted criterion application:".
- Do NOT use markdown tables. If you need definitions, define symbols inline in plain sentences (e.g., "Let x be ... [unit]").
- Do NOT rely on meta-slogans (e.g., "no symbol drift", "all constraints satisfied") as evidence. Demonstrate compliance by correct usage and explicit, auditable statements in the narrative.
- Avoid long quote blocks/backticks. If a point needs referencing the prompt, a precise paraphrase is acceptable.

COHERENCE REQUIREMENTS (mandatory for readability):
- Maintain narrative flow: Constraint-related evidence must be naturally integrated into the solution narrative, not presented as disconnected checklist items.
- Avoid abrupt topic switches: When introducing assumptions/boundaries/definitions, connect them logically to the preceding and following reasoning steps.
- Use transitional phrases: When moving from one constraint-related point to another, use natural transitions (e.g., "Given this assumption...", "To verify this boundary...", "Using this definition...") rather than abrupt labels.
- Contextual integration: Evidence for constraints must appear in contextually appropriate places (e.g., state assumptions when introducing the method, verify boundaries after stating the solution, define terms when first used).
- Do NOT interrupt the solution flow: If constraint-related content disrupts the logical sequence of the solution (e.g., inserting a boundary verification in the middle of a derivation without context), it fails the coherence requirement.

How to satisfy constraints WITHOUT rubric blocks (FAIRNESS: format does NOT matter; correctness does):
- Assumptions: state key assumptions actually used + 1-sentence impact each; include one counterfactual consequence; include at least one auditable verification using given/derived values (inline is fine). CRITICAL: The verification must be mathematically correct - if the computation is wrong, it fails regardless of format. You may provide more verifications than required if all are correct.
- Boundary Conditions: restate all hard constraints from the prompt; include ≥1 mechanical/auditable verification (substitution, equality, counting, etc.) in narrative. CRITICAL: The verification must be mathematically correct - if the substitution/check is wrong, it fails. You may provide more verifications than required if all are correct. Format (inline vs separate) does NOT matter.
- Units Standard: units on first mention for key variables; consistent units; include a short dimensional verification (one sentence/line).
- Cross-disciplinary Term Disambiguation: disambiguate terms; include at least one consequence that is logically correct and specific (e.g., unit mismatch, wrong formula, wrong direction). CRITICAL: If the consequence statement is factually wrong (e.g., says 'under-report' when it should be 'over-report'), it fails. Point to where you used the intended meaning (must be accurate - the referenced step must actually use the intended meaning).
- Intra-discipline Term Definitions: plain + formal definition/criterion (must be mathematically/logically correct) + theory note + apply criterion in one concrete step (must be accurate). Include one prompt-traceable rejected alternative with a problem-specific wrong consequence (must be logically correct). CRITICAL: If the formal definition is wrong (wrong equation/sign/relation) or the application is wrong, it fails regardless of format.
- Symbols & Constants Standardization: define final-result symbols clearly in narrative; cite constants with units + source/version when required; maintain symbol consistency.
- Applicability Range: state applicability range + outside-range failure mode with direction/trend and what becomes important (plain text).
{extra_symbol_guard}

Inline evidence tag guide (use sparingly; keep it natural):
{inline_tag_guide}

Recommended structure (you may adapt):
- Short setup / plan
- Derivation / reasoning (with equations if needed)
- Final Answer (clearly labeled)

Final Answer:
"""
        return prompt

    # If question already contains constraints, do not append Additional Constraints Summary to avoid duplication/disruption
    if constraints_text and not embedded_constraints:
        prompt = f"""Please solve the following problem. The problem includes specific constraints that should be followed in your answer.
    
Problem:
{question}

Additional Constraints Summary:
{constraints_text}

Please provide a detailed answer that:
1. Addresses all parts of the problem
2. Follows all specified constraints
3. Shows your reasoning and calculation steps (if applicable)
4. Uses proper units and notation
5. Write a coherent narrative. Avoid rubric/template blocks (e.g., "Evidence anchors:", "Symbol table:", "Checklist:") and avoid meta compliance slogans; integrate any necessary verification into the narrative.

Answer:
"""
    else:
        prompt = f"""Please solve the following problem.

Problem:
{question}

Please provide a detailed answer that:
1. Addresses all parts of the problem
2. Shows your reasoning and calculation steps (if applicable)
3. Uses proper units and notation
4. Write a coherent narrative. Avoid rubric/template blocks (e.g., "Evidence anchors:", "Symbol table:", "Checklist:") and avoid meta compliance slogans; integrate any necessary verification into the narrative.

Answer:
"""
    return prompt


def _load_oneshot_examples(
    file_path: str = "/root/code/multi_task_rollout/outputs_v2/test_set_v1_refine.json",
    limit_per_constraint: int = 3
) -> Dict[str, List[Dict[str, str]]]:
    """
    Select one one-shot example for each single constraint from the refined dataset.
    Only select "single constraint" problems (constraint field does not contain comma/semicolon).
    Returns {normalized_constraint: [{"question": ..., "answer": ...}, ...]}.
    """
    import json
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def _inner() -> Dict[str, List[Dict[str, str]]]:
        examples: Dict[str, List[Dict[str, str]]] = {}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Support multiple key names: corpus, selected_300_no_numexp, extra_numexp_questions
            if isinstance(data, list):
                items = data
            else:
                # Merge all available problem lists
                all_items = []
                for key in ['corpus', 'selected_300_no_numexp', 'extra_numexp_questions']:
                    if key in data and isinstance(data[key], list):
                        all_items.extend(data[key])
                items = all_items
        except Exception:
            return examples

        def norm(c: str) -> str:
            return normalize_constraint_name(c)

        for item in items:
            constraint_field = item.get("constraints", "")
            if not isinstance(constraint_field, str):
                continue
            # Only select single constraints (no Chinese/English commas/dunhao)
            if any(ch in constraint_field for ch in [",", "，", "、", ";", "；"]):
                continue
            key = norm(constraint_field)
            if not key:
                continue
            examples.setdefault(key, [])
            if len(examples[key]) < limit_per_constraint:
                examples[key].append(
                    {
                        "question": item.get("question_with_constraints") or item.get("text", ""),
                        "answer": item.get("answer", "")
                    }
                )
        return examples

    return _inner()


def _constraint_descriptions() -> Dict[str, str]:
    """Normalize constraint -> brief description (simplified version for validation prompt)"""
    return {
        "assumptions": "Assumptions: List and apply idealizations or ignored terms, explain their impact on the solution.",
        "boundary conditions": "Boundary Conditions: Clearly state initial/boundary/far-field/contact conditions and use or verify them in the solution.",
        "applicability range": "Applicability Range: Provide interval/order-of-magnitude bounds, ensure dimensional consistency, and explain trends or failures outside the range.",
        "units and notation": "Units Standard: Use SI units with scientific notation, include units on first mention, and maintain consistent units and significant figures in final results.",
        "cross-disciplinary term disambiguation": "Cross-disciplinary Term Disambiguation: Fix ambiguous term meanings and use them consistently, do not mix different meanings.",
        "intra-discipline term definitions": "Intra-discipline Term Definitions: Use recognized definitions or criteria and apply them in the solution/calculation.",
        "symbols and constants standardization": "Symbols & Constants Standardization: Maintain symbol consistency (avoid slogan-like self-checks), provide constants with version/values/units and cite them consistently.",
        "variable naming consistency": "Variable Naming Consistency: Include units on first mention, maintain consistent names and units throughout, do not use multiple symbols for the same quantity.",
        "numerical methods": "Numerical Methods: Describe algorithm, iteration/update formulas, convergence criteria, and demonstrate steps/stability/accuracy checks.",
        "experimental methods": "Experimental Methods: Describe controls/repetition/calibration/uncertainty handling, and demonstrate these steps in calculations or design."
    }


def _strict_required_evidence(norm: str) -> Dict[str, Any]:
    """Required evidence points for each constraint in strict mode (returns structured data)"""
    mapping = {
        "Assumptions": [
            ("point_1", "Assumption statement and impact: (a) state ≥1 key assumption OR explicitly state 'assumptions as given in the prompt' if the prompt already lists them; (b) for each stated assumption, briefly explain how it affects/enables the method or result (1 sentence each); (c) if the prompt/constraints text lists ≥2 assumptions, you must explicitly cover ≥2 of them (each with its own 1-sentence impact). FAIRNESS: Do NOT penalize for providing MORE assumptions than required if they are correct; do NOT reward for providing FEWER assumptions than required."),
            ("point_2", "Hard evidence requirement (format): you must clearly reference at least ONE concrete assumption from the prompt/constraints and show where you used it in THIS solution. Exact quoting/backticks are NOT required; a precise paraphrase or a short inline quote is acceptable as long as the linkage is auditable. FAIRNESS: The linkage must be explicit and auditable, but the format (inline vs separate section) does NOT matter. Judge based on whether the linkage exists, not on how it is formatted."),
            ("point_3", "Assumption rules: (a) it is ALWAYS allowed to restate/quote assumptions that are explicitly stated in the prompt/constraints text; that does NOT count as inventing assumptions; (b) do NOT invent assumptions not stated in the prompt unless strictly necessary; if you add any, label it 'extra assumption' and justify necessity in 1 line; (c) if you use implicit/default assumptions that are standard in the discipline but not explicitly stated in the prompt (e.g., standard gravity g = 9.81 m/s², standard temperature 298 K, ideal gas behavior, Earth's gravitational field), you must explicitly state them and label them as 'standard assumption' or 'implicit assumption' with a brief justification (1 line). FAIRNESS: Restating prompt assumptions is acceptable and should NOT be penalized. Only unlabeled, unjustified extra assumptions or unstated implicit assumptions should fail this point."),
            ("point_4", "Counterfactual consequence (format): include one 1-line counterfactual consequence for at least ONE assumption (e.g., 'If violated, then ...'), stating what would change in method/term/unit/equation. No fixed template/backticks required. FAIRNESS: The counterfactual must be logically correct and specific to the assumption. Do NOT fail for providing multiple counterfactuals if they are all correct; do NOT fail for brevity if the single counterfactual is correct and specific."),
            ("point_5", "Sensitivity analysis (format): add ONE 1–2 line sensitivity statement that names the exact step/formula/result that would change and how (directional or structural change), using THIS problem's symbols/values. Must include at least one problem symbol/value; MUST specify a concrete alternative equation/term/decision rule or correction term (not vague). No backticks required. FAIRNESS: The sensitivity statement must be mathematically/logically correct. If multiple sensitivity statements are provided, judge based on whether at least ONE is correct and specific. Do NOT penalize for providing more than one if they are all correct."),
            ("point_6", "Auditable assumption verification (format): provide at least ONE auditable/mechanical verification using given OR derived values that would typically fail if the assumption were false (e.g., compute a ratio like S/Km, verify v/c≪1, compare neglected vs kept term). Format: may be inline in narrative; do NOT require a literal `Check:` prefix or backticks. If truly unverifiable from given data, you MAY state what quantity is missing and how violation would bias the result (optional; does NOT replace required verification when verifiable). FAIRNESS: The verification must be mathematically correct. If the computation/check is wrong, this point fails regardless of format. If multiple verifications are provided, judge based on whether at least ONE is correct. Do NOT penalize for providing more verifications if they are all correct."),
            ("point_7", "Semantic correctness check (STRICT): Each assumption must be physically/mathematically plausible for THIS problem context. If an assumption contradicts known physical laws, experimental facts, or the problem setup itself (e.g., assuming steady-state when the problem explicitly mentions time-dependent conditions), this point fails. The assumption must be both necessary AND reasonable. Additionally, verify that: (a) the assumption is consistent with the problem's stated conditions and constraints; (b) the assumption does not contradict any explicit values or relationships given in the prompt; (c) if the assumption involves neglecting a term, that term must actually be negligible under the problem conditions (not just stated to be negligible). If any assumption violates these criteria, this point fails."),
            ("point_8", "Extended reasoning check: The impact/verification statements must demonstrate actual understanding of WHY the assumption matters, not just generic statements. For example, saying 'if we neglect friction, the system accelerates' is too generic; you must explain HOW it affects THIS specific calculation/result using problem-specific context."),
        ],
        "Boundary Conditions": [
            ("point_1", "Boundary extraction and citation (format): (a) extract and list ALL hard constraints/boundaries from the prompt (values, domains, conservation, formatting restrictions such as 'only these symbols', 'no extra terms'); (b) you must clearly reference at least TWO concrete hard constraints from the prompt. Exact backtick quoting is NOT required; precise paraphrase is acceptable as long as it is auditable. FAIRNESS: The extraction must be complete and accurate. Do NOT penalize for providing MORE constraints than required if they are all correct; do NOT reward for providing FEWER constraints than required. The format (list vs inline) does NOT matter."),
            ("point_2", "Boundary rules: do NOT add boundaries/conditions that are not stated in the prompt/constraints text; if you introduce an extra condition, label it as an assumption (and it may still be penalized if irrelevant). If the task truly has no meaningful boundary/initial/format constraints, state 'N/A' and justify in one line (rare). FAIRNESS: Restating prompt boundaries is acceptable and should NOT be penalized. Only unlabeled, unjustified extra boundaries should fail this point."),
            ("point_3", "Verification (format): provide ≥1 mechanical/auditable verification that the boundaries/constraints are satisfied (e.g., substitution, equality check, counting symbols, balancing atoms/charge, unit cancellation, etc.). Do NOT require a checklist block; verification can be inline in narrative. If there are ≥2 constraints, you should address at least two distinct checks (can be separate sentences). FAIRNESS: The verification must be mathematically/logically correct. If the verification computation is wrong, this point fails regardless of format. If multiple verifications are provided, judge based on whether at least ONE is correct. Do NOT penalize for providing more verifications if they are all correct. The verification must be explicit and auditable, but the format does NOT matter."),
            ("point_4", "Semantic correctness check: The stated boundaries must be physically/mathematically meaningful for THIS problem. If a boundary condition contradicts the problem physics (e.g., stating a boundary that would violate conservation laws or contradicts the setup), this point fails. The boundaries must align with the actual problem requirements."),
            ("point_5", "Extended reasoning check: The verification must show WHY the boundary is satisfied in the context of the solution method, not just mechanically substitute. For example, if verifying C(x→∞,t)=C0, you must explain the physical meaning in THIS problem context, not just substitute and state equality."),
        ],
        "Applicability Range": [
            ("point_1", "Applicability range and failure analysis (format): (a) if a model/approximation/formula is used, state an applicability range as an explicit interval/inequality/order-of-magnitude (numbers or symbols). Prefer symbolic bounds derived from the setup over inventing numeric ranges not stated in the prompt; (b) explicitly name the approximation/model you used and tie the stated range to it (range must mention at least one problem symbol); (c) state at least one failure mode outside the range AND the direction of error/trend; (d) the failure mode must name WHAT becomes important outside the range (not just 'may fail'). No backticks required; must be explicit in plain text."),
            ("point_2", "N/A justification: if no model/approximation/formula with an applicability range is used in the solution, state 'N/A' and provide a brief justification explaining why a range claim would be meaningless or inapplicable here (e.g., 'No approximation/model used; exact solution applies universally' or 'Problem uses fundamental definitions without range limitations'). The justification should be clear and specific to this problem context."),
        ],
        "Units Standard": [
            ("point_1", "Numerical unit requirements: if numerical quantities are present or requested: (a) give units on first mention for ALL key variables (≥2), (b) the FINAL answer line must include the target unit, and (c) handle sig figs/decimal places/notation exactly as requested (state the rule you used)."),
            ("point_2", "Unit self-check: provide a 1-line dimensional/unit self-check that explicitly names the target quantity (e.g., 'ρ has unit g/cm^3 because ...')."),
            ("point_3", "Dimensionless handling: if a quantity is dimensionless by definition (e.g., relative atomic mass, ratios, probabilities), explicitly state 'dimensionless' and do NOT assign physical units."),
            ("point_4", "Symbolic unit convention: if the task is symbolic/expressional: (a) explicitly state the unit convention for each symbol at first mention in text (e.g., P_X [bar]) AND justify whether the final result is dimensionless or why units are omitted; (b) if the problem forbids extra terms/symbols, you may explain standard-state conventions in text but must keep the FINAL expression in the allowed symbols only (no P°/p° insertion)."),
            ("point_5", "Validator intent note: do NOT fail this constraint for small arithmetic/value mistakes unless they cause unit/sig-fig violations; numerical correctness is evaluated elsewhere."),
        ],
        "Cross-disciplinary Term Disambiguation": [
            ("point_1", "Term identification and coverage: (a) disambiguate the ambiguity points explicitly mentioned in THIS item's constraint text (highest priority). If the constraint text lists multiple points (e.g., 'intramolecular vs intermolecular', 'hydrogen bonding', 'p=pressure'), you must cover ALL of them; (b) additionally, disambiguate at least ONE other high-risk symbol/term that appears in the prompt or your own answer (e.g., p, K, n, Z, A, group, equilibrium), if any. If you cannot find any, explicitly state 'No additional high-risk terms present' and justify in 1 line; (c) you must disambiguate AT LEAST TWO terms total (including those listed in the constraint text). If the constraint text contains only one term, you must add at least one more from the prompt/answer. FAIRNESS: Do NOT penalize for providing MORE disambiguations than required if they are all correct; do NOT reward for providing FEWER disambiguations than required."),
            ("point_2", "Term format and consequence (format): for each term, clearly state intended meaning and non-intended meaning; for at least ONE term, state a 1-line consequence (what would go wrong) if non-intended meaning were used. No fixed `Term: ...` template required. FAIRNESS: The consequence must be logically correct and specific (e.g., unit mismatch, wrong formula, wrong direction). If the consequence statement is factually wrong (e.g., says 'under-report' when it should be 'over-report', or gives wrong units), this point fails regardless of format. If multiple consequences are provided, judge based on whether at least ONE is correct. Do NOT penalize for providing more consequences if they are all correct."),
            ("point_3", "Consistency evidence (format): point to one specific step/formula/sentence in your solution where you use the intended meaning (must be specific and auditable). Exact backtick quoting not required. FAIRNESS: The evidence must be accurate - the referenced step must actually use the intended meaning. If the referenced step contradicts the intended meaning, this point fails. The format (inline vs separate) does NOT matter."),
            ("point_4", "N/A justification: you may claim 'No ambiguous terms detected' ONLY if the constraint text does not list any ambiguity points and the prompt defines all key symbols; give a 1-line justification. FAIRNESS: The N/A claim must be factually correct. If the constraint text explicitly lists ambiguous terms, claiming N/A is incorrect and this point fails."),
            ("point_5", "Semantic correctness check: The disambiguation must be factually correct within the domain. If the stated 'intended meaning' is actually wrong for this context (e.g., stating 'p' means momentum when the problem clearly uses 'p' for pressure throughout), this point fails. The disambiguation must align with actual usage in the problem/solution."),
            ("point_6", "Extended reasoning check: The consequence statement must demonstrate understanding of the actual impact, not generic statements. For example, saying 'wrong units' is insufficient; you must explain HOW it affects THIS specific calculation/result (e.g., 'would give force in N·s instead of N, leading to incorrect torque calculation')."),
            ("point_7", "Ambiguity analysis for different meanings (NEW): If a term has multiple possible meanings within the problem context (even if not explicitly listed in the constraint text), you must analyze how the solution would differ under each plausible interpretation. For at least ONE such term, provide: (a) the different possible meanings within this problem context, AND (b) a brief analysis (1-2 sentences) of how the solution/result would change if each different meaning were used. This demonstrates understanding of the ambiguity's actual impact. FAIRNESS: The analysis must be logically correct. If the analysis incorrectly describes how the solution would change, this point fails."),
        ],
        "Intra-discipline Term Definitions": [
            ("point_1", "Term definition and theory: (a) provide BOTH: (i) a plain-language description (1–2 sentences) of the term/criterion, AND (ii) a formal definition/criterion (equation/inequality/classification rule). If the prompt provides a definition, restate it without changing meaning; (b) add a short generalization/theory note (1–2 sentences): why this definition matters and what it controls/assumes (e.g., what would change if the definition/criterion were different). FAIRNESS: The formal definition must be mathematically/logically correct. If the formal definition is wrong (e.g., wrong equation, wrong sign, wrong relation), this point fails regardless of format. Do NOT penalize for providing MORE theory notes if they are all correct; do NOT reward for providing FEWER theory notes if the minimum requirement is met."),
            ("point_2", "Definition application link (format): explicitly link definition/criterion → one concrete step in THIS solution (derive an equation, justify a method choice, or make a classification decision using the criterion). The link must mention problem-specific symbols/data. No backticks required. FAIRNESS: The link must be accurate - the referenced step must actually use the definition/criterion correctly. If the application is wrong (e.g., uses wrong formula, wrong classification), this point fails. The format (inline vs separate) does NOT matter."),
            ("point_3", "Rejected alternative analysis (format): state at least ONE plausible rejected alternative that is TRACEABLE to the prompt/options (or a contrast explicitly implied there), and explain (i) why it fails under the formal criterion AND (ii) what wrong conclusion/result it would produce for THIS problem using at least one problem-specific symbol/value. No requirement to use the literal label 'Rejected alternative:' or backticks; traceability and problem-specific wrong consequence are mandatory. FAIRNESS: The rejected alternative must be traceable to the prompt, and the wrong consequence must be logically correct. If the wrong consequence is factually incorrect (e.g., says it would give a higher value when it would actually give a lower value), this point fails. If multiple rejected alternatives are provided, judge based on whether at least ONE is correct and traceable. Do NOT penalize for providing more alternatives if they are all correct."),
            ("point_4", "Symbol usage rule: do NOT introduce new abstract symbols (e.g., 'let f∈{a,b,c}', 'R=...') solely to state the criterion. Prefer plain-language rules and reuse the prompt's terms/symbols; introducing extra symbols may cause failure under 'Symbols & Constants Standardization'. FAIRNESS: This point fails only if new abstract symbols are introduced SOLELY for stating the criterion (not if they are used in the actual solution). If the symbols are used in the solution and properly declared, this point should pass."),
            ("point_5", "Semantic correctness check (STRICT): The formal definition must be correct within the discipline's conventions. If the definition contradicts standard terminology (e.g., defining 'activation energy' incorrectly), this point fails. The definition must align with accepted disciplinary knowledge. Additionally verify: (a) the definition's mathematical form (if any) is correct (correct signs, units, relationships); (b) the definition's scope and applicability match standard usage (e.g., if defining a classification criterion, it must correctly distinguish the categories); (c) the definition does not contradict any explicit definitions given in the prompt; (d) if the definition involves a threshold or boundary, the threshold value and direction (>, <, ≥, ≤) must be correct. If any aspect of the definition is factually incorrect, this point fails."),
            ("point_6", "Extended reasoning check: The theory note must explain WHY the definition matters in THIS problem context, not just generic statements. For example, saying 'this criterion determines the classification' is too generic; you must explain HOW it affects THIS specific solution step/result."),
            ("point_7", "Application depth check: The application link must show actual USE of the definition/criterion to derive or justify something, not just mention it. Simply stating 'we use this definition' without showing HOW it affects the derivation/method choice/classification decision fails this point."),
            ("point_8", "Semantic correctness check for application (STRICT): When applying the definition/criterion in the solution, verify that: (a) the application uses the definition correctly (correct formula, correct classification rule, correct threshold comparison); (b) the application produces results that are consistent with the definition (e.g., if using a definition to classify, the classification must follow from the definition's logic); (c) if the application involves numerical evaluation, the values used must be consistent with the definition's requirements; (d) the application does not contradict the definition itself (e.g., claiming to use a definition but applying a different criterion). If the application contradicts or misuses the definition, this point fails."),
            ("point_9", "Ambiguity analysis for different definitions (NEW): If the term/criterion has multiple possible definitions or interpretations within the discipline that could apply to this problem, you must analyze how the solution would differ under each plausible definition. For at least ONE such case, provide: (a) the different possible definitions/interpretations, AND (b) a brief analysis (1-2 sentences) of how the solution/result would change if each different definition were used. This demonstrates understanding of definitional choices and their consequences. FAIRNESS: The analysis must be logically correct. If the analysis incorrectly describes how the solution would change, this point fails."),
        ],
        "Symbols & Constants Standardization": [
            ("point_1", "Symbol declaration (format): ensure all symbols used in the FINAL result are defined clearly somewhere in the narrative (a separate symbol table is optional but NOT required). If you introduce unclear intermediate symbols/variables, define them too. You are NOT required to invent symbols just to satisfy this constraint."),
            ("point_2", "Constant declaration: (a) if physical/mathematical constants are used (e.g., R, N_A, g, k_B): provide value + units AND a standard source/version (e.g., SI definition / CODATA) unless the prompt provides it explicitly; (b) if a conversion/offset is stated as exact by definition in the prompt (e.g., °C↔°F relation, 9/5, 32 °F), treat it as given; do NOT require an external source citation beyond 'exact by definition'."),
            ("point_3", "Derived symbol handling: derived products like RT do NOT need their own symbol-table entry if written explicitly as R·T and both R and T are declared; but if you treat RT as a standalone symbol, you must declare it."),
            ("point_4", "Symbol restrictions and consistency (format): (a) do not introduce extra symbols into the FINAL expression when the problem restricts allowed symbols; if needed, explain conventions in text instead; (b) ensure there is NO symbol drift (each symbol keeps one meaning throughout). A 'no drift' sentence is NOT required; consistency must be demonstrable from the solution."),
        ],
        "Variable Naming Consistency": [
            ("point_1", "Symbol consistency: use one symbol per quantity; no symbol/unit drift across the solution. The same symbol must represent the same quantity throughout the entire solution. For example, if P_O2 is used to denote partial pressure of O2 at the beginning, it must maintain this meaning throughout."),
            ("point_2", "Unit consistency: (a) if you define a variable with units at first use, keep the same units thereafter; (b) unit conversions are allowed ONLY if you explicitly state the conversion step (e.g., `x = 0.80 mm = 8.0×10^-4 m`) and then use one unit consistently afterward. Do not mix units for the same variable without explicit conversion."),
            ("point_3", "Semantic correctness check (STRICT): Verify that the symbol usage is semantically correct and consistent with domain conventions. Specifically: (a) if a symbol is used to represent a physical quantity, it must be used consistently with its physical meaning (e.g., if 'v' is defined as velocity, it must not be used as volume later); (b) if a symbol represents a dimensionless quantity, it must be used consistently as dimensionless throughout; (c) if a symbol represents a vector/tensor, it must be used consistently with its tensor rank; (d) the symbol must align with standard notation conventions for the discipline (e.g., in chemistry, 'n' typically means moles, not number of particles). If the symbol usage contradicts its defined meaning or standard conventions, this point fails."),
        ],
        "Numerical Methods": [
            ("point_1", "Method type and order: name the exact algorithm/order (e.g., explicit midpoint/Heun/RK2, RK4, Newton, bisection) and state it is a numerical approximation (not a closed-form solution)."),
            ("point_2", "Update formula using this problem's symbols: write one step of the update with the symbols given in the prompt (e.g., y_n, h, k1, k2, x_n, v_n). If the method is multi-stage/piecewise (e.g., update v then x), show those formulas."),
            ("point_3", "Step size and convergence/stability reason: give a rationale or condition (e.g., h·ω0 ≪ 1, CFL-like) and explain why the chosen step keeps the method stable/convergent. If an initial guess matters (Newton/optimization), note why it is reasonable for this problem."),
            ("point_4", "At least one full numeric iteration: plug this problem's numbers into the update (include k1/k2 or f(x_n,y_n)), not just symbols; show the computed next state."),
            ("point_5", "Error analysis: (a) provide numeric absolute/relative error or gap to analytic/benchmark value; saying 'smaller error' without a number is insufficient; (b) state the method's order or error order and relate it to the chosen step/problem scale (e.g., RK2 is O(h^2) and is stable for the stated h here); (c) briefly state what would change if the step were halved (e.g., expected error scaling O(h^p) or stability margin), or compare against a second step size/iteration count if already given. A one-line expectation is enough but must be problem-grounded."),
            ("point_6", "Intermediate numerical checkpoints (≥2): include at least TWO intermediate numerical values or checkpoints that can be verified (e.g., a table showing intermediate values at different steps, a residual/stability check with specific numerical values, or the computed state after one or more iterations with actual numbers). Generic text like 'iterate until convergence' or statements without numeric substitution → FAIL."),
        ],
        "Experimental Methods": [
            ("point_1", "Measurement objective: name the target quantity (e.g., g, ΔH, refractive index) and the physical/chemical relation used to infer it."),
            ("point_2", "≥2 design elements tied to this problem's variables: e.g., controls/blank, calibration, repeated trials, systematic variation, random/systematic error control—each referenced with this problem's quantities (L, T, m, V, λ, etc.)."),
            ("point_3", "Executable data-taking steps: spell out an actionable sequence (e.g., 'measure 3 lengths, each timed for 10 periods, take the mean' or 'blank + standard calibration, then measure unknown'), not just generic 'repeat measurements.'"),
            ("point_4", "Uncertainty propagation to the target: show how measurement spread or instrument resolution flows to the final quantity (e.g., σ_T → σ_g). If using linearization/fit, cite the formula or slope uncertainty."),
            ("point_5", "At least two auditable anchors: list key raw/intermediate numbers (means, stdev/SEM, slope/intercept, or a specific subtraction like blank correction) so others can recompute. Purely verbal descriptions → FAIL."),
            ("point_6", "Result validation: (a) state how you judge reliability (repeatability, consistency, residuals/linearity, control comparison) rather than only giving a number without validation; (b) explicitly say why these design choices satisfy the problem's stated constraints (e.g., small-angle assumption, neglecting air drag, sufficient instrument resolution, blank baseline removal). Missing this linkage → FAIL."),
            ("point_7", "One explicit limitation/mitigation: name one plausible systematic/edge-case issue for this setup and give a one-line mitigation or why it is negligible (problem-specific, not generic)."),
        ],
    }
    
    points = mapping.get(norm, [])
    if not points:
        return {
            "points": [],
            "formatted_text": "(No strict evidence rules found.)"
        }
    
    # Generate formatted text (for backward compatibility)
    formatted_lines = []
    for i, (pid, text) in enumerate(points, 1):
        formatted_lines.append(f"{i}) {text}")
    formatted_text = "\n".join(formatted_lines)
    
    return {
        "points": points,  # [(point_id, text), ...]
        "formatted_text": formatted_text
    }


def build_single_constraint_validation_prompt(
    question: str,
    constraint: str,
    model_answer: str,
    subject: str,
    reference_answer: str = "",
    include_reason: bool = False,
    prompt_mode: Optional[str] = None,
    raw_constraints_text: str = "",
) -> str:
    """Build validation prompt for single constraint (supports one-shot, default output YES/NO only, optional reason)"""
    mode = (prompt_mode or VALIDATOR_PROMPT_MODE).lower()
    # Ensure mode is strict or loose, otherwise use strict as default
    if mode not in ["strict", "loose"]:
        mode = "strict"
    subject_name_map = {
        "physics": "Physics",
        "chemistry": "Chemistry",
        "biology": "Biology",
        "materials": "Materials",
        "protein": "Protein"
    }
    subject_name = subject_name_map.get(subject, subject)
    
    normalized_constraint = normalize_constraint_name(constraint)
    examples = _load_oneshot_examples()
    desc_map = _constraint_descriptions()
    desc = desc_map.get(normalized_constraint, constraint)

    # few-shot: take at most 3 examples for this constraint
    shots = examples.get(normalized_constraint, [])[:3]
    shot_blocks = []
    for shot in shots:
        shot_blocks.append(
            f"Example Question:\n{shot.get('question','')}\nExample Answer:\n{shot.get('answer','')}\nExample Verdict: YES"
        )
    oneshot_block = "\n\n".join(shot_blocks)
    if oneshot_block:
        oneshot_block = "Few-shot examples for this constraint (expected YES):\n" + oneshot_block + "\n"

    reason_hint = "Do not output any reason." if not include_reason else "If NO, optionally add one short reason (<=60 chars) after a new line starting with 'REASON:'."
    ref_block = f"Reference Answer (gold):\n{reference_answer}\n\n" if reference_answer else ""

    raw_block = f"Original constraints text (for reference):\n{raw_constraints_text}\n\n" if raw_constraints_text else ""

    if mode in ["strict", "loose"]:
        req_data = _strict_required_evidence(normalized_constraint)
        req_points = req_data.get("points", []) if isinstance(req_data, dict) else []
        req_text = req_data.get("formatted_text", "") if isinstance(req_data, dict) else req_data
        
        if req_points:
            # Get key point definitions for loose mode
            loose_config = LOOSE_KEY_POINTS.get(normalized_constraint, {})
            main_points = set(loose_config.get("main", []))
            secondary_points = set(loose_config.get("secondary", []))
            
            # Build point list text, identify main and secondary
            point_list_lines = []
            for i, (pid, text) in enumerate(req_points, 1):
                if pid in main_points:
                    point_list_lines.append(f"Point {i} ({pid}) [MAIN - REQUIRED]: {text}")
                elif pid in secondary_points:
                    point_list_lines.append(f"Point {i} ({pid}) [SECONDARY - OPTIONAL]: {text}")
                else:
                    point_list_lines.append(f"Point {i} ({pid}): {text}")
            point_list_text = "\n".join(point_list_lines)
            
            point_output_template = "\n".join([
                f"POINT_{i}: YES or NO [if NO, brief reason: which specific requirement is missing/incorrect]"
                for i in range(1, len(req_points) + 1)
            ])
            
            # Set mode label
            mode_label = "STRICT" if mode == "strict" else "LOOSE"
            
            # Set validation strategy according to mode
            if mode == "strict":
                decision_policy = """Decision policy (STRICT):
- Evaluate EACH point above independently. For each point, check if the Model Answer provides explicit evidence.
- Do NOT infer missing evidence. If you cannot find explicit text/equations for a point, that point = NO.
- Generic boilerplate without problem-specific symbols/data does NOT count as evidence when a point requires linking to THIS problem.
- NUMERICAL PRECISION MASK (TEMPORARY / MASK4SIG): Do NOT mark a point NO solely because intermediate/final numerical values are not exactly 4 significant figures, or solely because the Model Answer uses approximate-language markers such as "approximately", "about", "roughly", "around", "~", or "≈". Still require units where relevant, explicit numerical values when the evidence point needs a quantitative check, dimensionally valid calculations, and mathematically/physically correct numerical reasoning. If a numerical value, unit conversion, equation, or quantitative conclusion is substantively wrong or missing, mark the relevant point NO.
- TRUTHFULNESS CHECK (applies to EVERY point): If the Model Answer claims a verification/check/derivation but the surrounding work contradicts it (e.g., substitution doesn't actually satisfy the boundary, a stated unit check is wrong, a 'consistent symbols' claim is contradicted by symbol drift, a rejected alternative is not actually traceable to the prompt), you MUST mark the relevant point as NO.
- SEMANTIC CORRECTNESS CHECK (applies to relevant points): If a point involves domain knowledge (assumptions, boundaries, definitions, terminology, variable naming), verify that the content is factually correct within the discipline. For example: (a) assumptions must be physically/mathematically plausible for THIS problem and consistent with problem conditions; (b) boundary conditions must align with problem physics; (c) definitions must match standard disciplinary terminology with correct mathematical forms; (d) term disambiguations must reflect actual usage in the problem/solution; (e) variable naming must be semantically consistent (symbols must represent the same physical quantity/type throughout, align with standard notation conventions). For "Assumptions", "Variable Naming Consistency", and "Intra-discipline Term Definitions" constraints, apply STRICT semantic correctness checks: verify assumptions are necessary and reasonable, symbols are used consistently with their defined meanings and domain conventions, and definitions/applications are factually correct. If the content contradicts known facts, physical laws, standard conventions, or the problem setup itself, mark that point NO.
- EXTENDED REASONING CHECK (applies to relevant points): When evaluating evidence, verify that it demonstrates actual understanding, not generic statements. For numerical reasoning or quantitative impact statements: (a) If the Model Answer claims an impact but uses vague language (e.g., "would give wrong units" without explicit numerical values), mark that point NO. The Model Answer must provide explicit numerical values where numerical impact is claimed; (b) If the Model Answer performs verification but uses vague language (e.g., "approximately satisfies" instead of explicit numerical substitution), mark that point NO; (c) If the Model Answer discusses consequences but uses vague quantitative language instead of explicit numerical impact, mark that point NO. Generic statements without explicit numerical support (when numerical support is required) fail this check.
- COHERENCE CHECK (applies to relevant points): Constraint-related evidence must be naturally integrated into the solution narrative. If evidence appears as disconnected checklist items, abrupt topic switches without transitions, or interrupts the logical flow of the solution, mark the relevant point NO. Evidence should appear in contextually appropriate places with natural transitions.
- FORMAT CHECK (MODERATE STRICTNESS - overall assessment): Evaluate whether the Model Answer follows format guidelines:
  * Format violations: If the answer includes prohibited rubric/template blocks (e.g., "Evidence anchors:", "Symbol table:", "Checklist:" sections), markdown tables for definitions, or relies heavily on meta-slogans without substantive evidence, this is a format violation. However, minor format issues (e.g., occasional use of backticks for emphasis) should NOT cause failure.
  * Coherence violations: If the answer has severe coherence issues (e.g., constraint-related evidence appears as completely disconnected checklist items, abrupt topic switches without any logical connection, or significant disruption to the solution flow), this is a coherence violation. Minor coherence issues (e.g., somewhat abrupt transitions that don't severely disrupt flow) should NOT cause failure.
  * Judgment criteria: Only fail FORMAT check if there are SIGNIFICANT violations that seriously impact readability OR coherence. Do NOT fail for minor formatting preferences or slight coherence imperfections. If the answer is generally readable and maintains reasonable narrative flow despite some format/coherence imperfections, pass the FORMAT check.
  * Effect on verdict: If FORMAT check fails due to significant violations, you may mark relevant evidence points as NO if they depend on the problematic format/coherence. However, if the evidence content is correct and auditable despite format issues, do NOT automatically fail those points - judge based on substance.
- Do NOT award points for rubric-like packaging (templates, headings, slogans). Judge the substance: the referenced claim must be correct and supported by the work shown.
- FAIRNESS PRINCIPLE (CRITICAL): Judge based on whether the requirement is MET, NOT on how much text is written. If a model provides MORE evidence than required (e.g., 3 verifications when 1 is required), do NOT penalize if all are correct. If a model provides MINIMAL but CORRECT evidence that meets the requirement, award the point. The format (inline vs separate section, short vs long) does NOT matter - only correctness and completeness of the requirement matter.
- UNIFIED STANDARD FOR ALL JUDGE MODELS (CRITICAL): This decision policy applies EQUALLY to all judge models (GPT and Gemini). Both models MUST use the EXACT SAME criteria: (a) do not enforce exactly-4-significant-figures as a standalone requirement; (b) do not fail solely for approximate-language markers; (c) still apply substantive numerical, unit, and semantic checks with IDENTICAL strictness regardless of judge model identity. Do NOT apply different standards based on which model is judging. The rubric defines objective requirements that are model-independent.
- Overall verdict = YES only if ALL points are YES; otherwise NO."""
            else:  # loose mode
                # For "Applicability Range", in loose mode all points must also pass
                if normalized_constraint == "Applicability Range":
                    decision_policy = """Decision policy (LOOSE - but ALL points required for this constraint):
- Evaluate EACH point above independently. For each point, check if the Model Answer provides explicit evidence.
- Do NOT infer missing evidence. If you cannot find explicit text/equations for a point, that point = NO.
- Generic boilerplate without problem-specific symbols/data does NOT count as evidence when a point requires linking to THIS problem.
- NUMERICAL PRECISION MASK (TEMPORARY / MASK4SIG): Do NOT mark a point NO solely because intermediate/final numerical values are not exactly 4 significant figures, or solely because the Model Answer uses approximate-language markers such as "approximately", "about", "roughly", "around", "~", or "≈". Still require units where relevant, explicit numerical values when the evidence point needs a quantitative check, dimensionally valid calculations, and mathematically/physically correct numerical reasoning. If a numerical value, unit conversion, equation, or quantitative conclusion is substantively wrong or missing, mark the relevant point NO.
- TRUTHFULNESS CHECK: Do not accept claims on faith. If a claimed check/derivation is incorrect or contradicted by the shown work, that point = NO.
- SEMANTIC CORRECTNESS CHECK: Verify that the applicability range and failure mode statements are physically/mathematically meaningful and correct for THIS problem, with explicit numerical values where applicable.
- FORMAT CHECK (MODERATE STRICTNESS): Follow the format guidelines described above. Only fail if there are SIGNIFICANT format or coherence violations that seriously impact readability. Minor issues should NOT cause failure.
- UNIFIED STANDARD FOR ALL JUDGE MODELS (CRITICAL): This decision policy applies EQUALLY to all judge models (GPT and Gemini). Both models MUST use the EXACT SAME criteria: (a) do not enforce exactly-4-significant-figures as a standalone requirement; (b) do not fail solely for approximate-language markers; (c) still apply substantive numerical, unit, and semantic checks with IDENTICAL strictness regardless of judge model identity. Do NOT apply different standards based on which model is judging. The rubric defines objective requirements that are model-independent.
- Overall verdict = YES only if ALL points are YES; otherwise NO.
- Note: For this constraint (Applicability Range), even in LOOSE mode, ALL points must pass because both point_1 and point_2 are critical."""
                else:
                    decision_policy = f"""Decision policy (LOOSE):
- Evaluate EACH point above independently. For each point, check if the Model Answer provides explicit evidence.
- Do NOT infer missing evidence. If you cannot find explicit text/equations for a point, that point = NO.
- Generic boilerplate without problem-specific symbols/data does NOT count as evidence when a point requires linking to THIS problem.
- NUMERICAL PRECISION MASK (TEMPORARY / MASK4SIG): Do NOT mark a point NO solely because intermediate/final numerical values are not exactly 4 significant figures, or solely because the Model Answer uses approximate-language markers such as "approximately", "about", "roughly", "around", "~", or "≈". Still require units where relevant, explicit numerical values when the evidence point needs a quantitative check, dimensionally valid calculations, and mathematically/physically correct numerical reasoning. If a numerical value, unit conversion, equation, or quantitative conclusion is substantively wrong or missing, mark the relevant point NO.
- TRUTHFULNESS CHECK: If the Model Answer's claimed verification/justification is wrong or contradicted by its own steps, mark that point NO (even if the template/phrasing looks correct).
- SEMANTIC CORRECTNESS CHECK (applies to relevant MAIN points): If a point involves domain knowledge, verify that the content is factually correct within the discipline. For "Assumptions", "Variable Naming Consistency", and "Intra-discipline Term Definitions" constraints, apply STRICT semantic correctness checks: verify assumptions are necessary and reasonable, symbols are used consistently with their defined meanings and domain conventions, and definitions/applications are factually correct. If the content contradicts known facts, physical laws, standard conventions, or the problem setup itself, mark that point NO.
- EXTENDED REASONING CHECK (applies to relevant MAIN points): When evaluating evidence, verify that it demonstrates actual understanding, not generic statements. Generic statements without problem-specific context fail this check. For numerical reasoning or quantitative statements: If the Model Answer uses vague language (e.g., "approximately", "about", "roughly") instead of explicit numerical values when quantitative evidence is required, mark that point NO. The judge must verify that numerical claims are supported by explicit values when quantitative evidence is required.
- FORMAT CHECK (MODERATE STRICTNESS): Evaluate whether the Model Answer follows format guidelines. Only fail if there are SIGNIFICANT violations (prohibited rubric blocks, markdown tables, severe coherence issues) that seriously impact readability. Minor format/coherence imperfections should NOT cause failure.
- FAIRNESS PRINCIPLE (CRITICAL): Judge based on whether the requirement is MET, NOT on how much text is written. If a model provides MORE evidence than required (e.g., 3 verifications when 1 is required), do NOT penalize if all are correct. If a model provides MINIMAL but CORRECT evidence that meets the requirement, award the point. The format (inline vs separate section, short vs long) does NOT matter - only correctness and completeness of the requirement matter.
- UNIFIED STANDARD FOR ALL JUDGE MODELS (CRITICAL): This decision policy applies EQUALLY to all judge models (GPT and Gemini). Both models MUST use the EXACT SAME criteria: (a) do not enforce exactly-4-significant-figures as a standalone requirement; (b) do not fail solely for approximate-language markers; (c) still apply substantive numerical, unit, and semantic checks with IDENTICAL strictness regardless of judge model identity. Do NOT apply different standards based on which model is judging. The rubric defines objective requirements that are model-independent.
- Overall verdict = YES only if ALL MAIN points (marked as [MAIN - REQUIRED]) are YES. Secondary points (marked as [SECONDARY - OPTIONAL]) can be NO without affecting the overall verdict.
- Main points that must pass: {', '.join(sorted(main_points)) if main_points else 'None'}
- Secondary points (optional): {', '.join(sorted(secondary_points)) if secondary_points else 'None'}"""
            
            format_block = """Format Guidelines (for reference when evaluating):
- Model answers should NOT include rubric/template blocks or fixed labels such as: "Evidence anchors:", "Symbol table:", "Checklist:", "Rejected alternative:", "Accepted criterion application:".
- Model answers should NOT use markdown tables for definitions (should define symbols inline in plain sentences).
- Model answers should NOT rely on meta-slogans (e.g., "no symbol drift", "all constraints satisfied") as evidence without substantive content.
- Constraint-related evidence should be naturally integrated into the solution narrative, not presented as disconnected checklist items.
- Evidence should appear in contextually appropriate places with natural transitions (e.g., "Given this assumption...", "To verify this boundary...", "Using this definition...").
- The solution should maintain narrative flow without abrupt topic switches or significant disruption to the logical sequence.
"""
            
            prompt = f"""You are an expert validator for university-level {subject_name} problems. Be {mode_label}.

Constraint (normalized: {normalized_constraint}):
{desc}
{raw_block}

{format_block}
Required evidence points (evaluate EACH point independently):
{point_list_text}

{decision_policy}

Problem:
{question}

{ref_block}Model Answer:
{model_answer}

{oneshot_block}Output format ({mode_label} - you MUST follow this format exactly):
{point_output_template}
OVERALL: YES or NO
OVERALL_REASON: [if OVERALL is NO, list which points failed and why, in <=150 chars]

Example output:
POINT_1: YES
POINT_2: NO [missing update formula with problem-specific symbols like y_n, h]
POINT_3: YES
POINT_4: NO [only symbolic iteration shown, no numeric substitution]
POINT_5: YES
POINT_6: YES
OVERALL: NO
OVERALL_REASON: Points 2,4 failed: missing problem-specific update formula and numeric iteration with actual numbers.

Important: You MUST output each POINT_X line, then OVERALL line, then OVERALL_REASON line. Do NOT skip any point.
"""
        else:
            # If no points are defined, use old format
            mode_label = "STRICT" if mode == "strict" else "LOOSE"
            format_block_old = """Format Guidelines (for reference when evaluating):
- Model answers should NOT include rubric/template blocks or fixed labels such as: "Evidence anchors:", "Symbol table:", "Checklist:", "Rejected alternative:", "Accepted criterion application:".
- Model answers should NOT use markdown tables for definitions (should define symbols inline in plain sentences).
- Model answers should NOT rely on meta-slogans (e.g., "no symbol drift", "all constraints satisfied") as evidence without substantive content.
- Constraint-related evidence should be naturally integrated into the solution narrative, not presented as disconnected checklist items.
- Evidence should appear in contextually appropriate places with natural transitions (e.g., "Given this assumption...", "To verify this boundary...", "Using this definition...").
- The solution should maintain narrative flow without abrupt topic switches or significant disruption to the logical sequence.
"""
            prompt = f"""You are an expert validator for university-level {subject_name} problems. Be {mode_label}.

Constraint (normalized: {normalized_constraint}):
{desc}
{raw_block}

{format_block_old}
Required evidence for YES:
{req_text}

Decision policy ({mode_label}):
- Treat EVERY bullet under 'Required evidence for YES' as mandatory. Do NOT infer missing evidence.
- You must be able to point to explicit text/equations in the Model Answer for each bullet; otherwise verdict = NO.
- Generic boilerplate without problem-specific symbols/data does NOT count as evidence when a bullet requires linking to THIS problem.
- NUMERICAL PRECISION MASK (TEMPORARY / MASK4SIG): Do NOT mark a point NO solely because intermediate/final numerical values are not exactly 4 significant figures, or solely because the Model Answer uses approximate-language markers such as "approximately", "about", "roughly", "around", "~", or "≈". Still require units where relevant, explicit numerical values when the evidence point needs a quantitative check, dimensionally valid calculations, and mathematically/physically correct numerical reasoning. If a numerical value, unit conversion, equation, or quantitative conclusion is substantively wrong or missing, mark the relevant point NO.
- TRUTHFULNESS CHECK: If the Model Answer asserts a check/derivation but it is incorrect or contradicted by the work shown, verdict = NO (do not accept template compliance).
- SEMANTIC CORRECTNESS CHECK: If the content involves domain knowledge (assumptions, boundaries, definitions, terminology), verify that it is factually correct within the discipline. If the content contradicts known facts, physical laws, or the problem setup, verdict = NO.
- EXTENDED REASONING CHECK: When evaluating evidence, verify that it demonstrates actual understanding, not generic statements. Generic statements without problem-specific context fail this check. For numerical reasoning or quantitative statements: If the Model Answer uses vague language (e.g., "approximately", "about", "roughly") instead of explicit numerical values when quantitative evidence is required, verdict = NO. The judge must verify that numerical claims are supported by explicit values when quantitative evidence is required.
- COHERENCE CHECK: Constraint-related evidence must be naturally integrated into the solution narrative. If evidence appears as disconnected checklist items or interrupts the logical flow, verdict = NO.
- FORMAT CHECK (MODERATE STRICTNESS): Evaluate whether the Model Answer follows format guidelines (no rubric blocks, no markdown tables, natural narrative flow). Only fail if there are SIGNIFICANT violations that seriously impact readability or coherence. Minor format/coherence imperfections should NOT cause failure.
- Do NOT penalize the Model Answer for restating assumptions/boundaries that are explicitly stated in the Problem/constraints text.
- If the Model Answer adds assumptions/boundaries NOT stated in the prompt for constraints about assumptions/boundaries, that is a failure unless explicitly justified as necessary.
- FAIRNESS PRINCIPLE (CRITICAL): Judge based on whether the requirement is MET, NOT on how much text is written. If a model provides MORE evidence than required, do NOT penalize if all are correct. If a model provides MINIMAL but CORRECT evidence that meets the requirement, award the point. The format (inline vs separate section, short vs long) does NOT matter - only correctness and completeness of the requirement matter.
- UNIFIED STANDARD FOR ALL JUDGE MODELS (CRITICAL): This decision policy applies EQUALLY to all judge models (GPT and Gemini). Both models MUST use the EXACT SAME criteria: (a) do not enforce exactly-4-significant-figures as a standalone requirement; (b) do not fail solely for approximate-language markers; (c) still apply substantive numerical, unit, and semantic checks with IDENTICAL strictness regardless of judge model identity. Do NOT apply different standards based on which model is judging. The rubric defines objective requirements that are model-independent.
- For Units Standard: Do NOT fail solely for not maintaining exactly 4 significant figures or for approximate-language markers. Still fail substantive unit errors, missing units required by the constraint, or numerical errors that change the answer/constraint evidence.
- For Symbols & Constants Standardization: treat 'given in prompt' or 'exact by definition' as an acceptable source when the prompt says so; do not demand external citations in that case.

Problem:
{question}

{ref_block}Model Answer:
{model_answer}

{oneshot_block}Output instructions:
- Respond ONLY with 'YES' or 'NO'.
- If NO, add one short reason (<=80 chars) on a new line starting with 'REASON:'.
- If any required evidence is missing/contradicted/undefined, verdict = NO.
"""
        return prompt
    
    # Default mode: simple prompt without strict evidence requirements (used when mode is not strict or loose)
    prompt = f"""You are an expert validator for university-level {subject_name} problems. Decide if the model answer satisfies the constraint.

Constraint (normalized: {normalized_constraint}):
{desc}
{raw_block}

Problem:
{question}

{ref_block}Model Answer:
{model_answer}

{oneshot_block}Output instructions:
- Respond with ONLY 'YES' or 'NO'.
- {reason_hint}
"""
    return prompt

def build_adjusted_prompt_for_model(
    question: str,
    constraint: str,
    model_answer: str,
    subject: str,
    model_name: str,
    reference_answer: str = "",
    include_reason: bool = False,
    prompt_mode: Optional[str] = None,
    raw_constraints_text: str = "",
) -> str:
    """Build adjusted prompt for specific model (based on analysis report adjustment suggestions)"""
    
    # First build base prompt
    base_prompt = build_single_constraint_validation_prompt(
        question=question,
        constraint=constraint,
        model_answer=model_answer,
        subject=subject,
        reference_answer=reference_answer,
        include_reason=include_reason,
        prompt_mode=prompt_mode,
        raw_constraints_text=raw_constraints_text,
    )
    # Format / Fairness: do NOT apply judge-model-specific strictness/leniency adjustments.
    # The rubric itself already defines what is required; calibration should not depend on judge identity.
    return base_prompt


def build_multi_constraint_validation_prompt(
    question: str,
    constraints: List[str],
    model_answer: str,
    subject: str,
    reference_answer: str = "",
    include_reason: bool = False,
    prompt_mode: Optional[str] = None,
    raw_constraints_text: str = "",
) -> str:
    """Build comprehensive validation prompt for multiple constraints"""
    mode = (prompt_mode or VALIDATOR_PROMPT_MODE).lower()
    # Ensure mode is strict or loose, otherwise use strict as default
    if mode not in ["strict", "loose"]:
        mode = "strict"
    subject_name_map = {
        "physics": "Physics",
        "chemistry": "Chemistry",
        "biology": "Biology",
        "materials": "Materials",
        "protein": "Protein"
    }
    subject_name = subject_name_map.get(subject, subject)
    
    desc_map = _constraint_descriptions()
    normalized = [normalize_constraint_name(c) for c in constraints]
    constraint_lines = []
    for i, (c_raw, c_norm) in enumerate(zip(constraints, normalized), 1):
        desc = desc_map.get(c_norm, c_raw)
        constraint_lines.append(f"{i}. ({c_norm}) {desc}")
    constraints_text = "\n".join(constraint_lines)

    # few-shot: take 1 example for each involved constraint, at most 3 total
    examples = _load_oneshot_examples()
    shot_blocks = []
    for c_norm in normalized:
        if c_norm in examples and examples[c_norm]:
            shot = examples[c_norm][0]
            shot_blocks.append(
                f"[{c_norm}] Example Question:\n{shot.get('question','')}\nExample Answer:\n{shot.get('answer','')}\nExample Verdict: YES"
            )
        if len(shot_blocks) >= 3:
            break
    oneshot_block = "\n\n".join(shot_blocks)
    if oneshot_block:
        oneshot_block = "Few-shot examples (expected YES):\n" + oneshot_block + "\n"

    reason_hint = "Do not output any reason." if not include_reason else "If NO, optionally add one short reason (<=60 chars) after a new line starting with 'REASON:'."
    ref_block = f"Reference Answer (gold):\n{reference_answer}\n\n" if reference_answer else ""

    raw_block = f"Original constraints text (for reference):\n{raw_constraints_text}\n\n" if raw_constraints_text else ""

    if mode in ["strict", "loose"]:
        # Include strict/loose required-evidence bullets per constraint to avoid overall-judge leniency.
        req_blocks = []
        for c_norm in normalized:
            req_data = _strict_required_evidence(c_norm)
            req_text_item = req_data.get("formatted_text", "") if isinstance(req_data, dict) else req_data
            if req_text_item:
                req_blocks.append(f"[{c_norm}] Required evidence for YES:\n{req_text_item}")
        req_text = "\n\n".join(req_blocks).strip()
        if not req_text:
            req_text = "(No strict evidence rules found.)"

        mode_label = "STRICT" if mode == "strict" else "LOOSE"
        loose_note = ""
        if mode == "loose":
            loose_note = "\n- Note: In LOOSE mode, only MAIN points (marked as [MAIN - REQUIRED]) need to pass for each constraint, except for 'Applicability Range' which requires ALL points even in LOOSE mode."
        
        prompt = f"""You are an expert validator for university-level {subject_name} problems. Be {mode_label}.

Constraints (normalized) – ALL must be satisfied for YES:
{constraints_text}
{raw_block}

Per-constraint required evidence (ALL mandatory):
{req_text}

Decision policy ({mode_label}):
- Overall verdict = YES only if you can find explicit evidence for EVERY required-evidence bullet for EVERY listed constraint.
- Do NOT infer missing evidence. If uncertain, answer NO.
- If NO, report only the first blocking issue in REASON (<=80 chars).{loose_note}

Problem:
{question}

{ref_block}Model Answer:
{model_answer}

{oneshot_block}Output instructions:
- Respond ONLY with 'YES' or 'NO'.
- If NO, add one short reason (<=80 chars) on a new line starting with 'REASON:' (summarize the first blocking issue).
- Overall verdict = YES only if EVERY listed constraint is satisfied; otherwise NO.
"""
        return prompt

    prompt = f"""You are an expert validator for university-level {subject_name} problems. Decide if the model answer satisfies ALL listed constraints.

Constraints (normalized):
{constraints_text}
{raw_block}

Problem:
{question}

{ref_block}Model Answer:
{model_answer}

{oneshot_block}Output instructions:
- Respond with ONLY 'YES' or 'NO'.
- {reason_hint}
"""
    return prompt


def build_answer_correctness_prompt(
    question: str,
    reference_answer: str,
    model_answer: str,
    subject: str
) -> str:
    """Build answer correctness validation prompt (only compare answer content, not involving constraints)"""
    subject_name_map = {
        "physics": "Physics",
        "chemistry": "Chemistry",
        "biology": "Biology",
        "materials": "Materials",
        "protein": "Protein"
    }
    subject_name = subject_name_map.get(subject, subject)
    
    prompt = f"""You are an expert validator for university-level {subject_name} problems. Your task is to compare the model's answer with the reference answer to determine if they are equivalent.

Reference Answer:
{reference_answer}

Model's Answer:
{model_answer}

Please evaluate whether the model's answer is equivalent to the reference answer. Focus ONLY on:
1. Are the final numerical results the same (if applicable)?
2. Are the key conclusions the same?
3. Are the main concepts and reasoning equivalent?

Note: 
- You should NOT check constraint compliance (that is handled separately).
- You should ONLY compare the answer content itself.
- Minor formatting differences are acceptable, but the core content must be equivalent.

Please provide your evaluation in the following format:

**First, provide a simple verdict:**
YES or NO

If YES (the answers are equivalent):
- The model's answer is correct and equivalent to the reference answer.

If NO (the answers are different):
- Provide a brief explanation of the key differences.

Your response should start with either YES or NO on a separate line.
"""
    return prompt


def _parse_validation_result(validation_result: str, req_points: List[tuple]) -> Dict[str, Any]:
    """Parse validation result from single model, extract point validation and overall validation"""
    # Check if validation_result is None
    if validation_result is None:
        logger.warning(f"Model returned None response, unable to parse validation result")
        return {
            "point_validation": {},
            "overall_status": "UNKNOWN",
            "overall_verdict": "UNKNOWN",
            "overall_reason": "Model returned None response, unable to parse",
            "full_response": ""
        }
    
    # Ensure validation_result is a string
    if not isinstance(validation_result, str):
        logger.warning(f"Model returned non-string type response ({type(validation_result).__name__}), converting to string")
        validation_result = str(validation_result)
    
    point_validation = {}
    overall_status = "UNKNOWN"
    overall_verdict = "UNKNOWN"
    overall_reason = None
    
    current_mode = VALIDATOR_PROMPT_MODE.lower()
    # Ensure mode is strict or loose, otherwise use strict as default
    if current_mode not in ["strict", "loose"]:
        current_mode = "strict"
    
    # If points are defined, try to parse point-by-point validation
    if req_points and current_mode in ["strict", "loose"]:
        # Extract validation for each point
        for i, (pid, _) in enumerate(req_points, 1):
            patterns = [
                rf"POINT[_\s]*{i}\s*:\s*(YES|NO)(?:\s*\[([^\]]+)\])?",
                rf"Point[_\s]*{i}\s*:\s*(YES|NO)(?:\s*\[([^\]]+)\])?",
                rf"{i}\s*\.\s*(YES|NO)(?:\s*\[([^\]]+)\])?",
            ]
            verdict = None
            reason = None
            for pattern in patterns:
                match = re.search(pattern, validation_result, re.IGNORECASE | re.MULTILINE)
                if match:
                    verdict = match.group(1).upper()
                    if len(match.groups()) > 1 and match.group(2):
                        reason = match.group(2).strip()
                    break
            
            if verdict:
                point_validation[pid] = {
                    "status": "PASS" if verdict == "YES" else "FAIL",
                    "verdict": verdict,
                    "reason": reason
                }
            else:
                point_validation[pid] = {
                    "status": "UNKNOWN",
                    "verdict": "UNKNOWN",
                    "reason": None
                }
        
        # Extract overall validation
        overall_patterns = [
            r"OVERALL\s*:\s*(YES|NO)",
            r"Overall\s*:\s*(YES|NO)",
            r"Final\s*:\s*(YES|NO)",
        ]
        for pattern in overall_patterns:
            match = re.search(pattern, validation_result, re.IGNORECASE)
            if match:
                overall_verdict = match.group(1).upper()
                overall_status = "PASS" if overall_verdict == "YES" else "FAIL"
                break
        
        # Extract overall reason
        reason_pattern = r"OVERALL_REASON\s*:\s*\[([^\]]+)\]"
        reason_match = re.search(reason_pattern, validation_result, re.IGNORECASE)
        if reason_match:
            overall_reason = reason_match.group(1).strip()
    
    # If no point-by-point validation or parsing failed, use old overall validation logic
    if overall_status == "UNKNOWN":
        # If response is empty, log warning
        if not validation_result or not validation_result.strip():
            logger.warning(f"Model returned empty response, unable to parse validation result")
            return {
                "point_validation": point_validation,
                "overall_status": "UNKNOWN",
                "overall_verdict": "UNKNOWN",
                "overall_reason": "Model returned empty response, unable to parse",
                "full_response": validation_result
            }
        
        yes_no_match = re.search(r'^\s*(YES|NO)\s*$|^\*\*.*?:\s*(YES|NO)\s*$|YES|NO', 
                                validation_result.strip(), re.IGNORECASE | re.MULTILINE)
        if yes_no_match:
            overall_verdict = yes_no_match.group(0).upper().strip()
            overall_status = "PASS" if "YES" in overall_verdict else "FAIL"
            overall_verdict = "YES" if overall_status == "PASS" else "NO"
    
    return {
        "point_validation": point_validation,
        "overall_status": overall_status,
        "overall_verdict": overall_verdict,
        "overall_reason": overall_reason,
        "full_response": validation_result
    }


def _vote_on_points(
    point_results: List[Dict[str, Any]], 
    req_points: List[tuple]
) -> Dict[str, Any]:
    """Vote on point validation results from multiple models"""
    voted_points = {}
    
    for pid, _ in req_points:
        votes = []
        reasons = []
        for result in point_results:
            pv = result.get("point_validation", {}).get(pid, {})
            status = pv.get("status", "UNKNOWN")
            votes.append(status)
            if pv.get("reason"):
                reasons.append(pv.get("reason"))
        
        # Count votes
        pass_count = votes.count("PASS")
        fail_count = votes.count("FAIL")
        unknown_count = votes.count("UNKNOWN")
        total_judges = len(votes)
        
        # Strict voting: must all pass to be PASS, one wrong is FAIL
        if pass_count == total_judges:
            # All judges pass
            final_status = "PASS"
            final_verdict = "YES"
        elif fail_count > 0 or unknown_count > 0:
            # Any FAIL or UNKNOWN is FAIL
            final_status = "FAIL"
            final_verdict = "NO"
        else:
            # Other cases, mark as UNKNOWN
            final_status = "UNKNOWN"
            final_verdict = "UNKNOWN"
        
        # Collect all reasons, merge
        all_reasons = [r for r in reasons if r]
        reason_text = "; ".join(all_reasons) if all_reasons else None
        
        voted_points[pid] = {
            "status": final_status,
            "verdict": final_verdict,
            "reason": reason_text,
            "votes": {
                "PASS": pass_count,
                "FAIL": fail_count,
                "UNKNOWN": unknown_count
            }
        }
    
    return voted_points


def _vote_on_overall(
    overall_results: List[Dict[str, Any]]
) -> tuple:
    """Vote on overall validation results from multiple models"""
    votes = []
    reasons = []
    
    for result in overall_results:
        status = result.get("overall_status", "UNKNOWN")
        votes.append(status)
        if result.get("overall_reason"):
            reasons.append(result.get("overall_reason"))
    
    # Count votes
    pass_count = votes.count("PASS")
    fail_count = votes.count("FAIL")
    unknown_count = votes.count("UNKNOWN")
    total_judges = len(votes)
    
    # Strict voting: must all pass to be PASS, one wrong is FAIL
    if pass_count == total_judges:
        # All judges pass
        final_status = "PASS"
        final_verdict = "YES"
    elif fail_count > 0 or unknown_count > 0:
        # Any FAIL or UNKNOWN is FAIL
        final_status = "FAIL"
        final_verdict = "NO"
    else:
        final_status = "UNKNOWN"
        final_verdict = "UNKNOWN"
    
    # Merge all reasons
    all_reasons = [r for r in reasons if r]
    reason_text = "; ".join(all_reasons) if all_reasons else None
    
    return final_status, final_verdict, reason_text, {
        "PASS": pass_count,
        "FAIL": fail_count,
        "UNKNOWN": unknown_count
    }


async def validate_single_constraint_async(
    question: str,
    constraint: str,
    answer: str,
    subject: str,
    reference_answer: str = "",
    include_reason: bool = False,
    validator_model: str = "gpt-5",
    timeout: int = 6000,
    raw_constraints_text: str = "",
    use_multi_judge: bool = False,
    judge_models: Optional[List[str]] = None,
    run_responses_api_async_func = None,  # Pass async API call function
    test_model_name: str = "",  # Compatible with caller parameters (currently not used)
    **_ignored_kwargs,
) -> Dict[str, Any]:
    """Asynchronously validate single constraint (supports point-by-point validation and multi-model voting)
    
    Args:
        use_multi_judge: Whether to use multi-model voting mechanism
        judge_models: List of models for voting, default is ["gemini-3-flash", "gpt-5.1"]
        run_responses_api_async_func: Async API call function
    """
    if run_responses_api_async_func is None:
        raise ValueError("run_responses_api_async_func must be provided")
    
    # If using multi-model voting, asynchronously call multiple models in parallel
    if use_multi_judge:
        if judge_models is None:
            judge_models = ["gemini-3-flash", "gpt-5.1"]
        
        normalized_constraint = normalize_constraint_name(constraint)
        req_data = _strict_required_evidence(normalized_constraint)
        req_points = req_data.get("points", []) if isinstance(req_data, dict) else []
        
        # Asynchronously call multiple models in parallel, each model uses adjusted prompt
        async def judge_with_model_async(model_name: str) -> Dict[str, Any]:
            try:
                # Build adjusted prompt for each model
                prompt = build_adjusted_prompt_for_model(
                    question=question,
                    constraint=constraint,
                    model_answer=answer,
                    subject=subject,
                    model_name=model_name,
                    reference_answer=reference_answer,
                    include_reason=include_reason,
                    prompt_mode=VALIDATOR_PROMPT_MODE,
                    raw_constraints_text=raw_constraints_text,
                )
                
                validation_result = await run_responses_api_async_func(
                    model=model_name,
                    input_text=prompt,
                    timeout=timeout
                )
                return _parse_validation_result(validation_result, req_points)
            except Exception as e:
                logger.error(f"Model {model_name} validation failed: {e}")
                return {
                    "point_validation": {pid: {"status": "UNKNOWN", "verdict": "UNKNOWN", "reason": None} for pid, _ in req_points} if req_points else {},
                    "overall_status": "UNKNOWN",
                    "overall_verdict": "UNKNOWN",
                    "overall_reason": f"Error: {str(e)}",
                    "full_response": ""
                }
        
        # Use asyncio.gather to execute all judges in parallel (fully asynchronous)
        judge_tasks = [judge_with_model_async(model) for model in judge_models]
        results = await asyncio.gather(*judge_tasks, return_exceptions=True)
        
        # Process results
        model_results_map = {}
        for i, result in enumerate(results):
            model_name = judge_models[i]
            if isinstance(result, Exception):
                logger.error(f"Judge task execution failed ({model_name}): {result}")
                error_result = {
                    "point_validation": {pid: {"status": "UNKNOWN", "verdict": "UNKNOWN", "reason": None} for pid, _ in req_points} if req_points else {},
                    "overall_status": "UNKNOWN",
                    "overall_verdict": "UNKNOWN",
                    "overall_reason": f"Error: {str(result)}",
                    "full_response": "",
                    "model_name": model_name
                }
                results[i] = error_result
                model_results_map[model_name] = error_result
            else:
                result["model_name"] = model_name
                model_results_map[model_name] = result
        
        # Vote on points
        voted_points = _vote_on_points(results, req_points) if req_points else {}
        
        # Vote on overall
        overall_status, overall_verdict, overall_reason, overall_votes = _vote_on_overall(results)
        
        # If there is point-by-point validation, recalculate overall based on point voting results (if overall is UNKNOWN)
        if voted_points and overall_status == "UNKNOWN":
            current_mode = VALIDATOR_PROMPT_MODE.lower()
            # Ensure mode is strict or loose, otherwise use strict as default
            if current_mode not in ["strict", "loose"]:
                current_mode = "strict"
            if current_mode == "strict":
                # strict mode: all points must pass
                point_statuses = [pv.get("status") for pv in voted_points.values()]
                if all(s == "PASS" for s in point_statuses):
                    overall_status = "PASS"
                    overall_verdict = "YES"
                elif any(s == "FAIL" for s in point_statuses):
                    overall_status = "FAIL"
                    overall_verdict = "NO"
        
        # Build detailed validation results for each model
        individual_judge_results = {}
        for model_name in judge_models:
            model_result = model_results_map.get(model_name, {})
            individual_judge_results[model_name] = {
                "overall": {
                    "status": model_result.get("overall_status", "UNKNOWN"),
                    "verdict": model_result.get("overall_verdict", "UNKNOWN"),
                    "reason": model_result.get("overall_reason")
                },
                "point_validation": model_result.get("point_validation", {}),
                "full_response": model_result.get("full_response", "")
            }
        
        result = {
            "status": overall_status,
            "verdict": overall_verdict,
            "full_response": f"[Multi-judge results from {', '.join(judge_models)}]",
            "judge_votes": {
                "overall": overall_votes,
                "models": judge_models
            },
            "individual_judges": individual_judge_results  # Save detailed validation results for each model
        }
        
        if voted_points:
            result["point_validation"] = voted_points
        
        if overall_reason:
            result["reason"] = overall_reason
        
        return result
    
    # Original single model logic (async version)
    prompt = build_single_constraint_validation_prompt(
        question,
        constraint,
        answer,
        subject,
        reference_answer=reference_answer,
        include_reason=include_reason,
        prompt_mode=VALIDATOR_PROMPT_MODE,
        raw_constraints_text=raw_constraints_text,
    )
    
    try:
        validation_result = await run_responses_api_async_func(
            model=validator_model,
            input_text=prompt,
            timeout=timeout
        )
        
        normalized_constraint = normalize_constraint_name(constraint)
        req_data = _strict_required_evidence(normalized_constraint)
        req_points = req_data.get("points", []) if isinstance(req_data, dict) else []
        
        point_validation = {}
        overall_status = "UNKNOWN"
        overall_verdict = "UNKNOWN"
        overall_reason = None
        
        # Parse results (same as sync version)
        if req_points and VALIDATOR_PROMPT_MODE.lower() in ["strict", "loose"]:
            parsed = _parse_validation_result(validation_result, req_points)
            point_validation = parsed.get("point_validation", {})
            overall_status = parsed.get("overall_status", "UNKNOWN")
            overall_verdict = parsed.get("overall_verdict", "UNKNOWN")
            overall_reason = parsed.get("overall_reason")
        else:
            # Simple mode: only check YES/NO
            yes_no_match = re.search(r'^\s*(YES|NO)\s*$|^\*\*.*?:\s*(YES|NO)\s*$|YES|NO', 
                                    validation_result.strip(), re.IGNORECASE | re.MULTILINE)
            if yes_no_match:
                verdict = yes_no_match.group(0).upper().strip()
                overall_status = "PASS" if "YES" in verdict else "FAIL"
                overall_verdict = "YES" if overall_status == "PASS" else "NO"
        
        result = {
            "status": overall_status,
            "verdict": overall_verdict,
            "full_response": validation_result
        }
        
        if point_validation:
            result["point_validation"] = point_validation
        
        if overall_reason:
            result["reason"] = overall_reason
        
        return result
        
    except Exception as e:
        logger.error(f"constraint validation failed: {e}")
        return {
            "status": "ERROR",
            "verdict": "UNKNOWN",
            "message": str(e),
            "full_response": ""
        }


def validate_single_constraint(
    question: str,
    constraint: str,
    answer: str,
    subject: str,
    reference_answer: str = "",
    include_reason: bool = False,
    validator_model: str = "gpt-5",
    timeout: int = 6000,
    raw_constraints_text: str = "",
    use_multi_judge: bool = False,
    judge_models: Optional[List[str]] = None,
    test_model_name: str = "",  # Compatible with caller parameters (currently not used)
    **_ignored_kwargs,
) -> Dict[str, Any]:
    """Validate single constraint (supports point-by-point validation and multi-model voting)
    
    Args:
        use_multi_judge: Whether to use multi-model voting mechanism
        judge_models: List of models for voting, default is ["gemini-3-flash", "gpt-5.1"]
    """
    # If using multi-model voting, call multiple models in parallel
    if use_multi_judge:
        if judge_models is None:
            judge_models = ["gemini-3-flash", "gpt-5.1"]
        
        prompt = build_single_constraint_validation_prompt(
            question,
            constraint,
            answer,
            subject,
            reference_answer=reference_answer,
            include_reason=include_reason,
            prompt_mode=VALIDATOR_PROMPT_MODE,
            raw_constraints_text=raw_constraints_text,
        )
        
        normalized_constraint = normalize_constraint_name(constraint)
        req_data = _strict_required_evidence(normalized_constraint)
        req_points = req_data.get("points", []) if isinstance(req_data, dict) else []
        
        # Call multiple models in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def judge_with_model(model_name: str) -> Dict[str, Any]:
            try:
                validation_result = run_responses_api(
                    model=model_name,
                    input_text=prompt,
                    timeout=timeout
                )
                return _parse_validation_result(validation_result, req_points)
            except Exception as e:
                logger.error(f"Model {model_name} validation failed: {e}")
                return {
                    "point_validation": {pid: {"status": "UNKNOWN", "verdict": "UNKNOWN", "reason": None} for pid, _ in req_points} if req_points else {},
                    "overall_status": "UNKNOWN",
                    "overall_verdict": "UNKNOWN",
                    "overall_reason": f"Error: {str(e)}",
                    "full_response": ""
                }
        
        # Execute all judges in parallel
        with ThreadPoolExecutor(max_workers=len(judge_models)) as executor:
            future_to_model = {executor.submit(judge_with_model, model): model for model in judge_models}
            results = []
            model_results_map = {}  # Save detailed results for each model
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    result["model_name"] = model_name  # Add model name
                    results.append(result)
                    model_results_map[model_name] = result
                except Exception as e:
                    logger.error(f"⚠️  Judge task execution failed: {e}")
                    error_result = {
                        "point_validation": {pid: {"status": "UNKNOWN", "verdict": "UNKNOWN", "reason": None} for pid, _ in req_points} if req_points else {},
                        "overall_status": "UNKNOWN",
                        "overall_verdict": "UNKNOWN",
                        "overall_reason": f"Error: {str(e)}",
                        "full_response": "",
                        "model_name": model_name
                    }
                    results.append(error_result)
                    model_results_map[model_name] = error_result
        
        # Vote on points
        voted_points = _vote_on_points(results, req_points) if req_points else {}
        
        # Vote on overall
        overall_status, overall_verdict, overall_reason, overall_votes = _vote_on_overall(results)
        
        # If there is point-by-point validation, recalculate overall based on point voting results (if overall is UNKNOWN)
        if voted_points and overall_status == "UNKNOWN":
            current_mode = VALIDATOR_PROMPT_MODE.lower()
            # Ensure mode is strict or loose, otherwise use strict as default
            if current_mode not in ["strict", "loose"]:
                current_mode = "strict"
            if current_mode == "strict":
                # strict mode: all points must pass
                point_statuses = [pv.get("status") for pv in voted_points.values()]
                if all(s == "PASS" for s in point_statuses):
                    overall_status = "PASS"
                    overall_verdict = "YES"
                elif any(s == "FAIL" for s in point_statuses):
                    overall_status = "FAIL"
                    overall_verdict = "NO"
        
        # Build detailed validation results for each model
        individual_judge_results = {}
        for model_name in judge_models:
            model_result = model_results_map.get(model_name, {})
            individual_judge_results[model_name] = {
                "overall": {
                    "status": model_result.get("overall_status", "UNKNOWN"),
                    "verdict": model_result.get("overall_verdict", "UNKNOWN"),
                    "reason": model_result.get("overall_reason")
                },
                "point_validation": model_result.get("point_validation", {}),
                "full_response": model_result.get("full_response", "")
            }
        
        result = {
            "status": overall_status,
            "verdict": overall_verdict,
            "full_response": f"[Multi-judge results from {', '.join(judge_models)}]",
            "judge_votes": {
                "overall": overall_votes,
                "models": judge_models
            },
            "individual_judges": individual_judge_results  # Save detailed validation results for each model
        }
        
        if voted_points:
            result["point_validation"] = voted_points
        
        if overall_reason:
            result["reason"] = overall_reason
        
        return result
    
    # Original single model logic
    prompt = build_single_constraint_validation_prompt(
        question,
        constraint,
        answer,
        subject,
        reference_answer=reference_answer,
        include_reason=include_reason,
        prompt_mode=VALIDATOR_PROMPT_MODE,
        raw_constraints_text=raw_constraints_text,
    )
    
    # Check if prompt is empty
    if not prompt or not prompt.strip():
        logger.error(f"Built prompt is empty, cannot call API")
        return {
            "status": "ERROR",
            "verdict": "UNKNOWN",
            "message": "Built prompt is empty",
            "full_response": ""
        }
    
    try:
        validation_result = run_responses_api(
            model=validator_model,
            input_text=prompt,
            timeout=timeout
        )
        
        # Check if validation_result is None or if it is a string
        if validation_result is None:
            logger.warning(f"API returned None response, unable to parse validation result")
            return {
                "status": "ERROR",
                "verdict": "UNKNOWN",
                "message": "API returned None response",
                "full_response": ""
            }
        
        # Ensure validation_result is a string
        if not isinstance(validation_result, str):
            logger.warning(f"API returned non-string type response ({type(validation_result).__name__}), converting to string")
            validation_result = str(validation_result)
        
        normalized_constraint = normalize_constraint_name(constraint)
        req_data = _strict_required_evidence(normalized_constraint)
        req_points = req_data.get("points", []) if isinstance(req_data, dict) else []
        
        point_validation = {}
        overall_status = "UNKNOWN"
        overall_verdict = "UNKNOWN"
        overall_reason = None
        
        # Get current mode
        current_mode = VALIDATOR_PROMPT_MODE.lower()
        # Ensure mode is strict or loose, otherwise use strict as default
        if current_mode not in ["strict", "loose"]:
            current_mode = "strict"
        
    # If points are defined, try to parse point-by-point validation
        if req_points and current_mode in ["strict", "loose"]:
            # Extract validation for each point
            for i, (pid, _) in enumerate(req_points, 1):
                # Try multiple formats: POINT_1: YES, POINT_1:NO, POINT 1: YES, etc.
                patterns = [
                    rf"POINT[_\s]*{i}\s*:\s*(YES|NO)(?:\s*\[([^\]]+)\])?",
                    rf"Point[_\s]*{i}\s*:\s*(YES|NO)(?:\s*\[([^\]]+)\])?",
                    rf"{i}\s*\.\s*(YES|NO)(?:\s*\[([^\]]+)\])?",
                ]
                verdict = None
                reason = None
                for pattern in patterns:
                    match = re.search(pattern, validation_result, re.IGNORECASE | re.MULTILINE)
                    if match:
                        verdict = match.group(1).upper()
                        if len(match.groups()) > 1 and match.group(2):
                            reason = match.group(2).strip()
                        break
                
                if verdict:
                    point_validation[pid] = {
                        "status": "PASS" if verdict == "YES" else "FAIL",
                        "verdict": verdict,
                        "reason": reason
                    }
                else:
                    # If not found, mark as UNKNOWN
                    point_validation[pid] = {
                        "status": "UNKNOWN",
                        "verdict": "UNKNOWN",
                        "reason": None
                    }
            
            # Extract overall validation
            overall_patterns = [
                r"OVERALL\s*:\s*(YES|NO)",
                r"Overall\s*:\s*(YES|NO)",
                r"Final\s*:\s*(YES|NO)",
            ]
            for pattern in overall_patterns:
                match = re.search(pattern, validation_result, re.IGNORECASE)
                if match:
                    overall_verdict = match.group(1).upper()
                    overall_status = "PASS" if overall_verdict == "YES" else "FAIL"
                    break
            
            # Extract overall reason
            reason_pattern = r"OVERALL_REASON\s*:\s*\[([^\]]+)\]"
            reason_match = re.search(reason_pattern, validation_result, re.IGNORECASE)
            if reason_match:
                overall_reason = reason_match.group(1).strip()
            
            # If OVERALL is not found but all point validations are found, derive from point validations
            if overall_status == "UNKNOWN" and point_validation:
                # Decide validation logic according to mode
                if current_mode == "strict":
                    # strict mode: all points must pass
                    point_statuses = [pv.get("status") for pv in point_validation.values()]
                    if all(s == "PASS" for s in point_statuses):
                        overall_status = "PASS"
                        overall_verdict = "YES"
                    elif any(s == "FAIL" for s in point_statuses):
                        overall_status = "FAIL"
                        overall_verdict = "NO"
                else:  # loose mode
                    # Get key point definitions for loose mode
                    loose_config = LOOSE_KEY_POINTS.get(normalized_constraint, {})
                    main_points = set(loose_config.get("main", []))
                    
                    # For "Applicability Range", in loose mode all points must also pass
                    if normalized_constraint == "Applicability Range":
                        point_statuses = [pv.get("status") for pv in point_validation.values()]
                        if all(s == "PASS" for s in point_statuses):
                            overall_status = "PASS"
                            overall_verdict = "YES"
                        elif any(s == "FAIL" for s in point_statuses):
                            overall_status = "FAIL"
                            overall_verdict = "NO"
                    else:
                        # loose mode: only check main points
                        main_point_statuses = [
                            point_validation[pid].get("status")
                            for pid in main_points
                            if pid in point_validation
                        ]
                        if main_point_statuses:
                            if all(s == "PASS" for s in main_point_statuses):
                                overall_status = "PASS"
                                overall_verdict = "YES"
                            elif any(s == "FAIL" for s in main_point_statuses):
                                overall_status = "FAIL"
                                overall_verdict = "NO"
                        else:
                            # If main points validation is not found, fall back to all points
                            point_statuses = [pv.get("status") for pv in point_validation.values()]
                            if all(s == "PASS" for s in point_statuses):
                                overall_status = "PASS"
                                overall_verdict = "YES"
                            elif any(s == "FAIL" for s in point_statuses):
                                overall_status = "FAIL"
                                overall_verdict = "NO"
        
        # If no point-by-point validation or parsing failed, use old overall validation logic
        if overall_status == "UNKNOWN":
            yes_no_match = re.search(r'^\s*(YES|NO)\s*$|^\*\*.*?:\s*(YES|NO)\s*$|YES|NO', 
                                    validation_result.strip(), re.IGNORECASE | re.MULTILINE)
            if yes_no_match:
                overall_verdict = yes_no_match.group(0).upper().strip()
                overall_status = "PASS" if "YES" in overall_verdict else "FAIL"
                overall_verdict = "YES" if overall_status == "PASS" else "NO"
        
        result = {
            "status": overall_status,
            "verdict": overall_verdict,
            "full_response": validation_result
        }
        
        # If there are point-by-point validation results, add to result
        if point_validation:
            result["point_validation"] = point_validation
        
        # If there is overall reason, add to result
        if overall_reason:
            result["reason"] = overall_reason
        
        return result
    except Exception as e:
        logger.error(f"constraint validation failed: {e}")
        return {
            "status": "ERROR",
            "message": str(e)
        }


def validate_multi_constraints(
    question: str,
    constraints: List[str],
    answer: str,
    subject: str,
    reference_answer: str = "",
    include_reason: bool = False,
    validator_model: str = "gpt-5",
    timeout: int = 6000,
    raw_constraints_text: str = "",
    use_multi_judge: bool = False,
    judge_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate comprehensive situation of multiple constraints (supports multi-model voting)
    
    Args:
        use_multi_judge: Whether to use multi-model voting mechanism
        judge_models: List of models for voting, default is ["gemini-3-flash", "gpt-5.1"]
    """
    prompt = build_multi_constraint_validation_prompt(
        question,
        constraints,
        answer,
        subject,
        reference_answer=reference_answer,
        include_reason=include_reason,
        prompt_mode=VALIDATOR_PROMPT_MODE,
        raw_constraints_text=raw_constraints_text,
    )
    
    # Check if prompt is empty
    if not prompt or not prompt.strip():
        logger.error(f"Built prompt is empty, cannot call API")
        return {
            "status": "ERROR",
            "verdict": "UNKNOWN",
            "message": "Built prompt is empty",
            "full_response": ""
        }
    
    # If using multi-model voting, call multiple models in parallel
    if use_multi_judge:
        if judge_models is None:
            judge_models = ["gemini-3-flash", "gpt-5.1"]
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def judge_with_model(model_name: str) -> Dict[str, Any]:
            try:
                validation_result = run_responses_api(
                    model=model_name,
                    input_text=prompt,
                    timeout=timeout
                )
                return _parse_multi_constraint_result(validation_result, constraints)
            except Exception as e:
                logger.error(f"Model {model_name} validation failed: {e}")
                return {
                    "status": "UNKNOWN",
                    "verdict": "UNKNOWN",
                    "per_constraint": {},
                    "full_response": f"Error: {str(e)}",
                    "model_name": model_name
                }
        
        # Execute all judges in parallel
        with ThreadPoolExecutor(max_workers=len(judge_models)) as executor:
            future_to_model = {executor.submit(judge_with_model, model): model for model in judge_models}
            results = []
            model_results_map = {}
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    result["model_name"] = model_name
                    results.append(result)
                    model_results_map[model_name] = result
                except Exception as e:
                    logger.error(f"⚠️  Judge task execution failed: {e}")
                    error_result = {
                        "status": "UNKNOWN",
                        "verdict": "UNKNOWN",
                        "per_constraint": {},
                        "full_response": f"Error: {str(e)}",
                        "model_name": model_name
                    }
                    results.append(error_result)
                    model_results_map[model_name] = error_result
        
        # Vote on overall
        overall_status, overall_verdict, overall_reason, overall_votes = _vote_on_overall(results)
        
        # Vote on per_constraint
        voted_per_constraint = {}
        if results and results[0].get("per_constraint"):
            # Collect per_constraint results from all models
            all_per_constraint = {}
            for result in results:
                per_const = result.get("per_constraint", {})
                for constraint_name, status in per_const.items():
                    if constraint_name not in all_per_constraint:
                        all_per_constraint[constraint_name] = []
                    all_per_constraint[constraint_name].append(status)
            
            # Vote on each constraint
            for constraint_name, statuses in all_per_constraint.items():
                pass_count = sum(1 for s in statuses if s == "PASS")
                fail_count = sum(1 for s in statuses if s == "FAIL")
                if pass_count > fail_count:
                    voted_per_constraint[constraint_name] = "PASS"
                elif fail_count > pass_count:
                    voted_per_constraint[constraint_name] = "FAIL"
                else:
                    voted_per_constraint[constraint_name] = "UNKNOWN"
        
        return {
            "status": overall_status,
            "verdict": overall_verdict,
            "per_constraint": voted_per_constraint,
            "full_response": f"[Multi-judge results from {', '.join(judge_models)}]",
            "multi_judge": True,
            "judge_results": {
                "models": judge_models,
                "votes": overall_votes,
                "model_results": model_results_map
            }
        }
    
    try:
        validation_result = run_responses_api(
            model=validator_model,
            input_text=prompt,
            timeout=timeout
        )
        
        # Parse YES/NO
        yes_no_match = re.search(r'^\s*(YES|NO)\s*$|^\*\*.*?:\s*(YES|NO)\s*$|YES|NO',
                                validation_result.strip(), re.IGNORECASE | re.MULTILINE)
        overall = None
        if yes_no_match:
            verdict = yes_no_match.group(0).upper().strip()
            overall = "PASS" if "YES" in verdict else "FAIL"

        # Additionally parse individual constraint validation (optional, if model returns per-constraint)
        per_constraint = {}
        for c in constraints:
            norm = normalize_constraint_name(c)
            # Try to find pattern like "[constraint] YES/NO"
            m = re.search(rf"\[{re.escape(norm)}\]\s*(YES|NO)", validation_result, re.IGNORECASE)
            if m:
                per_constraint[norm] = "PASS" if m.group(1).upper() == "YES" else "FAIL"
        # If no individual results found, apply overall result to each by default
        if overall and not per_constraint:
            for c in constraints:
                norm = normalize_constraint_name(c)
                per_constraint[norm] = overall

        # If there are per-constraint results, derive overall from per-constraint results:
        # Any FAIL is overall FAIL; all PASS is overall PASS; otherwise keep UNKNOWN
        if per_constraint:
            vals = list(per_constraint.values())
            if all(v == "PASS" for v in vals):
                overall = "PASS"
            elif any(v == "FAIL" for v in vals):
                overall = "FAIL"

        return {
            "status": overall or "UNKNOWN",
            "verdict": "YES" if overall == "PASS" else ("NO" if overall == "FAIL" else "UNKNOWN"),
            "per_constraint": per_constraint,
            "full_response": validation_result
        }
    except Exception as e:
        logger.error(f"constraint validation failed: {e}")
        return {
            "status": "ERROR",
            "verdict": "UNKNOWN",
            "message": str(e),
            "full_response": "",
            "per_constraint": {}
        }


def _parse_multi_constraint_result(validation_result: str, constraints: List[str]) -> Dict[str, Any]:
    """Parse multi-constraint validation result"""
    # Parse YES/NO
    yes_no_match = re.search(r'^\s*(YES|NO)\s*$|^\*\*.*?:\s*(YES|NO)\s*$|YES|NO',
                            validation_result.strip(), re.IGNORECASE | re.MULTILINE)
    overall = None
    if yes_no_match:
        verdict = yes_no_match.group(0).upper().strip()
        overall = "PASS" if "YES" in verdict else "FAIL"

    # Additionally parse individual constraint validation (optional, if model returns per-constraint)
    per_constraint = {}
    for c in constraints:
        norm = normalize_constraint_name(c)
        # Try to find pattern like "[constraint] YES/NO"
        m = re.search(rf"\[{re.escape(norm)}\]\s*(YES|NO)", validation_result, re.IGNORECASE)
        if m:
            per_constraint[norm] = "PASS" if m.group(1).upper() == "YES" else "FAIL"
    # If no individual results found, apply overall result to each by default
    if overall and not per_constraint:
        for c in constraints:
            norm = normalize_constraint_name(c)
            per_constraint[norm] = overall

    # If there are per-constraint results, derive overall from per-constraint results:
    # Any FAIL is overall FAIL; all PASS is overall PASS; otherwise keep UNKNOWN
    if per_constraint:
        vals = list(per_constraint.values())
        if all(v == "PASS" for v in vals):
            overall = "PASS"
        elif any(v == "FAIL" for v in vals):
            overall = "FAIL"

    return {
        "status": overall or "UNKNOWN",
        "verdict": "YES" if overall == "PASS" else ("NO" if overall == "FAIL" else "UNKNOWN"),
        "per_constraint": per_constraint,
        "full_response": validation_result
        }


def validate_answer_correctness(
    question: str,
    reference_answer: str,
    model_answer: str,
    subject: str,
    validator_model: str = "gpt-5",
    timeout: int = 6000
) -> Dict[str, Any]:
    """Validate answer correctness"""
    prompt = build_answer_correctness_prompt(question, reference_answer, model_answer, subject)
    
    # Check if prompt is empty
    if not prompt or not prompt.strip():
        logger.error(f"Built prompt is empty, cannot call API")
        return {
            "status": "ERROR",
            "verdict": "UNKNOWN",
            "message": "Built prompt is empty",
            "full_response": ""
        }
    
    try:
        validation_result = run_responses_api(
            model=validator_model,
            input_text=prompt,
            timeout=timeout
        )
        
        # Parse YES/NO
        yes_no_match = re.search(r'^\s*(YES|NO)\s*$|^\*\*.*?:\s*(YES|NO)\s*$|YES|NO', 
                                validation_result.strip(), re.IGNORECASE | re.MULTILINE)
        if yes_no_match:
            verdict = yes_no_match.group(0).upper().strip()
            status = "PASS" if "YES" in verdict else "FAIL"
            return {
                "status": status,
                "verdict": "YES" if status == "PASS" else "NO",
                "full_response": validation_result
            }
        
        return {
            "status": "UNKNOWN",
            "verdict": "UNKNOWN",
            "full_response": validation_result
        }
    except Exception as e:
        logger.error(f"Answer correctness validation failed: {e}")
        return {
            "status": "ERROR",
            "message": str(e)
        }


def evaluate_problem(
    problem: Dict[str, Any],
    model_name: str,
    validator_model: str = "gpt-5",
    timeout: int = 6000,
    answer_cache: Optional[Dict[str, str]] = None,
    cache_file: Optional[str] = None,
    progress_bar: Optional[tqdm] = None,
    problem_id: Optional[str] = None,
    use_multi_judge: bool = False,
    judge_models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Evaluate single problem
    
    Args:
        problem: Problem dictionary, may contain problem_id field
        model_name: Model name
        validator_model: Validator model name
        timeout: Timeout duration
        answer_cache: Answer cache dictionary
        cache_file: Cache file path
        progress_bar: Progress bar object
        problem_id: Problem ID (if not provided, will be read from problem dict or generated using default method)
        use_multi_judge: Whether to use multi-model voting mechanism
        judge_models: List of models for voting (default: ['gemini-3-flash', 'gpt-5.1'])
    """
    # Support multiple formats:
    # - refined_corpus.json: question/answer separated
    # - sciQA_if_full.json: question_with_constraints is whole text (contains question + constraints)
    # - Old format: text field
    question = problem.get("question", "")
    raw_qwc = problem.get("question_with_constraints", "")
    # Original constraint string/structure (used to parse "constraint name list" for judge)
    raw_constraints = problem.get("constraints", "")
    constraints = parse_constraints(raw_constraints)
    constraints_str = raw_constraints if isinstance(raw_constraints, str) else ""
    # Record raw constraint text for prompt direct reference (prefer detailed constraints in question_with_constraints)
    raw_constraints_text = ""
    if isinstance(raw_constraints, str):
        raw_constraints_text = raw_constraints
    elif isinstance(raw_constraints, list):
        raw_constraints_text = json.dumps(raw_constraints, ensure_ascii=False)
    elif isinstance(raw_constraints, dict):
        raw_constraints_text = json.dumps(raw_constraints, ensure_ascii=False)

    # Store normalized constraint list for later use
    problem["_normalized_constraints"] = constraints
    problem["_constraints_str"] = constraints_str
    problem["_raw_constraints_text"] = raw_constraints_text
    reference_answer = problem.get("answer", "")
    subject = problem.get("subject", "unknown")
    
    # Get problem_id (prefer parameter, then from problem dict, finally use default generation)
    if not problem_id:
        problem_id = problem.get("problem_id")
    
    # Extract "pure question" for judge/validator (do not mix constraints in), but still keep raw full text for answer model
    # 1) Prefer parsing from question_with_constraints (also get detailed constraints text for judge reference)
    if raw_qwc and isinstance(raw_qwc, str):
        parsed = parse_text_field(raw_qwc)
        if parsed.get("question") and not question:
            question = parsed["question"]
        if parsed.get("answer") and not reference_answer:
            reference_answer = parsed["answer"]
        # Use detailed constraints from question_with_constraints to override raw_constraints_text (only for "reference text", does not affect constraint name parsing)
        if parsed.get("constraints"):
            raw_constraints_text = parsed["constraints"]
    # 2) If no question field, try parsing from text field
    if not question:
        text = problem.get("text", "")
        if text:
            if "question:" in text.lower():
                parts = re.split(r'(?i)(question|constraints|answer)[:：]\s*', text)
                for i in range(1, len(parts), 2):
                    key = parts[i].lower()
                    if i + 1 < len(parts):
                        value = parts[i + 1].strip()
                        if i + 2 < len(parts):
                            next_key_idx = value.lower().find(parts[i + 2].lower() + ":")
                            if next_key_idx > 0:
                                value = value[:next_key_idx].strip()
                        if key == "question":
                            question = value
                        elif key == "answer" and not reference_answer:
                            reference_answer = value
    
    if not question:
        return {
            "status": "ERROR",
            "message": "Problem missing"
        }
    
    # 1. Let model answer question (use cache)
    logger.debug(f"  Using model to generate answer...")
    # Route A: For the model under test input, try to use "original full text" (question_with_constraints / text), do not split it.
    # judge/validator Problem still uses pure question extracted above.
    raw_for_answer = ""
    if isinstance(raw_qwc, str) and raw_qwc.strip():
        raw_for_answer = raw_qwc
    else:
        raw_for_answer = problem.get("text", "") or question

    answer_prompt = build_answer_prompt(raw_for_answer, constraints_str)
    
    # Check cache (use problem_id to generate cache key, ensure consistency with refined_corpus.json and evaluation_results.jsonl IDs)
    cache_key = get_answer_cache_key(raw_for_answer, constraints_str, model_name, problem_id=problem_id)
    model_answer = None
    
    if answer_cache and cache_key in answer_cache:
        model_answer = answer_cache[cache_key]
        logger.debug(f"  Loaded answer from cache")
        if progress_bar:
            progress_bar.set_postfix({"status": "Generating answer (cached)"})
    else:
        try:
            if progress_bar:
                progress_bar.set_postfix({"status": "Generating answer"})
            model_answer = run_responses_api(
                model=model_name,
                input_text=answer_prompt,
                timeout=timeout
            )
            # Save to cache (include problem_id to ensure consistency)
            if cache_file:
                save_answer_to_cache(cache_file, cache_key, model_answer, problem_id=problem_id, model_name=model_name)
            if answer_cache is not None:
                answer_cache[cache_key] = model_answer
        except Exception as e:
            logger.error(f"  Model {model_name} failed to generate answer: {e}")
            return {
                "status": "ERROR",
                "message": f"Modelfailed to generate answer: {str(e)}"
            }
    
    # 2. Parse constraints
    # Prefer using already parsed normalized constraints (if stored earlier)
    constraints = problem.get("_normalized_constraints") or parse_constraints(constraints_str)
    is_single_constraint = len(constraints) == 1
    
    result = {
        "model": model_name,
        "model_answer": model_answer,
        "constraints": constraints,
        "is_single_constraint": is_single_constraint
    }
    
    def _reason_flag(key: str) -> bool:
        seed = f"{problem_id or ''}-{model_name}-{key}"
        return hash(seed) % 10 == 0

    # 3. Constraint validation
    logger.debug(f"  Validating constraint satisfaction...")
    if is_single_constraint:
        # Single constraint: check directly
        if progress_bar:
            progress_bar.set_postfix({"status": "Constraint validation"})
        constraint_result = validate_single_constraint(
            question,
            constraints[0],
            model_answer,
            subject,
            reference_answer,
            _reason_flag(constraints[0]),
            validator_model,
            timeout,
            raw_constraints_text=problem.get("_raw_constraints_text", ""),
            use_multi_judge=use_multi_judge,
            judge_models=judge_models,
        )
        result["constraint_validation"] = {
            "overall": constraint_result
        }
    else:
        # Multi-constraint: check each constraint separately + comprehensive check
        single_constraint_results = {}
        
        # Check each constraint
        for i, constraint in enumerate(constraints):
            if progress_bar:
                progress_bar.set_postfix({"status": "Constraint{i+1}/{len(constraints)}"})
            logger.debug(f"    Checking constraint: {constraint[:50]}...")
            constraint_result = validate_single_constraint(
                question,
                constraint,
                model_answer,
                subject,
                reference_answer,
                _reason_flag(constraint),
                validator_model,
                timeout,
                raw_constraints_text=problem.get("_raw_constraints_text", ""),
                use_multi_judge=use_multi_judge,
                judge_models=judge_models,
            )
            single_constraint_results[constraint] = constraint_result
        
        # Comprehensive check (keep overall verdict, and carry per_constraint)
        if progress_bar:
            progress_bar.set_postfix({"status": "Constraint Check"})
        logger.debug(f"    Comprehensively checking all constraints...")
        overall_result = validate_multi_constraints(
            question,
            constraints,
            model_answer,
            subject,
            reference_answer,
            _reason_flag("overall"),
            validator_model,
            timeout,
            raw_constraints_text=problem.get("_raw_constraints_text", ""),
            use_multi_judge=use_multi_judge,
            judge_models=judge_models,
        )
        
        # As long as any single constraint FAILs, combined constraint must FAIL; if all single constraints PASS, combined follows overall_result (if not given then PASS)
        individual_statuses = [res.get("status") for res in single_constraint_results.values()]
        aggregated_status = overall_result.get("status") or "UNKNOWN"
        if any(status == "FAIL" for status in individual_statuses):
            aggregated_status = "FAIL"
        elif all(status == "PASS" for status in individual_statuses):
            aggregated_status = "PASS" if aggregated_status == "UNKNOWN" else aggregated_status
        # Update combined result status and verdict
        overall_result["status"] = aggregated_status
        overall_result["verdict"] = "YES" if aggregated_status == "PASS" else ("NO" if aggregated_status == "FAIL" else overall_result.get("verdict", "UNKNOWN"))

        # Save complete constraint validation results
        result["constraint_validation"] = {
            "individual": single_constraint_results,  # Validation results for each constraint (individual validation)
            "overall": overall_result  # Comprehensive check results (includes per_constraint for statistics)
        }
    
    # 4. Answer correctness validation
    if reference_answer:
        logger.debug(f"  Validating answer correctness...")
        if progress_bar:
            progress_bar.set_postfix({"status": "Answer correctness"})
        correctness_result = validate_answer_correctness(
            question, reference_answer, model_answer, subject, validator_model, timeout
        )
        result["answer_correctness"] = correctness_result

        # A) Final gating: constraint compliance is only meaningful if the answer is correct.
        # If answer correctness fails, force overall constraint result to FAIL (prevents rubric/template gaming).
        try:
            if correctness_result.get("status") == "FAIL" and "constraint_validation" in result:
                cv = result.get("constraint_validation", {})
                overall_cv = cv.get("overall", {})
                overall_cv["status"] = "FAIL"
                overall_cv["verdict"] = "NO"
                # Keep original reasons if any, but add a clear gate marker.
                prev_reason = overall_cv.get("reason") or overall_cv.get("overall_reason") or ""
                gate_reason = "GATED_BY_ANSWER_CORRECTNESS: answer_correctness=FAIL"
                overall_cv["reason"] = (prev_reason + " | " + gate_reason).strip(" |") if prev_reason else gate_reason
                cv["overall"] = overall_cv
                result["constraint_validation"] = cv
                result["gated_by_answer_correctness"] = True
        except Exception:
            # Do not crash evaluation due to gating metadata
            pass
    
    if progress_bar:
        progress_bar.set_postfix({"status": "Complete"})
    
    return result


def evaluate_batch(
    problems: List[Dict[str, Any]],
    model_names: List[str],
    validator_model: str = "gpt-5",
    output_file: str = "model_evaluation_results.jsonl",
    resume: bool = False,
    start_index: int = 0,
    num_problems: Optional[int] = None,
    timeout: int = 6000,
    cache_dir: Optional[str] = None,
    prompt_version: Optional[str] = None,
    use_multi_judge: bool = False,
    judge_models: Optional[List[str]] = None
):
    """Batch evaluation
    
    Args:
        problems: List of problems
        model_names: List of model names
        validator_model: Validator model name
        output_file: Output file path
        resume: Whether to enable resume from checkpoint
        start_index: Start index
        num_problems: Number of problems to evaluate (None means evaluate all problems)
        timeout: API timeout duration
        cache_dir: Cache directory
        prompt_version: Prompt version
        use_multi_judge: Whether to use multi-model voting mechanism
        judge_models: List of models for voting (default: ['gemini-3-flash', 'gpt-5.1'])
    """
    # If num_problems is specified, truncate problem list
    # Note: actual_start_index must remain as start_index to ensure problem_id consistency
    if num_problems is not None:
        problems = problems[start_index:start_index + num_problems]
        actual_start_index = start_index  # Keep original start_index to ensure problem_id is correct
        logger.info(f"Starting from index, evaluating problems")
    else:
        problems = problems[start_index:]
        actual_start_index = start_index
        logger.info(f"Starting from index, evaluating problems")
    logger.info(f"Evaluating models: {', '.join(model_names)}")
    logger.info(f"Validator model: {validator_model}")
    if use_multi_judge:
        judge_models_list = judge_models or ["gemini-3-flash", "gpt-5.1"]
        logger.info(f"Multi-model voting: {', '.join(judge_models_list)}")
    logger.info(f"Validator mode: {VALIDATOR_PROMPT_MODE}")
    
    # Get prompt version
    if prompt_version is None:
        prompt_version = get_prompt_version_hash()
        logger.info(f"Prompt version: {prompt_version}")
    
    # Load answer cache
    answer_cache = {}
    cache_file = None
    if cache_dir:
        # Create separate cache file for each model
        for model_name in model_names:
            model_cache_file = os.path.join(cache_dir, f"answer_cache_{model_name}.jsonl")
            model_cache = load_answer_cache(model_cache_file)
            answer_cache.update(model_cache)
            cache_file = model_cache_file  # Use the last one, should actually process separately for each model
        logger.info(f"Loaded answer cache entries")
    
    # Load existing results (if resume is enabled)
    # Use (problem_id, model_name) as unique identifier
    existing_results = set()
    if resume and os.path.exists(output_file):
        logger.info(f"Resume enabled, detecting existing results: {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    problem_id = result.get("problem_id")
                    evaluation = result.get("evaluation", {})
                    model_name = evaluation.get("model")
                    if problem_id and model_name:
                        # Use (problem_id, model_name) as unique identifier
                        existing_results.add((problem_id, model_name))
                except json.JSONDecodeError:
                    pass
        logger.info(f"   - Existing evaluation results (problem×model combinations)")
    
    # Open output file (append mode)
    output_mode = "a" if (resume and os.path.exists(output_file)) else "w"
    output_stream = open(output_file, output_mode, encoding="utf-8")
    
    try:
        stats = defaultdict(lambda: {
            "total": 0,
            "constraint_pass": 0,
            "constraint_fail": 0,
            "correctness_pass": 0,
            "correctness_fail": 0,
            "errors": 0,
            "skipped": 0,
            "cache_hits": 0,
            "individual_constraints": defaultdict(lambda: {"pass": 0, "fail": 0, "total": 0})  # Individual statistics for each constraint
        })
        
        # Calculate total number of tasks (number of problems × number of models)
        total_tasks = len(problems) * len(model_names)
        
        # Use single tqdm to show overall progress (no longer nested multiple progress bars)
        with tqdm(total=total_tasks, desc="Evaluation progress", unit="task", ncols=120, 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
            for local_idx, problem in enumerate(problems):
                # Prefer using problem_id from problem dict (if already exists in refined_corpus.json)
                # Otherwise use default generation: subject_{original_idx + 1}
                problem_id = problem.get("problem_id")
                if not problem_id:
                    original_idx = actual_start_index + local_idx
                    problem_id = f"{problem.get('subject', 'unknown')}_{original_idx + 1}"
                else:
                    # If problem_id is read from problem, also calculate original_idx for display
                    original_idx = actual_start_index + local_idx
                
                for model_name in model_names:
                    # Check if already exists (based on problem_id + model_name)
                    task_key = (problem_id, model_name)
                    if resume and task_key in existing_results:
                        stats[model_name]["skipped"] += 1
                        pbar.update(1)
                        logger.debug(f"  Problem model combination already exists, skipping")
                        continue
                    
                    # Update progress bar to show current task
                    pbar.set_postfix({
                        "Problem": f"{local_idx + 1}/{len(problems)} (original: {original_idx + 1})",
                        "Model": model_name,
                        "Status": "Started"
                    })
                    
                    # Remove logger.info to avoid interrupting tqdm progress bar display
                    # logger.info(f"\n📖 Evaluating problem {local_idx + 1}/{len(problems)} (original index: {original_idx + 1}, {problem_id}) - Model: {model_name}")
                    
                    # Use separate cache file for each model
                    model_cache_file = None
                    if cache_dir:
                        model_cache_file = os.path.join(cache_dir, f"answer_cache_{model_name}.jsonl")
                        # Load cache for this model
                        model_cache = load_answer_cache(model_cache_file)
                    else:
                        model_cache = answer_cache
                    
                    evaluation_result = evaluate_problem(
                        problem, model_name, validator_model, timeout,
                        answer_cache=model_cache,
                        cache_file=model_cache_file,
                        progress_bar=pbar,
                        problem_id=problem_id,
                        use_multi_judge=use_multi_judge,
                        judge_models=judge_models
                    )
                    
                    if evaluation_result.get("status") == "ERROR":
                        stats[model_name]["errors"] += 1
                        logger.error(f"    evaluation failed: {evaluation_result.get('message')}")
                        pbar.update(1)
                        continue
                    
                    # Check if cache is used (use same cache key generation as in evaluate_problem)
                    cache_key = get_answer_cache_key(
                        problem.get("question", ""),
                        problem.get("constraints", ""),
                        model_name,
                        problem_id=problem_id
                    )
                    if model_cache and cache_key in model_cache:
                        stats[model_name]["cache_hits"] += 1
                    
                    # Statistics on constraint pass situation
                    constraint_validation = evaluation_result.get("constraint_validation", {})
                    if evaluation_result.get("is_single_constraint"):
                        overall_status = constraint_validation.get("overall", {}).get("status", "UNKNOWN")
                        if overall_status == "PASS":
                            stats[model_name]["constraint_pass"] += 1
                        elif overall_status == "FAIL":
                            stats[model_name]["constraint_fail"] += 1
                        
                        # Single constraint problem: statistics for this constraint
                        # Use constraints field from evaluation_result (parsed list)
                        constraints_list = evaluation_result.get("constraints", [])
                        if isinstance(constraints_list, list) and len(constraints_list) > 0:
                            # Single constraint problem should have only one constraint
                            constraint_name = constraints_list[0]
                            normalized_name = normalize_constraint_name(constraint_name) if isinstance(constraint_name, str) else normalize_constraint_name(str(constraint_name))
                            stats[model_name]["individual_constraints"][normalized_name]["total"] += 1
                            if overall_status == "PASS":
                                stats[model_name]["individual_constraints"][normalized_name]["pass"] += 1
                            elif overall_status == "FAIL":
                                stats[model_name]["individual_constraints"][normalized_name]["fail"] += 1
                    else:
                        overall_status = constraint_validation.get("overall", {}).get("status", "UNKNOWN")
                        if overall_status == "PASS":
                            stats[model_name]["constraint_pass"] += 1
                        elif overall_status == "FAIL":
                            stats[model_name]["constraint_fail"] += 1
                        
                        # Multi-constraint problem: statistics for each individual constraint
                        individual = constraint_validation.get("individual", {})
                        for constraint_name, result in individual.items():
                            normalized_name = normalize_constraint_name(constraint_name)
                            status = result.get("status", "UNKNOWN")
                            stats[model_name]["individual_constraints"][normalized_name]["total"] += 1
                            if status == "PASS":
                                stats[model_name]["individual_constraints"][normalized_name]["pass"] += 1
                            elif status == "FAIL":
                                stats[model_name]["individual_constraints"][normalized_name]["fail"] += 1
                    
                    # Statistics on answer correctness
                    answer_correctness = evaluation_result.get("answer_correctness", {})
                    correctness_status = answer_correctness.get("status", "UNKNOWN")
                    if correctness_status == "PASS":
                        stats[model_name]["correctness_pass"] += 1
                    elif correctness_status == "FAIL":
                        stats[model_name]["correctness_fail"] += 1
                    
                    stats[model_name]["total"] += 1
                    
                    # Save result immediately (save immediately after each problem×model combination completes)
                    result_record = {
                        "problem_id": problem_id,
                        "problem": problem,
                        "evaluation": evaluation_result
                    }
                    output_stream.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                    output_stream.flush()  # Ensure immediate write to disk
                    
                    pbar.update(1)
                    # Remove logger.info to avoid interrupting tqdm progress bar display
                    # logger.info(f"    ✅ Evaluation completed and saved")
        
        # Print brief statistics
        logger.info("\n" + "="*60)
        logger.info("📊 Evaluation statistics:")
        for model_name in model_names:
            model_stats = stats[model_name]
            logger.info(f"\nModel: {model_name}")
            logger.info(f"   Total problems: {model_stats['total']}")
            logger.info(f"   Skipped: {model_stats['skipped']}")
            logger.info(f"   Cache hits: {model_stats['cache_hits']}")
            logger.info(f"   Constraint validation: {model_stats['constraint_pass']} pass | {model_stats['constraint_fail']} fail")
            logger.info(f"   Answer correctness: {model_stats['correctness_pass']} pass | {model_stats['correctness_fail']} fail")
            logger.info(f"   Error count: {model_stats['errors']}")
            
            # Show individual pass rate for each constraint
            if model_stats['individual_constraints']:
                logger.info(f"\n   Individual Constraint Pass Rates:")
                # Sort by pass rate (from high to low)
                sorted_constraints = sorted(
                    model_stats['individual_constraints'].items(),
                    key=lambda x: (x[1]["pass"] / x[1]["total"] if x[1]["total"] > 0 else 0),
                    reverse=True
                )
                for constraint_name, constraint_stats in sorted_constraints:
                    pass_count = constraint_stats["pass"]
                    total_count = constraint_stats["total"]
                    pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0
                    logger.info(f"     • {constraint_name:<30} {pass_count:>2}/{total_count:<2} ({pass_rate:>5.1f}%)")
        logger.info("="*60)
        
    finally:
        output_stream.close()
    
    logger.info(f"\nEvaluation completed! Results saved to: {output_file}")
    
    # Generate detailed statistics report
    generate_statistics_report(output_file, logger, save_to_file=True)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Evaluation System - Evaluate different models on refined corpus"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="outputs/refined_corpus.json",
        help="Input problem file path(default: outputs/refined_corpus.json)"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="List of models to evaluate，e.g.: --models gpt-5 gemini-3"
    )
    
    parser.add_argument(
        "--validator_model",
        type=str,
        default="gpt-5",
        help="Model for validation(default: gpt-5, options: gpt-5, gemini-3)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="model_evaluation_results.jsonl",
        help="Output file path (default: model_evaluation_results.jsonl)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Enable resume from checkpoint (already evaluated problems will be skipped)"
    )
    
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index (default: 0)"
    )
    
    parser.add_argument(
        "--num_problems",
        type=int,
        default=None,
        help="Number of problems to evaluate(default: None, means evaluate all problems)"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="Log level (debug/info/warning/error, default: info)"
    )
    
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file path (optional)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="API call timeout duration (seconds, default: 120)"
    )
    
    parser.add_argument(
        "--prompt_version",
        type=str,
        default=None,
        help="Prompt version identifier (default: automatically calculated from prompt code)"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Answer cache directory (default: no cache)"
    )
    
    parser.add_argument(
        "--auto_output_dir",
        action="store_true",
        help="Automatically create output directory based on model, validator model and prompt version"
    )
    
    parser.add_argument(
        "--validator_mode",
        type=str,
        choices=["strict", "loose"],
        default=None,
        help="Validator prompt mode: strict (all points must pass), loose (only main points must pass). Default: use VALIDATOR_PROMPT_MODE environment variable or 'strict'"
    )
    
    parser.add_argument(
        "--use_multi_judge",
        action="store_true",
        help="Use multiple models for judging (voting mechanism)"
    )
    
    parser.add_argument(
        "--judge_models",
        type=str,
        nargs="+",
        default=None,
        help="List of models to use for multi-judge voting (default: ['gemini-3-flash', 'gpt-5.1'])"
    )
    
    args = parser.parse_args()
    
    # Set validator mode (if specified via command line, override environment variable)
    global VALIDATOR_PROMPT_MODE, ANSWER_PROMPT_MODE
    if args.validator_mode:
        mode = args.validator_mode.lower()
        if mode not in ["strict", "loose"]:
            logger.error(f"Invalid validator_mode: {mode}. Only 'strict' or 'loose' supported")
            sys.exit(1)
        VALIDATOR_PROMPT_MODE = mode
        ANSWER_PROMPT_MODE = VALIDATOR_PROMPT_MODE
        os.environ["VALIDATOR_PROMPT_MODE"] = VALIDATOR_PROMPT_MODE
        os.environ["ANSWER_PROMPT_MODE"] = ANSWER_PROMPT_MODE
        logger.info(f"Validator mode set to: {VALIDATOR_PROMPT_MODE}")
    else:
        # Validate mode in environment variable
        env_mode = VALIDATOR_PROMPT_MODE.lower()
        if env_mode not in ["strict", "loose"]:
            logger.warning(f"Environment variable VALIDATOR_PROMPT_MODE={env_mode} is not a valid mode (strict/loose), will use default value 'strict'")
            VALIDATOR_PROMPT_MODE = "strict"
            ANSWER_PROMPT_MODE = "strict"
    
    # Initialize logging
    setup_logging(args.log_level, args.log_file)
    
    # Get prompt version
    prompt_version = args.prompt_version or get_prompt_version_hash()
    
    # Determine output directory and file
    if args.auto_output_dir:
        # Create separate output directory for each model
        # Note: if multiple models, only use first model's directory structure
        # Should actually process separately for each model, but for simplicity, handle single model case first
        if len(args.models) == 1:
            base_output_dir = get_output_directory(
                args.output_dir,
                args.models[0],
                args.validator_model,
                prompt_version
            )
            output_file = os.path.join(base_output_dir, "evaluation_results.jsonl")
            cache_dir = os.path.join(base_output_dir, "cache")
        else:
            # When multiple models, use unified output directory
            base_output_dir = os.path.join(
                args.output_dir,
                f"multi_model_{args.validator_model}_prompt_{prompt_version}"
            )
            os.makedirs(base_output_dir, exist_ok=True)
            output_file = os.path.join(base_output_dir, args.output_file)
            cache_dir = os.path.join(base_output_dir, "cache")
    else:
        # Use specified output directory
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, args.output_file)
        cache_dir = args.cache_dir or os.path.join(args.output_dir, "cache")
    
    # Ensure cache directory exists
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Answer cache directory: {cache_dir}")
    
    logger.info(f"Output file: {output_file}")
    
    # Load problems
    problems = load_problems(args.input_file)
    if not problems:
        logger.error("No problems to evaluate")
        return
    
    # If num_problems is specified, check if it exceeds range
    if args.num_problems is not None:
        total_available = len(problems) - args.start_index
        if args.num_problems > total_available:
            logger.warning(f"Requested to evaluate problems, but only available (from index Started)")
            logger.info(f"   Will evaluate problems")
    
    # Start evaluation
    evaluate_batch(
        problems=problems,
        model_names=args.models,
        validator_model=args.validator_model,
        output_file=output_file,
        resume=args.resume,
        start_index=args.start_index,
        num_problems=args.num_problems,
        timeout=args.timeout,
        cache_dir=cache_dir,
        prompt_version=prompt_version,
        use_multi_judge=args.use_multi_judge,
        judge_models=args.judge_models
    )


if __name__ == "__main__":
    main()

