#!/usr/bin/env python3
"""
SciIF Local Model Testing

Test locally trained models (SFT/RL) on evaluation datasets.
Uses local model for answer generation and judge model for evaluation.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .evaluator import (
    setup_logging,
    build_answer_prompt,
    parse_constraints,
    normalize_constraint_name,
    validate_single_constraint,
    validate_multi_constraints,
    validate_answer_correctness,
    get_answer_cache_key,
    load_answer_cache,
    save_answer_to_cache,
    logger as eval_logger,
)
from .api_client import run_responses_api

# Setup logging
logger = eval_logger


class LocalModelWrapper:
    """Local model wrapper for generating answers (supports LoRA adapter, thread-safe)"""
    
    def __init__(self, base_model_path: str, adapter_path: Optional[str] = None, 
                 device: str = "cuda", torch_dtype=torch.bfloat16):
        """Load local model
        
        Args:
            base_model_path: Base model path
            adapter_path: LoRA adapter path (if using LoRA training)
            device: Device
            torch_dtype: Data type
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self._lock = threading.Lock()  # Thread lock to ensure thread safety during concurrent generation
        
        # If adapter path is provided, first check if it is a full model
        is_full_model = False
        if adapter_path and os.path.exists(adapter_path):
            # Check if it is a full model (has model.safetensors or pytorch_model.bin, but no adapter_config.json)
            has_model_file = (
                os.path.exists(os.path.join(adapter_path, "model.safetensors")) or
                os.path.exists(os.path.join(adapter_path, "pytorch_model.bin")) or
                os.path.exists(os.path.join(adapter_path, "model.safetensors.index.json"))
            )
            has_adapter_config = os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
            
            # Check model file size (full model should be large, e.g., >1GB)
            model_file_size = 0
            if os.path.exists(os.path.join(adapter_path, "model.safetensors")):
                model_file_size = os.path.getsize(os.path.join(adapter_path, "model.safetensors"))
            elif os.path.exists(os.path.join(adapter_path, "pytorch_model.bin")):
                model_file_size = os.path.getsize(os.path.join(adapter_path, "pytorch_model.bin"))
            
            # Full model judgment: has model file, no adapter_config, and file size >100MB
            if has_model_file and not has_adapter_config and model_file_size > 100 * 1024 * 1024:
                # This is a full model, load directly (do not load base model)
                is_full_model = True
                logger.info(f"Detected full model, loading directly: {adapter_path}")
                logger.info(f"   Model file size: {model_file_size / (1024**3):.2f} GB")
                logger.info(f"   Note: Full model loading may take some time, please wait patiently...")
                model_path = adapter_path
            elif has_model_file and model_file_size < 100 * 1024 * 1024:
                # Model file too small, may be RL training only saved partial content
                logger.warning(f"Detected model file but size is abnormal ({model_file_size / 1024:.2f} KB)")
                logger.warning(f"This may not be a full model, will use base model + SFT adapter")
                logger.info(f"Loading base model: {base_model_path}")
                model_path = base_model_path
            else:
                # This is LoRA adapter, need to load base model first
                logger.info(f"Loading base model: {base_model_path}")
                model_path = base_model_path
        else:
            # No adapter path, load base model directly
            logger.info(f"Loading base model: {base_model_path}")
            model_path = base_model_path
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model
        logger.info(f"   Starting to load model file (this may take a few minutes)...")
        try:
        self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True  # Reduce CPU memory usage
        )
            logger.info(f"   Model file loading completed")
        except Exception as e:
            logger.error(f"   Model loading failed: {e}")
            raise
        
        if is_full_model:
            logger.info("Full model loaded successfully")
        else:
            # If adapter path is provided and not a full model, try to load as LoRA adapter
        if adapter_path and os.path.exists(adapter_path):
                # Check if it is RL training output (has small model.safetensors but no adapter_config.json)
                has_small_model = (
                    os.path.exists(os.path.join(adapter_path, "model.safetensors")) and
                    os.path.getsize(os.path.join(adapter_path, "model.safetensors")) < 100 * 1024 * 1024
                )
                has_adapter_config = os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
                
                if has_small_model and not has_adapter_config:
                    # RL training output, but save incomplete
                    logger.error(f"RL training output incomplete (only partial files, {os.path.getsize(os.path.join(adapter_path, 'model.safetensors')) / 1024:.2f} KB)")
                    logger.error(f"This may be because PPOTrainer.save_model() only saved partial content (e.g., value head), not the full model")
                    logger.warning(f"Since RL training is based on SFT adapter, it is recommended to use SFT adapter for testing")
                    logger.warning(f"Or fix the RL training save logic to ensure adapter is saved correctly")
                    logger.warning(f"Will now use base model (does not include SFT or RL updates)")
                    # Do not load RL output, continue using base model
                elif has_adapter_config:
                    # This is LoRA adapter, try to load
                    logger.info(f"Loading LoRA adapter: {adapter_path}")
            try:
                from peft import PeftModel
                        # Try to load adapter, if fails continue using base model
                        try:
                            self.model = PeftModel.from_pretrained(self.model, adapter_path)
                            logger.info("LoRA adapter loaded successfully")
                        except Exception as adapter_error:
                            logger.warning(f"Failed to load LoRA adapter: {adapter_error}")
                            logger.warning(f"Will use base model (without adapter)")
                            # Continue using base model, do not raise exception
                    except ImportError:
                        logger.warning("peft not installed, cannot load LoRA adapter, will use base model")
                else:
                # Other cases, try to load as adapter
                logger.info(f"Attempting to load as LoRA adapter: {adapter_path}")
                    try:
                        from peft import PeftModel
                        try:
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                        logger.info("✅ LoRA adapter loaded successfully")
                        except Exception as adapter_error:
                        logger.warning(f"Load failed: {adapter_error}")
                        logger.warning(f"Will use base model")
            except ImportError:
                    logger.warning("peft not installed, will use base model")
        
        self.model.eval()
        logger.info(f"Model loading completed")
    
    def generate(self, prompt: str, max_new_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Generate answer (thread-safe)"""
        # Use lock to ensure thread safety
        with self._lock:
            # Build input
        messages = [
            {"role": "user", "content": prompt}
        ]
        
            # Use tokenizer apply_chat_template
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
                # If apply_chat_template fails, use prompt directly
            text = prompt
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
            # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Prevent repetitive generation
            )
        
                # Decode (only take newly generated part)
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            
                # Detailed log
                logger.info(f"Generation completed: input token count={inputs['input_ids'].shape[1]}, output token count={outputs[0].shape[0]}, newly generated token count={len(generated_ids)}")
            
            if len(generated_ids) == 0:
                    logger.warning(f"Generated 0 tokens! Input length={inputs['input_ids'].shape[1]}")
                    return ""  # Return empty string, let upper layer handle
            
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            answer = answer.strip()
            
                # Check if valid content is generated
            if not answer:
                    logger.warning(f"Generation result is empty string!")
                    logger.warning(f"  generated_ids length: {len(generated_ids)}")
                    logger.warning(f"  generated_ids content: {generated_ids[:10].tolist() if len(generated_ids) > 0 else 'empty'}")
                # Try not to decode special tokens
                try:
                    answer_no_skip = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                    logger.warning(f"  Result length without decoding special tokens: {len(answer_no_skip)}")
                    logger.warning(f"  Result preview without decoding special tokens: {answer_no_skip[:200]}")
                except Exception as e:
                    logger.warning(f"  Cannot decode (without decoding special tokens): {e}")
            
            return answer


def load_problems(file_path: str) -> List[Dict[str, Any]]:
    """Load problem file"""
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # New format: dict format, contains corpus key
            if isinstance(data, dict):
                all_problems = []
                for key in ['corpus', 'selected_300_no_numexp', 'extra_numexp_questions']:
                    if key in data and isinstance(data[key], list):
                        all_problems.extend(data[key])
                
                if all_problems:
                    logger.info(f"Successfully loaded problems")
                    return all_problems
                else:
                    logger.error(f"Dictionary format error: should contain corpus/selected_300_no_numexp/extra_numexp_questions keys")
                    return []
            
            # Old format: directly a list
            elif isinstance(data, list):
                logger.info(f"✅ Successfully loaded problems")
                return data
            else:
                logger.error(f"JSON file format error")
                return []
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return []


def parse_text_field(text: str) -> Dict[str, str]:
    """Parse question, constraints, answer from text field"""
    result = {
        "question": "",
        "constraints": "",
        "answer": ""
    }
    
    if not text:
        return result
    
    import re
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


def generate_answer_for_problem(
    problem: Dict[str, Any],
    model_wrapper: LocalModelWrapper,
    answer_cache: Optional[Dict[str, str]] = None,
    cache_file: Optional[str] = None,
    cache_lock: Optional[threading.Lock] = None,
    problem_id: Optional[str] = None,
    max_new_tokens: int = 8192,
    adapter_path: Optional[str] = None,
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """Generate answer for single problem (can be called concurrently)
    
    Returns:
        (model_answer, problem_id, problem_data_or_error_dict)
    """
    # Get problem_id
    if not problem_id:
        problem_id = problem.get("id") or problem.get("_problem_id") or problem.get("problem_id")
    
    # Parse problem content
    question = problem.get("question", "")
    constraints_str_simple = problem.get("constraints", "")
    
    # Extract from question_with_constraints
    qw = problem.get("question_with_constraints")
    raw_qw_for_answer = ""
    if isinstance(qw, str) and qw.strip():
        raw_qw_for_answer = qw
        parsed_qw = parse_text_field(qw)
        if parsed_qw["question"] and not question:
            question = parsed_qw["question"]
        if parsed_qw["constraints"] and not constraints_str_simple:
            constraints_str_simple = parsed_qw["constraints"]
    
    # Parse from text field
    text = problem.get("text", "")
    if text:
        parsed_text = parse_text_field(text)
        if parsed_text["question"] and not question:
            question = parsed_text["question"]
        if parsed_text["constraints"] and not constraints_str_simple:
            constraints_str_simple = parsed_text["constraints"]
    
    if not question:
        return None, problem_id, {
            "status": "ERROR",
            "message": "Problem missing",
            "problem_id": problem_id
        }
    
    # Use local model to generate answer
    question_for_answer = raw_qw_for_answer or question
    answer_prompt = build_answer_prompt(question_for_answer, constraints_str_simple)
    
    # Determine model type based on adapter_path
    if adapter_path and ("rl_output" in adapter_path or "rl" in adapter_path.lower()):
        model_name = "qwen3-8b-rl"
    else:
        model_name = "qwen3-8b-sft"
    
    cache_key = get_answer_cache_key(question_for_answer, constraints_str_simple, model_name, problem_id=problem_id)
    model_answer = None
    
    # Check cache (needs lock protection)
    if cache_lock:
        with cache_lock:
    if answer_cache and cache_key in answer_cache:
        model_answer = answer_cache[cache_key]
    else:
        if answer_cache and cache_key in answer_cache:
            model_answer = answer_cache[cache_key]
    
    if model_answer is None:
        try:
            logger.info(f"  Problem started generating answer...")
            logger.info(f"    Prompt length: {len(answer_prompt)}")
            model_answer = model_wrapper.generate(answer_prompt, max_new_tokens=max_new_tokens)
            logger.info(f"  Problem generation completed, answer length: {len(model_answer) if model_answer else 0}")
            
            # Check if generated answer is empty
            if not model_answer or not model_answer.strip():
                logger.warning(f"  Problem generated empty answer")
                logger.warning(f"    model_answer type: {type(model_answer)}")
                logger.warning(f"    model_answer value: {repr(model_answer)}")
                return None, problem_id, {
                    "status": "ERROR",
                    "message": "Model generated empty answer",
                    "problem_id": problem_id,
                    "details": f"Generated answer is empty string or only contains whitespace, type={type(model_answer).__name__}, length={len(model_answer) if model_answer else 0}"
                }
            
            # Save to cache (needs lock protection)
            if cache_lock:
                with cache_lock:
                    if cache_file:
                        save_answer_to_cache(cache_file, cache_key, model_answer, problem_id=problem_id, model_name=model_name)
                    if answer_cache is not None:
                        answer_cache[cache_key] = model_answer
            else:
            if cache_file:
                    save_answer_to_cache(cache_file, cache_key, model_answer, problem_id=problem_id, model_name=model_name)
            if answer_cache is not None:
                answer_cache[cache_key] = model_answer
        except Exception as e:
            logger.error(f"  Problem model failed to generate answer: {e}")
            import traceback
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            # Check if it is GPU memory error
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.error(f"  GPU memory error! Attempting to clear cache...")
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return None, problem_id, {
                "status": "ERROR",
                "message": f"Model failed to generate answer: {str(e)}",
                "problem_id": problem_id,
                "error_type": type(e).__name__,
                "traceback": error_trace
            }
    
    # Check again if answer is empty (case of reading from cache)
    if not model_answer or not model_answer.strip():
        logger.warning(f"  Problem answer read from cache is empty")
        return None, problem_id, {
            "status": "ERROR",
            "message": "Answer is empty (may be from cache)",
            "problem_id": problem_id
        }
    
    # No longer need to return problem_data, as evaluate_problem_with_local_model will extract directly from problem
    return model_answer, problem_id, None


def evaluate_problem_with_local_model(
    problem: Dict[str, Any],
    model_answer: str,
    problem_data: Optional[Dict[str, Any]] = None,
    validator_model: str = "gpt-5",
    timeout: int = 6000,
    problem_id: Optional[str] = None,
    use_multi_judge: bool = False,
    judge_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate single problem using answer generated by local model (validation part)
    
    Process question and constraints in the same way as run_other_models_on_refined_corpus.py
    """
    # Get problem_id
    if not problem_id:
        problem_id = problem.get("id") or problem.get("_problem_id") or problem.get("problem_id")
    
    subject = problem.get("subject", "unknown")
    
    # Extract question and constraints in the way of run_other_models_on_refined_corpus.py
    # Important: distinguish two types of constraint strings
    # - constraints_str_simple: simplified constraint list (for parsing constraints, e.g., "Boundary Conditions, Units Standard")
    # - constraints_str_detailed: detailed constraint description (for building prompt, contains full description)
    
    question = problem.get("question", "")
    constraints_str_simple = problem.get("constraints", "")  # Simplified format, for constraint parsing
    constraints_str_detailed = ""  # Detailed format, for building prompt
    reference_answer = problem.get("answer", "")
    
    # First try to extract from question_with_constraints (sciQA_if_full.json etc. formats)
    qw = problem.get("question_with_constraints")
    if isinstance(qw, str) and qw.strip():
        parsed_qw = parse_text_field(qw)
        if parsed_qw["question"] and not question:
            question = parsed_qw["question"]
        # Constraints parsed from question_with_constraints are complete constraint descriptions
        if parsed_qw["constraints"]:
            constraints_str_detailed = parsed_qw["constraints"]
        if parsed_qw["answer"] and not reference_answer:
            reference_answer = parsed_qw["answer"]
    
    # If there is text field, parse from text field
    text = problem.get("text", "")
    if text:
        parsed_text = parse_text_field(text)
        if parsed_text["question"]:
            question = parsed_text["question"]
        # Constraints parsed from text field are complete constraint descriptions, for building prompt
        if parsed_text["constraints"]:
            constraints_str_detailed = parsed_text["constraints"]
        if parsed_text["answer"] and not reference_answer:
            reference_answer = parsed_text["answer"]
    
    # If still no question, try old format parsing method
    if not question and text:
        import re
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
                    if key == "question" and not question:
                        question = value
                    elif key == "constraints" and not constraints_str_detailed:
                        constraints_str_detailed = value
                    elif key == "answer" and not reference_answer:
                        reference_answer = value
    
    # Verify if question exists
    if not question or not question.strip():
        return {
            "status": "ERROR",
            "message": "Problem missing, cannot perform validation",
            "problem_id": problem_id,
            "model_answer": model_answer
        }
    
    # Parse constraints (use simplified constraint string)
    constraints = parse_constraints(constraints_str_simple) if constraints_str_simple else []
    constraints = [normalize_constraint_name(c) for c in constraints]
    is_single_constraint = len(constraints) == 1
    
    result = {
        "model": "qwen3-8b-sft",
        "model_answer": model_answer,
        "constraints": constraints,
        "is_single_constraint": is_single_constraint,
        "problem_id": problem_id,
    }
    
    # 3. Constraint validation (use validate_single_constraint_simplified, consistent with run_other_models_on_refined_corpus.py)
    logger.debug(f"  Validating constraint satisfaction...")
    
    if is_single_constraint:
        constraint_result = validate_single_constraint(
            question=question,
            constraint=constraints[0],
            answer=model_answer,
            subject=subject,
            include_reason=True,
            validator_model=validator_model,
            timeout=timeout,
            raw_constraints_text=constraints_str_detailed,
            use_multi_judge=use_multi_judge,
            judge_models=judge_models,
        )
        # Simplify result (only keep status and verdict)
        constraint_result = {
            "status": constraint_result.get("status", "UNKNOWN"),
            "verdict": constraint_result.get("verdict", "UNKNOWN"),
        }
        result["constraint_validation"] = {
            "overall": constraint_result
        }
    else:
        # Multi-constraint: check each constraint separately
        single_constraint_results = {}
        for constraint in constraints:
            constraint_result = validate_single_constraint(
                question=question,
                constraint=constraint,
                answer=model_answer,
                subject=subject,
                include_reason=True,
                validator_model=validator_model,
                timeout=timeout,
                raw_constraints_text=constraints_str_detailed,
                use_multi_judge=use_multi_judge,
                judge_models=judge_models,
            )
            # Simplify result (only keep status and verdict)
            constraint_result = {
                "status": constraint_result.get("status", "UNKNOWN"),
                "verdict": constraint_result.get("verdict", "UNKNOWN"),
            }
            single_constraint_results[constraint] = constraint_result
        
        # Overall is AND of all constraints (consistent with run_other_models_on_refined_corpus.py)
        statuses = [res.get("status", "UNKNOWN") for res in single_constraint_results.values()]
        if statuses and all(s == "PASS" for s in statuses):
            overall_status = "PASS"
        elif any(s == "ERROR" for s in statuses):
            overall_status = "ERROR"
        elif any(s == "UNKNOWN" for s in statuses):
            overall_status = "UNKNOWN"
        else:
            overall_status = "FAIL"
        
        overall_result = {
            "status": overall_status,
            "verdict": "YES" if overall_status == "PASS" else ("NO" if overall_status == "FAIL" else "UNKNOWN"),
            "derived_from_individual": True,
        }
        
        result["constraint_validation"] = {
            "individual": single_constraint_results,
            "overall": overall_result
        }
    
    # 4. Answer correctness validation
    if reference_answer:
        correctness_result = validate_answer_correctness(
            question, reference_answer, model_answer, subject, validator_model, timeout
        )
        result["answer_correctness"] = correctness_result
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Test SFT model performance on 334 problems")
    
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/mnt/shared-storage-user/ai4sreason/suencheng/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5",
        help="Base model path"
    )
    
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="/root/code/multi_task_rollout/qwen3_8b_sft_output/v0-20260101-140752/checkpoint-171",
        help="LoRA adapter path (if using LoRA training)"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="/root/code/multi_task_rollout/sciQA_IF_334_classified.json",
        help="Input problem file path"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="/root/code/multi_task_rollout/qwen3_8b_sft_334_results.jsonl",
        help="Output result file path"
    )
    
    parser.add_argument(
        "--validator_model",
        type=str,
        default="gpt-5",
        help="Validator model name"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/root/code/multi_task_rollout/qwen3_8b_sft_334_cache",
        help="Cache directory"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index"
    )
    
    parser.add_argument(
        "--num_problems",
        type=int,
        default=None,
        help="Number of problems to evaluate (None means all)"
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of concurrent threads (for concurrent answer generation)"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens for answer generation (default: 8192)"
    )
    
    parser.add_argument(
        "--judge_only",
        action="store_true",
        help="Only perform validation, skip answer generation (use cached answers)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=6000,
        help="API call timeout duration (seconds, default: 6000)"
    )
    
    parser.add_argument(
        "--max_concurrent_judge",
        type=int,
        default=None,
        help="Maximum concurrency for Judge phase (default: max(64, --max_workers), supports high concurrency judge)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Enable resume from checkpoint (already judged problems will be skipped, continue judging unfinished problems)"
    )
    
    args = parser.parse_args()
    
    # Set judge concurrency (default: max(64, max_workers))
    if args.max_concurrent_judge is None:
        args.max_concurrent_judge = max(64, args.max_workers)
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_file = os.path.join(args.cache_dir, "answer_cache.jsonl")
    
    # Load problems
    problems = load_problems(args.input_file)
    if not problems:
        logger.error("No problems loaded")
        return
    
    # If num_problems is specified, truncate problem list
    if args.num_problems is not None:
        problems = problems[args.start_index:args.start_index + args.num_problems]
        logger.info(f"Starting from index, evaluating problems")
    else:
        problems = problems[args.start_index:]
        logger.info(f"Starting from index, evaluating problems")
    
    # Load answer cache
    answer_cache = load_answer_cache(cache_file) if os.path.exists(cache_file) else {}
    cache_lock = threading.Lock()  # Cache lock, for protecting concurrent writes
    logger.info(f"Loaded answer cache entries")
    
    # Check if using multi-model voting
    use_multi_judge = os.environ.get("USE_MULTI_JUDGE", "false").lower() == "true"
    judge_models = None
    if use_multi_judge:
        judge_models_str = os.environ.get("JUDGE_MODELS", "gemini-3-flash,gpt-5.1")
        judge_models = [m.strip() for m in judge_models_str.split(",")]
        logger.info(f"Using multi-model voting: {', '.join(judge_models)}")
    else:
        logger.info(f"Validator model: {args.validator_model}")
    
    answer_results = {}  # {problem_id: (model_answer, problem_data, error_dict)}
    
    if args.judge_only:
        # Validation-only mode: read answer from cache
        logger.info(f"Mode: Only perform validation (use cached answers)")
        logger.info(f"Loading answers from cache...")
        
        # Extract answer from cache (no need to build problem_data, as evaluate_problem_with_local_model will extract directly from problem)
        for problem in problems:
            problem_id = problem.get("id") or problem.get("_problem_id") or problem.get("problem_id")
            
            # Build cache_key in the way of run_other_models_on_refined_corpus.py
            question_for_answer = ""
            qw = problem.get("question_with_constraints")
            if isinstance(qw, str) and qw.strip():
                question_for_answer = qw
            else:
                question_for_answer = problem.get("question", "")
            
            constraints_for_prompt = problem.get("constraints", "")
            # Determine model type based on adapter_path
            if args.adapter_path and ("rl_output" in args.adapter_path or "rl" in args.adapter_path.lower()):
                model_name = "qwen3-8b-rl"
            else:
                model_name = "qwen3-8b-sft"
            cache_key = get_answer_cache_key(
                question_for_answer,
                constraints_for_prompt,
                model_name,
                problem_id=problem_id
            )
            
            # Try to get answer from cache
            model_answer = answer_cache.get(cache_key)
            if model_answer:
                answer_results[problem_id] = (model_answer, None, None)
        
        logger.info(f"Loaded answers from cache")
    else:
        # Normal mode: generate answer
        # Load local model
        model_wrapper = LocalModelWrapper(
            base_model_path=args.base_model_path,
            adapter_path=args.adapter_path,
            device=args.device
        )
        
        # Evaluate each problem
        logger.info(f"🚀 Starting to evaluate problems...")
        logger.info(f"🔧 Using concurrent threads to generate answers")
        
        # Step 1: concurrently generate all answers
        logger.info(f"📝 Step 1: Concurrently generating answers...")
        results_lock = threading.Lock()  # Results dict lock
    
    def generate_single_answer(idx_problem):
        idx, problem = idx_problem
        problem_id = problem.get("id") or problem.get("_problem_id") or problem.get("problem_id")
        try:
            model_answer, pid, problem_data_or_error = generate_answer_for_problem(
                problem=problem,
                model_wrapper=model_wrapper,
                answer_cache=answer_cache,
                cache_file=cache_file,
                cache_lock=cache_lock,
                problem_id=problem_id,
                max_new_tokens=args.max_new_tokens,
                adapter_path=args.adapter_path,
            )
            
            # Check if there is error message
            if problem_data_or_error and problem_data_or_error.get("status") == "ERROR":
                # Has error message, return error
                logger.warning(f"  ⚠️  [{idx+1}/{len(problems)}] Problem generation failed: unknown error")
                return idx, pid, None, None, problem_data_or_error
            
            # Check if answer is empty
            if not model_answer or not model_answer.strip():
                logger.warning(f"  ⚠️  [{idx+1}/{len(problems)}] Problem generated empty answer")
                return idx, pid, None, None, {
                    "status": "ERROR",
                    "message": "Answer generation failed: generated answer is empty",
                    "problem_id": pid
                }
            
            if model_answer is not None:
                logger.info(f"  ✅ [{idx+1}/{len(problems)}] Problem answer generation succeeded (length: characters)")
            return idx, pid, model_answer, problem_data_or_error, None
        except Exception as e:
            logger.error(f"  ❌ [{idx+1}/{len(problems)}] Problem generation failed: {e}")
            import traceback
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            return idx, problem_id, None, None, {
                "status": "ERROR",
                "message": f"Generation process exception: {str(e)}",
                "problem_id": problem_id,
                "error_type": type(e).__name__
            }
    
    # Use thread pool to concurrently generate answers
    logger.info(f"📋 Preparing to generate answers for problems...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(generate_single_answer, (i, p)): (i, p) for i, p in enumerate(problems)}
        logger.info(f"✅ Submitted generation tasks to thread pool")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Answer generation progress"):
            try:
                idx, problem_id, model_answer, problem_data_or_error, error_dict = future.result()
                # Show first 50 characters of answer for debugging
                answer_preview = (model_answer[:50] + "...") if model_answer and len(model_answer) > 50 else (model_answer or "None")
                logger.info(f"  📝 Task completed: problem_id={problem_id}, model_answerlength={len(model_answer) if model_answer else 0}, preview={answer_preview}, error_dict={'has None'}")
                # Use lock to protect results dict writes
                with results_lock:
                    if error_dict:
                        answer_results[problem_id] = (None, None, error_dict)
                    else:
                        answer_results[problem_id] = (model_answer, problem_data_or_error, None)
            except Exception as e:
                logger.error(f"  ❌ Failed to get task result: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"✅ Answer generation completed! Success: Failed: {sum(1 for v in answer_results.values() if v[0] is None)}")
    
    # Step 2: validate answers (high concurrency)
    if not args.judge_only:
        logger.info(f"🔍 Step 2: Validating answers（Maximum concurrency: {args.max_concurrent_judge}）...")
    else:
        logger.info(f"🔍 Starting to validate answers（Maximum concurrency: {args.max_concurrent_judge}）...")
        logger.info(f"📊 Will validate problems with answers (total problems)")
    
    # If using judge_only mode, only process problems with answers in cache
    problems_to_judge = []
    if args.judge_only:
        # Only process problems with answers in cache
        for problem in problems:
            problem_id = problem.get("id") or problem.get("_problem_id") or problem.get("problem_id")
            if problem_id in answer_results:
                problems_to_judge.append(problem)
    else:
        # Process all problems
        problems_to_judge = problems
    
    # If resume is enabled, load existing results, skip already judged problems
    judged_problem_ids = set()
    if args.resume and os.path.exists(args.output_file):
        logger.info(f"📂 Detected existing result file, loading already judged problems...")
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing_result = json.loads(line)
                        problem_id = existing_result.get("problem_id")
                        if not problem_id:
                            # Try to get from problem field
                            problem = existing_result.get("problem", {})
                            problem_id = problem.get("id") or problem.get("_problem_id") or problem.get("problem_id")
                        if problem_id:
                            judged_problem_ids.add(problem_id)
                    except json.JSONDecodeError:
                        continue
            logger.info(f"✅ Loaded already judged problems, will skip these problems")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load existing result file, will start judging from beginning")
    
    # Filter out already judged problems
    if judged_problem_ids:
        original_count = len(problems_to_judge)
        problems_to_judge = [
            p for p in problems_to_judge
            if (p.get("id") or p.get("_problem_id") or p.get("problem_id")) not in judged_problem_ids
        ]
        skipped_count = original_count - len(problems_to_judge)
        if skipped_count > 0:
            logger.info(f"⏭️  Skipped already judged problems, remaining problems need to be judged")
    
    # Prepare judge tasks
    def judge_single_problem(problem):
        """Judge task for single problem (can be called concurrently)"""
        problem_id = problem.get("id") or problem.get("_problem_id") or problem.get("problem_id")
        
        model_answer, _, error_dict = answer_results.get(problem_id, (None, None, None))
        
        if error_dict:
            # Error when generating answer
            result = error_dict
            result["problem"] = problem
            return problem_id, result
        elif model_answer is None or not model_answer.strip():
            if args.judge_only:
                # In judge_only mode, skip problems without answers
                logger.warning(f"  ⚠️  Problem has no answer in cache, skipping")
                return problem_id, None
            result = {
                "status": "ERROR",
                "message": "Answer generation failed: answer is empty or not generated",
                "problem_id": problem_id,
                "details": "model_answer is None or empty string"
            }
            result["problem"] = problem
            return problem_id, result
        else:
            # Normal validation (problem_data no longer needed, evaluate_problem_with_local_model will extract directly from problem)
            try:
                result = evaluate_problem_with_local_model(
                    problem=problem,
                    model_answer=model_answer,
                    problem_data=None,
                    validator_model=args.validator_model,
                    timeout=args.timeout,
                    problem_id=problem_id,
                    use_multi_judge=use_multi_judge,
                    judge_models=judge_models,
                )
                result["problem"] = problem
                return problem_id, result
            except Exception as e:
                logger.error(f"  ❌ Problem judge failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                result = {
                    "status": "ERROR",
                    "message": f"Judge failed: {str(e)}",
                    "problem_id": problem_id
                }
                result["problem"] = problem
                return problem_id, result
    
    # Use thread pool to concurrently execute judge
    file_lock = threading.Lock()  # File write lock
    # If resume is enabled and file exists, use append mode; otherwise use write mode
    file_mode = "a" if (args.resume and os.path.exists(args.output_file)) else "w"
    with open(args.output_file, file_mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.max_concurrent_judge) as executor:
            futures = {executor.submit(judge_single_problem, p): p for p in problems_to_judge}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Answer validation progress"):
                try:
                    problem_id, result = future.result()
                    if result is not None:
                        # Thread-safely write result
                        with file_lock:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f.flush()
                except Exception as e:
                    problem = futures[future]
                    problem_id = problem.get("id") or problem.get("_problem_id") or problem.get("problem_id")
                    logger.error(f"  ❌ Problem judge task exception: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    error_result = {
                        "status": "ERROR",
                        "message": f"Judge task exception: {str(e)}",
                        "problem_id": problem_id,
                        "problem": problem
                    }
                    with file_lock:
                        f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
            f.flush()
    
    logger.info(f"✅ Evaluation completed! Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

