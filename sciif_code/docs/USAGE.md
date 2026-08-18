# SciIF Usage Guide

## Overview

SciIF (Scientific Instruction Following Evaluation Framework) is designed to evaluate LLM performance on scientific instruction following tasks with comprehensive constraint validation.

## Basic Usage

### 1. Evaluate API Models

The simplest way to evaluate API models:

```bash
python scripts/evaluate_api_models.py \
    --input_file data/dataset.json \
    --models gpt-5 \
    --validator_model gpt-5 \
    --output_dir results
```

### 2. Test Local Models

For locally trained models:

```bash
python scripts/test_local_model.py \
    --base_model_path /path/to/base/model \
    --adapter_path /path/to/adapter \
    --input_file data/dataset.json \
    --output_file results.jsonl \
    --validator_model gpt-5
```

## Advanced Usage

### Resume from Checkpoint

If evaluation is interrupted, you can resume:

```bash
python scripts/evaluate_api_models.py \
    --input_file data/dataset.json \
    --models gpt-5 \
    --output_dir results \
    --resume  # Automatically skips already evaluated problems
```

### Limit Number of Problems

Test on a subset:

```bash
python scripts/evaluate_api_models.py \
    --input_file data/dataset.json \
    --models gpt-5 \
    --num_problems 100 \
    --start_index 0
```

### Custom Cache Directory

```bash
python scripts/evaluate_api_models.py \
    --input_file data/dataset.json \
    --models gpt-5 \
    --cache_dir /path/to/cache
```

## Configuration

### Environment Variables

Set API keys via environment variables:

```bash
export GPT5_API_KEY="your-key"
export GPT5_API_URL="https://api.example.com/v1"
```

### Config File

Edit `sciif/config.py` to add custom models:

```python
MODEL_MAP["custom-model"] = {
    "api_key": "your-key",
    "api_url": "https://api.example.com/v1",
    "model": "custom-model",
    "api_scheme": "boyue",
    "use_chat_completions": False,
}
```

## Output Format

Results are saved in JSONL format with detailed evaluation information:

- `problem_id`: Unique problem identifier
- `problem`: Original problem data
- `evaluation`: Evaluation results including:
  - `model`: Model name
  - `model_answer`: Generated answer
  - `constraint_validation`: Constraint validation results
  - `answer_correctness`: Answer correctness results

A statistics report is automatically generated as `*_statistics_report.txt`.

## Constraint Validation Modes

Control validation strictness:

```bash
export VALIDATOR_PROMPT_MODE=strict   # All points must pass
export VALIDATOR_PROMPT_MODE=loose     # Only main points must pass
export VALIDATOR_PROMPT_MODE=loose     # Only main points must pass
```

## Troubleshooting

### API Errors

If you encounter API errors:

1. Check API keys are set correctly
2. Verify API URLs are correct
3. Increase timeout: `--timeout 300`
4. Check network connectivity

### Local Model Issues

For local model problems:

1. Verify model paths are correct
2. Check GPU availability: `nvidia-smi`
3. Reduce workers: `--max_workers 2`
4. Check model format (full model vs adapter)

### Memory Issues

If running out of memory:

1. Reduce batch size
2. Use smaller models
3. Process in smaller chunks: `--num_problems 50`

