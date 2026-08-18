# SciIF: Scientific Instruction Following Evaluation Framework

A comprehensive evaluation framework for assessing Large Language Model (LLM) performance on scientific instruction following tasks with constraint validation.

## Features

- **Multi-model Evaluation**: Evaluate multiple API-based models on scientific instruction following tasks
- **Local Model Support**: Test locally trained models (SFT/RL) with LoRA adapter support
- **Constraint Validation**: Comprehensive constraint validation system with multiple validation modes
- **Answer Correctness Checking**: Automated answer correctness evaluation
- **Resume Support**: Resume interrupted evaluations from checkpoints
- **Caching**: Answer caching to avoid redundant API calls
- **Detailed Reports**: Generate comprehensive statistics and reports

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### With Local Model Support

```bash
pip install -r requirements.txt
pip install torch transformers peft accelerate
```

Or install as a package:

```bash
pip install -e .
```

## Quick Start

### 1. Configure Models

Edit `sciif/config.py` to configure your API models, or set environment variables:

```bash
export GPT5_API_KEY="your-api-key"
export GPT5_API_URL="https://api.example.com/v1"
```

### 2. Evaluate API Models

**Basic usage:**
```bash
python scripts/evaluate_api_models.py \
    --input_file data/test_dataset.json \
    --models gpt-5 gemini-3 \
    --validator_model gpt-5 \
    --output_dir results \
    --resume
```

**With strict/loose mode:**
```bash
python scripts/evaluate_api_models.py \
    --input_file data/test_dataset.json \
    --models gpt-5 \
    --validator_model gpt-5 \
    --validator_mode strict \  # or 'loose'
    --output_dir results
```

**With multi-model judge (voting):**
```bash
python scripts/evaluate_api_models.py \
    --input_file data/test_dataset.json \
    --models gpt-5 \
    --validator_model gpt-5 \
    --use_multi_judge \
    --judge_models gemini-3-flash gpt-5.1 \
    --output_dir results
```

### 3. Test Local Models

```bash
python scripts/test_local_model.py \
    --base_model_path /path/to/base/model \
    --adapter_path /path/to/adapter \
    --input_file data/test_dataset.json \
    --output_file results.jsonl \
    --validator_model gpt-5 \
    --resume
```

## Usage

### Evaluating API Models

The main evaluation script supports the following options:

```bash
python scripts/evaluate_api_models.py \
    --input_file <input_file> \          # Input dataset file (JSON/JSONL)
    --models <model1> <model2> ... \     # Models to evaluate
    --validator_model <validator> \      # Validator model (default: gpt-5)
    --output_dir <output_dir> \          # Output directory
    --output_file <output_file> \        # Output file name
    --resume \                           # Resume from checkpoint
    --start_index <index> \              # Start from specific index
    --num_problems <num> \               # Limit number of problems
    --timeout <seconds> \                # API timeout (default: 120)
    --cache_dir <cache_dir> \            # Cache directory
    --log_level <level>                  # Log level (debug/info/warning/error)
```

### Testing Local Models

For testing locally trained models:

```bash
python scripts/test_local_model.py \
    --base_model_path <base_model> \     # Base model path
    --adapter_path <adapter> \           # LoRA adapter path (optional)
    --input_file <input_file> \          # Input dataset
    --output_file <output_file> \       # Output file
    --validator_model <validator> \      # Validator model
    --max_workers <num> \               # Number of workers (default: 8)
    --max_concurrent_judge <num> \       # Concurrent judge requests (default: 64)
    --resume                             # Resume from checkpoint
```

## Dataset Format

The input dataset should be a JSON file with the following structure:

```json
[
  {
    "problem_id": "physics_1",
    "question": "What is the speed of light?",
    "constraints": "Units and Notation",
    "answer": "The speed of light is approximately 3.00 × 10^8 m/s.",
    "subject": "physics"
  }
]
```

Or a dictionary format:

```json
{
  "corpus": [
    {
      "problem_id": "physics_1",
      "question": "...",
      "constraints": "...",
      "answer": "...",
      "subject": "physics"
    }
  ]
}
```

## Configuration

### Model Configuration

Edit `sciif/config.py` to add or modify model configurations:

```python
MODEL_MAP = {
    "your-model": {
        "api_key": os.getenv("YOUR_MODEL_API_KEY", ""),
        "api_url": os.getenv("YOUR_MODEL_API_URL", "https://api.example.com/v1"),
        "model": "your-model",
        "api_scheme": "boyue",  # or "ailab"
        "use_chat_completions": False,
    },
}
```

### Environment Variables

You can also configure models via environment variables:

- `GPT5_API_KEY`: API key for GPT-5
- `GPT5_API_URL`: API URL for GPT-5
- `GEMINI3_API_KEY`: API key for Gemini-3
- `API_SCHEME`: Default API scheme (boyue/ailab)

## Output Format

The evaluation results are saved in JSONL format:

```json
{
  "problem_id": "physics_1",
  "problem": {
    "question": "...",
    "constraints": "...",
    "answer": "...",
    "subject": "physics"
  },
  "evaluation": {
    "model": "gpt-5",
    "model_answer": "...",
    "constraints": ["Units and Notation"],
    "is_single_constraint": true,
    "constraint_validation": {
      "overall": {
        "status": "PASS",
        "verdict": "YES"
      }
    },
    "answer_correctness": {
      "status": "PASS",
      "verdict": "YES"
    }
  }
}
```

A detailed statistics report is also generated automatically.

## Constraint Validation

The framework supports multiple constraint validation modes:

- **Strict Mode**: All constraint points must pass
- **Loose Mode**: Only main points must pass, secondary points can fail

Set via command-line argument (recommended):
```bash
--validator_mode strict  # or loose
```

Or via environment variable:
```bash
export VALIDATOR_PROMPT_MODE=strict  # or loose
```

## Multi-Model Judge (Voting)

The framework supports multi-model voting mechanism for more robust validation:

```bash
python scripts/evaluate_api_models.py \
    --input_file data/dataset.json \
    --models gpt-5 \
    --validator_model gpt-5 \
    --use_multi_judge \
    --judge_models gemini-3-flash gpt-5.1 \
    --output_dir results
```

When enabled, multiple judge models will evaluate each constraint independently, and the final verdict is determined by majority vote. This provides more reliable and robust evaluation results.

## Project Structure

```
sciif_code/
├── sciif/                  # Main package
│   ├── __init__.py
│   ├── evaluator.py        # Main evaluation logic
│   ├── api_client.py       # API client
│   ├── local_model.py      # Local model testing
│   └── config.py           # Configuration
├── scripts/                # Command-line scripts
│   ├── evaluate_api_models.py
│   └── test_local_model.py
├── tests/                  # Test files
├── docs/                   # Documentation
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Support

For issues and questions, please open an issue on GitHub.
