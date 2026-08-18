# SciIF: Scientific Instruction Following Evaluation Framework

A comprehensive evaluation framework for assessing LLM performance on scientific instruction following tasks with constraint validation.

## Overview

SciIF is designed to evaluate how well language models follow scientific instructions while respecting various constraints such as:

- **Assumptions**: Explicit and implicit assumptions in problem-solving
- **Boundary Conditions**: Constraints on values, domains, and initial conditions
- **Applicability Range**: Domain and context limitations
- **Units Standard**: Unit consistency and conversion requirements
- **Cross-disciplinary Term Disambiguation**: Clarifying terms across disciplines
- **Intra-discipline Term Definitions**: Standardizing terminology within a field
- **Symbols & Constants Standardization**: Consistent symbol usage
- **Variable Naming Consistency**: Uniform variable naming conventions
- **Numerical Methods**: Appropriate numerical techniques
- **Experimental Methods**: Proper experimental procedures

## Project Structure

```
sciif_code/
├── sciif/              # Core evaluation framework
│   ├── evaluator.py   # Main evaluation logic
│   ├── api_client.py  # API client for model inference
│   ├── local_model.py # Local model wrapper
│   └── config.py      # Configuration management
├── scripts/           # Evaluation scripts
└── tests/            # Test suite

sciif_benchmark.json  # Benchmark dataset (334 problems)
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
cd sciif_code
pip install -e .
```

### Configuration

The framework reads API configurations from `/root/code/instruct_gen_v1/src/config.py` or uses default settings.

Set environment variables:
```bash
export MAX_CONCURRENT=32          # Concurrent API calls
export VALIDATOR_PROMPT_MODE=strict # strict or loose
export ANSWER_PROMPT_MODE=strict
```

### Running Evaluations

#### Evaluate API Models

```bash
cd sciif_code/scripts
python evaluate_api_models.py \
    --benchmark ../sciif_benchmark.json \
    --models gpt-5.2 \
    --judge-models gemini-3-flash gpt-5.1 \
    --num-problems 50 \
    --output results.jsonl
```

#### Using Shell Scripts

```bash
# Evaluate GPT-5.2 on 50 problems
bash run_final_test.sh
```

### Analyzing Results

```bash
python analyze_results.py results.jsonl
```

This generates:
- `results_statistics.txt`: Plain text statistics
- `results_report.md`: Markdown report with detailed metrics

## Benchmark Dataset

The `sciif_benchmark.json` file contains 334 scientific problems with:
- Unique IDs in format `sciif_subject_xxxxx`
- Questions with constraints
- Reference answers
- All content in English

## Evaluation Metrics

- **Single Constraint Pass Rate**: Percentage of problems passing individual constraints
- **Multi-constraint Pass Rate**: Percentage of problems passing all constraints
- **Answer Correctness Rate**: Percentage of correct answers
- **Overall Pass Rate**: Combined metric

## License

[Specify your license here]
