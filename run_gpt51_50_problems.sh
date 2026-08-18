#!/bin/bash
# Test script: gpt-5.1 with 50 problems, 32 concurrent, strict mode, 2 judge models

set -e

# Configuration
INPUT_FILE="/root/code/multi_task_rollout/sciif_benchmark.json"
OUTPUT_DIR="/root/code/multi_task_rollout/gpt51_strict_2judges_50"
OUTPUT_FILE="evaluation_results.jsonl"
NUM_PROBLEMS=50
MAX_CONCURRENT=32
TIMEOUT=6000

# Models
ANSWER_MODEL="gpt-5.1"
JUDGE_MODELS=("gemini-3-flash" "gpt-5.2")

# Set environment variables
export VALIDATOR_PROMPT_MODE=strict
export ANSWER_PROMPT_MODE=strict
export MAX_CONCURRENT=$MAX_CONCURRENT

echo "=========================================="
echo "Test: gpt-5.1 Evaluation"
echo "=========================================="
echo "Input file: $INPUT_FILE"
echo "Answer model: $ANSWER_MODEL"
echo "Judge models: ${JUDGE_MODELS[@]}"
echo "Mode: strict"
echo "Number of problems: $NUM_PROBLEMS"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
cd /root/code/multi_task_rollout/sciif_code

python scripts/evaluate_api_models.py \
    --input_file "$INPUT_FILE" \
    --models "$ANSWER_MODEL" \
    --validator_mode strict \
    --use_multi_judge \
    --judge_models "${JUDGE_MODELS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --output_file "$OUTPUT_FILE" \
    --num_problems "$NUM_PROBLEMS" \
    --timeout "$TIMEOUT" \
    --log_level info \
    2>&1 | tee "$OUTPUT_DIR/evaluation.log"

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="

# Generate statistics and report
RESULT_FILE="$OUTPUT_DIR/$OUTPUT_FILE"
if [ -f "$RESULT_FILE" ]; then
    echo ""
    echo "Generating statistics and report..."
    python3 /root/code/multi_task_rollout/analyze_results.py "$RESULT_FILE"
    echo ""
    echo "Statistics and reports generated!"
    echo "   Results: $RESULT_FILE"
    echo "   Text report: ${RESULT_FILE%.jsonl}_report.txt"
    echo "   Markdown report: ${RESULT_FILE%.jsonl}_report.md"
else
    echo "Result file not found: $RESULT_FILE"
fi
