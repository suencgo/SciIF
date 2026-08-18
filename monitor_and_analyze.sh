#!/bin/bash
# Monitor evaluation progress and generate statistics when complete

RESULT_FILE="/root/code/multi_task_rollout/gpt52_strict_2judges_50/evaluation_results.jsonl"
TARGET_COUNT=50
LOG_FILE="/root/code/multi_task_rollout/run_gpt52_50.log"

echo "Monitoring evaluation progress..."
echo "Target: $TARGET_COUNT problems"
echo "Result file: $RESULT_FILE"
echo ""

while true; do
    if [ -f "$RESULT_FILE" ]; then
        CURRENT_COUNT=$(wc -l < "$RESULT_FILE" 2>/dev/null || echo "0")
        echo -ne "\rCurrent progress: $CURRENT_COUNT / $TARGET_COUNT problems"
        
        if [ "$CURRENT_COUNT" -ge "$TARGET_COUNT" ]; then
            echo ""
            echo ""
            echo "Evaluation completed! Generating statistics..."
            echo ""
            
            # Wait a bit to ensure file is fully written
            sleep 5
            
            # Generate statistics
            python3 /root/code/multi_task_rollout/analyze_results.py "$RESULT_FILE"
            
            echo ""
            echo "Statistics generated!"
            break
        fi
    else
        echo -ne "\rWaiting for result file to be created..."
    fi
    
    # Check if process is still running
    if ! pgrep -f "run_gpt52_50_problems.py" > /dev/null; then
        if [ -f "$RESULT_FILE" ]; then
            CURRENT_COUNT=$(wc -l < "$RESULT_FILE" 2>/dev/null || echo "0")
            echo ""
            echo ""
            echo "Evaluation process stopped. Current progress: $CURRENT_COUNT / $TARGET_COUNT"
            if [ "$CURRENT_COUNT" -gt 0 ]; then
                echo "Generating statistics for completed results..."
                python3 /root/code/multi_task_rollout/analyze_results.py "$RESULT_FILE"
            fi
        else
            echo ""
            echo "Evaluation process stopped and no results file found."
            echo "Check log: $LOG_FILE"
        fi
        break
    fi
    
    sleep 10
done
