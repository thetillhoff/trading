#!/bin/bash
# Monitor grid search progress in real-time
#
# Usage:
#   ./scripts/monitor_progress.sh [progress_file]
#
# If no file specified, looks for latest progress.json in results/

PROGRESS_FILE="${1:-}"

if [ -z "$PROGRESS_FILE" ]; then
    # Find most recent progress.json
    PROGRESS_FILE=$(find results -name "progress.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -z "$PROGRESS_FILE" ]; then
        echo "No progress.json found in results/"
        echo "Usage: $0 [progress_file]"
        exit 1
    fi
    
    echo "Monitoring: $PROGRESS_FILE"
    echo "---"
fi

# Watch progress with formatted output
watch -n 1 -c "
    if [ -f '$PROGRESS_FILE' ]; then
        echo '=== Grid Search Progress ==='
        jq -r '
            \"Timestamp: \" + .timestamp,
            \"Progress: \" + (.completed|tostring) + \"/\" + (.total_tasks|tostring) + \" tasks (\" + (.completion_pct|tostring) + \"%)\" ,
            \"Running: \" + (.running|tostring),
            \"Pending: \" + (.pending|tostring),
            \"Failed: \" + (.failed|tostring),
            \"Avg task time: \" + (.avg_task_time_s|tostring) + \"s\",
            \"ETA: \" + (.eta_minutes|tostring) + \" minutes\"
        ' '$PROGRESS_FILE' 2>/dev/null || cat '$PROGRESS_FILE'
    else
        echo 'Waiting for progress file...'
    fi
"
