#!/usr/bin/env bash

set -euo pipefail  # Exit on error, undefined vars, and pipe failures

echo "Running ML platforms experiments..."
ROOT_DATA_DIR="../eda"
OUTLIER_DIRS=(
    "$ROOT_DATA_DIR/ml-experients-with-outliers2025-05-31_142307/imputation_global"
    "$ROOT_DATA_DIR/ml-experients-with-outliers2025-05-31_142307/imputation_user"
)

WITHOUT_OUTLIER_DIRS=(
    "$ROOT_DATA_DIR/ml-experients-without-outliers2025-05-31_143027/imputation_global"
    "$ROOT_DATA_DIR/ml-experients-without-outliers2025-05-31_143027/imputation_user"
)

# Create a key-value pair for each dataset attribute and name: dataset_name, filename_prefix
declare -A DATASETS=(
    ["platform_id"]="dataset_1_full"
    ["session_id"]="dataset_2_full"
    ["video_id"]="dataset_3_full"
)

DATASET_NAME_ATTR="_IL_filtered.csv"

# Create logs directory
mkdir -p logs

# Function to run a command with proper logging
run_command() {
    local cmd="$1"
    local logfile="$2"
    local description="$3"
    
    echo "=========================================="
    echo "Running: $description"
    echo "Command: $cmd"
    echo "Log file: $logfile"
    echo "Started at: $(date)"
    echo "=========================================="
    
    # Run command and capture both stdout and stderr
    if eval "$cmd" > "$logfile" 2>&1; then
        echo "✅ SUCCESS: Command completed successfully"
        echo "📁 Output saved to: $logfile"
    else
        local exit_code=$?
        echo "❌ FAILED: Command failed with exit code $exit_code"
        echo "📁 Error log saved to: $logfile"
        echo "Last few lines of error:"
        tail -10 "$logfile" | sed 's/^/  /'
        return $exit_code
    fi
    echo "Finished at: $(date)"
    echo ""
}

# Form commands for outlier datasets
echo "🔧 Preparing outlier dataset commands..."
OUTLIER_COMMANDS=()
OUTLIER_LOGS=()
OUTLIER_DESCRIPTIONS=()

for dir in "${OUTLIER_DIRS[@]}"; do
    # Extract imputation method from directory name
    imputation_method=$(basename "$dir")
    
    for dataset_key in "${!DATASETS[@]}"; do
        dataset_file="${DATASETS[$dataset_key]}_with_outliers$DATASET_NAME_ATTR"
        full_path="$dir/$dataset_file"
        
        # Check if file exists
        if [[ ! -f "$full_path" ]]; then
            echo "⚠️  WARNING: File not found: $full_path"
            continue
        fi
        
        # Fixed command - removed invalid flags and fixed syntax
        cmd="python ml_platforms_runner.py -d \"$full_path\" -s 3 -e -o \"${dataset_key}-${imputation_method}-outliers-estop\""
        log_name="logs/${dataset_key}_${imputation_method}_with_outliers.log"
        description="Dataset: $dataset_key | Imputation: $imputation_method | With Outliers"
        
        OUTLIER_COMMANDS+=("$cmd")
        OUTLIER_LOGS+=("$log_name")
        OUTLIER_DESCRIPTIONS+=("$description")
    done
done

# Form commands for datasets without outliers
echo "🔧 Preparing without-outlier dataset commands..."
WITHOUT_OUTLIER_COMMANDS=()
WITHOUT_OUTLIER_LOGS=()
WITHOUT_OUTLIER_DESCRIPTIONS=()

for dir in "${WITHOUT_OUTLIER_DIRS[@]}"; do
    # Extract imputation method from directory name
    imputation_method=$(basename "$dir")
    
    for dataset_key in "${!DATASETS[@]}"; do
        dataset_file="${DATASETS[$dataset_key]}_without_outliers$DATASET_NAME_ATTR"
        full_path="$dir/$dataset_file"
        
        # Check if file exists
        if [[ ! -f "$full_path" ]]; then
            echo "⚠️  WARNING: File not found: $full_path"
            continue
        fi
        
        # Fixed command - removed invalid flags and fixed syntax
        cmd="python ml_platforms_runner.py -d \"$full_path\" -s 3 -e -o \"${dataset_key}-${imputation_method}-without-outliers-estop\""
        log_name="logs/${dataset_key}_${imputation_method}_without_outliers.log"
        description="Dataset: $dataset_key | Imputation: $imputation_method | Without Outliers"
        
        WITHOUT_OUTLIER_COMMANDS+=("$cmd")
        WITHOUT_OUTLIER_LOGS+=("$log_name")
        WITHOUT_OUTLIER_DESCRIPTIONS+=("$description")
    done
done

# Summary
total_commands=$((${#OUTLIER_COMMANDS[@]} + ${#WITHOUT_OUTLIER_COMMANDS[@]}))
echo "📊 Summary:"
echo "  Outlier commands: ${#OUTLIER_COMMANDS[@]}"
echo "  Without-outlier commands: ${#WITHOUT_OUTLIER_COMMANDS[@]}"
echo "  Total commands: $total_commands"
echo ""

# Ask for confirmation
read -p "🚀 Ready to run $total_commands experiments? This may take several hours. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted by user"
    exit 0
fi

# Track overall progress
current_cmd=0
start_time=$(date +%s)

# Run each outlier command in series
if [[ ${#OUTLIER_COMMANDS[@]} -gt 0 ]]; then
    echo "🎯 Running outlier experiments..."
    for i in "${!OUTLIER_COMMANDS[@]}"; do
        current_cmd=$((current_cmd + 1))
        echo "📈 Progress: $current_cmd/$total_commands"
        
        run_command "${OUTLIER_COMMANDS[$i]}" "${OUTLIER_LOGS[$i]}" "${OUTLIER_DESCRIPTIONS[$i]}"
        
        # Show time elapsed
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        echo "⏱️  Time elapsed: $((elapsed / 60))m $((elapsed % 60))s"
        echo ""
    done
else
    echo "⚠️  No outlier commands to run"
fi

# Run each without-outlier command in series
if [[ ${#WITHOUT_OUTLIER_COMMANDS[@]} -gt 0 ]]; then
    echo "🎯 Running without-outlier experiments..."
    for i in "${!WITHOUT_OUTLIER_COMMANDS[@]}"; do
        current_cmd=$((current_cmd + 1))
        echo "📈 Progress: $current_cmd/$total_commands"
        
        run_command "${WITHOUT_OUTLIER_COMMANDS[$i]}" "${WITHOUT_OUTLIER_LOGS[$i]}" "${WITHOUT_OUTLIER_DESCRIPTIONS[$i]}"
        
        # Show time elapsed
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        echo "⏱️  Time elapsed: $((elapsed / 60))m $((elapsed % 60))s"
        echo ""
    done
else
    echo "⚠️  No without-outlier commands to run"
fi

# Final summary
end_time=$(date +%s)
total_elapsed=$((end_time - start_time))
echo "🎉 All commands executed successfully!"
echo "⏱️  Total time: $((total_elapsed / 3600))h $((total_elapsed % 3600 / 60))m $((total_elapsed % 60))s"
echo "📁 All logs saved in: logs/"
echo "📊 Check individual experiment results in their respective experiment_results_* directories"