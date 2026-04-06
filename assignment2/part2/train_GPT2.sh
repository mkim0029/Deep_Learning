#!/bin/bash
#SBATCH --account=def-sfabbro_gpu
#SBATCH --job-name=GPT2
#SBATCH --output=GPT2_%j.out
#SBATCH --error=GPT2_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=a100:1

set -euo pipefail

# Resolve project root robustly on Slurm nodes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"

source /home/minjihk/scratch/ML/bin/activate

GPT2_LOG_DIR="${GPT2_LOG_DIR:-$PROJECT_ROOT/gpt2_logs}"
ASSETS_DIR="${ASSETS_DIR:-$PROJECT_ROOT/assets}"
TXT_FILE="${TXT_FILE:-$ASSETS_DIR/book_EN_grimms_fairy_tales.txt}"
mkdir -p "$GPT2_LOG_DIR"

if [ ! -d "$ASSETS_DIR" ]; then
  echo "Assets directory not found at $ASSETS_DIR"
  exit 1
fi

if [ ! -f "$TXT_FILE" ]; then
  echo "Training text file not found at $TXT_FILE"
  exit 1
fi

echo "Starting GPT2 training at $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "ASSETS_DIR(shared)=$ASSETS_DIR"
echo "TXT_FILE(shared)=$TXT_FILE"
echo "GPT2_LOG_DIR(shared)=$GPT2_LOG_DIR"

START_TIME=$(date +%s)

python "$PROJECT_ROOT/train.py" \
  --txt_file "$TXT_FILE" \
  --log_dir "$GPT2_LOG_DIR"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Job finished at $(date)."
echo "Total execution time: $(($DURATION / 3600))h $(($DURATION % 3600 / 60))m $(($DURATION % 60))s"
