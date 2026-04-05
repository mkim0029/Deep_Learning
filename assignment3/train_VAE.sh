#!/bin/bash
#SBATCH --account=def-sfabbro_gpu
#SBATCH --job-name=vae
#SBATCH --output=vae_%j.out
#SBATCH --error=vae_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=a100:1

set -euo pipefail

# Resolve project root robustly on Slurm nodes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/part1" ]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$SCRIPT_DIR"
fi

cd "$PROJECT_ROOT"

source /home/minjihk/scratch/ML/bin/activate

VAE_LOG_DIR="${VAE_LOG_DIR:-$PROJECT_ROOT/vae_logs}"
DATA="${DATA:-$PROJECT_ROOT/../data}"
mkdir -p "$VAE_LOG_DIR"

if [ ! -d "$DATA/MNIST" ]; then
  echo "MNIST dataset not found at $DATA/MNIST"
  echo "Compute nodes usually cannot download from the internet."
  echo "Download MNIST once on a login node, then re-submit:"
  echo "python - <<'PY'"
  echo "from torchvision.datasets import MNIST"
  echo "MNIST(root='$DATA', train=True, download=True)"
  echo "MNIST(root='$DATA', train=False, download=True)"
  echo "PY"
  exit 1
fi

# Stage training inputs to node-local storage to reduce network I/O bottlenecks.
TMP_BASE="${SLURM_TMPDIR:-/tmp/${USER}/slurm_${SLURM_JOB_ID:-manual}}"
STAGE_DIR="$TMP_BASE/vae_train"
mkdir -p "$STAGE_DIR"

DATA_LOCAL="$STAGE_DIR/$(basename "$DATA")"
VAE_LOG_DIR_LOCAL="$STAGE_DIR/$(basename "$VAE_LOG_DIR")"
mkdir -p "$DATA_LOCAL"
mkdir -p "$VAE_LOG_DIR_LOCAL"

echo "Staging inputs to local storage: $STAGE_DIR"
cp -a "$VAE_LOG_DIR/." "$VAE_LOG_DIR_LOCAL/" || true
if [ -d "$DATA" ]; then
  cp -a "$DATA/." "$DATA_LOCAL/"
else
  echo "No existing data directory at $DATA. Dataset will be downloaded to $DATA_LOCAL by train_pl.py"
fi

echo "Starting VAE training at $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA(src)=$DATA"
echo "DATA(local)=$DATA_LOCAL"
echo "VAE_LOG_DIR(src)=$VAE_LOG_DIR"
echo "VAE_LOG_DIR(local)=$VAE_LOG_DIR_LOCAL"

START_TIME=$(date +%s)

python "$PROJECT_ROOT/part1/train_pl.py" \
  --z_dim 20 \
  --num_filters 32 \
  --lr 1e-3 \
  --batch_size 128 \
  --data_dir "$DATA_LOCAL" \
  --epochs 80 \
  --num_workers 4 \
  --log_dir "$VAE_LOG_DIR_LOCAL" 

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Syncing logs/checkpoints back to persistent storage: $VAE_LOG_DIR"
cp -a "$VAE_LOG_DIR_LOCAL/." "$VAE_LOG_DIR/"

echo "Job finished at $(date)."
echo "Total execution time: $(($DURATION / 3600))h $(($DURATION % 3600 / 60))m $(($DURATION % 60))s"
