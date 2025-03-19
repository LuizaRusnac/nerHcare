# Stop the script if any command fails
set -e

echo "Starting model evaluation..."

# Create logs directory if not exists
mkdir -p logs

# Run evaluation and save logs
python evaluate.py 2>&1 | tee logs/evaluation.log

echo "Evaluation complete. Logs saved to logs/evaluation.log"