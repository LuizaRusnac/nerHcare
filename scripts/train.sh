#!/bin/bash

# Stop on errors
set -e

echo "Starting NER model training..."

# Create necessary directories
mkdir -p logs output models/saved_ner_model

# Activate virtual environment (if using one)
# source venv/bin/activate  # Uncomment if using a virtual environment

# Run training
python train.py 2>&1 | tee logs/training.log

echo "Training complete. Logs saved to logs/training.log"