import logging
import os
from config import CFG

# Create logs directory if it doesn't exist
os.makedirs(CFG.LOG_DIR, exist_ok=True)

def setup_logger(name, log_file, level=logging.INFO):
    """Sets up a logger with file and console output."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Define loggers for different modules
train_logger = setup_logger("train", os.path.join(CFG.LOG_DIR, "training.log"))
eval_logger = setup_logger("evaluate", os.path.join(CFG.LOG_DIR, "evaluation.log"))
inference_logger = setup_logger("inference", os.path.join(CFG.LOG_DIR, "inference.log"))

# Example usage
if __name__ == "__main__":
    train_logger.info("Training started...")
    eval_logger.info("Evaluation started...")
    inference_logger.info("Inference request received...")