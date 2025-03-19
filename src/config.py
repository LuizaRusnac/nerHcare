import torch

class CFG:
    # General settings
    MODEL_NAME = "bert-base-uncased"  # Pre-trained BERT model
    MODEL_DIR = "models/saved_ner_model_test"  # Directory to save/load the trained model

    # Dataset
    DATASET_NAME = "ktgiahieu/maccrobat2018_2020"  # Hugging Face dataset
    DATASET_DIR = "data"
    TEST_SIZE = 0.2  # Train-test split ratio

    # Training settings
    BATCH_SIZE = 16  # Adjust based on GPU memory
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 1
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500
    GRADIENT_ACCUMULATION_STEPS = 1  # Increase if batch size is small

    # Device settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging & output directories
    OUTPUT_DIR = "output"
    LOG_DIR = "logs"

    # FastAPI settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
