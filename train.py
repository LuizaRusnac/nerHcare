import os
import torch
from transformers import Trainer, TrainingArguments
from src.model import NERModel
from src.data_loader import load_data, create_label_mappings, preprocess_data, create_data_loader, save_label_mappings, save_data_pkl
from src.config import CFG
from src.logger import train_logger

def train():
    """Trains the NER model using Hugging Face's Trainer API."""

    # Load dataset
    print("Loading dataset...")
    train_data, test_data = load_data()

    # Create label mappings
    print("Creating label mappings...")
    label_to_id, id_to_label = create_label_mappings(train_data)

    #Save label mapping
    print("Saving label mappings...")
    save_label_mappings(label_to_id, id_to_label, CFG.MODEL_DIR)

    # Preprocess data (tokenization + alignment)
    print("Tokenizing datasets...")
    train_tokenized, test_tokenized = preprocess_data(train_data, test_data, label_to_id)

    # Create DataLoaders
    # print("Creating DataLoaders...")
    # train_dataloader = create_data_loader(train_tokenized, batch_size=CFG.BATCH_SIZE)
    # test_dataloader = create_data_loader(test_tokenized, batch_size=CFG.BATCH_SIZE)

    # Initialize model
    print("Initializing model...")
    model = NERModel(model_name=CFG.MODEL_NAME, num_labels=len(label_to_id))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=CFG.OUTPUT_DIR,
        num_train_epochs=CFG.NUM_EPOCHS,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        per_device_eval_batch_size=CFG.BATCH_SIZE,
        warmup_steps=CFG.WARMUP_STEPS,
        weight_decay=CFG.WEIGHT_DECAY,
        logging_dir=CFG.LOG_DIR,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CFG.LEARNING_RATE,
        fp16=True if torch.cuda.is_available() else False,
    )

    # Define Trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
    )

    # Train model
    print("Starting training...")
    trainer.train()

    # Save trained model
    print(f"Saving model to {CFG.MODEL_DIR}...")
    model.save(CFG.MODEL_DIR)

    print("Training complete!")

if __name__ == "__main__":
    train_logger.info("Training started...")
    try:
        train()
        train_logger.info("Training completed successfully.")
    except Exception as e:
        train_logger.error(f"Training failed: {e}")
