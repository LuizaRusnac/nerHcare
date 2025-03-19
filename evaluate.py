import torch
import numpy as np
from transformers import Trainer
from sklearn.metrics import precision_recall_fscore_support
from model import NERModel
from data_loader import load_data, create_label_mappings, preprocess_data, load_data_pkl, load_label_mappings

# Configuration
MODEL_DIR = "models\saved_ner_model_test"

def compute_metrics(predictions, labels):
    """
    Computes precision, recall, and F1-score for NER evaluation.
    :param predictions: Model predictions
    :param labels: True labels
    :return: Dictionary of precision, recall, and F1-score
    """
    predictions = np.argmax(predictions, axis=2)  # Convert logits to label indices

    true_labels, true_predictions = [], []

    for lbl_seq, pred_seq in zip(labels, predictions):
        for lbl, pred in zip(lbl_seq, pred_seq):
            if lbl != -100:  # Ignore padding labels
                true_labels.append(lbl)
                true_predictions.append(pred)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average='weighted')

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def evaluate():
    """Loads the trained model and evaluates it on the test dataset."""
    print("Loading dataset...")
    test_tokenized = load_data_pkl('/processed/test_dataset.pkl')

    print("Loading label mappings...")
    label_to_id, id_to_label = load_label_mappings(MODEL_DIR)

    print("Loading trained model...")
    model = NERModel.load(MODEL_DIR, num_labels=len(label_to_id))

    # Use Hugging Face Trainer for evaluation
    trainer = Trainer(model=model.model)
    print("Evaluating model...")
    results = trainer.evaluate(eval_dataset=test_tokenized)

    # Compute detailed metrics
    print("Computing precision, recall, and F1-score...")
    logits, labels, _ = trainer.predict(test_tokenized)
    metrics = compute_metrics(logits, labels)

    print("Evaluation Results:")
    print(metrics)

if __name__ == "__main__":
    evaluate()