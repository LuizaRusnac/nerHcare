import torch
from transformers import Trainer, pipeline
from src.model import NERModel
from src.data_loader import load_label_mappings
from src.config import CFG

# Configuration
MODEL_DIR = "models/saved_ner_model"

def align_labels(results, id_to_label):
    words = []
    labels = []
    words.append(results[0]['word'])
    labels.append(id_to_label[results[0]['entity'].replace("LABEL_", "")])

    for result in results[1:]:
        word = result['word']
        label = id_to_label[result['entity'].replace("LABEL_", "")]
        if word.startswith("##"):
            words[-1] += word[2:]
        else:
            words.append(word)
            labels.append(label)

    # Pair the correctly reconstructed words with their labels
    return words, labels

def predict(text, id_to_label):
    """
    Performs Named Entity Recognition (NER) inference on a given text.
    :param text: Input sentence for NER
    :param model: Trained BERT-based NER model
    :param tokenizer: Corresponding tokenizer
    :param id_to_label: Mapping of label indices to label names
    :return: List of tokens with their predicted entity labels
    """

    ner_pipeline = pipeline("ner", model=MODEL_DIR)
    ner_results = ner_pipeline(text)
    words, labels = align_labels(ner_results, id_to_label)
    
    return zip(words, labels)

def main():
    """Loads the trained model and runs inference on user-input text."""
    print("Loading label mappings...")
    _, id_to_label = load_label_mappings(MODEL_DIR)

    while True:
        text = input("\nEnter a sentence for NER (or type 'exit' to quit): ")

        if text.lower() == "exit":
            print("Exiting NER inference.")
            break

        results = predict(text, id_to_label)

        print("\nNamed Entity Recognition Results:")
        for word, label in results:
            print(f"{word}: {label}")

if __name__ == "__main__":
    main()
