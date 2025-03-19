import torch
from transformers import BertForTokenClassification, BertTokenizerFast
from src.config import CFG

class NERModel:
    def __init__(self, model_name="bert-base-uncased", num_labels=None, device=None):
        """
        Initializes the NER model.
        :param model_name: Pretrained model name (default: bert-base-uncased)
        :param num_labels: Number of unique entity labels
        :param device: Device (CPU/GPU) for model execution
        """
        if num_labels is None:
            raise ValueError("num_labels must be specified.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)

    def save(self, save_directory):
        """
        Saves the model and tokenizer to the specified directory.
        :param save_directory: Path to save the model
        """
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        print(f"Model and tokenizer saved to {save_directory}.")

    @classmethod
    def load(cls, model_directory, num_labels=None, device=None):
        """
        Loads a trained model and tokenizer from a directory.
        :param model_directory: Path to the saved model
        :param device: Device to load the model onto
        :param num_labels: The number of labels of the dataset
        :return: Loaded NERModel instance
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_directory} on {device}...")
        model = cls(model_name=model_directory, num_labels=num_labels, device=device)
        return model
