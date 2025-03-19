import pickle
import json
import os
import torch
from datasets import load_dataset
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from src.config import CFG

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(CFG.MODEL_NAME)

def load_data(dataset_name=CFG.DATASET_NAME, test_size=0.2):
    """Loads and splits the dataset into training and testing sets."""
    dataset = load_dataset(dataset_name)['train']
    train_test_split = dataset.train_test_split(test_size=test_size)
    return train_test_split['train'], train_test_split['test']

def save_data_pkl(data, name='dataset.pkl'):
    # Save the datasets using pickle
    name = CFG.DATASET_DIR + '/' + name
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def load_data_pkl(name='dataset.pkl'):
    name = CFG.DATASET_DIR + '/' + name
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def create_label_mappings(dataset):
    """Creates label-to-ID and ID-to-label mappings."""
    unique_labels = sorted(set(tag for tags in dataset['tags'] for tag in tags))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label

def save_label_mappings(label_to_id, id_to_label, output_dir):
    """Save label mappings as a JSON file."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    label_mappings = {
        "label_to_id": label_to_id,
        "id_to_label": id_to_label
    }
    with open(os.path.join(output_dir, "label_mappings.json"), "w") as f:
        json.dump(label_mappings, f, indent=4)
    print(f"Label mappings saved to {os.path.join(output_dir, 'label_mappings.json')}")

def load_label_mappings(model_dir):
    """Load label mappings from JSON file."""
    with open(os.path.join(model_dir, "label_mappings.json"), "r") as f:
        label_mappings = json.load(f)
    return label_mappings["label_to_id"], label_mappings["id_to_label"]

def tokenize_and_align_labels(examples, label_to_id):
    """Tokenizes input texts and aligns entity labels with subword tokens."""
    tokenized_inputs = tokenizer(
        examples['tokens'],
        is_split_into_words=True,
        padding='longest',
        truncation=True
    )

    labels = []
    for i, label in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(tokenized_inputs['input_ids'][i])
        current_label = None

        for j, word_id in enumerate(word_ids):
            if word_id is None:
                continue  # Skip non-word tokens
            if j == 0 or word_ids[j] != word_ids[j - 1]:  # Start of a new word
                current_label = label[word_id]
                label_ids[j] = label_to_id.get(current_label, -100)
            else:
                label_ids[j] = label_to_id.get(current_label, -100)

        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def preprocess_data(train_data, test_data, label_to_id):
    """Tokenizes and aligns labels for train and test datasets."""
    train_tokenized = train_data.map(lambda x: tokenize_and_align_labels(x, label_to_id), batched=True)
    test_tokenized = test_data.map(lambda x: tokenize_and_align_labels(x, label_to_id), batched=True)

    # Save processed datasets
    with open("data/processed/train_dataset.pkl", "wb") as f:
        pickle.dump(train_tokenized, f)
    with open("data/processed/test_dataset.pkl", "wb") as f:
        pickle.dump(test_tokenized, f)

    return train_tokenized, test_tokenized

def create_data_loader(tokenized_dataset, batch_size=8):
    """Creates a PyTorch DataLoader for batching."""
    def collate_fn(batch):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_masks = torch.tensor([item['attention_mask'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': labels}

    return DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=collate_fn)

def verify_labels(tokenized_datasets, num_samples=1):
    for idx in range(num_samples):
        # Extract the input_ids, attention masks, and labels for each entry
        input_ids = tokenized_datasets['input_ids'][idx]
        labels = tokenized_datasets['labels'][idx]
        
        # Get the tokens from the tokenizer
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Print tokens along with their corresponding labels
        print(f"Sample {idx + 1}:")
        for token, label_id in zip(tokens[:100], labels[:100]):
            # Convert label ID back to label using id_to_label mapping
            label = id_to_label.get(label_id, "O")  # Default to "O" for -100
            print(f"{token:20} -> {label}")
        print("\n" + "-"*50 + "\n")

# # Call the verification function
# verify_labels(test_tokenized)

# Main execution for loading and preprocessing data
if __name__ == "__main__":
    print("Loading dataset...")
    train_data, test_data = load_data()

    print("Creating label mappings...")
    label_to_id, id_to_label = create_label_mappings(train_data)

    print("Tokenizing datasets...")
    train_tokenized, test_tokenized = preprocess_data(train_data, test_data, label_to_id)

    print("Creating DataLoaders...")
    train_dataloader = create_data_loader(train_tokenized)
    test_dataloader = create_data_loader(test_tokenized)

    print("Data preparation complete!")