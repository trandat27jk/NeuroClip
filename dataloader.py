import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# File paths
image_dir = "./data/convert_img"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
# Load dataset
with open("neuronClip_data.json", encoding="utf-8") as file:
    data = json.load(file)

testing_data = pd.read_csv("testing_data.csv")
data_target = pd.read_csv("processed_data.csv")
train_test = data_target[-700:]


class NeuralClipDataset(Dataset):
    def __init__(self, dataset, image_dir, tokenizer, max_length=148):
        self.dataset = dataset
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            selected_version = random.randint(1, 5)
            pmid = int(list(self.dataset[index].keys())[0])
            text = self.dataset[index][str(pmid)].get(
                f"Version {selected_version}", None
            )
            if text is None:
                raise KeyError(f"Version {selected_version} not found for pmid {pmid}")

            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )["input_ids"][0]

            # Load and preprocess image
            image_file = os.path.join(self.image_dir, f"pmid_{pmid}.npy")
            if not os.path.exists(image_file):
                raise FileNotFoundError(f"Image file {image_file} not found")

            image = np.expand_dims(np.load(image_file), axis=0)
            image = image / (np.max(image) + 1e-9)  # Normalize between 0 and 1
            image = np.nan_to_num(image, copy=False)  # Replace NaN with 0

            return torch.Tensor(image), tokens
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            return None


class Testing_data(Dataset):
    def __init__(self, testing_ids, image_dir, dataset, tokenizer):
        super().__init__()
        self.testing_ids = testing_ids
        self.image_dir = image_dir
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.testing_ids)

    def __getitem__(self, idx):
        testing_ids = self.testing_ids.iloc[idx, 0]
        title = self.dataset[self.dataset["article-id"] == testing_ids][
            "abstract"
        ].iloc[0]
        title_tensor = self.tokenizer(
            title,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=63,
        )["input_ids"][0]
        image_file = os.path.join(self.image_dir, f"pmid_{testing_ids}.npy")
        image = np.expand_dims(np.load(image_file), axis=0)
        image = image / (np.max(image) + 1e-9)
        image = np.nan_to_num(image, copy=False)

        return torch.tensor(image), title_tensor


class Train_test_data(Dataset):
    def __init__(self, image_dir, dataset, tokenizer):
        super().__init__()
        self.testing_ids = dataset
        self.image_dir = image_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.testing_ids)

    def __getitem__(self, idx):
        title = self.testing_ids.iloc[idx, 0]
        testing_ids = self.testing_ids["article-id"].iloc[idx]
        title_tensor = self.tokenizer(
            title,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=50,
        )["input_ids"][0]
        image_file = os.path.join(self.image_dir, f"pmid_{testing_ids}.npy")
        image = np.expand_dims(np.load(image_file), axis=0)
        image = image / (np.max(image) + 1e-9)
        image = np.nan_to_num(image, copy=False)

        return torch.tensor(image), title_tensor


# Instantiate the dataset
train_dataset = NeuralClipDataset(data, image_dir, tokenizer)
test_dataset = Testing_data(testing_data, image_dir, data_target, tokenizer)
train_test_dataset = Train_test_data(image_dir, train_test, tokenizer)
