import json
import torch
from llama import Llama, Dialog
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader

class HHDataset(Dataset):
    def __init__(self, file_path):
        # TODO add support for multiple file_paths
        self.data = self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #TODO change this so it returns lines that will end up in this format:
        dialogs: List[Dialog] = [
            [{"role": "user", "content": "hey, how are you, what is the recipe of mayonnaise?"}],
        ]

        return self.data[idx]

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            data = [json.loads(line) for line in lines]
        return data

#TODO add ability to pass in batch_idx and logits to write to them? need to figure out bc batch_idx != idx
#TODO add support so just returns tokenized text rather than text
