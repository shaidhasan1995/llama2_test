import json
import torch
from llama import Llama, Dialog
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
import re

class HHDataset(Dataset):
    def __init__(self, folder_paths):
        self.data = []
        for path in folder_paths:
            self.data.extend(self.load_data(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #TODO test in logit completion
        text = self.data[idx]['text']
        reformatted_group = self.reformat_dataset_grouped(text)
        return reformatted_group

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            data = [json.loads(line) for line in lines]
        return data
    
    @staticmethod
    def reformat_dataset_grouped(text):
        reformatted_group = []

        # Split the text based on the 'Human:' and 'Assistant:' tags
        parts = re.split(r'\n\n(Human|Assistant): ', text)

        # Remove the first empty element if it exists
        if parts and parts[0] == '':
            parts = parts[1:]

        # Mapping of original speaker to new role
        role_map = {'Human': 'user', 'Assistant': 'assistant'}

        # Process and reformat each part
        for i in range(0, len(parts), 2):
            if i + 1 >= len(parts):
                break  # Avoid IndexError if the number of parts is odd

            speaker = parts[i]
            content = parts[i + 1].strip()

            # Create a new dictionary for the reformatted text
            reformatted_group.append({"role": role_map.get(speaker, "unknown"), "content": content})

        return reformatted_group
    

#TODO later add collate_fn based off generator code and returns tokenized text rather than just text
# before doing it ctrl f repo if has collator
class TokenizationCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        pass
