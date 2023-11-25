import json
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, file_path):
        self.data = self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            data = [json.loads(line) for line in lines]
        return data

# Assuming your data is in a file named 'data.txt'
file_path = "./hh-rlhf/harmless-base/human_test.txt"

# Create an instance of your custom dataset
dataset = MyDataset(file_path)

# Create a PyTorch data loader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example usage in a loop
for batch in dataloader:
    # Access your data in the batch
    print(batch)

    print("\n \n")
