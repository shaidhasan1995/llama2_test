import json
import torch
from llama import Llama, Dialog
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
import re
from llama.tokenizer import Tokenizer

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
B_INST, E_INST = "[INST]", "[/INST]"


class HHDataset(Dataset):
    def __init__(self, folder_paths, params):
        #TODO add bool to be passed in that determines if calls reformat_dataset_grouped as offline kd dataset will alr be preprocessed
        self.params = params
        self.tokenizer = Tokenizer(model_path='tokenizer.model')
        self.data = []
        for path in folder_paths:
            self.data.extend(self.load_data(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        dialog = self.reformat_dataset_grouped(text)
        # print("dialog", dialog)
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        roles_are_correct = all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        )
        if not roles_are_correct:
            i = 0
            while i < len(dialog):
                expected_role = "user" if i % 2 == 0 else "assistant"
                if dialog[i]["role"] != expected_role:
                    dialog.pop(i)
                else:
                    i += 1
        # assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        #     [msg["role"] == "assistant" for msg in dialog[1::2]]
        # ), (
        #     "model only supports 'system', 'user' and 'assistant' roles, "
        #     "starting with 'system', then 'user' and alternating (u/a/u/a/u...)", dialog
        # )
        dialog_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                    bos=True,
                    eos=True,
                )
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        # assert ( TODO check removing this doesnt break anything
        #     dialog[-1]["role"] == "user"
        # ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += self.tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            bos=True,
            eos=False,
        )
        return dialog_tokens

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
    

class TokenizationCollator:
    def __init__(self, params, device = None):
        self.params = params
        self.tokenizer = Tokenizer(model_path='tokenizer.model')
        self.device = device


    def __call__(self, prompt_tokens):
        max_gen_len = 1
        params = self.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        if max_prompt_len >params.max_seq_len:
            # print(f"max prompt len is {max_prompt_len} while max seq len is {params.max_seq_len}")
            prompt_tokens = [t[:params.max_seq_len] for t in prompt_tokens]
            #truncate 
        # assert max_prompt_len <= params.max_seq_len, f"max prompt len is {max_prompt_len} while max seq len is {params.max_seq_len}"
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)
        # print("total_len", total_len)

        pad_id = self.tokenizer.pad_id
        if self.device is None:
            tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)
            for k, t in enumerate(prompt_tokens):
                tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)
        else:
            tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device = self.device)
            for k, t in enumerate(prompt_tokens):
                tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device = self.device)

        batch = {'tokens' : tokens, 'min_prompt_len' : min_prompt_len, 'total_len' : total_len}
        return batch
