# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
import fire
from llama import Llama, Dialog
from torch.utils.data import Dataset, DataLoader
import os
from rlhf_data_loader import HHDataset

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8, #TODO try more vals
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    base_path = "./hh-rlhf2/"
    train_file_name = 'train.jsonl.gz'
    val_file_name = 'test.jsonl.gz' # using test split as validation
    folders = ['harmless-base', 'helpful-base', 'helpful-online', 'helpful-rejection-sampled']
    train_folders = [os.path.join(base_path, folder, train_file_name) for folder in folders]
    val_folders = [os.path.join(base_path, folder, val_file_name) for folder in folders]
    combined = train_folders + val_folders
    print("combined", combined)
    dataset = HHDataset(combined)
    dataloader = DataLoader(dataset, batch_size=max_batch_size, num_workers = 8, shuffle=False)
    #then save those logits in a format friendly with the hh-rlhf dataset that shows what the logits for each token were 
    for idx, batch in enumerate(dataloader):
        print(batch, idx)
        # dialogs: List[Dialog] = [
        #     [{"role": "user", "content": "hey, how are you, what is the recipe of mayonnaise?"}],
        # ]

        logits = generator.next_logits(
            batch,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=0,
            top_p=top_p,
        )
        # print("logits.shape", logits.shape) yields bs, tokens, vocab_len tensor
        # TODO here save this tensor along with the text, see if possible to save token_id, logits? whatever allows for easy dataloading in the future




if __name__ == "__main__":
    fire.Fire(main)
