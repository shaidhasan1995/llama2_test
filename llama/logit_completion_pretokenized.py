# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
import fire
from llama import Llama2, Dialog # llama2 is version I made for running pretokenization already
import torch

from torch.utils.data import Dataset, DataLoader
import os
from rlhf_data_loader import HHDataset, TokenizationCollator
import torch.multiprocessing as mp

def main(
    ckpt_dir: str,
    tokenizer_path: str,
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
    mp.set_start_method('spawn', force=True)
    generator = Llama2.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    base_path = "./hh-rlhf2/"
    train_file_name = 'all_train.txt'
    val_file_name = 'all_test.txt' # using test split as validation
    folders = ['harmless-base', 'helpful-base', 'helpful-online', 'helpful-rejection-sampled']
    train_folders = [os.path.join(base_path, folder, train_file_name) for folder in folders]
    val_folders = [os.path.join(base_path, folder, val_file_name) for folder in folders]
    combined = train_folders + val_folders
    dataset = HHDataset(combined, generator.model.params)
    collate_fn = TokenizationCollator(generator.model.params)
    dataloader = DataLoader(dataset, batch_size=max_batch_size, collate_fn = collate_fn, num_workers = 8, shuffle=False)
    sample_id = 0
    save_dir = "./kd-data/all"
    os.makedirs(save_dir, exist_ok=True)
    accumulated_data = []
    #then save those logits in a format friendly with the hh-rlhf dataset that shows what the logits for each token were 
    print("len(dataloader)", len(dataloader))
    for idx, batch in enumerate(dataloader):
        logits = generator.next_logits(batch)
        print("idx", idx)
        print("logits.shape", logits.shape) #yields batch_size, tokens, vocab_len tensor
        print("tokens.shape", batch['tokens'].shape)
        #TODO figure out why tokens is all same shape?
        torch.set_printoptions(threshold=5000)

        for sample_idx in range(logits.size(0)):
            sample_logits = logits[sample_idx]
            sample_tokens = batch['tokens'][sample_idx]
            save_data = {
                'logits': sample_logits,
                'tokens': sample_tokens
            }
            save_path = os.path.join(save_dir, f'sample_{sample_id}.pt')
            torch.save(save_data, save_path)

            sample_id += 1


        #TODO do sanity check decode all tokens and argmax and decode logits make sure makes sense




if __name__ == "__main__":
    fire.Fire(main)
