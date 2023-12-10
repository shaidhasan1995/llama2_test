import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# import pytorch_lightning as pl
import wandb  # Importing WandB for logging
from llama.model import ModelArgs, Transformer
from model2 import SimpleTransformer
import torch.nn.functional as F
from rlhf_data_loader import HHDataset, TokenizationCollator
import os
from llama import Llama2, Dialog # llama3 is version I made for using pl dist already
import torch.multiprocessing as mp
from torch import nn


# Define a function to calculate accuracy
class StudentPLModule(nn.Module):
    def __init__(self, hparams):
        super(StudentPLModule, self).__init__()
        self.hparams = hparams
        if self.hparams.loss_fn == 'smooth':
            self.loss_fn = nn.SmoothL1Loss(beta=1.0)
        elif self.hparams.loss_fn == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid loss type specified {self.hparams.loss_fn}")
        
        mp.set_start_method('spawn', force=True)
        self.teacher_model = Llama2.build(
            ckpt_dir="llama-2-7b-chat/",
            tokenizer_path="tokenizer.model",
            max_seq_len=self.hparams.max_seq_len,
            max_batch_size=self.hparams.max_batch_size,
        )
        print("teacher number of params", sum(p.numel() for p in self.teacher_model.model.parameters()))

        # self.student_model = Transformer(hparams)
        self.student_model = SimpleTransformer(hparams)
        self.student_model.weight_initialization()

        # print("student number of params requires grad", sum(p.numel() for p in self.student_model.parameters() if p.requires_grad))
        print("student number of params", sum(p.numel() for p in self.student_model.parameters()))

        self.collate_fn = TokenizationCollator(self.student_model.params, None)
        



    def training_step(self, batch, batch_idx):
        loss = self.eval_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.eval_step(batch, batch_idx, "val")
        return loss
    
    def eval_step(self, batch, batch_idx, stage):
        tokens = batch['tokens'].to('cuda')
        # print("tokens.dtype", tokens.dtype) int 64
        min_prompt_len = batch['min_prompt_len']
        total_len = batch['total_len']
        prev_pos = 0
        if min_prompt_len == total_len:
            student_logits = self.student_model.forward(tokens, prev_pos, learning = (stage == "train"))
            teacher_logits = self.teacher_model.model.forward(tokens, prev_pos)
            #very unlikely to happen - tbh shouldnt during training 
        else:
            # print("tokens.shape", tokens.shape)
            if self.hparams.all_logits:
                student_logits = self.student_model.forward(tokens, prev_pos, learning = (stage == "train"))
                teacher_logits = self.teacher_model.model.forward(tokens, prev_pos)
            else:
                #TODO check tokens[:, prev_pos:min_prompt_len]
                student_logits = self.student_model.forward(tokens[:, prev_pos:min_prompt_len], prev_pos, learning = (stage == "train"))
                teacher_logits = self.teacher_model.model.forward(tokens[:, prev_pos:min_prompt_len], prev_pos)
        
        # print("student_logits.shape", student_logits.shape)
        # print("teacher_logits.shape", teacher_logits.shape)
        teacher_logits = teacher_logits.clone()

        loss = self.loss_fn(student_logits, teacher_logits)
        return loss

        
    def configure_optimizers(self):
        all_params = list(self.student_model.parameters())
        # print(all_params)
        optimizer_parameters = [
            {'params': all_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.learning_rate}  # Weight decay for other parameters
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                        T_0=self.hparams.epochs+1,
                                                                        eta_min=self.hparams.learning_rate / 10)
        return [optimizer], [lr_scheduler]
    

    def setup(self, stage=None):
        if stage == "fit":          
            base_path = "./hh-rlhf2/"
            train_file_name = 'all_train.txt'
            val_file_name = 'all_test.txt' # using test split as validation
            folders = ['harmless-base', 'helpful-base', 'helpful-online', 'helpful-rejection-sampled']
            train_folders = [os.path.join(base_path, folder, train_file_name) for folder in folders]
            val_folders = [os.path.join(base_path, folder, val_file_name) for folder in folders]
            self.train_ds = HHDataset(train_folders, self.student_model.params)
            self.val_ds = HHDataset(val_folders, self.student_model.params)
            print("len(self.train_ds)", len(self.train_ds))
            print("len(self.val_ds)", len(self.val_ds))
        else:
            raise ValueError(f"Have not implemented stage {stage} yet")


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.max_batch_size, collate_fn=self.collate_fn, num_workers=self.hparams.num_workers, persistent_workers=True, drop_last = True, shuffle = False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.max_batch_size, collate_fn=self.collate_fn, num_workers=self.hparams.num_workers, persistent_workers=True, drop_last = True, shuffle = False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.max_batch_size, collate_fn=self.collate_fn, num_workers=self.hparams.num_workers, persistent_workers=True, drop_last = False, shuffle = False)

# TODO in dataloader callers add support for includes_rejected and make train shuffle
    

