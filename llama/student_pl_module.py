import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytorch_lightning as pl
import wandb  # Importing WandB for logging
from llama.model import ModelArgs, Transformer
import torch.nn.functional as F
from rlhf_data_loader import HHDataset, TokenizationCollator
import os
from llama import Llama3, Dialog # llama3 is version I made for using pl dist already
import torch.multiprocessing as mp

# Define a function to calculate accuracy
class StudentPLModule(pl.LightningModule):
    def __init__(self, hparams):
        super(StudentPLModule, self).__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        if self.hparams.loss_fn == 'smooth':
            self.loss_fn = self.smooth_loss
        elif self.hparams.loss_fn == 'cross_entropy':
            self.loss_fn = self.cross_entropy_loss
        else:
            raise ValueError(f"Invalid loss type specified {self.hparams.loss_fn}")
        
        #TODO may need to change how teacher model is loaded
        #TODO may need this line? mp.set_start_method('spawn', force=True)
        # mp.set_start_method('spawn', force=True)
        # self.teacher_model = Llama3.build(
        #     ckpt_dir="llama-2-7b-chat/",
        #     tokenizer_path="tokenizer.model",
        #     max_seq_len=self.hparams.max_seq_len,
        #     max_batch_size=self.hparams.max_batch_size,
        # )
        self.student_model = Transformer(hparams)
        self.collate_fn = TokenizationCollator(self.student_model.params)



    def smooth_loss(self, student_logits, teacher_logits):
        beta = 1.0
        return F.smooth_l1_loss(student_logits, teacher_logits, beta=beta)

    # Cross-entropy loss function
    def cross_entropy_loss(self, student_logits, teacher_logits):
        return F.cross_entropy(student_logits, teacher_logits)

    # Training step
    def training_step(self, batch, batch_idx):
        loss = self.eval_step(batch, batch_idx, "train")
        return loss

    # Validation step
    def validation_step(self, batch, batch_idx):
        loss = self.eval_step(batch, batch_idx, "val")
        return loss
    
    def eval_step(self, batch, batch_idx, step_type):
        tokens = batch['tokens'].to('cuda')
        min_prompt_len = batch['min_prompt_len']
        total_len = batch['total_len']
        prev_pos = 0
        if min_prompt_len == total_len:
            student_logits = self.student_model.forward(tokens, prev_pos, learning = True)
            teacher_logits = self.teacher_model.forward(tokens, prev_pos)
            #very unlikely to happen - tbh shouldnt during training 
        else:
            #TODO figure this out may need to not do prev_pos:min_prompt_len, tbh i think is fine check after sanity check
            student_logits = self.student_model.forward(tokens[:, prev_pos:min_prompt_len], prev_pos, learning = True)
            teacher_logits = self.teacher_model.forward(tokens[:, prev_pos:min_prompt_len], prev_pos)
        
        print("student_logits.shape", student_logits.shape, "teacher_logits.shape", teacher_logits.shape)
        loss = self.loss_fn(student_logits, teacher_logits)
        
        self.log(f"{step_type}_loss", loss)
        return loss

        
    def configure_optimizers(self):
        all_params = list(self.student_model.named_parameters())

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
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, collate_fn=self.collate_fn, num_workers=self.hparams.num_workers, persistent_workers=True, drop_last = True, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, collate_fn=self.collate_fn, num_workers=self.hparams.num_workers, persistent_workers=True, drop_last = True, shuffle = False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, collate_fn=self.collate_fn, num_workers=self.hparams.num_workers, persistent_workers=True, drop_last = False, shuffle = False)

# TODO in dataloader callers add support for includes_rejected 
    

