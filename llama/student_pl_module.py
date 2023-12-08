import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytorch_lightning as pl
import wandb  # Importing WandB for logging
from llama.model import ModelArgs, Transformer
import torch.nn.functional as F




# Define a function to calculate accuracy
class StudentPLModule(pl.LightningModule):
    def __init__(self, hparams):
        super(StudentPLModule, self).__init__()
        self.hparams = hparams
        self.model = Transformer(hparams)
        if self.hparams.loss_fn == 'smooth':
            self.loss_fn = self.smooth_loss
        elif self.hparams.loss_fn == 'cross_entropy':
            self.loss_fn = self.cross_entropy_loss
        else:
            raise ValueError(f"Invalid loss type specified {self.hparams.loss_fn}")

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
            text, teacher_logits = batch
            #TODO do things with text that are done in generation.py (or add option to dataloader to do this)
            student_logits = self.model(tokens, start_pos=0)
            
            print("student_logits.shape", student_logits.shape, "teacher_logits.shape", teacher_logits.shape)
            loss = self.loss_fn(student_logits, teacher_logits)
            
            self.log(f"{step_type}_loss", loss)
            return loss

        
    def configure_optimizers(self):
        all_params = list(self.model.named_parameters())

        optimizer_parameters = [
            {'params': all_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.learning_rate}  # Weight decay for other parameters
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                        T_0=self.hparams.epochs+1,
                                                                        eta_min=self.hparams.learning_rate / 10)
        return [optimizer], [lr_scheduler]
# TODO in dataloader callers add support for includes_rejected 
    