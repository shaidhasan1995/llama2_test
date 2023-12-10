# from pytorch_lightning import loggers, Trainer, seed_everything
# from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import os
from student_pl_module import StudentPLModule
# from pytorch_lightning.callbacks import ModelCheckpoint
import time
import datetime
from datetime import datetime
from typing import Optional
from tqdm import tqdm
import torch
import wandb
import logging
import numpy as np

def main(args):
    logging.basicConfig(filename='logs/debug.log', level=logging.DEBUG)
    if args.wandb:
        wandb.init(dir="logs/", 
           name=args.wandb_log_name, 
           entity=args.wandb_entity, 
           project=args.wandb_project_name)
    timestamp = int(time.time())
    dt_object = datetime.fromtimestamp(timestamp)
    dt_string = dt_object.strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = f'./logs/{args.wandb_log_name}_{dt_string}'

    model_trainer = StudentPLModule(args)
    model_trainer.cuda()
    model_trainer.setup(stage = "fit")
    model_trainer.train()
    # model_trainer = model_trainer.to(torch.float32)
    train_dataloader = model_trainer.train_dataloader()
    val_dataloader = model_trainer.val_dataloader()
    optimizers, lr_schedulers = model_trainer.configure_optimizers()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc='Training')):
        # for batch_idx, batch in enumerate(train_dataloader):
            loss = model_trainer.training_step(batch, batch_idx)
            if args.wandb:
                wandb.log({"train_loss": loss.item()}, step=epoch * len(train_dataloader) + batch_idx)
            # print("loss", loss, "batch_idx", batch_idx)

            loss.backward()

            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()


        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc='Validation')):
            loss = model_trainer.validation_step(batch, batch_idx)
            if args.wandb:
                wandb.log({"val_loss": loss.item()}, step=epoch * len(train_dataloader) + batch_idx)
            
            

        if epoch % args.epoch_save_rate == 0:
            # Check if the checkpoint directory exists, and if not, create it
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            checkpoint_path = os.path.join(checkpoint_dir, f'epoch-{epoch:02d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_trainer.student_model.state_dict(),
                'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
                # Include other relevant information if needed
            }, checkpoint_path)



if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser()
    
    parser.add_argument("--loss_fn", help="loss function for KD",
                        type=str, default='smooth')
    parser.add_argument("--includes_rejected", help="whether or not it includes rejected RLHF samples",
                        action="store_true", default=False)
    parser.add_argument("--all_logits", help="uses all logits for KD instead of those truncated to min batch len",
                        action="store_true", default=False)
    
    parser.add_argument('--dim', type=int, default=4096, help='Dimension size')
    parser.add_argument('--n_layers', type=int, default=32, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=32, help='Number of heads')
    parser.add_argument('--n_kv_heads', type=int, default=None, help='Number of key/value heads')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size (defined later by tokenizer)')
    parser.add_argument('--multiple_of', type=int, default=256, help='Multiple for SwiGLU hidden layer size')
    parser.add_argument('--ffn_dim_multiplier', type=float, default=None, help='FFN dimension multiplier')
    parser.add_argument('--norm_eps', type=float, default=1e-5, help='Normalization epsilon')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Maximum sequence length')

    # KD HPARAMS ####################################################
    parser.add_argument("-lr", "--learning_rate", help="learning rate",
                        type=float, default=3e-4)
    parser.add_argument("--weight_decay", help="weight_decay",
                        type=float, default=0)
    parser.add_argument("-bs", "--max_batch_size", help="batch size",
                        type=int, default=32)
    parser.add_argument("--num_workers", help="number of workers",
                        type=int, default=8)
    parser.add_argument("-ep", "--epochs", help="epoch per validation cycle",
                        type=int, default=200)
    parser.add_argument("--epoch_save_rate", help="rate at which to save every X epochs",
                        type=int, default=5)
    parser.add_argument("--wandb_log_name", help="wandb_log_name",
                        default=None)
    parser.add_argument("--wandb_entity", help="wandb_entity",
                        default='nlp_final_project_uva')
    parser.add_argument("--wandb_project_name", help="wandb_project_name",
                        default='final_project')
    parser.add_argument("--wandb", help="whether wandb is on",
                        action="store_true", default=False)
    args = parser.parse_args()
    main(args=args)
