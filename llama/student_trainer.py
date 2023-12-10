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
    torch.autograd.set_detect_anomaly(True) #TODO uncomment this
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
    model_trainer = model_trainer.to(torch.float32)
    train_dataloader = model_trainer.train_dataloader()
    val_dataloader = model_trainer.val_dataloader()
    optimizers, lr_schedulers = model_trainer.configure_optimizers()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc='Training')):
        # for batch_idx, batch in enumerate(train_dataloader):
            loss = model_trainer.training_step(batch, batch_idx)
            if args.wandb:
                wandb.log({"train_loss": loss.item()}, step=epoch * len(train_dataloader) + batch_idx)
            print("loss", loss, "batch_idx", batch_idx)

            #TODO comment this out
            for name, param in model_trainer.student_model.named_parameters():#TODO comment this block out
                weight_mean = param.data.mean().item()
                weight_std = param.data.std().item()
                weight_norm = param.data.norm().item()

                # Log the weight statistics
                logging.debug(f"Layer: {name}, Weight before backpropMean: {weight_mean}, Weight Std: {weight_std}, Weight Norm: {weight_norm}")


            loss.backward()

            for name, param in model_trainer.student_model.named_parameters():#TODO comment this block out
                if param.grad is not None:
                    grad = param.grad
                    grad_mean = grad.mean().item()
                    grad_std = grad.std().item()
                    grad_norm = grad.norm().item()

                    # Log the statistics
                    logging.debug(f"Layer: {name}, Gradient Mean: {grad_mean}, Gradient Std: {grad_std}, Gradient Norm: {grad_norm}")
                    logging.debug(f"Layer: {name}, Gradient Mean: {grad_mean}, Gradient Std: {grad_std}, Gradient Norm: {grad_norm}")
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        logging.warning(f"Inf or NaN gradient in layer: {name}")
                else:
                    # Log that this parameter has no gradient
                    logging.debug(f"No gradient for {name}")
            # for name, param in model_trainer.student_model.named_parameters(): 
            #     if param.grad is not None:
            #         # Convert the gradients to a string representation
            #         grad_string = np.array2string(param.grad.cpu().detach().numpy())
            #         # Log the name and gradient of the parameter
            #         logging.debug(f"Gradient of {name}: {grad_string}")
            #     else:
            #         # Log that this parameter has no gradient
            #         logging.debug(f"No gradient for {name}")

            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    print("Learning rate:", param_group['lr'])
                # for name, param in model_trainer.student_model.named_parameters():
                #     if param.device.type != 'cuda' or param.dtype != torch.float32:
                #         print(f"Parameter {name} is on {param.device} with dtype {param.dtype}")
                # for name, buffer in model_trainer.student_model.named_buffers():
                #     if buffer.device.type != 'cuda' or buffer.dtype != torch.float32:
                #         print(f"Buffer {name} is on {buffer.device} with dtype {buffer.dtype}")
                optimizer.step()
                optimizer.zero_grad()
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()


            for name, param in model_trainer.student_model.named_parameters():#TODO comment this block out
                weight_mean = param.data.mean().item()
                weight_std = param.data.std().item()
                weight_norm = param.data.norm().item()

                # Log the weight statistics
                logging.debug(f"Layer: {name}, Weights after backprop Mean: {weight_mean}, Weight Std: {weight_std}, Weight Norm: {weight_norm}")


        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc='Validation')):
            loss = model_trainer.validation_step(batch, batch_idx)
            if args.wandb:
                wandb.log({"val_loss": loss.item()}, step=epoch * len(train_dataloader) + batch_idx)

        if epoch % args.epoch_save_rate == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch-{epoch:02d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_trainer.student_model.state_dict(),
                'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
                # Include other relevant information if needed
            }, checkpoint_path)
    
    # TODO test ckpt every 5 ep and logging manually


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser()
    
    parser.add_argument("--loss_fn", help="loss function for KD",
                        type=str, default='smooth')
    parser.add_argument("--includes_rejected", help="whether or not it includes rejected RLHF samples",
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
