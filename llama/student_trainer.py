from pytorch_lightning import loggers, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import os
from student_pl_module import StudentPLModule
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import datetime
from typing import Optional



def main(args):
    seed_everything(33)
    wandb_logger = WandbLogger(save_dir="logs/", 
                                    name=f'{args.wandb_log_name}',
                                    entity=f'{args.wandb_entity}',
                                    project=f'{args.wandb_project_name}')
    timestamp = int(time.time())
    dt_object = datetime.fromtimestamp(timestamp)
    dt_string = dt_object.strftime("%Y-%m-%d_%H-%M-%S")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./logs/{args.wandb_log_name}_{dt_string}',
        filename='epoch-{epoch:02d}',
        save_top_k=-1,  # Set to -1 to save all checkpoints
        every_n_epochs=args.epoch_save_rate  # Save every 5 epochs
    )
    model = StudentPLModule(args)
    trainer = Trainer(gpus=-1, max_epochs = args.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback]) 
    trainer.fit(model)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser()
    
    parser.add_argument("--loss_fn", help="loss function for KD",
                        type=float, default=3e-4)
    parser.add_argument("--includes_rejected", help="whether or not it includes rejected RLHF samples",
                        action="store_true", default=False)
    # TODO mess with below args to make model less than 700M params, particularly n_layers and multiple_of
    parser.add_argument('--dim', type=int, default=4096, help='Dimension size')
    parser.add_argument('--n_layers', type=int, default=32, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=32, help='Number of heads')
    parser.add_argument('--n_kv_heads', type=Optional[int], default=None, help='Number of key/value heads')
    parser.add_argument('--vocab_size', type=int, default=-32000, help='Vocabulary size (defined later by tokenizer)')
    parser.add_argument('--multiple_of', type=int, default=256, help='Multiple for SwiGLU hidden layer size')
    parser.add_argument('--ffn_dim_multiplier', type=Optional[float], default=None, help='FFN dimension multiplier')
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
    args = parser.parse_args()
    main(args=args)
