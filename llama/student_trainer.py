from pytorch_lightning import loggers, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import os
from student_pl_module import StudentPLModule
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import datetime



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
    # TODO add support for all below args
    # class ModelArgs:
    # dim: int = 4096
    # n_layers: int = 32
    # n_heads: int = 32
    # n_kv_heads: Optional[int] = None
    # vocab_size: int = -1  # defined later by tokenizer
    # multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    # ffn_dim_multiplier: Optional[float] = None
    # norm_eps: float = 1e-5

    # max_batch_size: int = 32
    # max_seq_len: int = 2048
    # KD HPARAMS ####################################################
    parser.add_argument("-lr", "--learning_rate", help="learning rate",
                        type=float, default=3e-4)
    parser.add_argument("--weight_decay", help="weight_decay",
                        type=float, default=0)
    parser.add_argument("-bs", "--batch_size", help="batch size",
                        type=int, default=32)
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
