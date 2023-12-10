singularity exec --env HF_HOME=$HF_HOME --env PYTHONPATH=~/.local/lib/python3.8/site-packages:/usr/local/lib/python3.8/site-packages/ --nv /scratch/abg4br/containers/nlp_fp.sif torchrun --nproc_per_node 1 student_trainer.py \
--max_seq_len 1024 \
--dim 2048 \
--n_layers 20 \
--multiple_of 64 \
--ffn_dim_multiplier 0.25 \
--loss_fn 'cross_entropy' \
-bs 32 \
--num_workers 8 \
-ep 200 \
--epoch_save_rate 1 \
-lr 0.000003 \
--weight_decay 0 \
--wandb_log_name "debug"

#--wandb

#TODO add hparams for model size, may need to mess with them
#TODO play with bs
# ~/.local/lib/python3.8/site-packages: