singularity exec --env HF_HOME=$HF_HOME --env PYTHONPATH=/usr/local/lib/python3.8/site-packages/ --nv /scratch/ssb3vk/MLIA/mlia_fp.sif [command_such_as_python] [your_code_file.with_extention] \
--max_seq_len 2048 \
--dim 2048 \
--n_layers 20 \
--multiple_of 64 \
--ffn_dim_multiplier 0.25 \
--loss_fn 'smooth' \
-bs 32 \
--num_workers 8 \
-ep 200 \
--epoch_save_rate 5 \
-lr 0.0003 \
--weight_decay 0 \
--wandb_log_name "debug"

#dont worry about lines 2-14 they are just hparams for this specific project, replace them with the hparams you sent
