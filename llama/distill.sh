singularity exec --env HF_HOME=$HF_HOME --env PYTHONPATH=~/.local/lib/python3.8/site-packages:/usr/local/lib/python3.8/site-packages/ --nv /scratch/abg4br/containers/nlp_fp.sif python3 student_trainer.py \
--loss_fn 'smooth' \
-bs 32 \
--num_workers 8 \
-ep 200 \
--epoch_save_rate 5 \
-lr 0.0003 \
--weight_decay 0 \
--wandb_log_name "debug"
#TODO add hparams for model size

# ~/.local/lib/python3.8/site-packages: