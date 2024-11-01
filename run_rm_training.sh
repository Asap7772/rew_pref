# --dataset Asap7772/llama_3.1_binary_train_processed_hhstyle \

accelerate launch --config_file configs/accelerate/zero2.yaml \
           train_reward_model.py \
           --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
           --dataset Asap7772/llama_3.1_binary_train_small_processed_hhstyle \
           --batch_size 1 \
           --eval_interval 1000 \
           --lr 0.00001 \
           --weight_decay 0 \
           --gradient_checkpointing \
           --checkpoint_dir checkpoints_linearlr_rerun \
           --seq_length 1052 \
           --project_name rewpref_lmsys_09_29