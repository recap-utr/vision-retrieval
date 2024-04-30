# --model_name_or_path "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft" \
# --model_type swinv2 \
apptainer run --nv ~/poetry.sif run python ../pretrain/run_mim_no_trainer.py \
    --model_name_or_path "microsoft/swinv2-tiny-patch4-window8-256" \
    --dataset_name kblw/treemap_sat \
    --output_dir ./treemap_sat_mim_no_tr/ \
    --report_to wandb \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --num_train_epochs 60 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --seed 1337 \
    --with_tracking \
    # --overwrite_output_dir \
    # --remove_unused_columns False \
    # --label_names bool_masked_pos \
    # --do_train \
    # --logging_strategy steps \
    # --logging_steps 10 \
    # --evaluation_strategy epoch \
    # --save_strategy epoch \
    # --load_best_model_at_end True \
    # --save_total_limit 3 \
    # --cache_dir /data/kbartz/huggingface_cache/