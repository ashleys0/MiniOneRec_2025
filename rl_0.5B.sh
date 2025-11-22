#!/bin/bash
# Modified for 2×A6000 (48GB) with Qwen2-0.5B
# Optimized batch sizes for 2 GPUs while maintaining effective batch size

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    # Modified: 2 GPUs instead of 4
    # default test_during_training: False
    accelerate launch \
            --config_file ./config/zero2_opt.yaml \
            --num_processes 2 --main_process_port 29503 \
            rl.py \
        --model_path output_dir/qwen2_0.5B_sft \
        --train_batch_size 32 \
        --eval_batch_size 64 \
        --num_train_epochs 0.05 \
        --gradient_accumulation_steps 4 \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --info_file ${info_file} \
        --category ${category} \
        --sample_train False \
        --eval_step 0.1 \
        --reward_type ranking \
        --num_generations 16 \
        --mask_all_zero False \
        --dynamic_sampling False \
        --sync_ref_model True \
        --beam_search True \
        --test_during_training True \
        --test_beam 16 \
        --temperature 1.0 \
        --learning_rate 1e-5 \
        --add_gt False \
        --beta 1e-3 \
        --dapo False \
        --output_dir output_dir/qwen2_0.5B_rl \
        --wandb_project minionerec_0.5B\
        --wandb_run_name rl_0.5B_2xa6000_exp1_toyrun \
        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
done
