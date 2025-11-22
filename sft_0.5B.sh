#!/bin/bash
# Modified for 2×A6000 (48GB) with Qwen2-0.5B
# Optimized batch sizes for 2 GPUs while maintaining effective batch size

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}

    # Modified: 2 GPUs instead of 8
    torchrun --nproc_per_node 2 \
            sft.py \
            --base_model Qwen/Qwen2-0.5B-Instruct \
            --batch_size 1024 \
            --micro_batch_size 12 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir/qwen2_0.5B_sft \
            --wandb_project minionerec_0.5B \
            --wandb_run_name sft_0.5B_2xa6000 \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
            --item_meta_path ./data/Amazon/index//Industrial_and_Scientific.item.json \
            --freeze_LLM False
done
