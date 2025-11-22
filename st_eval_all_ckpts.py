#!/usr/bin/env python
"""
Wrapper script to evaluate all checkpoints and compare results.
Leverages quick_eval_sft.py for actual evaluation logic.

Usage: python eval_all_sft_ckpts_compare.py --category Industrial_and_Scientific
"""

import os
import json
import subprocess
import pandas as pd
import fire
from datetime import datetime


def main(
    category: str = "Industrial_and_Scientific",
    base_dir: str = "output_dir/qwen2_0.5B_sft",
    batch_size: int = 8,
    num_beams: int = 20,
):
    """
    Evaluate all checkpoints using quick_eval_sft.py and compare results.
    This is a thin wrapper that orchestrates evaluation and aggregation.
    """

    # Auto-detect checkpoints
    checkpoints = []

    # Check main directory
    if os.path.exists(os.path.join(base_dir, "config.json")):
        checkpoints.append({
            "name": "Main Directory",
            "path": base_dir,
            "description": "Best model loaded at end"
        })

    # Check for checkpoint subdirectories
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                checkpoints.append({
                    "name": item,
                    "path": item_path,
                    "description": f"Checkpoint: {item}"
                })

    if not checkpoints:
        print(f"❌ No valid checkpoints found in {base_dir}")
        return

    print("=" * 80)
    print(f"Found {len(checkpoints)} checkpoint(s) to evaluate")
    print("=" * 80)
    for ckpt in checkpoints:
        print(f"  - {ckpt['name']}: {ckpt['path']}")
    print()

    results = []

    # Evaluate each checkpoint using quick_eval_sft.py
    for idx, ckpt in enumerate(checkpoints, 1):
        print(f"\n[{idx}/{len(checkpoints)}] Evaluating: {ckpt['name']}")
        print("-" * 80)

        cmd = [
            "python", "st_quick_eval.py",
            "--model_path", ckpt['path'],
            "--category", category,
            "--batch_size", str(batch_size),
            "--num_beams", str(num_beams),
        ]

        try:
            subprocess.run(cmd, check=True)

            # Load and preserve results
            result_file = f"quick_eval_results_{category}.json"
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    result_data = json.load(f)

                results.append({
                    "Checkpoint": ckpt['name'],
                    "Path": ckpt['path'],
                    "Val_NDCG@5": round(result_data['validation']['NDCG@5'], 4),
                    "Val_NDCG@10": round(result_data['validation']['NDCG@10'], 4),
                    "Val_HR@5": round(result_data['validation']['HR@5'], 4),
                    "Val_HR@10": round(result_data['validation']['HR@10'], 4),
                    "Test_NDCG@5": round(result_data['test']['NDCG@5'], 4),
                    "Test_NDCG@10": round(result_data['test']['NDCG@10'], 4),
                    "Test_HR@5": round(result_data['test']['HR@5'], 4),
                    "Test_HR@10": round(result_data['test']['HR@10'], 4),
                })

                # Preserve individual result file
                new_name = f"results_{ckpt['name'].replace(' ', '_').replace('-', '_').lower()}_{category}.json"
                os.rename(result_file, new_name)

        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed: {e}")
            continue

    # Generate comparison table
    if not results:
        print("❌ No results collected.")
        return

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print("\nVALIDATION SET:")
    print("-" * 80)
    # val_df = df[['Checkpoint', 'Val_NDCG@5', 'Val_NDCG@10', 'Val_HR@5', 'Val_HR@10']].copy()
    # val_df.columns = ['Checkpoint', 'NDCG@5', 'NDCG@10', 'HR@5', 'HR@10']
    val_df = df[['Checkpoint', 'Val_NDCG@10', 'Val_HR@5']].copy()
    val_df.columns = ['Checkpoint', 'NDCG@10', 'HR@5']
    print(val_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print("\n\nTEST SET:")
    print("-" * 80)
    # test_df = df[['Checkpoint', 'Test_NDCG@5', 'Test_NDCG@10', 'Test_HR@5', 'Test_HR@10']].copy()
    # test_df.columns = ['Checkpoint', 'NDCG@5', 'NDCG@10', 'HR@5', 'HR@10']
    test_df = df[['Checkpoint', 'Test_NDCG@10', 'Test_HR@5', 'Test_HR@10']].copy()
    test_df.columns = ['Checkpoint', 'NDCG@10', 'HR@5', 'HR@10']
    print(test_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print("\n\nBEST CHECKPOINT FOR EACH METRIC:")
    print("-" * 80)
    # for metric in ['Val_NDCG@10', 'Test_NDCG@10', 'Val_HR@10', 'Test_HR@10']:
    for metric in ['Val_NDCG@10', 'Val_HR@5', 'Test_NDCG@10', 'Test_HR@5', 'Test_HR@10']:
        best_idx = df[metric].idxmax()
        print(f"{metric:20s}: {df.loc[best_idx, 'Checkpoint']:20s} ({df.loc[best_idx, metric]:.4f})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"checkpoint_comparison_{category}_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
