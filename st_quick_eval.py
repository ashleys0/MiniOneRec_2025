"""
Quick evaluation script for SFT models
Evaluates NDCG@5, NDCG@10, HR@5, HR@10 on validation and test sets
Usage: python quick_eval_sft.py --model_path <path> --category <category>
"""

import torch
import json
import math
import fire
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
from data import EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor


def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)


def compute_metrics(predictions, targets, topk_list=[5, 10]):
    """
    Compute NDCG and HR metrics
    Args:
        predictions: list of lists, each inner list contains ranked predictions
        targets: list of target items
        topk_list: list of k values to compute metrics for
    """
    metrics = {f"NDCG@{k}": [] for k in topk_list}
    metrics.update({f"HR@{k}": [] for k in topk_list})

    for pred_list, target in zip(predictions, targets):
        # Find position of target in predictions
        target = target.strip("\"\n ")
        pred_list = [p.strip("\"\n ") for p in pred_list]

        try:
            pos = pred_list.index(target)
        except ValueError:
            pos = len(pred_list)  # Not found

        # Compute metrics for each k
        for k in topk_list:
            if pos < k:
                metrics[f"NDCG@{k}"].append(1.0 / math.log2(pos + 2))
                metrics[f"HR@{k}"].append(1.0)
            else:
                metrics[f"NDCG@{k}"].append(0.0)
                metrics[f"HR@{k}"].append(0.0)

    # Average the metrics
    for key in metrics:
        metrics[key] = np.mean(metrics[key])

    return metrics


def evaluate_dataset(model, tokenizer, dataset, info_file, base_model,
                     batch_size=8, num_beams=20, max_new_tokens=128, device="cuda"):
    """
    Evaluate model on a dataset
    """
    # Build constraint hash dict from info file
    with open(info_file, 'r') as f:
        info = f.readlines()
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
        info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]

    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]

    if base_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3

    # Build hash_dict for constrained generation
    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])

    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []

    # Prepare generation config
    generation_config = GenerationConfig(
        num_beams=num_beams,
        length_penalty=0.0,
        num_return_sequences=num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )

    # Get all data
    encodings = [dataset[i] for i in range(len(dataset))]
    test_data = dataset.get_all()

    # Batch evaluation
    all_predictions = []
    num_batches = (len(encodings) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        batch_encodings = encodings[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        # Pad batch
        maxLen = max([len(_["input_ids"]) for _ in batch_encodings])
        padding_encodings = {"input_ids": []}
        attention_mask = []

        for enc in batch_encodings:
            L = len(enc["input_ids"])
            padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + enc["input_ids"])
            attention_mask.append([0] * (maxLen - L) + [1] * L)

        # Generate
        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                base_model=base_model
            )
            logits_processor = LogitsProcessorList([clp])

            generation_output = model.generate(
                torch.tensor(padding_encodings["input_ids"]).to(device),
                attention_mask=torch.tensor(attention_mask).to(device),
                generation_config=generation_config,
                logits_processor=logits_processor,
            )

        batched_completions = generation_output[:, maxLen:]

        if base_model.lower().find("llama") > -1:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)

        output = [_.split("Response:\n")[-1].strip() for _ in output]
        batch_predictions = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        all_predictions.extend(batch_predictions)

    # Extract targets
    targets = []
    for sample in test_data:
        if isinstance(sample['output'], list):
            targets.append(sample['output'][0])
        else:
            targets.append(sample['output'])

    # Compute metrics
    metrics = compute_metrics(all_predictions, targets, topk_list=[5, 10])

    return metrics


def main(
    model_path: str,
    category: str = "Industrial_and_Scientific",
    val_file: str = None,
    test_file: str = None,
    info_file: str = None,
    batch_size: int = 8,
    num_beams: int = 20,
    max_new_tokens: int = 128,
    seed: int = 42,
):
    """
    Quick evaluation of SFT model on validation and test sets

    Args:
        model_path: Path to the model checkpoint
        category: Dataset category
        val_file: Path to validation file (if None, auto-detected)
        test_file: Path to test file (if None, auto-detected)
        info_file: Path to info file (if None, auto-detected)
        batch_size: Batch size for evaluation
        num_beams: Number of beams for beam search
        max_new_tokens: Maximum number of tokens to generate
        seed: Random seed
    """
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Auto-detect files if not provided
    if val_file is None:
        import glob
        val_files = glob.glob(f"./data/Amazon/valid/{category}*.csv")
        if not val_files:
            raise ValueError(f"No validation file found for category {category}")
        val_file = val_files[0]
        print(f"Auto-detected validation file: {val_file}")

    if test_file is None:
        import glob
        test_files = glob.glob(f"./data/Amazon/test/{category}*.csv")
        if not test_files:
            raise ValueError(f"No test file found for category {category}")
        test_file = test_files[0]
        print(f"Auto-detected test file: {test_file}")

    if info_file is None:
        import glob
        info_files = glob.glob(f"./data/Amazon/info/{category}*.txt")
        if not info_files:
            raise ValueError(f"No info file found for category {category}")
        info_file = info_files[0]
        print(f"Auto-detected info file: {info_file}")

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books"
    }
    category_name = category_dict[category]

    # Load model and tokenizer
    print(f"\nLoading model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Evaluate on validation set
    print(f"\n{'='*60}")
    print(f"Evaluating on VALIDATION set")
    print(f"{'='*60}")
    val_dataset = EvalSidDataset(
        train_file=val_file,
        tokenizer=tokenizer,
        max_len=2560,
        category=category_name,
        test=True,
        K=0,
        seed=seed
    )

    val_metrics = evaluate_dataset(
        model, tokenizer, val_dataset, info_file, model_path,
        batch_size=batch_size, num_beams=num_beams, max_new_tokens=max_new_tokens, device=device
    )

    print("\nValidation Set Results:")
    print(f"  NDCG@5:  {val_metrics['NDCG@5']:.4f}")
    print(f"  NDCG@10: {val_metrics['NDCG@10']:.4f}")
    print(f"  HR@5:    {val_metrics['HR@5']:.4f}")
    print(f"  HR@10:   {val_metrics['HR@10']:.4f}")

    # Evaluate on test set
    print(f"\n{'='*60}")
    print(f"Evaluating on TEST set")
    print(f"{'='*60}")
    test_dataset = EvalSidDataset(
        train_file=test_file,
        tokenizer=tokenizer,
        max_len=2560,
        category=category_name,
        test=True,
        K=0,
        seed=seed
    )

    test_metrics = evaluate_dataset(
        model, tokenizer, test_dataset, info_file, model_path,
        batch_size=batch_size, num_beams=num_beams, max_new_tokens=max_new_tokens, device=device
    )

    print("\nTest Set Results:")
    print(f"  NDCG@5:  {test_metrics['NDCG@5']:.4f}")
    print(f"  NDCG@10: {test_metrics['NDCG@10']:.4f}")
    print(f"  HR@5:    {test_metrics['HR@5']:.4f}")
    print(f"  HR@10:   {test_metrics['HR@10']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Category: {category}")
    print(f"\n{'Metric':<15} {'Validation':<15} {'Test':<15}")
    print(f"{'-'*45}")
    print(f"{'NDCG@5':<15} {val_metrics['NDCG@5']:<15.4f} {test_metrics['NDCG@5']:<15.4f}")
    print(f"{'NDCG@10':<15} {val_metrics['NDCG@10']:<15.4f} {test_metrics['NDCG@10']:<15.4f}")
    print(f"{'HR@5':<15} {val_metrics['HR@5']:<15.4f} {test_metrics['HR@5']:<15.4f}")
    print(f"{'HR@10':<15} {val_metrics['HR@10']:<15.4f} {test_metrics['HR@10']:<15.4f}")
    print(f"{'='*60}\n")

    # Save results to JSON
    results = {
        "model_path": model_path,
        "category": category,
        "validation": val_metrics,
        "test": test_metrics,
        "config": {
            "batch_size": batch_size,
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
        }
    }

    output_file = f"quick_eval_results_{category}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
