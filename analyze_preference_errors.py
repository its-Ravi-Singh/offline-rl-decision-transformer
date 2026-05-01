import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import PreferencePairDataset, load_preference_pairs
from models.preference_model import PreferenceModel, preference_logits
from train import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="preference_results/preference_pairs.npz")
    parser.add_argument("--model", default="preference_results/preference_model.pth")
    parser.add_argument("--out-dir", default="preference_results")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--state-dim", type=int, default=11)
    parser.add_argument("--act-dim", type=int, default=3)
    args = parser.parse_args()

    pairs = load_preference_pairs(args.pairs)
    dataset = PreferencePairDataset(pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    device = get_device()

    model = PreferenceModel(state_dim=args.state_dim, act_dim=args.act_dim)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    rows = []
    idx_start = 0
    with torch.no_grad():
        for batch in loader:
            current = {k: v.to(device) for k, v in batch.items()}
            logits = preference_logits(model, current)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1).cpu().numpy()
            confidence = probs.max(dim=-1).values.cpu().numpy()
            labels = batch["labels"].numpy()
            batch_size = len(labels)
            for offset in range(batch_size):
                i = idx_start + offset
                rows.append(
                    {
                        "idx": i,
                        "label": int(labels[offset]),
                        "clean_label": int(pairs.get("clean_labels", labels)[i]),
                        "pred": int(pred[offset]),
                        "correct": int(pred[offset] == labels[offset]),
                        "clean_correct": int(pred[offset] == pairs.get("clean_labels", labels)[i]),
                        "confidence": float(confidence[offset]),
                        "return_gap": float(pairs.get("return_gaps", np.zeros(len(dataset)))[i]),
                    }
                )
            idx_start += batch_size

    correct = np.asarray([r["correct"] for r in rows], dtype=np.float32)
    clean_correct = np.asarray([r["clean_correct"] for r in rows], dtype=np.float32)
    confidence = np.asarray([r["confidence"] for r in rows], dtype=np.float32)
    gaps = np.asarray([r["return_gap"] for r in rows], dtype=np.float32)
    possible_label_errors = [
        r
        for r in rows
        if r["label"] != r["clean_label"] or (not r["correct"] and r["confidence"] > 0.8)
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "preference_error_cases.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(possible_label_errors)

    print("Preference error analysis")
    print(f"pairs: {len(rows)}")
    print(f"accuracy vs noisy labels: {correct.mean():.3f}")
    print(f"accuracy vs clean return labels: {clean_correct.mean():.3f}")
    print(f"mean confidence: {confidence.mean():.3f}")
    print(f"mean return gap correct: {gaps[correct == 1].mean():.3f}")
    print(f"mean return gap incorrect: {gaps[correct == 0].mean():.3f}")
    print(f"flagged cases: {len(possible_label_errors)}")
    print(f"saved flagged cases to {csv_path}")


if __name__ == "__main__":
    main()
