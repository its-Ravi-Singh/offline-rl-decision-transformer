# train the preference model on segment pairs
# generates pairs from offline data and trains a bradley-terry style model

import argparse
import os

os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.join(os.getcwd(), "minari_datasets"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import minari
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.dataset import (
    MinariTrajectoryBuffer,
    PreferencePairDataset,
    build_preference_pairs,
    save_preference_pairs,
)
from models.preference_model import PreferenceModel, preference_logits
from train import get_device


def load_minari_data(ds_id):
    print(f"Loading {ds_id}...")
    try:
        return minari.load_dataset(ds_id, download=True)
    except Exception:
        minari.download_dataset(ds_id)
        return minari.load_dataset(ds_id)


def train_preference_model(model, train_loader, val_loader, epochs, out_dir):
    device = get_device()
    model.to(device)
    # used same lr as the policy models, seemed to work ok
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_acc": []}

    best_acc = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = preference_logits(model, batch)
            loss = loss_fn(logits, batch["labels"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        val_acc = evaluate_preference_model(model, val_loader, device)
        avg_loss = total / max(len(train_loader), 1)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            # keep a copy of the best weights
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"loss={avg_loss:.4f} val_acc={val_acc:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "preference_model.pth"))
    plot_history(history, out_dir)
    return history


def evaluate_preference_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = preference_logits(model, batch).argmax(dim=-1)
            correct += int((pred == batch["labels"]).sum().item())
            total += int(batch["labels"].numel())
    return correct / max(total, 1)


def plot_history(history, out_dir):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], color="#2F80ED", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_acc"], color="#27AE60", label="val accuracy")
    ax2.set_ylabel("Validation accuracy")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "preference_training.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved preference training plot to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="mujoco/hopper/medium-v0")
    parser.add_argument("--target-rtg", type=float, default=3000.0)
    parser.add_argument("--num-pairs", type=int, default=10000)
    parser.add_argument("--segment-len", type=int, default=20)
    parser.add_argument("--min-return-gap", type=float, default=1.0)
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-dir", default="preference_results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mds = load_minari_data(args.dataset_id)
    buffer = MinariTrajectoryBuffer(mds, target_rtg=args.target_rtg)
    pairs = build_preference_pairs(
        buffer,
        num_pairs=args.num_pairs,
        segment_len=args.segment_len,
        target_rtg=args.target_rtg,
        min_return_gap=args.min_return_gap,
        label_noise=args.label_noise,
    )
    pair_path = os.path.join(args.out_dir, "preference_pairs.npz")
    save_preference_pairs(pairs, pair_path)
    print(f"Saved preference pairs to {pair_path}")

    dataset = PreferencePairDataset(pairs)
    # 80/20 split
    val_size = max(int(0.2 * len(dataset)), 1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = PreferenceModel(state_dim=buffer.state_dim, act_dim=buffer.act_dim)
    train_preference_model(model, train_loader, val_loader, args.epochs, args.out_dir)


if __name__ == "__main__":
    main()
