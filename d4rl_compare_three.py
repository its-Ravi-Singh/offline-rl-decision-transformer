import csv
import os

os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.join(os.getcwd(), "minari_datasets"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import minari
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import (
    MinariTrajectoryBuffer,
    SequenceTrajectoryDataset,
    TrajectoryDataset,
    WeightedSequenceTrajectoryDataset,
)
from evaluate import evaluate
from models.decision_transformer import DecisionTransformer
from models.perception_transformer import PerceptionTransformer
from models.preference_model import PreferenceModel
from train import get_device, train
from utils.check import save_model


OUT_DIR = "d4rl_three_way_results"
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "512"))
CONTEXT_LEN = int(os.environ.get("CONTEXT_LEN", "8"))
N_EVAL = int(os.environ.get("N_EVAL", "5"))
TARGET_RTG = float(os.environ.get("TARGET_RTG", "3000.0"))
ENV_NAME = os.environ.get("ENV_NAME", "Hopper-v4")
MAX_WINDOWS = os.environ.get("MAX_WINDOWS", "10000")
MAX_WINDOWS = None if MAX_WINDOWS in (None, "", "0") else int(MAX_WINDOWS)
MAX_TRANSITIONS = os.environ.get("MAX_TRANSITIONS")
MAX_TRANSITIONS = None if MAX_TRANSITIONS in (None, "", "0") else int(MAX_TRANSITIONS)
PREFERENCE_MODEL_FILE = os.environ.get(
    "PREFERENCE_MODEL_FILE", "preference_results/preference_model.pth"
)

SPLITS = {
    "simple": "mujoco/hopper/simple-v0",
    "medium": "mujoco/hopper/medium-v0",
    "expert": "mujoco/hopper/expert-v0",
}

MODELS = {
    "decision_transformer": {
        "display": "Decision Transformer",
        "color": "#2F80ED",
    },
    "decision_transformer_preference": {
        "display": "DT + Preference",
        "color": "#27AE60",
    },
    "perception_transformer": {
        "display": "Perception Transformer",
        "color": "#EB5757",
    },
}


def load_minari_data(ds_id):
    print(f"Loading {ds_id}...", flush=True)
    try:
        return minari.load_dataset(ds_id, download=True)
    except Exception:
        minari.download_dataset(ds_id)
        return minari.load_dataset(ds_id)


def build_preference_weights(buffer, dataset):
    if not os.path.exists(PREFERENCE_MODEL_FILE):
        raise FileNotFoundError(PREFERENCE_MODEL_FILE)

    device = get_device()
    model = PreferenceModel(state_dim=buffer.state_dim, act_dim=buffer.act_dim)
    model.load_state_dict(torch.load(PREFERENCE_MODEL_FILE, map_location=device))
    model.to(device)
    model.eval()

    scores = []
    loader = DataLoader(dataset, batch_size=256)
    with torch.no_grad():
        for states, actions, rtgs, timesteps, mask in loader:
            states = states.to(device)
            actions = actions.to(device)
            rtgs = rtgs.to(device)
            mask = mask.to(device)
            scores.append(model(states, actions, rtgs, mask=mask).cpu().numpy())

    scores = np.concatenate(scores, axis=0)
    centered = scores - scores.mean()
    scale = scores.std() if scores.std() > 1e-6 else 1.0
    return (1.0 + 0.5 * np.tanh(centered / scale)).astype(np.float32)


def make_dataset(buffer, model_key):
    if model_key in {"decision_transformer", "decision_transformer_preference"}:
        dataset = SequenceTrajectoryDataset(
            buffer,
            context_len=CONTEXT_LEN,
            target_rtg=TARGET_RTG,
            stride=1,
            max_windows=MAX_WINDOWS,
        )
        if model_key == "decision_transformer_preference":
            weights = build_preference_weights(buffer, dataset)
            print(
                f"Preference weights: mean={weights.mean():.3f}, "
                f"min={weights.min():.3f}, max={weights.max():.3f}",
                flush=True,
            )
            dataset = WeightedSequenceTrajectoryDataset(dataset, weights)
        return dataset

    return TrajectoryDataset(
        buffer,
        target_rtg=TARGET_RTG,
        max_samples=MAX_TRANSITIONS if MAX_TRANSITIONS is not None else MAX_WINDOWS,
    )


def run_model(split_label, buffer, model_key):
    spec = MODELS[model_key]
    dataset = make_dataset(buffer, model_key)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    if model_key == "perception_transformer":
        model = PerceptionTransformer(state_dim=buffer.state_dim, act_dim=buffer.act_dim)
    else:
        model = DecisionTransformer(
            state_dim=buffer.state_dim,
            act_dim=buffer.act_dim,
            context_len=CONTEXT_LEN,
        )
        model.set_state_stats(buffer.state_mean, buffer.state_std)

    safe_name = f"{model_key}_{split_label}"
    print(f"\nTraining {spec['display']} on {split_label}", flush=True)
    losses = train(
        model,
        loader,
        epochs=EPOCHS,
        plot_path=os.path.join(OUT_DIR, f"loss_{safe_name}.png"),
        plot_title=f"{spec['display']} Training Loss - {split_label.title()}",
        log_every=max(EPOCHS // 5, 1),
    )
    metrics = evaluate(
        model,
        env_name=ENV_NAME,
        num_episodes=N_EVAL,
        target_rtg_max=TARGET_RTG,
        record_video=False,
    )
    model_path = os.path.join(OUT_DIR, f"model_{safe_name}.pth")
    save_model(model, model_path)
    return losses, metrics, model_path


def save_summary(records):
    path = os.path.join(OUT_DIR, "summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "dataset_id",
                "model",
                "epochs",
                "batch_size",
                "context_len",
                "max_windows",
                "eval_episodes",
                "avg_return",
                "std_return",
                "min_return",
                "max_return",
                "model_path",
            ]
        )
        for record in records:
            metrics = record["metrics"]
            writer.writerow(
                [
                    record["split"],
                    record["dataset_id"],
                    record["model"],
                    EPOCHS,
                    BATCH_SIZE,
                    CONTEXT_LEN,
                    "" if MAX_WINDOWS is None else MAX_WINDOWS,
                    N_EVAL,
                    f"{metrics['avg_return']:.3f}",
                    f"{metrics['std_return']:.3f}",
                    f"{metrics['min_return']:.3f}",
                    f"{metrics['max_return']:.3f}",
                    record["model_path"],
                ]
            )
    print(f"Saved summary to {path}", flush=True)


def plot_summary(records):
    split_labels = list(SPLITS)
    x_pos = np.arange(len(split_labels))
    width = 0.25
    offsets = {
        "decision_transformer": -width,
        "decision_transformer_preference": 0.0,
        "perception_transformer": width,
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_key, spec in MODELS.items():
        model_records = [r for r in records if r["model_key"] == model_key]
        means = [r["metrics"]["avg_return"] for r in model_records]
        stds = [r["metrics"]["std_return"] for r in model_records]
        ax.bar(
            x_pos + offsets[model_key],
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=spec["color"],
            label=spec["display"],
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([label.title() for label in split_labels])
    ax.set_ylabel(f"{ENV_NAME} return")
    ax.set_title("D4RL Hopper Benchmark: DT vs DT + Preference vs Perception")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "three_way_comparison.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved plot to {path}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []
    for split_label, dataset_id in SPLITS.items():
        mds = load_minari_data(dataset_id)
        buffer = MinariTrajectoryBuffer(mds, target_rtg=TARGET_RTG)
        print(
            f"{split_label}: episodes={len(buffer)}, steps={buffer.num_steps}, "
            f"returns={buffer.return_stats}",
            flush=True,
        )
        for model_key, spec in MODELS.items():
            losses, metrics, model_path = run_model(split_label, buffer, model_key)
            records.append(
                {
                    "split": split_label,
                    "dataset_id": dataset_id,
                    "model_key": model_key,
                    "model": spec["display"],
                    "losses": losses,
                    "metrics": metrics,
                    "model_path": model_path,
                }
            )

    save_summary(records)
    plot_summary(records)

    print("\nD4RL Three-Way Benchmark")
    print("-" * 80)
    for record in records:
        metrics = record["metrics"]
        print(
            f"{record['split']:<8} {record['model']:<24} "
            f"avg={metrics['avg_return']:.1f} std={metrics['std_return']:.1f} "
            f"min/max={metrics['min_return']:.0f}/{metrics['max_return']:.0f}"
        )


if __name__ == "__main__":
    main()
