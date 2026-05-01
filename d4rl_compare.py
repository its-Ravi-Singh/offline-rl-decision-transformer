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
from torch.utils.data import DataLoader

from data.dataset import MinariTrajectoryBuffer, SequenceTrajectoryDataset, TrajectoryDataset
from evaluate import evaluate
from models.decision_transformer import DecisionTransformer
from models.perception_transformer import PerceptionTransformer
from train import train
from utils.check import save_model


OUT_DIR = "d4rl_results"
EPOCHS = int(os.environ.get("EPOCHS", "30"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
CONTEXT_LEN = int(os.environ.get("CONTEXT_LEN", "20"))
N_EVAL = int(os.environ.get("N_EVAL", "10"))
TARGET_RTG = float(os.environ.get("TARGET_RTG", "3000.0"))
ENV_NAME = os.environ.get("ENV_NAME", "Hopper-v4")
MAX_WINDOWS = os.environ.get("MAX_WINDOWS")
MAX_WINDOWS = None if MAX_WINDOWS in (None, "", "0") else int(MAX_WINDOWS)
MAX_TRANSITIONS = os.environ.get("MAX_TRANSITIONS")
MAX_TRANSITIONS = None if MAX_TRANSITIONS in (None, "", "0") else int(MAX_TRANSITIONS)

SPLITS = {
    "simple": "mujoco/hopper/simple-v0",
    "medium": "mujoco/hopper/medium-v0",
    "expert": "mujoco/hopper/expert-v0",
}


MODELS = {
    "decision_transformer": {
        "display": "Decision Transformer",
        "class": DecisionTransformer,
        "dataset": "sequence",
        "color": "#2F80ED",
    },
    "perception_transformer": {
        "display": "Perception Transformer",
        "class": PerceptionTransformer,
        "dataset": "transition",
        "color": "#EB5757",
    },
}


def load_minari_data(ds_id):
    print(f"Loading {ds_id}...")
    try:
        return minari.load_dataset(ds_id, download=True)
    except Exception:
        minari.download_dataset(ds_id)
        return minari.load_dataset(ds_id)


def make_dataset(buffer, model_key):
    if MODELS[model_key]["dataset"] == "sequence":
        return SequenceTrajectoryDataset(
            buffer,
            context_len=CONTEXT_LEN,
            target_rtg=TARGET_RTG,
            stride=CONTEXT_LEN,
            max_windows=MAX_WINDOWS,
        )

    return TrajectoryDataset(
        buffer,
        target_rtg=TARGET_RTG,
        max_samples=MAX_TRANSITIONS if MAX_TRANSITIONS is not None else MAX_WINDOWS,
    )


def run_model(split_label, buffer, model_key):
    spec = MODELS[model_key]
    nice_split = split_label.title()
    dataset = make_dataset(buffer, model_key)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    kwargs = {"state_dim": buffer.state_dim, "act_dim": buffer.act_dim}
    if model_key == "decision_transformer":
        kwargs["context_len"] = CONTEXT_LEN
    model = spec["class"](**kwargs)

    safe_name = f"{model_key}_{split_label}"
    print(f"\n  Training {spec['display']} on {split_label}")
    losses = train(
        model,
        loader,
        epochs=EPOCHS,
        plot_path=os.path.join(OUT_DIR, f"loss_{safe_name}.png"),
        plot_title=f"{spec['display']} Training Loss - {nice_split}",
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


def run_split(label, ds_id):
    mds = load_minari_data(ds_id)
    buffer = MinariTrajectoryBuffer(mds, target_rtg=TARGET_RTG)
    print(
        f"{label}: episodes={len(buffer)}, steps={buffer.num_steps}, "
        f"return_stats={buffer.return_stats}"
    )
    records = []
    for model_key in MODELS:
        losses, metrics, model_path = run_model(label, buffer, model_key)
        records.append(
            {
                "split": label,
                "dataset_id": ds_id,
                "model_key": model_key,
                "model": MODELS[model_key]["display"],
                "losses": losses,
                "metrics": metrics,
                "model_path": model_path,
            }
        )
    return records


def plot_summary(records):
    nice = {"simple": "Simple", "medium": "Medium", "expert": "Expert"}

    fig, (ax_loss, ax_return) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Minari / D4RL Hopper Benchmark", fontsize=13)

    for record in records:
        label = f"{record['model']} / {nice[record['split']]}"
        ax_loss.plot(
            record["losses"],
            linewidth=2,
            label=label,
            color=MODELS[record["model_key"]]["color"],
            alpha={"simple": 0.55, "medium": 0.75, "expert": 1.0}[record["split"]],
        )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Action MSE")
    ax_loss.set_title("Training Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(frameon=False, fontsize=8)

    split_labels = list(SPLITS)
    x_pos = np.arange(len(split_labels))
    width = 0.36
    offsets = {
        "decision_transformer": -width / 2,
        "perception_transformer": width / 2,
    }
    for model_key, spec in MODELS.items():
        model_records = [r for r in records if r["model_key"] == model_key]
        means = [r["metrics"]["avg_return"] for r in model_records]
        stds = [r["metrics"]["std_return"] for r in model_records]
        bars = ax_return.bar(
            x_pos + offsets[model_key],
            means,
            width=width,
            yerr=stds,
            capsize=5,
            color=spec["color"],
            label=spec["display"],
        )
        for bar, value in zip(bars, means):
            ax_return.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax_return.set_xticks(x_pos)
    ax_return.set_xticklabels([nice[label] for label in split_labels])
    ax_return.set_ylabel(f"{ENV_NAME} return")
    ax_return.set_title(f"Evaluation over {N_EVAL} episodes")
    ax_return.grid(axis="y", alpha=0.25)
    ax_return.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "d4rl_comparison.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved plot to {path}")


def save_summary(records):
    path = os.path.join(OUT_DIR, "summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "dataset_id",
                "model",
                "env",
                "epochs",
                "batch_size",
                "context_len",
                "max_windows",
                "max_transitions",
                "eval_episodes",
                "target_rtg",
                "model_path",
                "avg_return",
                "std_return",
                "min_return",
                "max_return",
            ]
        )
        for record in records:
            metrics = record["metrics"]
            writer.writerow(
                [
                    record["split"],
                    record["dataset_id"],
                    record["model"],
                    ENV_NAME,
                    EPOCHS,
                    BATCH_SIZE,
                    CONTEXT_LEN,
                    "" if MAX_WINDOWS is None else MAX_WINDOWS,
                    "" if MAX_TRANSITIONS is None else MAX_TRANSITIONS,
                    N_EVAL,
                    f"{TARGET_RTG:.1f}",
                    record["model_path"],
                    f"{metrics['avg_return']:.3f}",
                    f"{metrics['std_return']:.3f}",
                    f"{metrics['min_return']:.3f}",
                    f"{metrics['max_return']:.3f}",
                ]
            )
    print(f"Saved summary to {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print(f"Offline RL model benchmark: {ENV_NAME}")
    print("=" * 70)

    records = []
    for label, ds_id in SPLITS.items():
        print(f"\n[{label.upper()}] {ds_id}")
        records.extend(run_split(label, ds_id))

    plot_summary(records)
    save_summary(records)

    print("\nSummary")
    for record in records:
        metrics = record["metrics"]
        print(
            f"{record['split']:<8} {record['model']:<24} "
            f"avg={metrics['avg_return']:.1f} std={metrics['std_return']:.1f}"
        )


if __name__ == "__main__":
    main()
