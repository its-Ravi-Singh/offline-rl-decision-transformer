import csv
import gc
import os

os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from evaluate import evaluate
from utils.check import load_model as load_weights
from models.decision_transformer import DecisionTransformer
from models.perception_transformer import PerceptionTransformer


OUT_DIR = "model_comparison"
ENV_NAME = os.environ.get("ENV_NAME", "Hopper-v4")
TARGET_RTG = float(os.environ.get("TARGET_RTG", "3000.0"))
NUM_EPISODES = int(os.environ.get("EVAL_EPISODES", "20"))

CHECKPOINTS = {
    "baseline_dt": {
        "label": "Baseline DT",
        "class": DecisionTransformer,
        "checkpoint": os.environ.get(
            "BASELINE_DT", "saved_models/decision_transformer_d4rl.pth"
        ),
        "color": "#005BBB",
    },
    "updated_dt": {
        "label": "Preference-updated DT",
        "class": DecisionTransformer,
        "checkpoint": os.environ.get(
            "UPDATED_DT", "saved_models/decision_transformer_d4rl_pref.pth"
        ),
        "color": "#2EAD4B",
    },
    "perception_transformer": {
        "label": "Perception Transformer",
        "class": PerceptionTransformer,
        "checkpoint": os.environ.get(
            "PERCEPTION_MODEL", "saved_models/perception_transformer_d4rl.pth"
        ),
        "color": "#E8392A",
    },
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_cls, checkpoint, device):
    if model_cls is DecisionTransformer:
        model = model_cls(state_dim=11, act_dim=3, context_len=int(os.environ.get("CONTEXT_LEN", "8")))
    else:
        model = model_cls(state_dim=11, act_dim=3)
    load_weights(model, checkpoint, map_location=device)
    model.to(device)
    model.eval()
    return model


def main():
    device = get_device()
    print(f"Evaluating on {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    for key, spec in CHECKPOINTS.items():
        checkpoint = spec["checkpoint"]
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        print(f"\nEvaluating {spec['label']}: {checkpoint}")
        model = load_model(spec["class"], checkpoint, device)
        metrics = evaluate(
            model,
            env_name=ENV_NAME,
            num_episodes=NUM_EPISODES,
            target_rtg_max=TARGET_RTG,
            record_video=False,
        )
        results[key] = {
            "label": spec["label"],
            "checkpoint": checkpoint,
            "avg_return": float(metrics["avg_return"]),
            "std_return": float(metrics["std_return"]),
            "min_return": float(metrics["min_return"]),
            "max_return": float(metrics["max_return"]),
        }
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    path = os.path.join(OUT_DIR, "dt_baseline_vs_updated.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "checkpoint", "avg_return", "std_return", "min_return", "max_return"])
        for key in CHECKPOINTS:
            r = results[key]
            writer.writerow(
                [
                    r["label"],
                    r["checkpoint"],
                    f"{r['avg_return']:.3f}",
                    f"{r['std_return']:.3f}",
                    f"{r['min_return']:.3f}",
                    f"{r['max_return']:.3f}",
                ]
            )
    print(f"Saved summary to {path}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [results[k]["label"] for k in CHECKPOINTS]
    means = [results[k]["avg_return"] for k in CHECKPOINTS]
    stds = [results[k]["std_return"] for k in CHECKPOINTS]
    colors = [CHECKPOINTS[k]["color"] for k in CHECKPOINTS]
    bars = ax.bar(names, means, yerr=stds, capsize=7, color=colors, width=0.55)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean, f"{mean:.1f}", ha="center", va="bottom")
    ax.set_title("Saved Policy Benchmark on Hopper-v4")
    ax.set_ylabel(f"Average Return ({NUM_EPISODES} episodes)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "dt_baseline_vs_updated.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    print("\nDT Comparison")
    print("-" * 72)
    for key in CHECKPOINTS:
        r = results[key]
        print(f"{r['label']:<24} avg={r['avg_return']:.1f} std={r['std_return']:.1f} min/max={r['min_return']:.0f}/{r['max_return']:.0f}")


if __name__ == "__main__":
    main()
