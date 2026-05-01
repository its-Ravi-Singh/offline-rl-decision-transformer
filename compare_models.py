"""Evaluate saved DecisionTransformer and PerceptionTransformer checkpoints.

Run:
    python3 compare_models.py
"""

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
from models.decision_transformer import DecisionTransformer
from models.perception_transformer import PerceptionTransformer


OUT_DIR = "model_comparison"
ENV_NAME = "Hopper-v4"
TARGET_RTG = 3000.0
NUM_EPISODES = int(os.environ.get("EVAL_EPISODES", "20"))

MODELS = {
    "Decision Transformer": {
        "class": DecisionTransformer,
        "checkpoint": "saved_models/decision_transformer_d4rl.pth",
        "color": "#005BBB",
    },
    "Perception Transformer": {
        "class": PerceptionTransformer,
        "checkpoint": "saved_models/perception_transformer_d4rl.pth",
        "color": "#E8392A",
    },
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_saved_model(model_cls, checkpoint, device):
    model = model_cls(state_dim=11, act_dim=3)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_summary(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "summary.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "checkpoint",
                "parameters",
                "episodes",
                "avg_return",
                "std_return",
                "min_return",
                "max_return",
            ]
        )
        for name, result in results.items():
            writer.writerow(
                [
                    name,
                    result["checkpoint"],
                    result["parameters"],
                    result["episodes"],
                    f"{result['avg_return']:.3f}",
                    f"{result['std_return']:.3f}",
                    f"{result['min_return']:.3f}",
                    f"{result['max_return']:.3f}",
                ]
            )

    print(f"Saved summary to {path}")


def save_return_plot(results):
    names = list(results.keys())
    means = [results[name]["avg_return"] for name in names]
    stds = [results[name]["std_return"] for name in names]
    colors = [MODELS[name]["color"] for name in names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        names,
        means,
        yerr=stds,
        capsize=7,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        width=0.55,
    )

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("Saved Model Evaluation on Hopper-v4")
    ax.set_ylabel(f"Average Return ({NUM_EPISODES} episodes)")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(OUT_DIR, "decision_vs_perception.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved return plot to {path}")


def main():
    device = get_device()
    print(f"Evaluating on {device}")

    results = {}
    for name, spec in MODELS.items():
        print(f"\nEvaluating {name}")
        checkpoint = spec["checkpoint"]
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)

        model = load_saved_model(spec["class"], checkpoint, device)
        metrics = evaluate(
            model,
            env_name=ENV_NAME,
            num_episodes=NUM_EPISODES,
            target_rtg_max=TARGET_RTG,
            record_video=False,
        )

        results[name] = {
            "checkpoint": checkpoint,
            "parameters": count_parameters(model),
            "episodes": NUM_EPISODES,
            "avg_return": float(metrics["avg_return"]),
            "std_return": float(metrics["std_return"]),
            "min_return": float(metrics["min_return"]),
            "max_return": float(metrics["max_return"]),
        }
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    save_summary(results)
    save_return_plot(results)

    print("\nSaved Model Comparison")
    print("-" * 78)
    print(f"{'Model':<26} {'Params':>12} {'Avg Return':>14} {'Std':>10} {'Min/Max':>16}")
    for name, result in results.items():
        print(
            f"{name:<26} "
            f"{result['parameters']:>12,} "
            f"{result['avg_return']:>14.1f} "
            f"{result['std_return']:>10.1f} "
            f"{result['min_return']:>6.0f}/{result['max_return']:<6.0f}"
        )

    best = max(results, key=lambda key: results[key]["avg_return"])
    print(f"\nBest saved checkpoint: {best}")


if __name__ == "__main__":
    main()
