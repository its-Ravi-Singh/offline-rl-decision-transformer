import os

os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault(
    "MINARI_DATASETS_PATH",
    os.path.join(os.getcwd(), "minari_datasets"),
)

import minari
from torch.utils.data import DataLoader

from data.dataset import TrajectoryDataset
from evaluate import evaluate
from models.perception_transformer import PerceptionTransformer
from train import train
from utils.check import save_model


EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1024"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "3"))
MIN_DELTA = float(os.environ.get("MIN_DELTA", "1e-3"))
DATASET_ID = os.environ.get("DATASET_ID", "mujoco/hopper/expert-v0")
ENV_NAME = os.environ.get("ENV_NAME", "Hopper-v4")
TARGET_RTG = float(os.environ.get("TARGET_RTG", "3000.0"))
MODEL_FILE = os.environ.get(
    "MODEL_FILE",
    "saved_models/perception_transformer_d4rl.pth",
)
PLOT_FILE = os.environ.get(
    "PLOT_FILE",
    "plots/perception_transformer_training_loss.png",
)


def load_minari_data(ds_id):
    print(f"Loading {ds_id} data...")
    try:
        return minari.load_dataset(ds_id, download=True)
    except Exception:
        minari.download_dataset(ds_id)
        return minari.load_dataset(ds_id)


def main():
    mds = load_minari_data(DATASET_ID)
    dataset = TrajectoryDataset(mds, target_rtg=TARGET_RTG)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PerceptionTransformer(state_dim=11, act_dim=3)

    print("Starting PerceptionTransformer training...")
    loss_history = train(
        model,
        loader,
        epochs=EPOCHS,
        plot_path=PLOT_FILE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA,
        log_every=1,
    )

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    save_model(model, MODEL_FILE)

    print("Testing the PerceptionTransformer...")
    results = evaluate(model, env_name=ENV_NAME, target_rtg_max=TARGET_RTG)

    print(f"\nFinal Loss: {loss_history[-1]:.4f}")
    print(f"Average Return: {results['avg_return']:.1f}")
    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved training plot to {PLOT_FILE}")


if __name__ == "__main__":
    main()
