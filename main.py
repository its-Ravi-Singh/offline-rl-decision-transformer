# CSE 676 Final Project — Gradient Gone Wild
# Hemanth Phani Srinivas Chilamkurthy, Ravi Rajaram Singh

import os

os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.join(os.getcwd(), "minari_datasets"))

import minari
from torch.utils.data import DataLoader

from models.decision_transformer import DecisionTransformer
from data.dataset import MinariTrajectoryBuffer, SequenceTrajectoryDataset
from train import train
from evaluate import evaluate
from utils.check import save_model


EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
CONTEXT_LEN = int(os.environ.get("CONTEXT_LEN", "20"))
DATASET_ID = os.environ.get("DATASET_ID", "mujoco/hopper/expert-v0")
ENV_NAME = os.environ.get("ENV_NAME", "Hopper-v4")
TARGET_RTG = float(os.environ.get("TARGET_RTG", "3000.0"))
NUM_EVAL = int(os.environ.get("NUM_EVAL", "20"))
RECORD_VIDEO = os.environ.get("RECORD_VIDEO", "1") not in {"0", "false", "False"}
MODEL_FILE = "saved_models/decision_transformer_d4rl.pth"


def load_minari_data(ds_id):
    print(f"Loading {ds_id} data...")
    try:
        mds = minari.load_dataset(ds_id, download=True)
    except Exception:
        mds = minari.download_dataset(ds_id)
        mds = minari.load_dataset(ds_id)
    return mds


def main():

    mds = load_minari_data(DATASET_ID)
    buffer = MinariTrajectoryBuffer(mds, target_rtg=TARGET_RTG)
    print(
        "Loaded offline benchmark: "
        f"{DATASET_ID}, episodes={len(buffer)}, steps={buffer.num_steps}, "
        f"returns={buffer.return_stats}"
    )
    ds = SequenceTrajectoryDataset(
        buffer,
        context_len=CONTEXT_LEN,
        target_rtg=TARGET_RTG,
        stride=CONTEXT_LEN,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


    model = DecisionTransformer(
        state_dim=buffer.state_dim,
        act_dim=buffer.act_dim,
        context_len=CONTEXT_LEN,
    )


    print("Starting training...")
    loss_history = train(model, loader, epochs=EPOCHS)


    print("Testing the model...")
    results = evaluate(
        model,
        env_name=ENV_NAME,
        num_episodes=NUM_EVAL,
        target_rtg_max=TARGET_RTG,
        record_video=RECORD_VIDEO,
    )


    os.makedirs("saved_models", exist_ok=True)
    save_model(model, MODEL_FILE)

    print(f"\nFinal Loss: {loss_history[-1]:.4f}")
    print(f"Average Return: {results['avg_return']:.1f}")


if __name__ == "__main__":
    main()
