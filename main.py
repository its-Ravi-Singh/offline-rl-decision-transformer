# CSE 676 Final Project — Gradient Gone Wild
# Ravi Rajaram Singh, Hemanth Phani Srinivas Chilamkurthy
#
# main training script for the decision transformer
# runs on hopper by default but can be changed via env vars

import os

os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.join(os.getcwd(), "minari_datasets"))

import minari
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.decision_transformer import DecisionTransformer
from models.preference_model import PreferenceModel
from data.dataset import MinariTrajectoryBuffer, SequenceTrajectoryDataset
from train import train
from evaluate import evaluate
from train import get_device
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
MAX_WINDOWS = os.environ.get("MAX_WINDOWS")
MAX_WINDOWS = None if MAX_WINDOWS in (None, "", "0") else int(MAX_WINDOWS)
UPDATED_MODEL_FILE = os.environ.get(
    "UPDATED_MODEL_FILE", "saved_models/decision_transformer_d4rl_pref.pth"
)
INIT_MODEL_FILE = os.environ.get(
    "INIT_MODEL_FILE", "saved_models/decision_transformer_d4rl.pth"
)
PREFERENCE_MODEL_FILE = os.environ.get(
    "PREFERENCE_MODEL_FILE", "preference_results/preference_model.pth"
)
USE_PREFERENCE_WEIGHTS = os.environ.get("USE_PREFERENCE_WEIGHTS", "1") not in {
    "0",
    "false",
    "False",
}


def load_minari_data(ds_id):
    print(f"loading dataset: {ds_id}")
    try:
        mds = minari.load_dataset(ds_id, download=True)
    except Exception:
        # sometimes the above fails first time, downloading manually fixes it
        mds = minari.download_dataset(ds_id)
        mds = minari.load_dataset(ds_id)
    return mds


def build_preference_weights(buffer, base_dataset, preference_model_path):
    print("scoring sequence windows with preference model...", flush=True)
    device = get_device()
    model = PreferenceModel(state_dim=buffer.state_dim, act_dim=buffer.act_dim)
    model.load_state_dict(torch.load(preference_model_path, map_location=device))
    model.to(device)
    model.eval()

    loader = DataLoader(base_dataset, batch_size=256)
    scores = []
    with torch.no_grad():
        for batch in loader:
            states, actions, rtgs, timesteps, mask = batch
            states = states.to(device)
            actions = actions.to(device)
            rtgs = rtgs.to(device)
            mask = mask.to(device)
            score = model(states, actions, rtgs, mask=mask).detach().cpu().numpy()
            scores.append(score)

    scores = np.concatenate(scores, axis=0)
    centered = scores - scores.mean()
    scale = scores.std() if scores.std() > 1e-6 else 1.0
    weights = 1.0 + 0.5 * np.tanh(centered / scale)
    print(
        f"preference weights: mean={weights.mean():.3f}, "
        f"min={weights.min():.3f}, max={weights.max():.3f}",
        flush=True,
    )
    return weights.astype(np.float32)


def main():

    mds = load_minari_data(DATASET_ID)
    buffer = MinariTrajectoryBuffer(mds, target_rtg=TARGET_RTG)
    print(
        f"episodes: {len(buffer)}, steps: {buffer.num_steps}, returns: {buffer.return_stats}",
        flush=True,
    )

    ds = SequenceTrajectoryDataset(
        buffer,
        context_len=CONTEXT_LEN,
        target_rtg=TARGET_RTG,
        stride=CONTEXT_LEN,
        max_windows=MAX_WINDOWS,
    )
    if USE_PREFERENCE_WEIGHTS and os.path.exists(PREFERENCE_MODEL_FILE):
        from data.dataset import WeightedSequenceTrajectoryDataset

        weights = build_preference_weights(buffer, ds, PREFERENCE_MODEL_FILE)
        ds = WeightedSequenceTrajectoryDataset(ds, weights)
        print(f"using preference weights from {PREFERENCE_MODEL_FILE}", flush=True)
    elif USE_PREFERENCE_WEIGHTS:
        print(f"preference model not found at {PREFERENCE_MODEL_FILE}, training vanilla DT", flush=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = DecisionTransformer(
        state_dim=buffer.state_dim,
        act_dim=buffer.act_dim,
        context_len=CONTEXT_LEN,
    )
    if os.path.exists(INIT_MODEL_FILE):
        print(f"loading initial DT weights from {INIT_MODEL_FILE}", flush=True)
        model.load_state_dict(torch.load(INIT_MODEL_FILE, map_location="cpu"))
    else:
        print(f"initial DT checkpoint not found at {INIT_MODEL_FILE}, training from scratch", flush=True)

    print("starting training...", flush=True)
    loss_history = train(model, loader, epochs=EPOCHS)

    print("evaluating...")
    results = evaluate(
        model,
        env_name=ENV_NAME,
        num_episodes=NUM_EVAL,
        target_rtg_max=TARGET_RTG,
        record_video=RECORD_VIDEO,
    )

    os.makedirs("saved_models", exist_ok=True)
    save_model(model, UPDATED_MODEL_FILE)
    print(f"saved to {UPDATED_MODEL_FILE}")

    print(f"\nfinal loss: {loss_history[-1]:.4f}")
    print(f"avg return: {results['avg_return']:.1f}")


if __name__ == "__main__":
    main()
