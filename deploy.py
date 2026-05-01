import argparse
import os

import gymnasium as gym
import numpy as np
import torch

from models.decision_transformer import DecisionTransformer
from utils.check import load_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(checkpoint, device):
    model = DecisionTransformer(state_dim=11, act_dim=3)
    load_model(model, checkpoint, map_location=device)
    model.to(device)
    model.eval()
    return model


def run_rollout(model, env_name, episodes, target_rtg_max, record_video, video_dir):
    render_mode = "rgb_array" if record_video else None
    env = gym.make(env_name, render_mode=render_mode)
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: episode_id == 0,
            disable_logger=True,
        )

    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        rtg = 1.0

        while not done:
            action = model.act(obs.astype(np.float32), target_rtg=rtg)
            action = np.clip(action, -1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            rtg -= reward / (target_rtg_max + 1e-8)
            done = terminated or truncated

        returns.append(float(ep_return))

    env.close()
    return {
        "episodes": episodes,
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "returns": returns,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="saved_models/decision_transformer_d4rl.pth")
    parser.add_argument("--env", default="Hopper-v4")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--target-rtg-max", type=float, default=3000.0)
    parser.add_argument("--video-dir", default="deployment_results")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    model = build_model(args.checkpoint, device)
    metrics = run_rollout(
        model=model,
        env_name=args.env,
        episodes=args.episodes,
        target_rtg_max=args.target_rtg_max,
        record_video=not args.no_video,
        video_dir=args.video_dir,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Episodes: {metrics['episodes']}")
    print(f"Average Return: {metrics['avg_return']:.1f}")
    print(f"Std Return: {metrics['std_return']:.1f}")
    print(f"Min/Max Return: {metrics['min_return']:.1f} / {metrics['max_return']:.1f}")
    if not args.no_video:
        print(f"Video directory: {args.video_dir}")


if __name__ == "__main__":
    main()
