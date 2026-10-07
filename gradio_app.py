# CSE 676 Final Project
# Team: Gradient Gone Wild
# Members: Ravi Rajaram Singh, Hemanth Phani Srinivas Chilamkurthy
#
# gradio demo
# tab 1: predict action from a state
# tab 2: run both models on the same episode and show the videos side by side

import glob
import json
import os
import subprocess
import sys

import gradio as gr
import gymnasium as gym
import imageio
import numpy as np
import torch

from models.decision_transformer import DecisionTransformer
from models.perception_transformer import PerceptionTransformer
from utils.check import load_model

DT_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "saved_models/decision_transformer_d4rl.pth")
PT_CHECKPOINT = os.environ.get("PT_CHECKPOINT", "saved_models/perception_transformer_d4rl.pth")
VIDEO_DIR = "demo_videos"
STATE_DIM = 11   # hopper has 11 state dims
ACT_DIM = 3      # and 3 action dims
CONTEXT_LEN = int(os.environ.get("CONTEXT_LEN", "8"))
TARGET_RTG = 3000.0

# Hopper renders at 125 fps, so a 20 step episode written at native speed is a
# 0.16 second file. We write the video ourselves and pick the frame rate.
FPS_REAL = 30    # normal playback
FPS_SLOW = 6     # for short episodes, so a fall is actually watchable


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_policies():
    device = get_device()

    dt = DecisionTransformer(state_dim=STATE_DIM, act_dim=ACT_DIM, context_len=CONTEXT_LEN)
    if os.path.exists(DT_CHECKPOINT):
        load_model(dt, DT_CHECKPOINT, map_location=device)
        print(f"loaded DT checkpoint from {DT_CHECKPOINT}")
    else:
        print("warning: no DT checkpoint, using random weights")
    dt.to(device).eval()

    pt = None
    if os.path.exists(PT_CHECKPOINT):
        pt = PerceptionTransformer(state_dim=STATE_DIM, act_dim=ACT_DIM)
        load_model(pt, PT_CHECKPOINT, map_location=device)
        pt.to(device).eval()
        print(f"loaded Perception checkpoint from {PT_CHECKPOINT}")
    else:
        print("warning: no Perception checkpoint, comparison tab will show DT only")

    return dt, pt


model, perception_model = load_policies()


def predict_action(target_rtg, *state_vals):
    state = np.array(state_vals, dtype=np.float32)
    with torch.no_grad():
        action = model.act(state, target_rtg=target_rtg)
    clipped = np.clip(action, -1.0, 1.0)
    out = f"raw action:\n{np.array2string(action, precision=4, separator=', ')}\n\n"
    out += f"clipped (what env sees):\n{np.array2string(clipped, precision=4, separator=', ')}"
    return out


def _rollout(policy, target_return, seed, use_history):
    """Run one episode and keep every rendered frame."""
    env = gym.make("Hopper-v4", render_mode="rgb_array")
    obs, _ = env.reset(seed=int(seed))

    rtg = float(target_return) / TARGET_RTG
    total = 0.0
    frames = []
    state_hist = [obs.astype(np.float32)]
    action_hist = []
    rtg_hist = [rtg]

    while True:
        with torch.no_grad():
            if use_history:
                a = policy.act(obs.astype(np.float32), target_rtg=rtg,
                               state_history=state_hist,
                               action_history=action_hist,
                               rtg_history=rtg_hist)
            else:
                a = policy.act(obs.astype(np.float32), target_rtg=rtg)

        a = np.clip(a, -1.0, 1.0)
        obs, rew, term, trunc, _ = env.step(a)
        total += rew
        frames.append(env.render())

        rtg -= rew / TARGET_RTG
        state_hist.append(obs.astype(np.float32))
        action_hist.append(a.astype(np.float32))
        rtg_hist.append(float(rtg))
        if term or trunc:
            break

    env.close()
    return frames, total


def _write(frames, path, fps, hold_last=0):
    if hold_last:
        frames = frames + [frames[-1]] * hold_last
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)


def run_comparison(target_return, seed, slow_short):
    """Called from the UI. Does the work in a subprocess (see note below)."""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(VIDEO_DIR, "*.mp4")):
        os.remove(f)

    # macOS crashes if mujoco renders off the main thread, and gradio callbacks
    # run on a worker thread - so the rendering happens in a fresh process.
    subprocess.run([sys.executable, __file__, "record",
                    str(target_return), str(int(seed)), "1" if slow_short else "0"])

    dt_path = os.path.join(VIDEO_DIR, "decision_transformer.mp4")
    pt_path = os.path.join(VIDEO_DIR, "perception_transformer.mp4")

    try:
        with open(os.path.join(VIDEO_DIR, "summary.json")) as f:
            s = json.load(f)
        summary = (
            f"Decision Transformer   return {s['dt_return']:8.1f}   {s['dt_steps']:4d} steps\n"
            f"Perception Transformer return {s['pt_return']:8.1f}   {s['pt_steps']:4d} steps\n\n"
            f"Same seed ({int(seed)}), same target return ({int(target_return)}), same eval loop.\n"
            f"The Decision Transformer sees the full history; the Perceiver sees only\n"
            f"the current state."
        )
        if s.get("dt_slowed"):
            summary += f"\n\nDT clip played at {FPS_SLOW} fps so the fall is visible."
    except Exception as e:
        summary = f"finished, but could not read the summary ({e})"

    return (dt_path if os.path.exists(dt_path) else None,
            pt_path if os.path.exists(pt_path) else None,
            summary)


def _record_subprocess(target_return, seed, slow_short):
    os.makedirs(VIDEO_DIR, exist_ok=True)

    dt_frames, dt_return = _rollout(model, target_return, seed, use_history=True)
    slowed = bool(slow_short) and len(dt_frames) < 150
    _write(dt_frames, os.path.join(VIDEO_DIR, "decision_transformer.mp4"),
           fps=FPS_SLOW if slowed else FPS_REAL,
           hold_last=12 if slowed else 0)

    if perception_model is not None:
        pt_frames, pt_return = _rollout(perception_model, target_return, seed, use_history=False)
        _write(pt_frames, os.path.join(VIDEO_DIR, "perception_transformer.mp4"), fps=FPS_REAL)
    else:
        pt_frames, pt_return = [], float("nan")

    with open(os.path.join(VIDEO_DIR, "summary.json"), "w") as f:
        json.dump({
            "dt_return": dt_return, "dt_steps": len(dt_frames),
            "pt_return": pt_return, "pt_steps": len(pt_frames),
            "dt_slowed": slowed,
        }, f)


# build the UI
with gr.Blocks(title="Hopper Offline RL Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Decision Transformer vs Perception Transformer — Hopper-v4")
    gr.Markdown("**CSE 676 Final** | Ravi Rajaram Singh, Hemanth Phani Srinivas Chilamkurthy")

    with gr.Tabs():

        with gr.Tab("Run Episode — side by side"):
            gr.Markdown(
                "Runs **both** trained policies on the same seed and target return, then shows "
                "each episode. Takes a few seconds.\n\n"
                "Both models were trained on the D4RL Hopper medium data. Their return varies "
                "between seeds, so try a few."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    rtg_slider = gr.Slider(100, 3000, value=3000, step=100, label="Target Return")
                    seed_input = gr.Number(value=0, precision=0, label="Episode seed (try 0, 2, 5, 7)")
                    slow_check = gr.Checkbox(value=True, label="Slow down short episodes so they are watchable")
                    run_btn = gr.Button("Run both & compare", variant="primary")
                    episode_out = gr.Textbox(label="Result", lines=8)
                with gr.Column(scale=2):
                    with gr.Row():
                        dt_video = gr.Video(label="Decision Transformer (726K params, sees history)")
                        pt_video = gr.Video(label="Perception Transformer (1.9M params, state only)")

            run_btn.click(fn=run_comparison,
                          inputs=[rtg_slider, seed_input, slow_check],
                          outputs=[dt_video, pt_video, episode_out],
                          api_name=False)

        with gr.Tab("Predict Action"):
            gr.Markdown("Enter a state and target RTG to see what action the model picks.")
            with gr.Row():
                with gr.Column():
                    rtg_input = gr.Number(value=1.0, label="Target RTG (normalized)")
                    gr.Markdown("**State (11 values)** — a fresh `env.reset()` looks like "
                                "`1.2477, -0.0046, -0.0048, 0.0031, 0.0041, 0.0011, 0.0023, "
                                "0.0004, 0.0044, 0.0032, -0.0050`")
                    state_inputs = [gr.Number(value=0.0, label=f"state[{i}]") for i in range(STATE_DIM)]
                    predict_btn = gr.Button("Get Action", variant="primary")
                with gr.Column():
                    action_output = gr.Textbox(label="Output", lines=6)
            predict_btn.click(fn=predict_action,
                              inputs=[rtg_input] + state_inputs,
                              outputs=action_output, api_name=False)


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "record":
        _record_subprocess(float(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1")
    else:
        demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "8000")))
