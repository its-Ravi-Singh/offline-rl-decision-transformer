# CSE 676 Final Project - Team: Gradient Gone Wild
# Team members: Ravi Singh, [Teammate Name]
#
# Gradio dashboard to test our trained Decision Transformer on Hopper-v4.
# You can plug in a state, pick a target RTG, and see what action the model predicts.
# There's also a "Run Hopper Episode" button that records a short video of the agent playing.

import os
import glob
import torch
import numpy as np
import gymnasium as gym
import gradio as gr

from models.decision_transformer import DecisionTransformer
from utils.check import load_model

# paths
CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "saved_models/decision_transformer_d4rl.pth")
VIDEO_DIR = "demo_videos"

# Hopper-v4 dims
STATE_DIM = 11
ACT_DIM = 3
CONTEXT_LEN = 20
TARGET_RTG = 3000.0


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_policy():
    device = get_device()
    model = DecisionTransformer(state_dim=STATE_DIM, act_dim=ACT_DIM, context_len=CONTEXT_LEN)
    if os.path.exists(CHECKPOINT):
        load_model(model, CHECKPOINT, map_location=device)
        print(f"Loaded checkpoint: {CHECKPOINT} on {device}")
    else:
        print(f"Warning: no checkpoint at {CHECKPOINT}, using random weights")
    model.to(device)
    model.eval()
    return model


model = load_policy()


# ---- action prediction ----

def predict_action(target_rtg, *state_vals):
    state = np.array(state_vals, dtype=np.float32)
    with torch.no_grad():
        action = model.act(state, target_rtg=target_rtg)
    clipped = np.clip(action, -1.0, 1.0)
    result = f"Raw action:\n{np.array2string(action, precision=4, separator=', ')}\n\n"
    result += f"Clipped (what env sees):\n{np.array2string(clipped, precision=4, separator=', ')}"
    return result


# ---- hopper video rollout ----

def run_hopper_episode(target_rtg_val):
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # remove old videos so we always serve the freshest one
    for old in glob.glob(os.path.join(VIDEO_DIR, "*.mp4")):
        os.remove(old)

    env = gym.make("Hopper-v4", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda ep: ep == 0,
        disable_logger=True,
    )

    obs, _ = env.reset()
    ep_reward = 0.0
    rtg = float(target_rtg_val) / TARGET_RTG   # normalize like training

    state_history = [obs.astype(np.float32)]
    action_history = [np.zeros(ACT_DIM, dtype=np.float32)]
    rtg_history = [rtg]
    done = False

    while not done:
        with torch.no_grad():
            try:
                a = model.act(
                    obs.astype(np.float32),
                    target_rtg=rtg,
                    state_history=state_history,
                    action_history=action_history,
                    rtg_history=rtg_history,
                )
            except TypeError:
                a = model.act(obs.astype(np.float32), target_rtg=rtg)

        a = np.clip(a, -1.0, 1.0)
        obs, rew, term, trunc, _ = env.step(a)
        ep_reward += rew
        rtg -= rew / TARGET_RTG
        state_history.append(obs.astype(np.float32))
        action_history.append(a.astype(np.float32))
        rtg_history.append(float(rtg))
        done = term or trunc

    env.close()

    # find the recorded mp4
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    video_path = videos[0] if videos else None

    summary = f"Episode finished!\nTotal return: {ep_reward:.1f}"
    return video_path, summary


# ---- gradio UI ----

with gr.Blocks(title="Decision Transformer - Hopper", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Decision Transformer — Hopper-v4 Demo")
    gr.Markdown("**CSE 676 Final Project** | Team: Gradient Gone Wild")

    with gr.Tabs():

        # Tab 1: action prediction
        with gr.Tab("Predict Action"):
            gr.Markdown("Enter a Hopper state and target RTG to see what action the model picks.")
            with gr.Row():
                with gr.Column():
                    rtg_input = gr.Number(value=1.0, label="Target Return-to-Go (normalized)")
                    gr.Markdown("**State Vector** — 11 values for Hopper-v4")
                    state_inputs = []
                    for i in range(STATE_DIM):
                        state_inputs.append(gr.Number(value=0.0, label=f"state[{i}]"))
                    predict_btn = gr.Button("Get Action", variant="primary")

                with gr.Column():
                    action_output = gr.Textbox(label="Model Output", lines=6)

            predict_btn.click(
                fn=predict_action,
                inputs=[rtg_input] + state_inputs,
                outputs=action_output,
            )

        # Tab 2: live hopper rollout with video
        with gr.Tab("Run Hopper Episode"):
            gr.Markdown("Click the button to run one Hopper episode and record a video of the agent.")
            with gr.Row():
                with gr.Column():
                    rtg_slider = gr.Slider(
                        minimum=100, maximum=3000, value=1500, step=100,
                        label="Target Return (unnormalized)"
                    )
                    run_btn = gr.Button("Run Episode & Record Video", variant="primary")
                    episode_summary = gr.Textbox(label="Episode Result", lines=3)

                with gr.Column():
                    video_output = gr.Video(label="Hopper Agent Video")

            run_btn.click(
                fn=run_hopper_episode,
                inputs=[rtg_slider],
                outputs=[video_output, episode_summary],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)
