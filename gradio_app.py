# CSE 676 Final Project
# Team: Gradient Gone Wild
# Members: Hemanth Phani Srinivas Chilamkurthy, Ravi Rajaram Singh
#
# gradio demo for testing our trained DT model
# tab 1: predict action from state
# tab 2: run a full hopper episode and record video

import os
import glob
import torch
import numpy as np
import gymnasium as gym
import gradio as gr

from models.decision_transformer import DecisionTransformer
from utils.check import load_model

CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "saved_models/decision_transformer_d4rl.pth")
VIDEO_DIR = "demo_videos"
STATE_DIM = 11   # hopper has 11 state dims
ACT_DIM = 3      # and 3 action dims
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
        print(f"loaded checkpoint from {CHECKPOINT}")
    else:
        # just use random weights if no checkpoint found
        print("warning: no checkpoint, using random weights")
    model.to(device)
    model.eval()
    return model


model = load_policy()


def predict_action(target_rtg, *state_vals):
    state = np.array(state_vals, dtype=np.float32)
    with torch.no_grad():
        action = model.act(state, target_rtg=target_rtg)
    clipped = np.clip(action, -1.0, 1.0)
    out = f"raw action:\n{np.array2string(action, precision=4, separator=', ')}\n\n"
    out += f"clipped (what env sees):\n{np.array2string(clipped, precision=4, separator=', ')}"
    return out


def run_hopper_episode(target_rtg_val):
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # delete old videos first so we dont serve stale ones
    for f in glob.glob(os.path.join(VIDEO_DIR, "*.mp4")):
        os.remove(f)

    env = gym.make("Hopper-v4", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda ep: ep == 0,
        disable_logger=True,
    )

    obs, _ = env.reset()
    ep_reward = 0.0
    rtg = float(target_rtg_val) / TARGET_RTG  # normalize the same way as training
    state_hist = [obs.astype(np.float32)]
    action_hist = [np.zeros(ACT_DIM, dtype=np.float32)]
    rtg_hist = [rtg]
    done = False

    while not done:
        with torch.no_grad():
            try:
                a = model.act(
                    obs.astype(np.float32),
                    target_rtg=rtg,
                    state_history=state_hist,
                    action_history=action_hist,
                    rtg_history=rtg_hist,
                )
            except TypeError:
                a = model.act(obs.astype(np.float32), target_rtg=rtg)

        a = np.clip(a, -1.0, 1.0)
        obs, rew, term, trunc, _ = env.step(a)
        ep_reward += rew
        rtg -= rew / TARGET_RTG
        state_hist.append(obs.astype(np.float32))
        action_hist.append(a.astype(np.float32))
        rtg_hist.append(float(rtg))
        done = term or trunc

    env.close()

    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    video_path = videos[0] if videos else None
    summary = f"done! total return: {ep_reward:.1f}"
    return video_path, summary


# build the UI
with gr.Blocks(title="DT Hopper Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Decision Transformer — Hopper-v4 Demo")
    gr.Markdown("**CSE 676 Final** | Hemanth Phani Srinivas Chilamkurthy, Ravi Rajaram Singh")

    with gr.Tabs():

        with gr.Tab("Predict Action"):
            gr.Markdown("Enter a state and target RTG to see what action the model picks.")
            with gr.Row():
                with gr.Column():
                    rtg_input = gr.Number(value=1.0, label="Target RTG (normalized)")
                    gr.Markdown("**State (11 values)**")
                    state_inputs = [gr.Number(value=0.0, label=f"state[{i}]") for i in range(STATE_DIM)]
                    predict_btn = gr.Button("Get Action", variant="primary")
                with gr.Column():
                    action_output = gr.Textbox(label="Output", lines=6)
            predict_btn.click(fn=predict_action, inputs=[rtg_input] + state_inputs, outputs=action_output, api_name=False)

        with gr.Tab("Run Hopper Episode"):
            gr.Markdown("Runs a full episode and records video. Takes a few seconds.")
            with gr.Row():
                with gr.Column():
                    rtg_slider = gr.Slider(100, 3000, value=1500, step=100, label="Target Return")
                    run_btn = gr.Button("Run & Record", variant="primary")
                    episode_out = gr.Textbox(label="Result", lines=2)
                with gr.Column():
                    video_out = gr.Video(label="Episode Video")
            run_btn.click(fn=run_hopper_episode, inputs=[rtg_slider], outputs=[video_out, episode_out], api_name=False)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)
