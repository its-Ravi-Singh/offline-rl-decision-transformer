import os
import torch
import numpy as np
import gradio as gr

from models.decision_transformer import DecisionTransformer
from utils.check import load_model

# load from env or just use the default checkpoint
CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT",
    "saved_models/decision_transformer_d4rl.pth",
)
STATE_DIM = int(os.environ.get("STATE_DIM", "11"))   # Hopper has 11 state dims
ACT_DIM = int(os.environ.get("ACT_DIM", "3"))        # and 3 action dims
CONTEXT_LEN = int(os.environ.get("CONTEXT_LEN", "20"))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_policy():
    device = get_device()
    model = DecisionTransformer(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        context_len=CONTEXT_LEN,
    )
    if os.path.exists(CHECKPOINT):
        load_model(model, CHECKPOINT, map_location=device)
    else:
        print(f"Warning: checkpoint not found at {CHECKPOINT}, using random weights")
    model.to(device)
    model.eval()
    return model


model = load_policy()


def predict_action(target_rtg, *state_vals):
    state = np.array(state_vals, dtype=np.float32)
    with torch.no_grad():
        action = model.act(state, target_rtg=target_rtg)
    clipped = np.clip(action, -1.0, 1.0)

    result = f"Raw Action:\n{np.array2string(action, precision=4, separator=', ')}\n\n"
    result += f"Clipped Action (what the env actually sees):\n{np.array2string(clipped, precision=4, separator=', ')}"
    return result


with gr.Blocks(title="Decision Transformer - Hopper Policy", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Decision Transformer - Hopper Policy")
    gr.Markdown("Plug in a Hopper state and target RTG below, then click **Get Action** to see the model's prediction.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Inputs")
            target_rtg = gr.Number(value=1.0, label="Target Return-to-Go")

            gr.Markdown("**State Vector** (11 values for Hopper-v4)")
            state_inputs = []
            for i in range(STATE_DIM):
                val = gr.Number(value=0.0, label=f"state[{i}]")
                state_inputs.append(val)

            predict_btn = gr.Button("Get Action", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Predicted Action")
            output_text = gr.Textbox(label="Output", lines=6)

    predict_btn.click(
        fn=predict_action,
        inputs=[target_rtg] + state_inputs,
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)
