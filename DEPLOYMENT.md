# Deployment

The main trained checkpoint is at:

```text
saved_models/decision_transformer_d4rl.pth
```

## Rollout Smoke Test

Run one live Hopper episode to make sure the model works end to end:

```bash
python3 deploy.py --no-video --episodes 1
```

Latest result we got:

```text
Checkpoint: saved_models/decision_transformer_d4rl.pth
Episodes: 1
Average Return: 25.0
```

## Gradio Web App

We set up a Gradio dashboard so you can test the model in your browser without writing any code.

Start it with:

```bash
MODEL_CHECKPOINT=saved_models/decision_transformer_d4rl.pth python3 gradio_app.py
```

Then open `http://127.0.0.1:8000` in your browser. You can plug in the 11-dimensional Hopper state, set a target return-to-go, and instantly see what action the model predicts.
