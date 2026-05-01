# Deployment

The current deployable Decision Transformer checkpoint is:

```text
saved_models/decision_transformer_d4rl.pth
```

## Rollout Smoke Test

Run one live Hopper rollout without recording video:

```bash
python3 deploy.py --no-video --episodes 1
```

Latest smoke result:

```text
Checkpoint: saved_models/decision_transformer_d4rl.pth
Episodes: 1
Average Return: 25.0
```

## Gradio Web App

Start the interactive Gradio dashboard:

```bash
MODEL_CHECKPOINT=saved_models/decision_transformer_d4rl.pth \
python3 gradio_app.py
```

This will launch a web server locally (usually at `http://127.0.0.1:8000`).
You can open this URL in your browser to interactively test the model. 
The dashboard lets you use sliders to tweak the 11-dimensional state vector and target return-to-go, instantly visualizing the model's predicted actions.
