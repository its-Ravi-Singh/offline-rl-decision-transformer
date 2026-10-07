# Offline RL Decision Transformer with Preference Learning

**Deep Learning and Reinforcement Learning — Team Gradient Gone Wild**

**Members:** Ravi Rajaram Singh, Hemanth Phani Srinivas Chilamkurthy

We built an offline RL pipeline for MuJoCo continuous control. The main idea was to train a Decision Transformer on Hopper trajectories and then add a preference learning pipeline to study what happens when preference labels are noisy or wrong.

## Highlights

- Trained return-conditioned transformer policies purely from fixed offline data, with no online environment interaction during training.
- Benchmarked three models on the D4RL Hopper simple, medium and expert splits.
- Fixed three bugs in the Decision Transformer pipeline (no state normalization, action history shifted by one step at evaluation, timesteps reset each window), which raised its Hopper return 6-20x on every split (24.0 to 476.1 on medium).
- Preference weighting helped on simple and expert and slightly hurt on medium.
- Deployed as a Gradio web demo with an action predictor and a live Hopper rollout video.

## Demo

![Decision Transformer vs Perception Transformer on Hopper](docs/hopper-comparison.gif)

One example rollout from the original checkpoints, recorded before the fixes below: the Decision Transformer falls early, while the Perception Transformer keeps hopping. The fixed Decision Transformer now hops too; current numbers are in the benchmark table.

## What We Did

1. Trained return-conditioned transformer policies from fixed offline data — no online environment interaction during training.
2. Compared Decision Transformer, DT + Preference, and Perception Transformer across Hopper simple, medium, and expert splits.
3. Generated segment-level preference pairs from offline trajectories.
4. Trained a preference model, analyzed its errors, and used it to reweight Decision Transformer training windows.

We started with CartPole for quick testing but moved to the real D4RL Hopper benchmark after the checkpoint feedback.

## Repository Structure

```text
.
|-- main.py                         # train + eval Decision Transformer on one dataset
|-- d4rl_compare.py                 # benchmark DT and Perception on all three Hopper splits
|-- d4rl_compare_three.py           # benchmark DT, DT + Preference, and Perception
|-- train.py                        # training loop shared by both models
|-- evaluate.py                     # live Gymnasium rollout evaluation
|-- train_perception.py             # train the Perception Transformer
|-- compare_models.py               # compare saved checkpoints side by side
|-- train_preference.py             # generate preference pairs, train preference model
|-- analyze_preference_errors.py    # see where the preference model gets it wrong
|-- serve.py                        # FastAPI inference API (/health, /act)
|-- static/                         # small web UI served by the API at /ui
|-- gradio_app.py                   # Gradio web demo (action predictor + Hopper video)
|-- data/
|   `-- dataset.py                  # trajectory buffer, sequence windows, preference pairs
|-- models/
|   |-- decision_transformer.py     # causal sequence model (our main policy)
|   |-- perception_transformer.py   # Perceiver-style comparison policy
|   `-- preference_model.py         # segment preference model
|-- d4rl_results/                   # benchmark plots, summaries, checkpoints
|-- saved_models/                   # deployable checkpoints
|-- docs/                           # project board screenshots and walkthrough GIF
|-- Dockerfile                      # container image for the inference API
`-- REPORT.md                       # project report
```

## Setup

We used Python 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Minari will download the datasets automatically on first run.

## Train the Decision Transformer

```bash
python3 main.py
```

Default settings:

| Setting | Value |
| --- | --- |
| Dataset | `mujoco/hopper/expert-v0` |
| Environment | `Hopper-v4` |
| Epochs | `50` |
| Batch size | `256` |
| Context length | `20` |
| Target RTG | `3000.0` |

Override with env vars:

```bash
DATASET_ID=mujoco/hopper/medium-v0 EPOCHS=30 python3 main.py
```

By default, if a trained preference model exists, `main.py` uses it to weight DT sequence windows. To train the normal DT without preference weighting:

```bash
USE_PREFERENCE_WEIGHTS=0 python3 main.py
```

## Run the Benchmark

```bash
EPOCHS=10 BATCH_SIZE=512 CONTEXT_LEN=8 N_EVAL=10 MAX_WINDOWS=10000 python3 d4rl_compare.py
```

Trains both models on all three Hopper splits and saves results to `d4rl_results/`.

For the final three-way benchmark:

```bash
EPOCHS=10 BATCH_SIZE=512 CONTEXT_LEN=8 N_EVAL=10 MAX_WINDOWS=10000 python3 d4rl_compare_three.py
```

This compares Decision Transformer, DT + Preference, and Perception Transformer.

Latest three-way benchmark (average return over 10 evaluation episodes, using the settings in the command above):

| Split | Decision Transformer | DT + Preference | Perception Transformer |
| --- | ---: | ---: | ---: |
| Simple | 834.9 | **861.0** | 544.6 |
| Medium | 476.1 | 442.4 | **552.3** |
| Expert | 429.7 | **571.4** | 47.7 |

DT + Preference was best on simple and expert, and the Perception Transformer was best on medium. Preference weighting helped the DT on two of three splits (+26 on simple, +142 on expert) and hurt it on medium (-34).

These runs use short training (10 epochs, 10,000 windows), so treat the numbers as a comparison between models rather than final benchmark scores. Well-tuned offline RL methods reach a few thousand on Hopper.

### Bugs we fixed

The first version of this benchmark scored the Decision Transformer at 60.4 / 24.0 / 68.6, meaning the hopper fell over within about 20 steps. Three problems caused it:

1. **No state normalization.** Hopper state values have very different scales. The model now stores the dataset mean and std with its weights and normalizes inside `forward`, so training, evaluation, the API and the demo all use the same numbers.
2. **Action history off by one at evaluation.** Evaluation put a zero action at the front of the history, so each past action sat next to the wrong state. In training, action `t` always sits next to state `t`.
3. **Timesteps restarted at 0.** Evaluation numbered every context window from 0, while training used the real episode step.

Training also uses overlapping windows now (`stride=1` instead of `stride=context_len`).

With the same data and training budget, the original code scores 16.5 on medium and the fixed code scores 418.9.

## Model Architecture

| Model | Hidden Dim | Layers | Heads | Main Input |
| --- | ---: | ---: | ---: | --- |
| Decision Transformer | 128 | 3 | 4 | RTG, state, action sequence |
| DT + Preference | 128 | 3 | 4 | Same DT architecture with weighted sequence loss |
| Perception Transformer | 256 | 2 | 4 | Current state and RTG |
| Preference Model | 128 | 2 | 4 | Paired trajectory segments |

Decision Transformer turns each timestep into three tokens: RTG, state, and action. With context length 8, that gives 24 tokens per window. Perception Transformer uses 8 learned latent tokens and cross-attention over the state and RTG tokens. The preference model scores trajectory segments and gives higher weights to windows that look better.

## Preference Learning

```bash
python3 train_preference.py --dataset-id mujoco/hopper/medium-v0 --num-pairs 10000 --segment-len 20 --epochs 20
```

Labels are based on segment return — whichever segment has higher total reward is labeled as preferred. You can also add noise to test how robust the model is:

```bash
python3 train_preference.py --label-noise 0.2 --num-pairs 10000
```

## Preference Error Analysis

```bash
python3 analyze_preference_errors.py
```

Checks where the model was wrong and how confident it was when it messed up. Saves flagged cases to `preference_results/preference_error_cases.csv`.

## Preference-Weighted DT

After preference training, DT windows are scored by the preference model. Higher-score windows get larger sample weights in the DT action MSE loss:

```text
higher preference score -> larger DT loss weight
lower preference score -> smaller DT loss weight
```

This is offline preference-weighted behavior cloning, not full online RLHF.

## Deployment

### Inference API (FastAPI + Docker)

`serve.py` serves the trained Decision Transformer over HTTP. Requests are validated with Pydantic, so a state with the wrong number of values is rejected with a 422 before it reaches the model.

| Endpoint | Method | What it does |
| --- | --- | --- |
| `/health` | GET | Status, checkpoint, device and model dimensions |
| `/act` | POST | Takes an 11-value Hopper state (plus optional history and target RTG) and returns the predicted action |
| `/ui` | GET | Small browser dashboard for trying `/act` |
| `/docs` | GET | Interactive OpenAPI docs |

Run it with the prebuilt image (built and tested by GitHub Actions on every change):

```bash
docker run -p 8000:8000 ghcr.io/its-ravi-singh/offline-rl-decision-transformer:latest
```

Or build it yourself:

```bash
docker build -t dt-api .
docker run -p 8000:8000 dt-api
```

Or run it without Docker:

```bash
pip install -r requirements.txt
uvicorn serve:app --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/act \
  -H "Content-Type: application/json" \
  -d '{"state": [1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "target_rtg": 1.0}'
```

The container reads `PORT` from the environment, so it runs as-is on hosts such as Google Cloud Run, Render or Railway.

### Hugging Face Space (live Hopper video)

`space/` holds a Dockerfile that runs the Gradio demo on a free Hugging Face Docker Space. MuJoCo renders with OSMesa (software OpenGL) because Spaces have no display. To publish it:

1. Create a new Space on Hugging Face with the **Docker** SDK (blank template).
2. Push this repo to the Space, with `space/Dockerfile` copied to the root as `Dockerfile` and a README that starts with:

```yaml
---
title: Hopper Offline RL
sdk: docker
app_port: 7860
---
```

### Gradio demo

Launch using Gradio:

```bash
MODEL_CHECKPOINT=saved_models/decision_transformer_d4rl.pth python3 gradio_app.py
```

Then open `http://127.0.0.1:8000`. Tab 1 lets you test the action predictor, Tab 2 runs a live Hopper episode and records a video.

## Project Management

We tracked the work on a GitHub Projects board (23 tasks in total) in our class organization. I was assigned 15 of them, some shared with my teammate, covering the proposal, dataset, model building, training, evaluation, documentation and improvements, from Mar 3 to Apr 30, 2026. All 15 are marked Done.

![Project board: My items and Backlog views](docs/project-board-proof.png)

![Project board walkthrough](docs/project-board-tour.gif)

The board lives in the private class organization, so these screenshots are the proof. The commit history in this repo shows the same work.

## References

- Chen et al. *Decision Transformer: Reinforcement Learning via Sequence Modeling*, NeurIPS 2021. [[arxiv]](https://arxiv.org/abs/2106.01345)
- Fu et al. *D4RL: Datasets for Deep Data-Driven Reinforcement Learning*, 2020. [[arxiv]](https://arxiv.org/abs/2004.07219)
- Christiano et al. *Deep Reinforcement Learning from Human Preferences*, NeurIPS 2017. [[arxiv]](https://arxiv.org/abs/1706.03741)
- Farama Foundation. Minari documentation. [[docs]](https://minari.farama.org/)
- Farama Foundation. Gymnasium MuJoCo environments. [[docs]](https://gymnasium.farama.org/environments/mujoco/)
