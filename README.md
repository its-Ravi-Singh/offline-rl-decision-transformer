# Offline RL Decision Transformer with Preference Learning

**CSE 676 Deep Learning — Team Gradient Gone Wild**

**Members:** Hemanth Phani Srinivas Chilamkurthy, Ravi Rajaram Singh

We built an offline RL pipeline for MuJoCo continuous control. The main idea was to train a Decision Transformer on Hopper trajectories and then add a preference learning pipeline to study what happens when preference labels are noisy or wrong.

## What We Did

1. Trained return-conditioned transformer policies from fixed offline data — no online environment interaction during training.
2. Compared our Decision Transformer against a simpler Perception Transformer across Hopper simple, medium, and expert splits.
3. Generated segment-level preference pairs from offline trajectories.
4. Trained a preference model and analyzed its errors before using it as a reward proxy.

We started with CartPole for quick testing but moved to the real D4RL Hopper benchmark after the checkpoint feedback.

## Repository Structure

```text
.
|-- main.py                         # train + eval Decision Transformer on one dataset
|-- d4rl_compare.py                 # benchmark both models on all three Hopper splits
|-- train.py                        # training loop shared by both models
|-- evaluate.py                     # live Gymnasium rollout evaluation
|-- train_perception.py             # train the Perception Transformer
|-- compare_models.py               # compare saved checkpoints side by side
|-- train_preference.py             # generate preference pairs, train preference model
|-- analyze_preference_errors.py    # see where the preference model gets it wrong
|-- deploy.py                       # quick rollout smoke test
|-- gradio_app.py                   # Gradio web demo (action predictor + Hopper video)
|-- data/
|   `-- dataset.py                  # trajectory buffer, sequence windows, preference pairs
|-- models/
|   |-- decision_transformer.py     # causal sequence model (our main policy)
|   |-- perception_transformer.py   # Perceiver-style comparison policy
|   `-- preference_model.py         # segment preference model
|-- d4rl_results/                   # benchmark plots, summaries, checkpoints
|-- saved_models/                   # deployable checkpoints
|-- DEPLOYMENT.md                   # how to run the demo
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

## Run the Benchmark

```bash
EPOCHS=10 BATCH_SIZE=512 CONTEXT_LEN=8 N_EVAL=10 MAX_WINDOWS=10000 python3 d4rl_compare.py
```

Trains both models on all three Hopper splits and saves results to `d4rl_results/`.

## Results

![D4RL Hopper benchmark comparison](d4rl_results/d4rl_comparison.png)

| Split | Model | Avg Return | Std Return | Min / Max |
| --- | --- | ---: | ---: | ---: |
| Simple | Decision Transformer | 18.9 | 0.2 | 18.6 / 19.3 |
| Simple | Perception Transformer | 201.2 | 2.3 | 197.9 / 204.5 |
| Medium | Decision Transformer | 23.1 | 0.8 | 21.7 / 24.9 |
| Medium | Perception Transformer | 552.8 | 2.3 | 550.5 / 559.1 |
| Expert | Decision Transformer | 54.8 | 0.9 | 53.4 / 56.5 |
| Expert | Perception Transformer | 78.3 | 2.2 | 75.5 / 82.2 |

These are bounded integration results, not final tuned D4RL numbers. We kept training short so the full pipeline would finish in reasonable time.

## Training Curves

| Split | Decision Transformer | Perception Transformer |
| --- | --- | --- |
| Simple | ![DT simple](d4rl_results/loss_decision_transformer_simple.png) | ![PT simple](d4rl_results/loss_perception_transformer_simple.png) |
| Medium | ![DT medium](d4rl_results/loss_decision_transformer_medium.png) | ![PT medium](d4rl_results/loss_perception_transformer_medium.png) |
| Expert | ![DT expert](d4rl_results/loss_decision_transformer_expert.png) | ![PT expert](d4rl_results/loss_perception_transformer_expert.png) |

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

## Deployment

Run a quick rollout:

```bash
python3 deploy.py --no-video --episodes 1
```

Launch the Gradio demo:

```bash
MODEL_CHECKPOINT=saved_models/decision_transformer_d4rl.pth python3 gradio_app.py
```

Then open `http://127.0.0.1:8000`. Tab 1 lets you test the action predictor, Tab 2 runs a live Hopper episode and records a video.

## Checkpoint Feedback & Task Checklist

- [x] Migrate DT baseline to D4RL benchmarks (Hopper) instead of CartPole.
- [x] Build the preference-pair generation pipeline from offline trajectories.
- [x] Implement and train the preference model.
- [x] Analyze preference model errors and build a detection script.
- [x] Benchmark both models, plot training curves, compare results.
- [x] Set up Gradio demo and rollout scripts.

## References

- Chen et al. *Decision Transformer: Reinforcement Learning via Sequence Modeling*, NeurIPS 2021. [[arxiv]](https://arxiv.org/abs/2106.01345)
- Fu et al. *D4RL: Datasets for Deep Data-Driven Reinforcement Learning*, 2020. [[arxiv]](https://arxiv.org/abs/2004.07219)
- Christiano et al. *Deep Reinforcement Learning from Human Preferences*, NeurIPS 2017. [[arxiv]](https://arxiv.org/abs/1706.03741)
- Farama Foundation. Minari documentation. [[docs]](https://minari.farama.org/)
- Farama Foundation. Gymnasium MuJoCo environments. [[docs]](https://gymnasium.farama.org/environments/mujoco/)
