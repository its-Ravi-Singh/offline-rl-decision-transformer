# Offline RL Decision Transformer with Preference Learning

Final project for CSE 676 Deep Learning — Team **Gradient Gone Wild**.

**Team Members:** Ravi Singh, [Teammate Name]


We built an offline RL pipeline for MuJoCo continuous control using Minari/D4RL datasets. The core idea was to train a Decision Transformer on Hopper trajectories and add a preference learning pipeline on top of it to study what happens when preference labels are noisy or wrong.

## What We Did

1. Trained return-conditioned transformer policies from fixed offline RL data (no online exploration).
2. Compared performance across Hopper simple, medium, and expert datasets.
3. Generated segment-level preference pairs from offline trajectories.
4. Trained a preference model and analyzed its error patterns before using it as a reward proxy.

We started with CartPole for prototyping but eventually moved everything to the real D4RL Hopper benchmark, which was one of the key pieces of feedback from the checkpoint.

## Repository Structure

```text
.
|-- main.py                         # train/evaluate Decision Transformer on one dataset
|-- d4rl_compare.py                 # benchmark DT and Perception Transformer on Hopper splits
|-- train.py                        # shared training loop
|-- evaluate.py                     # live Gymnasium MuJoCo evaluation
|-- train_perception.py             # train Perception Transformer checkpoint
|-- compare_models.py               # compare saved DT and Perception checkpoints
|-- train_preference.py             # generate preference pairs and train preference model
|-- analyze_preference_errors.py    # check where the preference model gets things wrong
|-- deploy.py                       # rollout smoke test
|-- gradio_app.py                   # Gradio web app for live model testing
|-- data/
|   `-- dataset.py                  # trajectory buffers, sequence windows, preference pairs
|-- models/
|   |-- decision_transformer.py     # causal sequence Decision Transformer
|   |-- perception_transformer.py   # Perceiver-style policy (our comparison model)
|   `-- preference_model.py         # Bradley-Terry segment preference model
|-- d4rl_results/                   # benchmark plots, summaries, and checkpoints
|-- saved_models/                   # deployable checkpoints
|-- DEPLOYMENT.md                   # deployment notes
`-- REPORT.md                       # full project report
```

## Setup

We used Python 3.10 for this project.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required packages:
- PyTorch
- Gymnasium with MuJoCo
- Minari (with HDF5/HuggingFace support)
- NumPy, Matplotlib
- Gradio (for the web demo)

Minari will auto-download the datasets on first use.

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
| Target RTG normalizer | `3000.0` |
| Output checkpoint | `saved_models/decision_transformer_d4rl.pth` |

You can override settings with environment variables:

```bash
DATASET_ID=mujoco/hopper/medium-v0 EPOCHS=30 python3 main.py
```

Walker2d also works through the same path:

```bash
DATASET_ID=mujoco/walker2d/medium-v0 ENV_NAME=Walker2d-v4 TARGET_RTG=5000 python3 main.py
```

## Run the Hopper Benchmark

To reproduce our recorded results:

```bash
EPOCHS=10 BATCH_SIZE=512 CONTEXT_LEN=8 N_EVAL=10 MAX_WINDOWS=10000 python3 d4rl_compare.py
```

This trains both models (Decision Transformer and Perception Transformer) on all three Hopper splits and saves results to `d4rl_results/`.

## Results

Latest bounded Hopper-v4 benchmark:

![D4RL Hopper benchmark comparison](d4rl_results/d4rl_comparison.png)

*Figure 1. Training loss curves and evaluation returns for both models across the three Hopper splits.*

| Setting | Value |
| --- | ---: |
| Epochs | 10 |
| Batch size | 512 |
| Context length | 8 |
| Samples per split | 10,000 |
| Eval episodes | 10 |
| Target RTG normalizer | 3000.0 |

| Split | Model | Avg Return | Std Return | Min / Max |
| --- | --- | ---: | ---: | ---: |
| Simple | Decision Transformer | 18.9 | 0.2 | 18.6 / 19.3 |
| Simple | Perception Transformer | 201.2 | 2.3 | 197.9 / 204.5 |
| Medium | Decision Transformer | 23.1 | 0.8 | 21.7 / 24.9 |
| Medium | Perception Transformer | 552.8 | 2.3 | 550.5 / 559.1 |
| Expert | Decision Transformer | 54.8 | 0.9 | 53.4 / 56.5 |
| Expert | Perception Transformer | 78.3 | 2.2 | 75.5 / 82.2 |

These are bounded integration results, not final tuned D4RL scores. We kept training short so the full pipeline could run in a reasonable amount of time.

## Training Curves

| Split | Decision Transformer | Perception Transformer |
| --- | --- | --- |
| Simple | ![DT simple loss](d4rl_results/loss_decision_transformer_simple.png) | ![PT simple loss](d4rl_results/loss_perception_transformer_simple.png) |
| Medium | ![DT medium loss](d4rl_results/loss_decision_transformer_medium.png) | ![PT medium loss](d4rl_results/loss_perception_transformer_medium.png) |
| Expert | ![DT expert loss](d4rl_results/loss_decision_transformer_expert.png) | ![PT expert loss](d4rl_results/loss_perception_transformer_expert.png) |

![Decision Transformer training loss](plots/training_loss.png)
*Figure 2. DT training loss from the single-model run.*

![Perception Transformer training loss](plots/perception_transformer_training_loss.png)
*Figure 3. Perception Transformer training loss.*

## Preference Learning

Generate preference pairs and train the preference model:

```bash
python3 train_preference.py \
  --dataset-id mujoco/hopper/medium-v0 \
  --num-pairs 10000 \
  --segment-len 20 \
  --epochs 20
```

Labels are assigned based on which segment had a higher total return:
- label `0`: left segment is better
- label `1`: right segment is better

You can also add some label noise to test robustness:

```bash
python3 train_preference.py --label-noise 0.2 --num-pairs 10000
```

Outputs go to `preference_results/`.

## Preference Error Analysis

After training the preference model, run:

```bash
python3 analyze_preference_errors.py
```

This checks where the model was wrong, how confident it was when it made mistakes, and saves flagged error cases to `preference_results/preference_error_cases.csv`. We wanted to understand these failure modes before using the model as a reward signal.

## Deployment

Run a quick rollout to check the trained model:

```bash
python3 deploy.py --no-video --episodes 1
```

Launch the Gradio web demo to test the model interactively:

```bash
MODEL_CHECKPOINT=saved_models/decision_transformer_d4rl.pth python3 gradio_app.py
```

Then open `http://127.0.0.1:8000` in your browser.

See `DEPLOYMENT.md` for more details.

## Checkpoint Feedback & Task Checklist

We divided the work among the team to address all the instructor feedback from the checkpoint:

- [x] **Dataset Migration:** Get the DT baseline working on D4RL benchmarks (Hopper) instead of CartPole.
- [x] **Data Pipeline:** Build the preference-pair generation pipeline from offline trajectories.
- [x] **Preference Model:** Implement and train the preference model on the generated pairs.
- [x] **Error Analysis:** Analyze what happens when the preference model makes mistakes.
- [x] **Evaluation:** Benchmark the models, plot training curves, and compare results.
- [x] **Deployment:** Set up the Gradio demo and rollout scripts.

## References

- Chen et al., 2021. *Decision Transformer: Reinforcement Learning via Sequence Modeling.*
- Fu et al., 2020. *D4RL: Datasets for Deep Data-Driven Reinforcement Learning.*
- Christiano et al., 2017. *Deep Reinforcement Learning from Human Preferences.*
- Farama Foundation, Minari and Gymnasium MuJoCo environments.
