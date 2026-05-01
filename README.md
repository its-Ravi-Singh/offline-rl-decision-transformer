# Offline RL Decision Transformer with Preference Learning

Final project for CSE 676 Deep Learning by team **Gradient Gone Wild**.

This repository implements an offline reinforcement learning pipeline for
continuous-control MuJoCo tasks using Minari/D4RL-style datasets. The main
implementation trains a causal Decision Transformer on Hopper trajectories, adds
a Perception Transformer comparison policy, and includes a preference-learning
pipeline for trajectory segment comparisons and error analysis.

## Project Scope

The project focuses on four goals:

1. Train return-conditioned transformer policies from fixed offline RL data.
2. Compare performance across Hopper simple, medium, and expert datasets.
3. Generate segment-level preference pairs from offline trajectories.
4. Train and audit a preference model before using preferences as reward
   feedback.

The current codebase is aligned with the original offline RL proposal rather
than a toy CartPole demonstration. Hopper-v4 is the recorded benchmark target,
and Walker2d can be run through the same single-model path.

## Repository Structure

```text
.
|-- main.py                         # train/evaluate Decision Transformer on one dataset
|-- d4rl_compare.py                 # benchmark DT and Perception Transformer on Hopper splits
|-- train.py                        # shared supervised action training loop
|-- evaluate.py                     # live Gymnasium MuJoCo evaluation
|-- train_perception.py             # train Perception Transformer checkpoint
|-- compare_models.py               # compare saved DT and Perception checkpoints
|-- train_preference.py             # generate preference pairs and train preference model
|-- analyze_preference_errors.py    # diagnose preference-label/model mistakes
|-- deploy.py                       # rollout deployment smoke test
|-- serve.py                        # FastAPI action service
|-- data/
|   `-- dataset.py                  # trajectory buffers, sequence windows, preference pairs
|-- models/
|   |-- decision_transformer.py     # causal sequence Decision Transformer
|   |-- perception_transformer.py   # Perceiver-style RTG-conditioned policy
|   `-- preference_model.py         # Bradley-Terry segment preference model
|-- d4rl_results/                   # recorded benchmark summaries, plots, checkpoints
|-- saved_models/                   # deployable checkpoints
|-- BENCHMARK_RESULTS.md            # concise benchmark table
|-- DEPLOYMENT.md                   # rollout/API deployment notes
`-- REPORT.md                       # project report
```

## Setup

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required packages are listed in `requirements.txt`:

- PyTorch
- Gymnasium with MuJoCo
- Minari with Hugging Face/HDF5 support
- NumPy
- Matplotlib
- FastAPI and Uvicorn

Minari datasets are downloaded automatically on first use. The scripts set local
cache paths for Minari, Matplotlib, and font caches so generated files stay
inside the project directory.

## Train the Decision Transformer

Train the default Decision Transformer on Hopper expert data:

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

Useful overrides:

```bash
DATASET_ID=mujoco/hopper/medium-v0 EPOCHS=30 CONTEXT_LEN=20 python3 main.py
```

Walker2d can be targeted with matching dataset and environment settings:

```bash
DATASET_ID=mujoco/walker2d/medium-v0 ENV_NAME=Walker2d-v4 TARGET_RTG=5000 python3 main.py
```

## Run the Hopper Benchmark

Run the bounded benchmark configuration used for the recorded results:

```bash
EPOCHS=10 BATCH_SIZE=512 CONTEXT_LEN=8 N_EVAL=10 MAX_WINDOWS=10000 python3 d4rl_compare.py
```

This trains both model families on Hopper simple, medium, and expert splits:

- `DecisionTransformer`: causal sequence model over RTG/state/action tokens
- `PerceptionTransformer`: Perceiver-style one-step policy conditioned on state
  and target RTG

Outputs are written to `d4rl_results/`:

- `summary.csv`
- `d4rl_comparison.png`
- one training-loss plot per model/split
- one checkpoint per model/split

## Recorded Benchmark Results

Latest bounded Hopper-v4 benchmark:

![D4RL Hopper benchmark comparison](d4rl_results/d4rl_comparison.png)

*Figure 1. Bounded Hopper benchmark summary showing training loss curves and
evaluation returns for Decision Transformer and Perception Transformer across
simple, medium, and expert splits.*

| Setting | Value |
| --- | ---: |
| Epochs | 10 |
| Batch size | 512 |
| Context length | 8 |
| Sampled windows/transitions per split | 10,000 |
| Evaluation episodes | 10 |
| Target RTG normalizer | 3000.0 |

| Split | Model | Avg Return | Std Return | Min / Max |
| --- | --- | ---: | ---: | ---: |
| Simple | Decision Transformer | 18.9 | 0.2 | 18.6 / 19.3 |
| Simple | Perception Transformer | 201.2 | 2.3 | 197.9 / 204.5 |
| Medium | Decision Transformer | 23.1 | 0.8 | 21.7 / 24.9 |
| Medium | Perception Transformer | 552.8 | 2.3 | 550.5 / 559.1 |
| Expert | Decision Transformer | 54.8 | 0.9 | 53.4 / 56.5 |
| Expert | Perception Transformer | 78.3 | 2.2 | 75.5 / 82.2 |

These are bounded integration results, not final tuned D4RL scores. The
configuration intentionally limits training samples so the full pipeline can be
run within a practical project timeline.

## Training Curves

The benchmark run saves one loss plot for each model and dataset split:

| Split | Decision Transformer | Perception Transformer |
| --- | --- | --- |
| Simple | ![Decision Transformer simple loss](d4rl_results/loss_decision_transformer_simple.png) | ![Perception Transformer simple loss](d4rl_results/loss_perception_transformer_simple.png) |
| Medium | ![Decision Transformer medium loss](d4rl_results/loss_decision_transformer_medium.png) | ![Perception Transformer medium loss](d4rl_results/loss_perception_transformer_medium.png) |
| Expert | ![Decision Transformer expert loss](d4rl_results/loss_decision_transformer_expert.png) | ![Perception Transformer expert loss](d4rl_results/loss_perception_transformer_expert.png) |

Single-checkpoint training scripts also save standalone curves:

![Decision Transformer training loss](plots/training_loss.png)

*Figure 2. Decision Transformer training loss from the single-model training
path.*

![Perception Transformer training loss](plots/perception_transformer_training_loss.png)

*Figure 3. Perception Transformer training loss from the standalone Perception
Transformer training path.*

## Preference Learning

Generate trajectory-segment preference pairs and train a preference model:

```bash
python3 train_preference.py \
  --dataset-id mujoco/hopper/medium-v0 \
  --num-pairs 10000 \
  --segment-len 20 \
  --epochs 20
```

Preference-pair labels are return-derived:

- label `0`: left segment has higher return
- label `1`: right segment has higher return

Controlled noise can be injected for robustness experiments:

```bash
python3 train_preference.py --label-noise 0.2 --num-pairs 10000
```

The preference pipeline writes:

- `preference_results/preference_pairs.npz`
- `preference_results/preference_model.pth`
- `preference_results/preference_training.png`

When generated, the preference training curve can be embedded with:

```markdown
![Preference model training](preference_results/preference_training.png)
```

## Preference Error Analysis

After training a preference model, run:

```bash
python3 analyze_preference_errors.py
```

The analysis compares predictions against noisy labels and clean return-derived
labels, reports confidence/error patterns, and saves flagged cases to:

```text
preference_results/preference_error_cases.csv
```

This is intended to identify harmful preference-model errors before using a
learned preference model as a reward proxy.

## Deployment

Run a rollout smoke test:

```bash
python3 deploy.py --no-video --episodes 1
```

Start the local FastAPI action service:

```bash
MODEL_CHECKPOINT=saved_models/decision_transformer_d4rl.pth \
uvicorn serve:app --host 127.0.0.1 --port 8000
```

See `DEPLOYMENT.md` for request examples and the latest recorded smoke-test
output.

## Generated Artifacts

Expected generated artifacts include:

- `saved_models/decision_transformer_d4rl.pth`
- `saved_models/perception_transformer_d4rl.pth`
- `plots/training_loss.png`
- `plots/perception_transformer_training_loss.png`
- `d4rl_results/d4rl_comparison.png`
- `d4rl_results/loss_decision_transformer_simple.png`
- `d4rl_results/loss_decision_transformer_medium.png`
- `d4rl_results/loss_decision_transformer_expert.png`
- `d4rl_results/loss_perception_transformer_simple.png`
- `d4rl_results/loss_perception_transformer_medium.png`
- `d4rl_results/loss_perception_transformer_expert.png`
- `d4rl_results/summary.csv`
- `model_comparison/summary.csv`
- `model_comparison/decision_vs_perception.png`
- `preference_results/preference_pairs.npz`
- `preference_results/preference_model.pth`
- `preference_results/preference_training.png`
- `preference_results/preference_error_cases.csv`

Large downloaded datasets, local virtual environments, Python bytecode, and
temporary caches are excluded from version control.

## Method Summary

1. Load fixed Minari/D4RL MuJoCo trajectories.
2. Compute undiscounted return-to-go for each timestep.
3. Train a causal Decision Transformer on fixed-length trajectory windows.
4. Train a Perception Transformer comparison policy on transition samples.
5. Evaluate trained policies in live Gymnasium MuJoCo environments.
6. Sample segment pairs from offline trajectories and label the higher-return
   segment as preferred.
7. Train a Bradley-Terry style preference model from segment comparisons.
8. Analyze preference mistakes using label noise, clean-label disagreement,
   confidence, and return gaps.

## Key Formulas

Return-to-go for timestep `t`:

```text
R_t = sum_{k=t}^{T} gamma^{k-t} r_k
```

The recorded Hopper runs use `gamma = 1.0` and normalize RTG as:

```text
rhat_t = R_t / R_target
```

Policy training uses masked action MSE:

```text
L_policy =
    (sum_i sum_t m_{i,t} ||ahat_{i,t} - a_{i,t}||_2^2)
    / (sum_i sum_t m_{i,t})
```

Preference learning scores two segments and applies Bradley-Terry comparison:

```text
P(left preferred) = exp(u_left) / (exp(u_left) + exp(u_right))
L_pref = -log softmax([u_left, u_right])_y
```

## Checkpoint Feedback & Task Checklist

We divided the remaining work among the team to make sure we addressed all the instructor feedback from the checkpoint. Everything is now complete:

- [x] **Dataset Migration:** Get the DT baseline working on the target D4RL benchmarks (Hopper) instead of the basic CartPole setup.
- [x] **Data Pipeline:** Build the preference-pair generation pipeline from the offline trajectories.
- [x] **Preference Model:** Implement and train the preference model on the generated pairs.
- [x] **Error Analysis:** Analyze what happens when the preference model makes mistakes and write a script to detect these errors.
- [x] **Evaluation:** Benchmark the models, plot the training curves, and compare results.
- [x] **Deployment:** Set up the FastAPI action service and rollout scripts.

## References

- Chen et al., 2021. *Decision Transformer: Reinforcement Learning via Sequence
  Modeling.*
- Fu et al., 2020. *D4RL: Datasets for Deep Data-Driven Reinforcement Learning.*
- Christiano et al., 2017. *Deep Reinforcement Learning from Human Preferences.*
- Farama Foundation, Minari dataset standard and Gymnasium MuJoCo environments.
