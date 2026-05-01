# Offline RL Decision Transformer with Preference Learning

## Executive Summary

This project studies offline reinforcement learning for continuous-control
MuJoCo tasks using return-conditioned transformer policies and preference
learning. The implementation uses Minari/D4RL-style Hopper datasets, trains a
causal Decision Transformer baseline, compares it with a Perception Transformer
policy, and adds a full preference-pair generation and preference-model analysis
pipeline.

The most important project improvement is methodological alignment. The code now
uses real offline RL datasets instead of toy control data, evaluates policies in
Hopper-v4, and includes the preference-learning components needed to study how
incorrect preference labels can affect downstream learning.

## Problem Statement

Offline reinforcement learning learns a policy from a fixed dataset without
collecting new experience during training. This setting is important when online
exploration is expensive, unsafe, or unavailable. In this project, the policy is
trained from fixed Hopper trajectories and evaluated in the matching Gymnasium
MuJoCo environment.

The project investigates four questions:

1. Can return-conditioned transformer policies learn useful actions from fixed
   Minari/D4RL Hopper trajectories?
2. How do results change across simple, medium, and expert data splits?
3. Can trajectory segment preferences be generated and modeled from offline
   data?
4. Which diagnostics reveal harmful preference-model errors before the model is
   used as a reward proxy?

## Dataset

The recorded benchmark uses the following Minari Hopper datasets:

| Split | Dataset ID | Evaluation Environment |
| --- | --- | --- |
| Simple | mujoco/hopper/simple-v0 | Hopper-v4 |
| Medium | mujoco/hopper/medium-v0 | Hopper-v4 |
| Expert | mujoco/hopper/expert-v0 | Hopper-v4 |

Each episode provides observations, actions, and rewards. The data loader stores
observations, actions, rewards, and an undiscounted return-to-go value computed
backward through each episode. Return-to-go is normalized by a target return
constant, currently 3000.0 for Hopper.

The implementation also supports Walker2d through the same single-model
training path when the dataset, environment, and target return are changed to
the Walker2d equivalents.

Walker2d support is implemented but does not yet have recorded benchmark
results in this repository.

## Model Summaries

The following summaries are generated from the PyTorch model classes using the
default Hopper dimensions: state dimension 11, action dimension 3, and context length 20
for the Decision Transformer.

### Decision Transformer Summary

| Component | Layer Type | Parameters |
| --- | --- | ---: |
| State embedding | Linear | 1,536 |
| Action embedding | Linear | 512 |
| RTG embedding | Linear | 256 |
| Timestep embedding | Embedding | 128,000 |
| Token-type embedding | Embedding | 384 |
| Transformer body | Transformer encoder | 594,816 |
| Final normalization | Layer normalization | 256 |
| Action head | Linear layer with Tanh | 387 |
| **Total** |  | **726,147** |

Input tensors:

| Input | Shape |
| --- | --- |
| States | batch by context length by 11 |
| Actions | batch by context length by 3 |
| RTGs | batch by context length, optionally with a final singleton channel |
| Timesteps | batch by context length |
| Attention mask | batch by context length |

Output tensor:

| Output | Shape |
| --- | --- |
| Predicted actions | batch by context length by 3 |

The model embeds RTG, state, and action tokens at every timestep, applies causal
transformer attention, and predicts continuous actions from the state-token
positions.

### Perception Transformer Summary

| Component | Layer Type | Parameters |
| --- | --- | ---: |
| State embedding | Linear | 3,072 |
| RTG embedding | Linear | 512 |
| Cross-attention | Multi-head attention | 263,168 |
| Cross-attention normalization | Layer normalization | 512 |
| Latent encoder | Transformer encoder | 1,579,520 |
| Action head | MLP with Tanh | 67,075 |
| **Total** |  | **1,916,419** |

Input tensors:

| Input | Shape |
| --- | --- |
| States | batch by 11 |
| RTGs | batch, optionally with a final singleton channel |

Output tensor:

| Output | Shape |
| --- | --- |
| Predicted actions | batch by 3 |

The model embeds the current state and target RTG, cross-attends learned latent
tokens to those inputs, processes the latents with a transformer encoder, and
predicts a continuous action.

### Preference Model Summary

| Component | Layer Type | Parameters |
| --- | --- | ---: |
| Input projection | Linear | 2,048 |
| Segment encoder | Transformer encoder | 396,544 |
| Score head | MLP scalar head | 16,897 |
| **Total** |  | **415,489** |

Input tensors for one segment:

| Input | Shape |
| --- | --- |
| States | batch by segment length by 11 |
| Actions | batch by segment length by 3 |
| RTGs | batch by segment length, optionally with a final singleton channel |
| Mask | optional batch by segment length |

Output tensor:

| Output | Shape |
| --- | --- |
| Segment score | batch |

For preference training, the same model scores the left and right trajectory
segments. The two scores are stacked into left-score and right-score logits and
trained with cross-entropy against the preferred segment label.

## Decision Transformer

The Decision Transformer is a causal sequence model over three token types per
timestep: a return-to-go token, a state token, and an action token.

Implementation details:

- state embedding: linear projection from the continuous observation vector
- action embedding: linear projection from the continuous action vector
- RTG embedding: linear projection from normalized return-to-go
- timestep embedding: learned embedding
- token-type embedding: distinguishes RTG, state, and action tokens
- sequence model: PyTorch transformer encoder with a causal mask
- output head: predicts continuous actions at state-token positions
- action range: bounded with Tanh

Training examples come from a sequence trajectory dataset, which samples
fixed-length trajectory windows and supplies a mask for padded timesteps. The
training loss is masked mean-squared error between predicted actions and dataset
actions.

## Perception Transformer

The Perception Transformer is a Perceiver-style comparison policy. It is not a
full causal sequence model. Instead, it predicts an action from the current state
and current target RTG.

Implementation details:

- input tokens: state token and RTG token
- learned latent array cross-attends to the input tokens
- transformer encoder processes the latent representation
- pooled latent representation predicts a continuous action
- action range: bounded with Tanh

This model is naturally compatible with the one-step evaluator. Under the
current bounded benchmark, it outperforms the sequence Decision Transformer.

## Preference Model

The preference model learns to choose between two trajectory segments. Segment
pairs are generated by sampling two fixed-length segments from offline
trajectories, computing the return of each segment, and assigning the preference
label to the higher-return segment.

Preference-pair generation:

1. Load full offline trajectories.
2. Sample two trajectory segments.
3. Compute undiscounted segment returns.
4. Assign label 0 if the left segment return is higher.
5. Assign label 1 if the right segment return is higher.
6. Optionally flip labels using the label-noise setting.
7. Save the pair dataset as compressed NumPy arrays.

Preference model details:

- inputs: segment states, segment actions, and segment RTGs
- encoder: transformer encoder over segment timesteps
- output: one scalar score per segment
- objective: cross-entropy over the left-segment and right-segment scores

The preference model is currently diagnostic. It is not yet connected back into
policy training or return relabeling.

## Mathematical Formulation

An offline trajectory is represented as:

$$
\tau = \{(s_0, a_0, r_0), (s_1, a_1, r_1), \ldots, (s_T, a_T, r_T)\}
$$

For every timestep t, the return-to-go is:

$$
R_t = \sum_{k=t}^{T} \gamma^{k-t} r_k
$$

The implementation uses gamma = 1.0, so R_t is the undiscounted future
return. For numerical stability, RTG is normalized before being passed into the
models:

$$
\hat{R}_t = \frac{R_t}{R_{\text{target}}}
$$

For Hopper, R_target = 3000.0 in the recorded experiments.

The Decision Transformer models the action distribution as a sequence modeling
problem:

$$
\hat{a}_t = f_{\theta}(R_{0:t}, s_{0:t}, a_{0:t-1}, t)
$$

where the hatted action term is the predicted continuous action at timestep t. The
supervised offline RL objective is masked action mean-squared error:

$$
\mathcal{L}_{\text{policy}}(\theta) =
\frac{\sum_i \sum_t m_{i,t}\|\hat{a}_{i,t} - a_{i,t}\|_2^2}
{\sum_i \sum_t m_{i,t}}
$$

Here m_{i,t} is 1 for real timesteps and 0 for padded timesteps. The
Perception Transformer uses the same action MSE objective, but predicts from a
single state and normalized RTG:

$$
\hat{a}_t = g_{\theta}(s_t, \hat{R}_t)
$$

For preference learning, each segment sigma contains a fixed-length sequence
of states, actions, and RTGs. Segment return is:

$$
G(\sigma) = \sum_{t \in \sigma} r_t
$$

The clean preference label for a left/right pair is:

$$
y =
\begin{cases}
0, & G(\sigma_{\text{left}}) > G(\sigma_{\text{right}}) \\
1, & \text{otherwise}
\end{cases}
$$

The preference model assigns one scalar score to each segment:

$$
u_{\text{left}} = h_{\phi}(\sigma_{\text{left}}), \qquad
u_{\text{right}} = h_{\phi}(\sigma_{\text{right}})
$$

The Bradley-Terry preference probability is:

$$
P_{\phi}(\text{left preferred}) =
\frac{\exp(u_{\text{left}})}
{\exp(u_{\text{left}}) + \exp(u_{\text{right}})}
$$

Equivalently, training uses the left and right segment scores as logits with
cross-entropy:

$$
\mathcal{L}_{\text{pref}}(\phi) =
-\log \operatorname{softmax}([u_{\text{left}}, u_{\text{right}}])_y
$$

## Training Methodology

Common policy training settings:

| Setting | Value |
| --- | ---: |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 |
| Scheduler | ReduceLROnPlateau |
| Loss | Action MSE |

Decision Transformer training:

1. Load Minari trajectories.
2. Compute return-to-go for every timestep.
3. Sample fixed-length sequence windows.
4. Build RTG, state, action, timestep, and mask tensors.
5. Predict actions for each state token.
6. Apply masked MSE over valid timesteps.

Perception Transformer training:

1. Load Minari trajectories.
2. Flatten trajectories into transition samples.
3. Pair each state/action sample with normalized RTG.
4. Predict the dataset action from the current state and RTG.
5. Apply MSE over continuous actions.

Preference model training:

1. Generate segment pairs from offline trajectories.
2. Split pairs into training and validation subsets.
3. Score left and right segments independently.
4. Train with cross-entropy against the preferred segment label.
5. Save the best validation-accuracy model.

## Benchmark Methodology

The benchmark trains both policy models on the Hopper simple, medium, and expert
splits.

Benchmark configuration:

| Setting | Value |
| --- | ---: |
| Environment | Hopper-v4 |
| Epochs | 10 |
| Batch size | 512 |
| Context length | 8 |
| Sampled windows/transitions per split | 10,000 |
| Evaluation episodes | 10 |
| Target RTG normalizer | 3000.0 |

This run is intentionally bounded so it can validate the complete pipeline
within a practical runtime. It should be interpreted as an integration benchmark,
not a final tuned D4RL score.

## Results

Latest recorded Hopper-v4 benchmark:

![D4RL Hopper benchmark comparison](d4rl_results/d4rl_comparison.png)

*Figure 1. Bounded Hopper benchmark summary. The left panel reports action-MSE
training curves, and the right panel reports live Hopper-v4 evaluation return
for Decision Transformer and Perception Transformer across simple, medium, and
expert dataset splits.*

| Split | Model | Avg Return | Std Return | Min / Max |
| --- | --- | ---: | ---: | ---: |
| Simple | Decision Transformer | 18.9 | 0.2 | 18.6 / 19.3 |
| Simple | Perception Transformer | 201.2 | 2.3 | 197.9 / 204.5 |
| Medium | Decision Transformer | 23.1 | 0.8 | 21.7 / 24.9 |
| Medium | Perception Transformer | 552.8 | 2.3 | 550.5 / 559.1 |
| Expert | Decision Transformer | 54.8 | 0.9 | 53.4 / 56.5 |
| Expert | Perception Transformer | 78.3 | 2.2 | 75.5 / 82.2 |

Recorded artifacts:

- d4rl_results/summary.csv
- d4rl_results/d4rl_comparison.png
- d4rl_results/model_decision_transformer_simple.pth
- d4rl_results/model_decision_transformer_medium.pth
- d4rl_results/model_decision_transformer_expert.pth
- d4rl_results/model_perception_transformer_simple.pth
- d4rl_results/model_perception_transformer_medium.pth
- d4rl_results/model_perception_transformer_expert.pth

The Perception Transformer performs better than the sequence Decision
Transformer in all three bounded Hopper settings. The best recorded bounded
return is the Perception Transformer on the medium split.

## Training Curves

The full benchmark saves per-split loss curves for both policy models. These
figures support two observations: all runs complete the supervised action
prediction objective, and low action MSE alone does not guarantee high live
rollout return.

| Split | Decision Transformer | Perception Transformer |
| --- | --- | --- |
| Simple | ![Decision Transformer simple loss](d4rl_results/loss_decision_transformer_simple.png) | ![Perception Transformer simple loss](d4rl_results/loss_perception_transformer_simple.png) |
| Medium | ![Decision Transformer medium loss](d4rl_results/loss_decision_transformer_medium.png) | ![Perception Transformer medium loss](d4rl_results/loss_perception_transformer_medium.png) |
| Expert | ![Decision Transformer expert loss](d4rl_results/loss_decision_transformer_expert.png) | ![Perception Transformer expert loss](d4rl_results/loss_perception_transformer_expert.png) |

Standalone training scripts also produce the following curves:

![Decision Transformer standalone training loss](plots/training_loss.png)

*Figure 2. Decision Transformer training loss from the single-model training
entry point.*

![Perception Transformer standalone training loss](plots/perception_transformer_training_loss.png)

*Figure 3. Perception Transformer training loss from the standalone Perception
Transformer entry point.*

## Preference Error Analysis

Preference labels and learned preference models can both be wrong. The error
analysis audits those errors by comparing:

- model accuracy against the training labels
- model accuracy against clean return-derived labels
- injected label flips
- model confidence
- return gaps for correct and incorrect predictions
- high-confidence incorrect predictions

For a preference pair with prediction yhat and label y, accuracy is:

$$
\text{accuracy} = \frac{1}{N}\sum_i \mathbf{1}[\hat{y}_i = y_i]
$$

The return gap used to rank ambiguous versus obvious preference comparisons is:

$$
\text{gap}_i =
\left|G(\sigma_{\text{left},i}) - G(\sigma_{\text{right},i})\right|
$$

A high-confidence wrong prediction is flagged when:

$$
\hat{y}_i \ne y_{\text{clean},i}
\quad \text{and} \quad
\max \operatorname{softmax}([u_{\text{left},i}, u_{\text{right},i}])
\ge \text{threshold}
$$

Flagged examples are saved as a preference error case table. The preference
training curve is generated after running preference-model training; it is not
included in the current checked artifacts unless that training stage has been
run locally.

These diagnostics matter because high-confidence preference mistakes can corrupt
downstream reward modeling or policy improvement. The current implementation
therefore keeps preference learning separate from policy updates until these
failure modes are measured.

## Deployment

The project includes two deployment paths: a rollout smoke test for live
environment evaluation and an HTTP action API for serving model predictions.

The API exposes:

- a health endpoint
- an action endpoint

The action endpoint accepts the current state, target RTG, and optional sequence
history. Supplying history is recommended for the Decision Transformer because it
was trained on trajectory windows.

## Limitations

- The recorded D4RL benchmark is bounded to 10,000 samples per split.
- Results are not averaged across random seeds.
- Hyperparameters are not exhaustively tuned.
- Walker2d support exists, but recorded Walker2d results are not included.
- Preference labels are return-derived proxy labels rather than human labels.
- The preference model is not yet used to relabel returns or improve the policy.
- The Decision Transformer underperforms the Perception Transformer in the
  current bounded benchmark.

## Next Steps

The next milestone should connect preference learning back into policy learning:

1. Use preference-model scores to re-rank or filter offline trajectory segments.
2. Relabel or reweight RTG targets using preference-model predictions.
3. Measure policy performance as injected preference noise increases.
4. Repeat Hopper experiments across multiple seeds and larger training budgets.
5. Add recorded Walker2d results using the same benchmark path.
6. Tune the Decision Transformer evaluator to maintain realistic sequence
   histories during rollout.

## Checkpoint Feedback & Task Checklist

We divided the remaining work among the team to make sure we addressed all the instructor feedback from the checkpoint. Everything is now complete:

- [x] **Dataset Migration:** Get the DT baseline working on the target D4RL benchmarks (Hopper) instead of the basic CartPole setup.
- [x] **Data Pipeline:** Build the preference-pair generation pipeline from the offline trajectories.
- [x] **Preference Model:** Implement and train the preference model on the generated pairs.
- [x] **Error Analysis:** Analyze what happens when the preference model makes mistakes and write a script to detect these errors.
- [x] **Evaluation:** Benchmark the models, plot the training curves, and compare results.
- [x] **Deployment:** Set up the FastAPI action service and rollout scripts.

## Conclusion

The repository now contains an end-to-end offline RL project: Minari/D4RL data
loading, return-conditioned transformer policies, live MuJoCo evaluation,
benchmark artifacts, preference-pair generation, preference-model training,
preference error analysis, and deployment utilities. The current results show
that the simpler Perception Transformer is stronger under the bounded benchmark,
while the preference-learning pipeline is ready for the next phase of using
learned preferences to guide policy improvement.

## References

1. Chen et al. *Decision Transformer: Reinforcement Learning via Sequence
   Modeling.* NeurIPS, 2021.
2. Fu et al. *D4RL: Datasets for Deep Data-Driven Reinforcement Learning.*
   arXiv:2004.07219, 2020.
3. Christiano et al. *Deep Reinforcement Learning from Human Preferences.*
   NeurIPS, 2017.
4. Farama Foundation. Minari dataset standard and Gymnasium MuJoCo
   environments.
