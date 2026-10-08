# Offline RL Decision Transformer with Preference Learning

**Deep Learning and Reinforcement Learning — Team Gradient Gone Wild**

**Members:** Ravi Rajaram Singh, Hemanth Phani Srinivas Chilamkurthy

We built an offline RL pipeline for MuJoCo continuous control. The main idea was to train a Decision Transformer on Hopper trajectories and then add a preference learning pipeline to study what happens when preference labels are noisy or wrong.

## Highlights

- Trained return-conditioned transformer policies purely from fixed offline data, with no online environment interaction during training.
- Benchmarked three models on the D4RL Hopper simple, medium and expert splits.
- Fixed three bugs in the Decision Transformer pipeline (no state normalization, action history shifted by one step at evaluation, timesteps reset each window), which raised its Hopper return 6-20x on every split (24.0 to 476.1 on medium).
- Preference weighting helped on simple and expert and slightly hurt on medium.
- Served as a FastAPI inference API in a Docker image that GitHub Actions builds, smoke-tests and publishes on every change, plus a local Gradio demo with a live Hopper rollout video.

## Demo

![Decision Transformer vs Perception Transformer on Hopper](docs/hopper-comparison.gif)

Same seed (0), same target return, both trained on the D4RL Hopper medium data. The camera follows the hopper.

| | Decision Transformer | Perception Transformer |
| --- | --- | --- |
| Parameters | 726K | 1.9M |
| Input | full history (last 8 steps) | current state only |
| Steps before falling | 160 | 175 |
| Return | 480.5 | 558.2 |

On 8 seeds (0 to 7) the Decision Transformer scored 453 to 488 and the Perception Transformer 547 to 558, so on this checkpoint the Perception Transformer is ahead. Both models still fall before the episode limit. The full benchmark is in the table below.

## Benchmark

```bash
EPOCHS=10 BATCH_SIZE=512 CONTEXT_LEN=8 N_EVAL=10 MAX_WINDOWS=10000 python3 d4rl_compare_three.py
```

Average return over 10 evaluation episodes on Hopper-v4:

| Split | Decision Transformer | DT + Preference | Perception Transformer |
| --- | ---: | ---: | ---: |
| Simple | 834.9 | **861.0** | 544.6 |
| Medium | 476.1 | 442.4 | **552.3** |
| Expert | 429.7 | **571.4** | 47.7 |

DT + Preference was best on simple and expert, and the Perception Transformer was best on medium. Preference weighting helped the DT on two of three splits (+26 on simple, +142 on expert) and hurt it on medium (-34). The simple split varies a lot between episodes (std above 200 for both DT models), so its ranking is not reliable.

These runs use short training (10 epochs, 10,000 windows), so treat the numbers as a comparison between models rather than final scores. Well-tuned offline RL methods reach a few thousand on Hopper. Raw numbers are in `d4rl_three_way_results/summary.csv`.

### Bugs we fixed

The first version of this benchmark scored the Decision Transformer at 60.4 / 24.0 / 68.6 (the numbers in `REPORT.md`), meaning the hopper fell over within about 20 steps. Three problems caused it:

1. **No state normalization.** Hopper state values have very different scales. The model now stores the dataset mean and std with its weights and normalizes inside `forward`, so training, evaluation, the API and the demo all use the same numbers.
2. **Action history off by one at evaluation.** Evaluation put a zero action at the front of the history, so each past action sat next to the wrong state. In training, action `t` always sits next to state `t`.
3. **Timesteps restarted at 0.** Evaluation numbered every context window from 0, while training used the real episode step.

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

### Gradio demo

Launch using Gradio:

```bash
MODEL_CHECKPOINT=saved_models/decision_transformer_d4rl.pth python3 gradio_app.py
```

Then open `http://127.0.0.1:8000`. Rendering needs MuJoCo with a display (or `MUJOCO_GL=osmesa` on a headless machine). Tab 1 lets you test the action predictor, Tab 2 runs a live Hopper episode and records a video.

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
