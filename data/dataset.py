from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def compute_rtg(rewards, gamma=1.0):
    rtg = np.zeros_like(rewards, dtype=np.float32)
    cum = 0.0
    for i in reversed(range(len(rewards))):
        cum = float(rewards[i]) + gamma * cum
        rtg[i] = cum
    return rtg


@dataclass
class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    rtg: np.ndarray
    return_sum: float


class MinariTrajectoryBuffer:
    """Keeps full offline trajectories for DT training and preference labels."""

    def __init__(self, minari_dataset, target_rtg=3000.0, gamma=1.0):
        self.target_rtg = float(target_rtg)
        self.gamma = float(gamma)
        self.trajectories: List[Trajectory] = []

        for ep in minari_dataset.iterate_episodes():
            obs = np.asarray(ep.observations[:-1], dtype=np.float32)
            act = np.asarray(ep.actions, dtype=np.float32)
            rew = np.asarray(ep.rewards, dtype=np.float32)
            length = min(len(obs), len(act), len(rew))
            if length == 0:
                continue

            obs = obs[:length]
            act = act[:length]
            rew = rew[:length]
            rtg = compute_rtg(rew, gamma=self.gamma)
            self.trajectories.append(
                Trajectory(
                    observations=obs,
                    actions=act,
                    rewards=rew,
                    rtg=rtg,
                    return_sum=float(rew.sum()),
                )
            )

        if not self.trajectories:
            raise ValueError("No valid episodes found in the Minari dataset.")

        self.state_dim = int(self.trajectories[0].observations.shape[-1])
        self.act_dim = int(self.trajectories[0].actions.shape[-1])
        self.num_steps = int(sum(len(t.rewards) for t in self.trajectories))
        returns = np.asarray([t.return_sum for t in self.trajectories], dtype=np.float32)
        self.return_stats = {
            "min": float(returns.min()),
            "mean": float(returns.mean()),
            "max": float(returns.max()),
        }

    def __len__(self):
        return len(self.trajectories)

    def iter_trajectories(self) -> Iterable[Trajectory]:
        return iter(self.trajectories)


class TrajectoryDataset(Dataset):
    """One-step supervised dataset kept for compatibility with earlier scripts."""

    def __init__(self, source, target_rtg=3000.0, max_samples: Optional[int] = None):
        buffer = (
            source
            if isinstance(source, MinariTrajectoryBuffer)
            else MinariTrajectoryBuffer(source, target_rtg=target_rtg)
        )
        self.states = torch.from_numpy(
            np.concatenate([t.observations for t in buffer.iter_trajectories()])
        ).float()
        self.actions = torch.from_numpy(
            np.concatenate([t.actions for t in buffer.iter_trajectories()])
        ).float()
        self.rtgs = torch.from_numpy(
            np.concatenate([t.rtg for t in buffer.iter_trajectories()])
            / (target_rtg + 1e-8)
        ).float()

        if max_samples is not None and len(self.states) > max_samples:
            generator = torch.Generator().manual_seed(0)
            keep = torch.randperm(len(self.states), generator=generator)[:max_samples]
            self.states = self.states[keep]
            self.actions = self.actions[keep]
            self.rtgs = self.rtgs[keep]

        print(f"Dataset loaded with {len(self.states)} transition samples")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        return self.states[i], self.actions[i], self.rtgs[i]


class SequenceTrajectoryDataset(Dataset):
    """Fixed-length trajectory windows for a causal Decision Transformer."""

    def __init__(
        self,
        source,
        context_len=20,
        target_rtg=3000.0,
        stride=1,
        max_windows: Optional[int] = None,
    ):
        self.buffer = (
            source
            if isinstance(source, MinariTrajectoryBuffer)
            else MinariTrajectoryBuffer(source, target_rtg=target_rtg)
        )
        self.context_len = int(context_len)
        self.target_rtg = float(target_rtg)
        self.windows: List[Tuple[int, int]] = []

        for traj_idx, traj in enumerate(self.buffer.iter_trajectories()):
            length = len(traj.rewards)
            if length <= self.context_len:
                self.windows.append((traj_idx, 0))
                continue
            for start in range(0, length - self.context_len + 1, stride):
                self.windows.append((traj_idx, start))

        if max_windows is not None and len(self.windows) > max_windows:
            rng = np.random.default_rng(0)
            keep = rng.choice(len(self.windows), size=max_windows, replace=False)
            self.windows = [self.windows[int(i)] for i in sorted(keep)]

        print(
            "Sequence dataset loaded with "
            f"{len(self.windows)} windows, context_len={self.context_len}"
        )

    @property
    def state_dim(self):
        return self.buffer.state_dim

    @property
    def act_dim(self):
        return self.buffer.act_dim

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        traj_idx, start = self.windows[i]
        traj = self.buffer.trajectories[traj_idx]
        end = min(start + self.context_len, len(traj.rewards))
        actual = end - start
        pad = self.context_len - actual

        states = np.zeros((self.context_len, self.state_dim), dtype=np.float32)
        actions = np.zeros((self.context_len, self.act_dim), dtype=np.float32)
        rtgs = np.zeros((self.context_len,), dtype=np.float32)
        timesteps = np.zeros((self.context_len,), dtype=np.int64)
        mask = np.zeros((self.context_len,), dtype=np.float32)

        states[pad:] = traj.observations[start:end]
        actions[pad:] = traj.actions[start:end]
        rtgs[pad:] = traj.rtg[start:end] / (self.target_rtg + 1e-8)
        timesteps[pad:] = np.arange(start, end, dtype=np.int64)
        mask[pad:] = 1.0

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(rtgs),
            torch.from_numpy(timesteps),
            torch.from_numpy(mask),
        )


class WeightedSequenceTrajectoryDataset(Dataset):
    """Sequence windows with an extra scalar weight per sample."""

    def __init__(self, base_dataset: SequenceTrajectoryDataset, weights: np.ndarray):
        if len(base_dataset) != len(weights):
            raise ValueError("weights must match the number of sequence windows")
        self.base_dataset = base_dataset
        self.weights = np.asarray(weights, dtype=np.float32)

    @property
    def buffer(self):
        return self.base_dataset.buffer

    @property
    def state_dim(self):
        return self.base_dataset.state_dim

    @property
    def act_dim(self):
        return self.base_dataset.act_dim

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, i):
        sample = self.base_dataset[i]
        return sample + (torch.tensor(self.weights[i], dtype=torch.float32),)


class PreferencePairDataset(Dataset):
    def __init__(self, pairs: Dict[str, np.ndarray]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs["labels"])

    def __getitem__(self, i):
        input_keys = (
            "left_states",
            "left_actions",
            "left_rtgs",
            "right_states",
            "right_actions",
            "right_rtgs",
        )
        return {
            key: torch.from_numpy(self.pairs[key][i]).float()
            for key in input_keys
        } | {"labels": torch.tensor(self.pairs["labels"][i], dtype=torch.long)}


def _segment(traj: Trajectory, start: int, length: int, target_rtg: float):
    end = start + length
    return {
        "states": traj.observations[start:end],
        "actions": traj.actions[start:end],
        "rtgs": traj.rtg[start:end] / (target_rtg + 1e-8),
        "return": float(traj.rewards[start:end].sum()),
    }


def build_preference_pairs(
    source,
    num_pairs=10000,
    segment_len=20,
    target_rtg=3000.0,
    min_return_gap=1.0,
    label_noise=0.0,
    seed=0,
):
    """Generate preference pairs by comparing segment returns from offline data."""

    buffer = (
        source
        if isinstance(source, MinariTrajectoryBuffer)
        else MinariTrajectoryBuffer(source, target_rtg=target_rtg)
    )
    rng = np.random.default_rng(seed)
    candidates = [
        (idx, len(traj.rewards) - segment_len)
        for idx, traj in enumerate(buffer.trajectories)
        if len(traj.rewards) >= segment_len
    ]
    if not candidates:
        raise ValueError("No trajectories are long enough for preference segments.")

    left_states, left_actions, left_rtgs = [], [], []
    right_states, right_actions, right_rtgs = [], [], []
    labels, clean_labels, gaps = [], [], []
    attempts = 0
    max_attempts = max(num_pairs * 50, 1000)

    while len(labels) < num_pairs and attempts < max_attempts:
        attempts += 1
        li = int(rng.integers(len(candidates)))
        ri = int(rng.integers(len(candidates)))
        left_traj_idx, left_max_start = candidates[li]
        right_traj_idx, right_max_start = candidates[ri]
        left_start = int(rng.integers(left_max_start + 1))
        right_start = int(rng.integers(right_max_start + 1))

        left = _segment(
            buffer.trajectories[left_traj_idx],
            left_start,
            segment_len,
            target_rtg,
        )
        right = _segment(
            buffer.trajectories[right_traj_idx],
            right_start,
            segment_len,
            target_rtg,
        )
        gap = left["return"] - right["return"]
        if abs(gap) < min_return_gap:
            continue

        clean = 0 if gap > 0 else 1
        label = clean
        if rng.random() < label_noise:
            label = 1 - label

        left_states.append(left["states"])
        left_actions.append(left["actions"])
        left_rtgs.append(left["rtgs"])
        right_states.append(right["states"])
        right_actions.append(right["actions"])
        right_rtgs.append(right["rtgs"])
        labels.append(label)
        clean_labels.append(clean)
        gaps.append(abs(gap))

    if len(labels) < num_pairs:
        print(f"Generated {len(labels)} pairs after {attempts} attempts.")

    return {
        "left_states": np.asarray(left_states, dtype=np.float32),
        "left_actions": np.asarray(left_actions, dtype=np.float32),
        "left_rtgs": np.asarray(left_rtgs, dtype=np.float32),
        "right_states": np.asarray(right_states, dtype=np.float32),
        "right_actions": np.asarray(right_actions, dtype=np.float32),
        "right_rtgs": np.asarray(right_rtgs, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "clean_labels": np.asarray(clean_labels, dtype=np.int64),
        "return_gaps": np.asarray(gaps, dtype=np.float32),
    }


def save_preference_pairs(pairs: Dict[str, np.ndarray], path: str):
    np.savez_compressed(path, **pairs)


def load_preference_pairs(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
