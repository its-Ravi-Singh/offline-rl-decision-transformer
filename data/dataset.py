import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.states = []
        self.actions = []

        for s, a in data:
            self.states.extend(s)
            self.actions.extend(a)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.states[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.long)  # IMPORTANT: long for CE loss
        )