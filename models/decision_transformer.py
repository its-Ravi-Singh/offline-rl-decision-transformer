import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, state_dim, act_dim=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)  # 2 outputs for classification
        )

    def forward(self, x):
        return self.net(x)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(state)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1)
            return action.item()
