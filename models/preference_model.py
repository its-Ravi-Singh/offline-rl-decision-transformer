import torch
import torch.nn as nn


class PreferenceModel(nn.Module):

    def __init__(self, state_dim=11, act_dim=3, hidden_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(state_dim + act_dim + 1, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states, actions, rtgs, mask=None):
        if rtgs.dim() == 2:
            rtgs = rtgs.unsqueeze(-1)
        x = torch.cat([states, actions, rtgs], dim=-1)
        x = self.input_proj(x)
        key_padding_mask = None if mask is None else mask == 0
        hidden = self.encoder(x, src_key_padding_mask=key_padding_mask)
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = mask.unsqueeze(-1).clamp_min(0.0)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.score_head(pooled).squeeze(-1)


def preference_logits(model, batch):
    left = model(batch["left_states"], batch["left_actions"], batch["left_rtgs"])
    right = model(batch["right_states"], batch["right_actions"], batch["right_rtgs"])
    return torch.stack([left, right], dim=-1)
