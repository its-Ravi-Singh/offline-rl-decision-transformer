import torch
import torch.nn as nn


class PerceptionTransformer(nn.Module):
    def __init__(
        self,
        state_dim=11,
        act_dim=3,
        hidden_dim=256,
        num_latents=8,
        num_heads=4,
        depth=2,
        dropout=0.1,
    ):
        super().__init__()

        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.rtg_embed = nn.Linear(1, hidden_dim)
        self.type_embed = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )

    def forward(self, states, rtgs):
        if rtgs.dim() == 1:
            rtgs = rtgs.unsqueeze(-1)

        state_tokens = self.state_embed(states).unsqueeze(1)
        rtg_tokens = self.rtg_embed(rtgs).unsqueeze(1)
        inputs = torch.cat([state_tokens, rtg_tokens], dim=1) + self.type_embed

        latents = self.latents.expand(states.shape[0], -1, -1)
        attended, _ = self.cross_attn(latents, inputs, inputs, need_weights=False)
        latents = self.cross_norm(latents + attended)
        latents = self.encoder(latents)

        pooled = latents.mean(dim=1)
        return self.head(pooled)

    def act(self, state, target_rtg=1.0):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            r = torch.tensor([[target_rtg]], dtype=torch.float32, device=device)
            action = self.forward(s, r)
        return action.squeeze(0).cpu().numpy()


PolicyNet = PerceptionTransformer
