import torch
import torch.nn as nn
import numpy as np


class DecisionTransformer(nn.Module):
    """Causal sequence model for offline continuous-control action prediction."""

    def __init__(
        self,
        state_dim=11,
        act_dim=3,
        hidden_dim=128,
        context_len=20,
        n_layers=3,
        n_heads=4,
        dropout=0.1,
        max_timestep=1000,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.context_len = context_len

        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(act_dim, hidden_dim)
        self.rtg_embed = nn.Linear(1, hidden_dim)
        self.timestep_embed = nn.Embedding(max_timestep, hidden_dim)
        self.type_embed = nn.Embedding(3, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Sequential(nn.Linear(hidden_dim, act_dim), nn.Tanh())

    def _causal_mask(self, length, device):
        return torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, states, actions=None, rtgs=None, timesteps=None, attention_mask=None):
        one_step = states.dim() == 2
        if one_step:
            states = states.unsqueeze(1)
            if actions is not None and actions.dim() == 2:
                actions = actions.unsqueeze(1)
            if rtgs is not None and rtgs.dim() == 1:
                rtgs = rtgs[:, None]

        if rtgs is None:
            raise ValueError("rtgs must be provided")
        if rtgs.dim() == 2:
            rtgs = rtgs.unsqueeze(-1)
        if actions is None:
            actions = torch.zeros(
                states.shape[0],
                states.shape[1],
                self.act_dim,
                dtype=states.dtype,
                device=states.device,
            )
        if timesteps is None:
            timesteps = torch.arange(states.shape[1], device=states.device)
            timesteps = timesteps.unsqueeze(0).repeat(states.shape[0], 1)
        timesteps = timesteps.clamp(max=self.timestep_embed.num_embeddings - 1)

        batch, seq_len = states.shape[:2]
        time_emb = self.timestep_embed(timesteps)
        rtg_tokens = self.rtg_embed(rtgs) + time_emb + self.type_embed.weight[0]
        state_tokens = self.state_embed(states) + time_emb + self.type_embed.weight[1]
        action_tokens = self.action_embed(actions) + time_emb + self.type_embed.weight[2]

        tokens = torch.stack((rtg_tokens, state_tokens, action_tokens), dim=2)
        tokens = tokens.reshape(batch, seq_len * 3, -1)
        causal_mask = self._causal_mask(tokens.shape[1], tokens.device)

        src_key_padding_mask = None
        if attention_mask is not None:
            token_mask = attention_mask.unsqueeze(-1).repeat(1, 1, 3)
            src_key_padding_mask = token_mask.reshape(batch, seq_len * 3) == 0

        hidden = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        hidden = self.norm(hidden)
        state_hidden = hidden[:, 1::3]
        pred_actions = self.action_head(state_hidden)
        return pred_actions[:, -1] if one_step else pred_actions

    def act(self, state, target_rtg=1.0, state_history=None, action_history=None, rtg_history=None):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if state_history is None:
                states = torch.as_tensor(
                    state, dtype=torch.float32, device=device
                ).view(1, 1, -1)
                actions = torch.zeros(1, 1, self.act_dim, dtype=torch.float32, device=device)
                rtgs = torch.tensor([[[target_rtg]]], dtype=torch.float32, device=device)
            else:
                states_np = list(state_history)[-self.context_len :]
                states = torch.as_tensor(
                    np.asarray(states_np, dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                if action_history is None:
                    actions = torch.zeros(
                        1, len(states_np), self.act_dim, dtype=torch.float32, device=device
                    )
                else:
                    actions_np = list(action_history)[-self.context_len :]
                    if len(actions_np) < len(states_np):
                        pad = [torch.zeros(self.act_dim).numpy()] * (len(states_np) - len(actions_np))
                        actions_np = pad + actions_np
                    actions = torch.as_tensor(
                        np.asarray(actions_np, dtype=np.float32),
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                if rtg_history is None:
                    rtgs = torch.full(
                        (1, len(states_np), 1),
                        float(target_rtg),
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    rtgs = torch.as_tensor(
                        list(rtg_history)[-self.context_len :],
                        dtype=torch.float32,
                        device=device,
                    ).view(1, -1, 1)

            action = self.forward(states, actions=actions, rtgs=rtgs)
        return action[:, -1].squeeze(0).cpu().numpy()


PolicyNet = DecisionTransformer
