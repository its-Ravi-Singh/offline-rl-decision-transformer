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

        # saved with the weights so inference uses the same normalization as training
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

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

    def set_state_stats(self, mean, std):
        self.state_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.state_std.copy_(torch.as_tensor(std, dtype=torch.float32))

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
        states = (states - self.state_mean) / self.state_std

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

    def act(self, state, target_rtg=1.0, state_history=None, action_history=None, rtg_history=None, timestep=None):
        self.eval()
        device = next(self.parameters()).device

        if state_history is None:
            state_history = [state]
        if action_history is None:
            action_history = []
        if rtg_history is None or len(rtg_history) != len(state_history):
            rtg_history = [target_rtg] * len(state_history)
        if timestep is None:
            timestep = len(state_history) - 1

        k = min(len(state_history), self.context_len)
        states = np.asarray(state_history[-k:], dtype=np.float32)
        rtgs = np.asarray(rtg_history[-k:], dtype=np.float32)

        # action i belongs to state i, the last one is the action we are predicting
        actions = np.zeros((k, self.act_dim), dtype=np.float32)
        past = list(action_history)[-(k - 1):] if k > 1 else []
        if len(past) > 0:
            actions[k - 1 - len(past):k - 1] = np.asarray(past, dtype=np.float32)

        timesteps = np.arange(timestep - k + 1, timestep + 1).clip(min=0)

        with torch.no_grad():
            s = torch.tensor(states, device=device).unsqueeze(0)
            a = torch.tensor(actions, device=device).unsqueeze(0)
            r = torch.tensor(rtgs, device=device).unsqueeze(0)
            t = torch.tensor(timesteps, device=device).unsqueeze(0)
            pred = self.forward(s, actions=a, rtgs=r, timesteps=t)
        return pred[0, -1].cpu().numpy()

PolicyNet = DecisionTransformer
