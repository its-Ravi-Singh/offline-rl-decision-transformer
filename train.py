# CSE 676 Final Project — Gradient Gone Wild
# Ravi Rajaram Singh, Hemanth Phani Srinivas Chilamkurthy
#
# shared training loop, used by main.py and d4rl_compare.py
# works for both the decision transformer and perception transformer

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # for mac m1/m2
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# masked mse — ignores padded timesteps
# had to write this manually bc pytorch doesnt have a built in one
def _masked_mse(pred, target, mask=None, sample_weight=None):
    loss = (pred - target).pow(2).mean(dim=-1)
    if mask is None:
        loss = loss.mean(dim=-1) if loss.dim() > 1 else loss
    else:
        loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    if sample_weight is not None:
        loss = loss * sample_weight
        return loss.sum() / sample_weight.sum().clamp_min(1.0)
    return loss.mean()


def train(
    model,
    dataloader,
    epochs=50,
    plot_path="plots/training_loss.png",
    plot_title=None,
    early_stopping_patience=None,
    min_delta=1e-4,
    log_every=10,
):
    device = get_device()
    model.to(device)
    print(f"training on {device}")

    loss_fn = nn.MSELoss()

    # tried SGD first but adamw converged way faster
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    losses = []
    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        model.train()
        running = 0.0

        for batch in dataloader:
            # decision transformer batch has 5 items, perception has 3
            if len(batch) == 5:
                states, actions, rtgs, timesteps, mask = batch
                states = states.to(device)
                actions = actions.to(device)
                rtgs = rtgs.to(device)
                timesteps = timesteps.to(device)
                mask = mask.to(device)
                pred = model(states, actions=actions, rtgs=rtgs, timesteps=timesteps, attention_mask=mask)
                loss = _masked_mse(pred, actions, mask)
            elif len(batch) == 6:
                states, actions, rtgs, timesteps, mask, sample_weight = batch
                states = states.to(device)
                actions = actions.to(device)
                rtgs = rtgs.to(device)
                timesteps = timesteps.to(device)
                mask = mask.to(device)
                sample_weight = sample_weight.to(device)
                pred = model(states, actions=actions, rtgs=rtgs, timesteps=timesteps, attention_mask=mask)
                loss = _masked_mse(pred, actions, mask, sample_weight=sample_weight)
            else:
                states, actions, rtgs = batch
                states = states.to(device)
                actions = actions.to(device)
                rtgs = rtgs.to(device)
                pred = model(states, rtgs=rtgs)
                loss = loss_fn(pred, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

        avg = running / max(len(dataloader), 1)
        losses.append(avg)
        scheduler.step(avg)

        if avg < best_loss - min_delta:
            best_loss = avg
            # just save the state dict, load it back at the end
            import copy
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"epoch {ep + 1}/{epochs} | loss: {avg:.5f}")

        if early_stopping_patience is not None and no_improve >= early_stopping_patience:
            print(f"early stopping at epoch {ep + 1}, best loss was {best_loss:.5f}")
            break

    # restore best weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Action MSE")
    if plot_title is None:
        plot_title = f"{model.__class__.__name__} Training Loss"
    plt.title(plot_title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"saved loss plot to {plot_path}")

    return losses
