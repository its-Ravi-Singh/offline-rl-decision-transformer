# CSE 676 Final Project — Gradient Gone Wild
# Hemanth Phani Srinivas Chilamkurthy, Ravi Rajaram Singh
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
def _masked_mse(pred, target, mask=None):
    loss = (pred - target).pow(2).mean(dim=-1)
    if mask is None:
        return loss.mean()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


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
            # save a copy of the weights at the best point
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
