import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    print(f"Training on {device}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    losses = []
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for ep in range(epochs):
        model.train()
        running = 0.0

        for batch in dataloader:
            if len(batch) == 5:
                states, actions, rtgs, timesteps, mask = batch
                states = states.to(device)
                actions = actions.to(device)
                rtgs = rtgs.to(device)
                timesteps = timesteps.to(device)
                mask = mask.to(device)
                pred = model(
                    states,
                    actions=actions,
                    rtgs=rtgs,
                    timesteps=timesteps,
                    attention_mask=mask,
                )
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

        improved = avg < best_loss - min_delta
        if improved:
            best_loss = avg
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"Epoch {ep + 1}/{epochs} | Loss: {avg:.5f}")

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                "Early stopping triggered at "
                f"epoch {ep + 1}; best loss: {best_loss:.5f}"
            )
            break

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

    return losses
