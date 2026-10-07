import torch


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, map_location=None):
    state = torch.load(path, map_location=map_location)
    # older DT checkpoints were saved before state normalization was added
    if "state_mean" in model.state_dict() and "state_mean" not in state:
        state["state_mean"] = model.state_mean
        state["state_std"] = model.state_std
    model.load_state_dict(state)
    model.eval()
    return model
