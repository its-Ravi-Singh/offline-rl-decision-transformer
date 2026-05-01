from __future__ import annotations

import os

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, model_validator

from models.decision_transformer import DecisionTransformer
from utils.check import load_model


CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT",
    "saved_models/decision_transformer_d4rl.pth",
)
TARGET_RTG_MAX = float(os.environ.get("TARGET_RTG_MAX", "3000.0"))
STATE_DIM = int(os.environ.get("STATE_DIM", "11"))
ACT_DIM = int(os.environ.get("ACT_DIM", "3"))
CONTEXT_LEN = int(os.environ.get("CONTEXT_LEN", "20"))


from typing import Optional, List

class ActionRequest(BaseModel):
    state: List[float] = Field(..., min_length=STATE_DIM, max_length=STATE_DIM)
    target_rtg: float = 1.0
    state_history: Optional[List[List[float]]] = None
    action_history: Optional[List[List[float]]] = None
    rtg_history: Optional[List[float]] = None

    @model_validator(mode="after")
    def validate_histories(self):
        if self.state_history is not None:
            for state in self.state_history:
                if len(state) != STATE_DIM:
                    raise ValueError(f"state_history rows must have length {STATE_DIM}")
        if self.action_history is not None:
            for action in self.action_history:
                if len(action) != ACT_DIM:
                    raise ValueError(f"action_history rows must have length {ACT_DIM}")
        return self


class ActionResponse(BaseModel):
    action: List[float]
    clipped_action: List[float]
    target_rtg: float


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_policy():
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT}")
    device = get_device()
    model = DecisionTransformer(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        context_len=CONTEXT_LEN,
    )
    load_model(model, CHECKPOINT, map_location=device)
    model.to(device)
    model.eval()
    return model, device


app = FastAPI(title="Decision Transformer Hopper Policy")
model, device = load_policy()

app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")

@app.get("/")
def root():
    return RedirectResponse(url="/ui")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoint": CHECKPOINT,
        "device": str(device),
        "target_rtg_max": TARGET_RTG_MAX,
        "state_dim": STATE_DIM,
        "act_dim": ACT_DIM,
        "context_len": CONTEXT_LEN,
    }


@app.post("/act", response_model=ActionResponse)
def act(payload: ActionRequest):
    try:
        state = np.asarray(payload.state, dtype=np.float32)
        action = model.act(
            state,
            target_rtg=payload.target_rtg,
            state_history=payload.state_history,
            action_history=payload.action_history,
            rtg_history=payload.rtg_history,
        )
        clipped = np.clip(action, -1.0, 1.0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "action": action.astype(float).tolist(),
        "clipped_action": clipped.astype(float).tolist(),
        "target_rtg": payload.target_rtg,
    }
