# Inference API for the trained Decision Transformer (Hopper).
# Only what serve.py needs: CPU PyTorch, FastAPI and the checkpoint.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_CHECKPOINT=saved_models/decision_transformer_d4rl.pth \
    PORT=8000

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt

COPY serve.py .
COPY models/ models/
COPY utils/ utils/
COPY static/ static/
COPY saved_models/decision_transformer_d4rl.pth saved_models/

EXPOSE 8000

# PORT lets hosts like Cloud Run, Render or Railway pick the port.
CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT}"]
