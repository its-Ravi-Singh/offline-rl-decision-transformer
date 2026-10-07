# Gradio demo with live Hopper videos, for a Hugging Face Docker Space.
# MuJoCo renders with OSMesa (software OpenGL) since Spaces have no display.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 libgl1 libglfw3 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    MPLBACKEND=Agg \
    PORT=7860

WORKDIR $HOME/app

COPY --chown=user space/requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user gradio_app.py .
COPY --chown=user models/ models/
COPY --chown=user utils/ utils/
COPY --chown=user saved_models/ saved_models/

EXPOSE 7860
CMD ["python", "gradio_app.py"]
