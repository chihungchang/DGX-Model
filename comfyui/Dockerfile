FROM nvcr.io/nvidia/pytorch:25.11-py3@sha256:4a85d8cf6fb3a943280960b8948cf4e9b6eca77b4414c68c9b2c7bb863f79b70

ARG COMFYUI_COMMIT=34d0629452cac83dc20aa3d84e45c9b60d9e36b3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/comfy-venv/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/Comfy-Org/ComfyUI.git comfyui \
    && cd comfyui \
    && git checkout --detach "${COMFYUI_COMMIT}" \
    && git rev-parse HEAD

RUN python -m venv --system-site-packages /opt/comfy-venv \
    && grep -v -E '^(torch|torchvision|torchaudio)$' /opt/comfyui/requirements.txt > /tmp/comfy-requirements.txt \
    && /opt/comfy-venv/bin/pip install --upgrade pip \
    && PIP_CONSTRAINT= /opt/comfy-venv/bin/pip install -r /tmp/comfy-requirements.txt \
    && rm -f /tmp/comfy-requirements.txt \
    && chown -R ubuntu:ubuntu /opt/comfyui /opt/comfy-venv

RUN /opt/comfy-venv/bin/pip install --no-deps torchaudio==2.9.1 \
    && /opt/comfy-venv/bin/python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)" \
    && chown -R ubuntu:ubuntu /opt/comfy-venv

RUN PIP_CONSTRAINT= /opt/comfy-venv/bin/pip install \
        addict==2.4.0 \
        albumentations==2.0.8 \
        ftfy==6.3.1 \
        fvcore==0.1.5.post20221221 \
        omegaconf==2.3.1 \
        opencv-python-headless==5.0.0.93 \
        scikit-image==0.26.0 \
        trimesh==4.12.2 \
        yacs==0.1.8 \
    && chown -R ubuntu:ubuntu /opt/comfy-venv

USER ubuntu
WORKDIR /opt/comfyui

EXPOSE 8188

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=3)"

ENTRYPOINT ["/opt/comfy-venv/bin/python", "main.py"]
CMD ["--listen", "0.0.0.0", "--port", "8188", "--base-directory", "/data", "--database-url", "sqlite:////data/user/comfyui.db", "--disable-auto-launch", "--preview-method", "auto"]
