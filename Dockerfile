# ──────────────────────────────────────────────────────────────────────────────
# Base: CUDA 12.4 + cuDNN 9 (matches your RTX 4060 Laptop + driver 550.54.14)
# Ubuntu 22.04 LTS for broadest package compatibility
# ──────────────────────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    # OpenCV runtime libs (headless — no display needed in container)
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Camera / V4L2 support
    v4l-utils \
    # Misc
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3       1

# ── Python dependencies (slim — only what main.py actually imports) ───────────
# Torch with CUDA 12.4 wheel index
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    transformers==4.57.6 \
    accelerate==1.12.0 \
    huggingface-hub==0.36.2 \
    safetensors==0.7.0 \
    tokenizers==0.22.2 \
    qwen-vl-utils==0.0.14 \
    opencv-python-headless==4.13.0.92 \
    Pillow==12.1.0

# ── App ───────────────────────────────────────────────────────────────────────
WORKDIR /app
COPY main.py utils.py ./

# ── HuggingFace cache — mount your host cache at runtime to avoid re-download
#    docker run -v ~/.cache/huggingface:/root/.cache/huggingface ...
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/hub

# Temp dir for frame files
ENV TMPDIR=/tmp

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["python", "main.py"]