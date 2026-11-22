# Wyoming TTS server for the fine-tuned GLaDOS voice, with HIP-graph decoding (faster-qwen3-tts) on ROCm.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 MIOPEN_FIND_MODE=FAST
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 sox \
 && rm -rf /var/lib/apt/lists/*

# Same torch line as train/Dockerfile, so both images share this layer.
RUN pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2

# transformers newer than 5.15.1 breaks qwen-tts-hf (Qwen3TTSTokenizerV2Encoder.upsample has no setter).
RUN pip install faster-qwen3-tts==0.5.4 transformers==5.15.1 wyoming==1.11.0 \
 && python -c "import torch, wyoming, faster_qwen3_tts, transformers; assert torch.version.hip, 'not a ROCm build'; print(torch.__version__, transformers.__version__)"

# Jobs may run as any uid (compose RUN_AS); torch and HF need a user name and a writable home.
ENV USER=glados HOME=/tmp
LABEL org.opencontainers.image.source=https://github.com/MyrikLD/glados-tts
COPY server.py /app/server.py
EXPOSE 10200
CMD ["python", "/app/server.py"]
