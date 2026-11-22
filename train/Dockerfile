# Fine-tuning image: data prep (CPU codec encoding) and training/eval on an AMD GPU via ROCm.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/data/hf MIOPEN_FIND_MODE=FAST
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 sox git \
 && rm -rf /var/lib/apt/lists/*

# torchaudio stopped at 2.11 and its binary loads only with the torch it was built for, so torch
# stays at 2.11 too.
RUN pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2

# qwen-tts-hf is the Transformers 5 build of qwen-tts (same `qwen_tts` module); official qwen-tts
# still pins Transformers 4.57. Newer Transformers than 5.15.1 break it.
RUN pip install qwen-tts-hf==0.1.1.post1 transformers==5.15.1 \
 && python -c "import torch, torchaudio, qwen_tts, transformers; assert torch.version.hip, 'not a ROCm build'; print(torch.__version__, torchaudio.__version__, transformers.__version__)"

# train.py patches sft_12hz.py by exact text, so the upstream commit is pinned.
ARG QWEN3_TTS_COMMIT=022e286b98fbec7e1e916cb940cdf532cd9f488e
RUN git clone https://github.com/QwenLM/Qwen3-TTS.git /opt/Qwen3-TTS \
 && git -C /opt/Qwen3-TTS checkout "$QWEN3_TTS_COMMIT" \
 && mkdir -p /app \
 && cp -r /opt/Qwen3-TTS/finetuning /app/finetuning

# export_gguf.py runs the GGUF converters of the qwen3-tts.cpp commit that server-gguf builds.
ARG QWEN3_TTS_CPP_COMMIT=8fa5b30604b67b82fd9df7c30c91e78b42b54571
RUN pip install gguf==0.19.0 tqdm==4.70.1 \
 && git clone https://github.com/predict-woo/qwen3-tts.cpp.git /opt/qwen3-tts.cpp \
 && git -C /opt/qwen3-tts.cpp checkout "$QWEN3_TTS_CPP_COMMIT"

# Jobs may run as any uid (compose RUN_AS); torch and HF need a user name and a writable home.
ENV USER=glados HOME=/tmp
LABEL org.opencontainers.image.source=https://github.com/MyrikLD/glados-tts
COPY export_gguf.py prep.py stt.py textnorm.py train.py /app/
COPY eval /app/eval
WORKDIR /app
CMD ["python", "/app/train.py"]
