# Wyoming TTS server for the fine-tuned GLaDOS voice on qwen3-tts.cpp (GGML) with the Vulkan backend.
# Takes a GGUF export of the checkpoint (train/export_gguf.py); needs no ROCm or torch.
FROM ubuntu:24.04 AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates libvulkan-dev glslc spirv-headers \
 && rm -rf /var/lib/apt/lists/*

# The patch is made against this commit (and the ggml commit it pins as a submodule).
ARG QWEN3_TTS_CPP_COMMIT=8fa5b30604b67b82fd9df7c30c91e78b42b54571
COPY qwen3-tts-cpp.patch /tmp/
RUN git clone https://github.com/predict-woo/qwen3-tts.cpp.git /src \
 && git -C /src checkout "$QWEN3_TTS_CPP_COMMIT" \
 && git -C /src submodule update --init \
 && git -C /src apply /tmp/qwen3-tts-cpp.patch
WORKDIR /src
# GGML_NATIVE=OFF: a native build targets the CI runner's CPU and dies with SIGILL on hosts that lack its instructions.
RUN cmake -S ggml -B ggml/build -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
 && cmake --build ggml/build -j"$(nproc)" \
 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j"$(nproc)" --target qwen3tts_shared qwen3-tts-cli \
 && mkdir -p /out/lib \
 && cp -a build/libqwen3tts.so* /out/lib/ \
 && cp build/qwen3-tts-cli /out/ \
 && find ggml/build -name '*.so*' -exec cp -a {} /out/lib/ \;

# Ubuntu rather than python:slim: Debian's Mesa 25.0 exposes no cooperative-matrix support on RDNA4.
FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip libvulkan1 mesa-vulkan-drivers libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --break-system-packages wyoming==1.11.0 numpy==2.5.5
LABEL org.opencontainers.image.source=https://github.com/MyrikLD/glados-tts
COPY --from=build /out /app
# 8 codec frames per streamed chunk, as in the torch server.
ENV LD_LIBRARY_PATH=/app/lib PYTHONUNBUFFERED=1 QWEN3_TTS_STREAM_CHUNK_FRAMES=8
COPY server.py /app/server.py
EXPOSE 10200
CMD ["python3", "/app/server.py"]
