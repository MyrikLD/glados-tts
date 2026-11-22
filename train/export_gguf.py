"""Exports a fine-tuned checkpoint for server-gguf: GGUF talker and vocoder, plus the voice as a speaker embedding."""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

CONVERTERS = Path(os.environ.get("QWEN3_TTS_CPP", "/opt/qwen3-tts.cpp")) / "scripts"
SPEAKER = "glados"


def main() -> None:
    ckpt = Path(os.environ["CKPT"])
    kind = os.environ.get("GGUF_TYPE", "q8_0")
    out = Path(os.environ.get("OUT") or ckpt / f"gguf-{kind}")
    out.mkdir(parents=True, exist_ok=True)

    # libqwen3tts loads these two names whatever the weight type is.
    for script, name, weights in (
        ("convert_tts_to_gguf.py", "qwen3-tts-0.6b-f16.gguf", kind),
        ("convert_tokenizer_to_gguf.py", "qwen3-tts-tokenizer-f16.gguf", "f16"),
    ):
        subprocess.run([sys.executable, CONVERTERS / script, "-i", ckpt, "-o", out / name, "-t", weights], check=True)

    # A custom-voice fine-tune keeps its voice in one row of the codec embedding. The prompt puts that row
    # where a voice-clone prompt puts the speaker x-vector, so the row works as a speaker embedding.
    spk_id = json.loads((ckpt / "config.json").read_text())["talker_config"]["spk_id"][SPEAKER]
    with safe_open(ckpt / "model.safetensors", "pt") as f:
        row = f.get_tensor("talker.model.codec_embedding.weight")[spk_id].float().numpy().astype("<f4")
    (out / f"{SPEAKER}.spk").write_bytes(struct.pack("<I", row.size) + row.tobytes())
    print(f"exported {ckpt} -> {out} ({kind}, speaker row {spk_id}, norm {np.linalg.norm(row):.2f})")


if __name__ == "__main__":
    main()
