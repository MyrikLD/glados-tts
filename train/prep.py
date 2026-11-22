"""Build the Qwen3-TTS fine-tuning set from one language's GLaDOS clips: 24 kHz wavs, transcripts, codec codes.

Env:
    META    clip list name|text, default /ref/metadata_clean.csv (from dataset/clean_metadata.py)
    FT_DIR  output folder, default /data/ft; use one folder per language
    THREADS CPU threads for codec encoding

/ref holds the clips of one language: wavs/ + the metadata files from dataset/.
"""

import json
import os
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import textnorm

REF_DIR = Path("/ref")
META = Path(os.environ.get("META", "/ref/metadata_clean.csv"))
FT = Path(os.environ.get("FT_DIR", "/data/ft"))
WAVS = FT / "wavs"
RATE = 24000
MIN_SEC, MAX_SEC = 1.0, 20.0
REF_SEC = 8.0


def load_metadata() -> dict[str, str]:
    rows = (line.split("|", 1) for line in META.read_text(encoding="utf-8").splitlines() if "|" in line)
    return {Path(name.strip()).stem: text.strip() for name, text in rows}


def process(name: str, text: str) -> dict:
    # Digits are read differently in every context, so the text must spell numbers out.
    if not textnorm.tokens(text) or textnorm.has_digits(text):
        return {"drop": "text"}
    source = REF_DIR / "wavs" / f"{name}.wav"
    if not source.exists():
        return {"drop": "no_audio"}
    target = WAVS / f"{name}.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(target)],
        check=True,
    )
    with wave.open(str(target)) as w:
        seconds = w.getnframes() / w.getframerate()
    if not MIN_SEC <= seconds <= MAX_SEC:
        target.unlink()
        return {"drop": "duration"}
    return {"audio": str(target), "text": text, "seconds": seconds}


def main() -> None:
    WAVS.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    print(f"{META}: {len(meta)} clips", flush=True)

    rows, drops = [], {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n, result in enumerate(pool.map(lambda item: process(*item), meta.items()), 1):
            if "drop" in result:
                drops[result["drop"]] = drops.get(result["drop"], 0) + 1
            else:
                rows.append(result)
            if n % 200 == 0:
                print(f"  {n}/{len(meta)} kept={len(rows)} drops={drops}", flush=True)
    print(f"kept {len(rows)} clips, {sum(r['seconds'] for r in rows) / 3600:.2f} h, drops={drops}", flush=True)

    # The reference clip only conditions the speaker embedding; a mid-length line works best.
    ref = min(rows, key=lambda r: abs(r["seconds"] - REF_SEC))
    print(f"ref_audio {Path(ref['audio']).name} {ref['seconds']:.1f}s", flush=True)

    raw_rows = [{"audio": r["audio"], "text": r["text"], "ref_audio": ref["audio"]} for r in rows]
    (FT / "train_raw.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw_rows))

    import torch
    from qwen_tts import Qwen3TTSTokenizer

    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    tokenizer = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cpu")
    with open(FT / "train_with_codes.jsonl", "w") as out:
        for i in range(0, len(raw_rows), 16):
            batch = raw_rows[i : i + 16]
            encoded = tokenizer.encode([r["audio"] for r in batch])
            out.writelines(
                json.dumps({**row, "audio_codes": codes.cpu().tolist()}, ensure_ascii=False) + "\n"
                for row, codes in zip(batch, encoded.audio_codes, strict=True)
            )
            if (i // 16) % 10 == 0:
                print(f"  codes {i + len(batch)}/{len(raw_rows)}", flush=True)
    print("PREP DONE", flush=True)


if __name__ == "__main__":
    main()
