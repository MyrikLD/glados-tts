"""Fine-tune Qwen3-TTS Base on one language of the GLaDOS set on ROCm, then make test samples for every epoch.

With STT_URI set (a Wyoming speech-to-text server), each sample is also transcribed and scored by error rate.
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import stt
import textnorm

FT = Path(os.environ.get("FT_DIR", "/data/ft"))
OUT = FT / os.environ.get("RUN", "run1")
BASE = os.environ.get("BASE", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
SPEAKER = "glados"
EVAL_DIR = Path(__file__).with_name("eval")
# The upstream sft_12hz.py does not train correctly as published; each pair is (exact upstream text, fix).
PATCHES = [
    ('log_with="tensorboard"', "log_with=None"),
    ('attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"'),
    # The 0.6B text embedding is 2048 wide and must go through text_projection to reach the 1024 talker.
    (
        "model.talker.model.text_embedding(input_text_ids)",
        "model.talker.text_projection(model.talker.model.text_embedding(input_text_ids))",
    ),
    # The model shifts labels itself; passing pre-shifted labels shifts them twice.
    ("labels=codec_0_labels[:, 1:],", "labels=None,"),
    ("hidden_states[codec_mask[:, :-1]]", "hidden_states[codec_mask[:, 1:]]"),
    (
        "loss = outputs.loss + 0.3 * sub_talker_loss",
        (
            "main_loss = torch.nn.functional.cross_entropy(outputs.logits.float().flatten(0, 1), codec_0_labels[:, 1:].flatten(), ignore_index=-100)\n"  # noqa: E501
            "                sub_talker_loss = torch.nn.functional.cross_entropy(sub_talker_logits.float().flatten(0, 1), talker_codec_ids[:, 1:].flatten())\n"  # noqa: E501
            "                loss = main_loss + 0.3 * sub_talker_loss"
        ),
    ),
]


def eval_texts() -> list[str]:
    """Test phrases from EVAL_TEXTS, or eval/<VOICE_LANG>.txt; one phrase per line."""
    path = Path(os.environ.get("EVAL_TEXTS") or EVAL_DIR / f"{os.environ['VOICE_LANG']}.txt")
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not texts:
        raise ValueError(f"no test phrases in {path}")
    return texts


def train_set() -> Path:
    """Keep only clips listed in META (default /ref/metadata_clean.csv) and take their text from it."""
    rows = (FT / "train_with_codes.jsonl").read_text().splitlines()
    meta = Path(os.environ.get("META", "/ref/metadata_clean.csv"))
    if os.environ.get("META_ONLY", "1") == "1" and meta.exists():
        known = {
            Path(name.strip()).stem: text.strip()
            for name, text in (line.split("|", 1) for line in meta.read_text(encoding="utf-8").splitlines() if "|" in line)
        }
        parsed = [json.loads(r) for r in rows]
        parsed = [{**r, "text": known[Path(r["audio"]).stem]} for r in parsed if Path(r["audio"]).stem in known]
        rows = [json.dumps(r, ensure_ascii=False) for r in parsed]
        if Path(parsed[0]["ref_audio"]).stem not in known:
            ref = min(parsed, key=lambda r: abs(len(r["audio_codes"]) / 12.5 - 8.0))["audio"]
            rows = [json.dumps({**r, "ref_audio": ref}, ensure_ascii=False) for r in parsed]
        print(f"ref_audio {Path(json.loads(rows[0])['ref_audio']).name}", flush=True)
    target = FT / f"train_used_{OUT.name}.jsonl"
    target.write_text("\n".join(rows) + "\n")
    print(f"training rows: {len(rows)}", flush=True)
    return target


def train(base_path: str) -> None:
    work = Path("/tmp/finetuning")
    shutil.copytree("/app/finetuning", work, dirs_exist_ok=True)
    script = work / "sft_12hz.py"
    source = script.read_text()
    for old, new in PATCHES:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    script.write_text(source)
    args = [
        sys.executable,
        "sft_12hz.py",
        "--init_model_path",
        base_path,
        "--output_model_path",
        str(OUT),
        "--train_jsonl",
        str(train_set()),
        "--batch_size",
        os.environ.get("BATCH", "2"),
        "--lr",
        os.environ.get("LR", "1e-5"),
        "--num_epochs",
        os.environ.get("EPOCHS", "5"),
        "--speaker_name",
        SPEAKER,
    ]
    print("train:", " ".join(args[1:]), flush=True)
    started = time.perf_counter()
    subprocess.run(args, check=True, cwd=work)
    print(f"TRAIN DONE in {(time.perf_counter() - started) / 60:.1f} min", flush=True)


def evaluate() -> None:
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    texts = eval_texts()
    samples = OUT / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    for ckpt in sorted(OUT.glob("checkpoint-epoch-*"), key=lambda p: int(p.name.rsplit("-", 1)[1])):
        tts = Qwen3TTSModel.from_pretrained(str(ckpt), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        tts.generate_custom_voice(text=texts[0], speaker=SPEAKER, max_new_tokens=textnorm.max_new_tokens(texts[0]))
        epoch = ckpt.name.rsplit("-", 1)[1]
        for t, text in enumerate(texts):
            started = time.perf_counter()
            wavs, sr = tts.generate_custom_voice(text=text, speaker=SPEAKER, max_new_tokens=textnorm.max_new_tokens(text))
            took = time.perf_counter() - started
            target = samples / f"ft-e{epoch}-t{t}.wav"
            sf.write(target, wavs[0], sr)
            seconds = len(wavs[0]) / sr
            line = f"  {target.name}: took={took:.1f}s audio={seconds:.1f}s rtf={took / seconds:.2f}"
            if os.environ.get("STT_URI"):
                heard = stt.transcribe(stt.pcm16k(target))
                line += f" wer={textnorm.error_rate(text, heard):.2f} stt={heard!r}"
            print(line, flush=True)
        print(f"  vram peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        del tts
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if os.environ.get("PRINT_SAMPLES") == "1":
        for target in sorted(samples.glob("*.wav")):
            opus = subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-i", str(target), "-c:a", "libopus", "-b:a", "32k", "-f", "ogg", "-"],
                check=True,
                capture_output=True,
            ).stdout
            print(f"SAMPLE {target.stem}.ogg {base64.b64encode(opus).decode()}", flush=True)


def main() -> None:
    import torch
    from huggingface_hub import snapshot_download

    print(f"torch {torch.__version__} hip {torch.version.hip} gpu {torch.cuda.get_device_name(0)}", flush=True)
    free, total = torch.cuda.mem_get_info()
    print(f"vram free {free / 2**30:.1f} / {total / 2**30:.1f} GiB", flush=True)
    if "probe" in sys.argv[1:]:
        print("PROBE DONE", flush=True)
        return
    if "eval" not in sys.argv[1:]:
        train(snapshot_download(BASE))
    evaluate()
    print("RUN DONE", flush=True)


if __name__ == "__main__":
    main()
