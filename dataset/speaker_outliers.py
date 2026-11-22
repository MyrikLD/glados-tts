"""Rank dataset clips by speaker-embedding similarity to the dataset's GLaDOS centroid.

Usage: python speaker_outliers.py <lang_dir> <out_csv>

<lang_dir> is an LJSpeech-style folder (wavs/ + metadata.csv). The output CSV is sorted
from the least similar clip, so the clips that do not sound like GLaDOS come first.
"""

import csv
import sys
from math import gcd
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, WavLMForXVector

SR = 16000
MODEL = "microsoft/wavlm-base-plus-sv"


def load(path: Path) -> np.ndarray:
    sr, x = wavfile.read(path)
    x = x.astype(np.float32) / 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    g = gcd(sr, SR)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def main() -> None:
    lang_dir = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = WavLMForXVector.from_pretrained(MODEL).to(device).eval()

    with open(lang_dir / "metadata.csv", encoding="utf-8") as f:
        rows = [(name, text) for name, text in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)]

    names, durs, embs = [], [], []
    with torch.inference_mode():
        for name, _ in rows:
            wav = load(lang_dir / "wavs" / f"{name}.wav")
            inp = fe(wav, sampling_rate=SR, return_tensors="pt").to(device)
            e = model(**inp).embeddings[0]
            embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
            names.append(name)
            durs.append(len(wav) / SR)

    emb = np.stack(embs)
    # Outliers pull a plain mean towards themselves, so the final centroid uses only the closest 70%.
    centroid = emb.mean(0) / np.linalg.norm(emb.mean(0))
    sim = emb @ centroid
    centroid = emb[sim >= np.quantile(sim, 0.3)].mean(0)
    centroid /= np.linalg.norm(centroid)
    sim = emb @ centroid

    text_of = dict(rows)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "sim", "dur", "text"])
        for i in np.argsort(sim):
            w.writerow([names[i], f"{sim[i]:.4f}", f"{durs[i]:.2f}", text_of[names[i]]])
    print(f"{len(names)} clips, sim median {np.median(sim):.3f}, p5 {np.quantile(sim, 0.05):.3f}")


if __name__ == "__main__":
    main()
