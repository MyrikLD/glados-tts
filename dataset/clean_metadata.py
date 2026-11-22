"""Build metadata_clean.csv: keep only clips of plain GLaDOS speech whose audio matches the text.

Usage:
    python clean_metadata.py <lang_dir> [--exclude exclude.txt] [--delivery delivery_notes.txt]
    python clean_metadata.py <lang_dir> --sim speaker_sim.csv --write-exclude exclude.txt

The rules below always apply; --exclude adds a saved list on top of them.

Rules:
  potato     potatos_* lines, the PotatOS episode with a processed voice
  sabotage   sp_sabotage_glados_* gibberish and screams
  nonspeech  the whole transcript is a note in [] or (), e.g. [смех]
  note       the transcript has a note, e.g. (beep beep beep), [yawn]; clips listed in
             --delivery are kept instead, with the note cut from the text, because their
             notes only describe how the line is said, e.g. [sarcastic]
  lowsim     speaker similarity below a duration-dependent threshold (needs --sim)
"""

import argparse
import csv
import re
from pathlib import Path

# Subtitle notes in [] or () mark sounds or delivery that the text does not say: (page flip), [laughs].
# Cutting a sound note from the text would leave the sound in the audio, so such clips are dropped.
# Both bracket kinds hold both note kinds, so delivery-only clips are listed by name.
NOTE = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def threshold(seconds: float) -> float:
    # Short clips give noisy embeddings, so they get a lower bar.
    if seconds < 1:
        return 0.80
    if seconds < 2:
        return 0.90
    if seconds < 4:
        return 0.92
    return 0.95


def reasons(name: str, text: str, sim: tuple[float, float] | None, delivery: set[str]) -> list[str]:
    why = []
    if name.startswith("potatos"):
        why.append("potato")
    if name.startswith("sp_sabotage_glados"):
        why.append("sabotage")
    if NOTE.search(text) and name not in delivery:
        why.append("nonspeech" if not NOTE.sub("", text).strip() else "note")
    if sim is not None and sim[0] < threshold(sim[1]):
        why.append("lowsim")
    return why


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lang_dir", type=Path)
    parser.add_argument("--exclude", type=Path, help="name|reason list to drop in addition to the rules")
    parser.add_argument("--delivery", type=Path, help="clips whose notes describe delivery only; their notes are cut, the clip stays")
    parser.add_argument("--sim", type=Path, help="speaker_sim.csv from speaker_outliers.py")
    parser.add_argument("--write-exclude", type=Path, help="save the computed name|reason list here")
    args = parser.parse_args()

    meta = [line.split("|", 1) for line in (args.lang_dir / "metadata.csv").read_text(encoding="utf-8").splitlines() if "|" in line]
    sims = {}
    if args.sim:
        with open(args.sim, encoding="utf-8") as f:
            sims = {r["name"]: (float(r["sim"]), float(r["dur"])) for r in csv.DictReader(f)}

    delivery = set(args.delivery.read_text().split()) if args.delivery else set()
    excluded = {}
    for name, text in meta:
        if why := reasons(name, text, sims.get(name), delivery):
            excluded[name] = "+".join(why)
    if args.exclude:
        for line in args.exclude.read_text().splitlines():
            if "|" in line:
                name, why = line.split("|", 1)
                excluded.setdefault(name, why)

    if args.write_exclude:
        args.write_exclude.write_text("".join(f"{n}|{r}\n" for n, r in sorted(excluded.items())))

    kept = [(n, " ".join(NOTE.sub("", t).split())) for n, t in meta if n not in excluded]
    out = args.lang_dir / "metadata_clean.csv"
    out.write_text("".join(f"{n}|{t}\n" for n, t in kept), encoding="utf-8")
    print(f"kept {len(kept)}, dropped {len(meta) - len(kept)} -> {out}")


if __name__ == "__main__":
    main()
