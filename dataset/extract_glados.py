"""Extract GLaDOS voice lines from Portal 2 into LJSpeech-style datasets, one per language.

Usage: python extract_glados.py <Portal 2 dir> <out dir> <language>...

<language> is the game's own name for it: english, russian, french, german, spanish, ...
Output per language: <out>/<code>/wavs/<name>.wav (PCM 16-bit mono), metadata.csv (name|text)
and no_text.txt (clips that have no subtitle, mostly cut content).

Steam downloads the voice-over of a dubbed language only while the game is set to that
language, so switch the game language and let Steam update before extracting it.
"""

import argparse
import csv
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import vpk

# Game language name -> ISO 639-1 code used for the output folder and for TTS/STT.
CODES = {
    "english": "en",
    "russian": "ru",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "latam": "es",
    "italian": "it",
    "portuguese": "pt",
    "brazilian": "pt",
    "japanese": "ja",
    "korean": "ko",
    "koreana": "ko",
    "schinese": "zh",
    "tchinese": "zh",
    "polish": "pl",
    "czech": "cs",
    "dutch": "nl",
    "danish": "da",
    "finnish": "fi",
    "norwegian": "no",
    "swedish": "sv",
    "hungarian": "hu",
    "romanian": "ro",
    "turkish": "tr",
    "ukrainian": "uk",
    "bulgarian": "bg",
    "greek": "el",
    "thai": "th",
}
GAME_DIRS = ["portal2", "portal2_dlc1"]
VO_PREFIX = "sound/vo/glados/"

ENTRY_RE = re.compile(r'^\s*"(glados\.[^"]+)"\s*$', re.I | re.M)
WAVE_RE = re.compile(r'"wave"\s+"([^"]+)"', re.I)
SUB_RE = re.compile(r'^\s*"(glados\.[^"]+)"\s+"(.*)"\s*$', re.I)
TAG_RE = re.compile(r"<[^>]*>")
# Each subtitle starts with formatting tags and then "<speaker>: ".
SPEAKER_RE = re.compile(r"^(?:<[^>]*>)+\s*([^:<]{1,24}?)\s*:\s*")


def soundscript_waves(game: Path) -> dict[str, str]:
    """Map soundscript entry name -> wav path inside sound/."""
    result = {}
    for d in GAME_DIRS:
        for f in sorted((game / d / "scripts").glob("game_sounds*.txt")):
            text = f.read_text(encoding="utf-8", errors="replace")
            matches = list(ENTRY_RE.finditer(text))
            for i, m in enumerate(matches):
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                waves = WAVE_RE.findall(text[m.end() : end])
                if waves:
                    wave = waves[0].lstrip("*#@)(^<>!?").replace("\\", "/")
                    result[m.group(1).lower()] = "sound/" + wave.lower()
    return result


def subtitles(game: Path, lang: str) -> dict[str, str]:
    """Map entry name -> GLaDOS's subtitle text; lines of other speakers are left out."""
    raw = {}
    for d in GAME_DIRS:
        f = game / d / "resource" / f"subtitles_{lang}.txt"
        if f.exists():
            for line in f.read_text(encoding="utf-16").splitlines():
                if m := SUB_RE.match(line):
                    raw[m.group(1).lower()] = m.group(2)
    if not raw:
        raise SystemExit(f"no subtitles_{lang}.txt in {game}")

    speakers = Counter(m.group(1) for text in raw.values() if (m := SPEAKER_RE.match(text)))
    glados = speakers.most_common(1)[0][0] if speakers else None
    result, others = {}, Counter()
    for key, text in raw.items():
        if m := SPEAKER_RE.match(text):
            if m.group(1) != glados:
                others[m.group(1)] += 1
                continue
            text = text[m.end() :]
        result[key] = " ".join(TAG_RE.sub("", text).replace("\\n", " ").split())
    print(f"{lang}: speaker {glados!r}, skipped other speakers {dict(others)}")
    return result


def voice_paks(game: Path, lang: str) -> list[str]:
    """English is the base game; a dub lives only in its own paks, so no English audio leaks in."""
    if lang == "english":
        return GAME_DIRS
    return [f"{d}_{lang}" for d in GAME_DIRS if (game / f"{d}_{lang}" / "pak01_dir.vpk").exists()]


def open_paks(game: Path, names: list[str]) -> dict[str, vpk.VPK]:
    """Map wav path -> pak that holds its final version; later paks override earlier ones."""
    owner = {}
    for name in names:
        p = vpk.open(str(game / name / "pak01_dir.vpk"))
        for path in p:
            if path.startswith(VO_PREFIX):
                owner[path] = p
    return owner


def convert(data: bytes, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", "pipe:0", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        input=data,
        check=True,
    )


def extract(game: Path, out: Path, lang: str, waves: dict[str, str]) -> None:
    owner = open_paks(game, voice_paks(game, lang))
    if not owner:
        raise SystemExit(f"{lang}: no GLaDOS voice-over found; set the game to this language in Steam and let it download")
    subs = subtitles(game, lang)
    target = out / CODES.get(lang, lang)
    wav_dir = target / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows, jobs, unmatched = [], [], []
    for path, pak in sorted(owner.items()):
        stem = Path(path).stem
        keys = [k for k, w in waves.items() if w == path] or [f"glados.{stem}"]
        text = next((subs[k] for k in keys if k in subs), "")
        # vpk reads are not thread-safe, so only ffmpeg runs in parallel.
        jobs.append((pak[path].read(), wav_dir / f"{stem}.wav"))
        if text:
            rows.append((stem, text))
        else:
            unmatched.append(stem)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: convert(*j), jobs))

    with open(target / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL).writerows(rows)
    (target / "no_text.txt").write_text("\n".join(unmatched), encoding="utf-8")
    print(f"{lang} -> {target}: {len(jobs)} wav, {len(rows)} with text, {len(unmatched)} without")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("game", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("languages", nargs="+", metavar="language")
    args = parser.parse_args()

    waves = soundscript_waves(args.game)
    for lang in args.languages:
        extract(args.game, args.out, lang.lower(), waves)


if __name__ == "__main__":
    main()
