# glados-tts

A GLaDOS voice for Home Assistant in any language Portal 2 was dubbed into: Qwen3-TTS 0.6B fine-tuned on the game's voice-over, served as a streaming Wyoming TTS on an AMD GPU (ROCm).

The repository has no game audio and no subtitle text. You make the dataset from your own copy of Portal 2. The published model is Russian.

## Languages

You need two things for a language:

- **The dub.** Portal 2 has GLaDOS voice-over in English, French, German, Spanish and Russian. Steam downloads a dub only while the game is set to that language, so switch it in the game's properties and let Steam update before you extract.
- **Qwen3-TTS support.** The model speaks Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish and Italian. All five dubs are in that list.

Every step takes the language from `VOICE_LANG` (an ISO 639-1 code such as `ru` or `en`). Only `train/eval/<code>.txt` (four test phrases) is per language; it is there for all five dubs (`en`, `fr`, `de`, `es`, `ru`). `EVAL_TEXTS` points to other phrases.

## Layout

| Path                          | What it does                                                                                              |
| ----------------------------- | --------------------------------------------------------------------------------------------------------- |
| `dataset/extract_glados.py`   | Reads Portal 2 VPKs and subtitles, writes `<code>/wavs/` + `metadata.csv` (LJSpeech format)               |
| `dataset/speaker_outliers.py` | Scores every clip by speaker similarity (WavLM x-vector) to find clips that are not GLaDOS's normal voice |
| `dataset/clean_metadata.py`   | Writes `metadata_clean.csv`: drops PotatOS lines, gibberish, and low-similarity clips                     |
| `dataset/exclude.txt`         | The exclusion list used for the published model (clip name and reason)                                    |
| `dataset/delivery_notes.txt`  | Clips whose subtitle notes describe delivery only, so they are kept with the note cut                     |
| `train/`                      | Image and scripts for data prep (`prep.py`) and fine-tuning + evaluation (`train.py`)                     |
| `server/`                     | Image and script for the Wyoming TTS server                                                               |
| `deploy/`                     | Compose files for the GPU host                                                                            |

## Dataset

`extract_glados.py` needs the `vpk` package and `ffmpeg`. Give it the game's names of the languages (`english`, `russian`, `french`, `german`, `spanish`). For a dub it reads only that language's paks, so no English audio gets in. It matches each `sound/vo/glados/` clip to its subtitle through the soundscripts, takes GLaDOS's lines only (the subtitles also hold a few lines of Caroline), and converts the audio to 16-bit mono WAV. Clips without a subtitle (mostly cut content) go to `no_text.txt`, not to `metadata.csv`. The game audio is 80 kbps MP3 inside `.wav`, so this is the quality ceiling.

```sh
pip install vpk
python dataset/extract_glados.py "<Portal 2 dir>" ./data russian english
python dataset/clean_metadata.py data/ru --exclude dataset/exclude.txt --delivery dataset/delivery_notes.txt
```

The rules (PotatOS, gibberish, subtitle notes) always apply, to the text of the language at hand. A subtitle note such as `(page flip)` means there is an extra sound in the clip; cutting the note from the text would teach the model to make that sound, so the whole clip goes. The exception is `delivery_notes.txt`: clips whose notes only say how the line is said (`[sarcastic]`, `[impressed]`); there the note is cut and the clip stays. Both `[]` and `()` hold both kinds of note, so these clips are listed by name. `exclude.txt` adds the saved list on top; clip names are the same in every language, so it applies to all of them. Its `lowsim` entries were measured on the Russian recording; for another language, make its own list:

```sh
pip install numpy scipy torch transformers
python dataset/speaker_outliers.py data/en data/en/speaker_sim.csv
python dataset/clean_metadata.py data/en --sim data/en/speaker_sim.csv --delivery dataset/delivery_notes.txt --write-exclude data/en/exclude.txt
```

Why the cleaning matters: 192 of the 1185 subtitled lines come from the PotatOS episode, where the voice is processed. With them in the set, the model sometimes switches to that voice.

## Training

Neither job needs anything but the clips. To get an error rate for every epoch's test samples, point `STT_URI=tcp://host:port` at a speech-to-text server that speaks the [Wyoming protocol](https://github.com/OHF-Voice/wyoming) and knows the language, for example `rhassp /wyoming-whisper`. The client is `train/stt.py`. The rate catches a broken run (silence, endless noise) without listening; it says nothing about how close the voice is.

Training is a full fine-tune with AdamW. With 4 GiB of free VRAM it runs out of memory; with 19.5 GiB free on a 24 GiB card it works. Stop other GPU tenants first.

1. `prep` (once per language): takes the clips in `metadata_clean.csv`, drops those shorter than 1 s or longer than 20 s or with digits in the text, resamples to 24 kHz and encodes the codec tokens into `ft/<code>/train_with_codes.jsonl`.
2. `train`: trains 5 epochs, then makes the test phrases with every epoch's checkpoint into `ft/<code>/<RUN>/samples/`, scored by error rate if `STT_URI` is set (words, or characters for Chinese and Japanese).

```sh
cd deploy
export VOICE_LANG=ru DATA_DIR=... CLIPS_DIR=.../data/ru RENDER_GID=$(getent group render | cut -d: -f3)
docker compose -f train.compose.yml --profile prep up prep
RUN=run1 STT_URI=tcp://... docker compose -f train.compose.yml up train
```

The upstream `finetuning/sft_12hz.py` does not train correctly as published. `train.py` patches it at run time (see `PATCHES`). Without the patches the 0.6B model crashes or learns to output silence that never ends.

## Server

```sh
VOICE_LANG=ru DATA_DIR=... CKPT=/data/ft/ru/run1/checkpoint-epoch-4 RENDER_GID=$(getent group render | cut -d: -f3) \
  docker compose -f deploy/server.compose.yml up -d
```

Add it in Home Assistant as a Wyoming integration on port 10200. Text is split into sentences, and each sentence is streamed in chunks of 8 codec frames. The first audio comes in about 450 ms on an RX 7900 XTX. The process uses about 4.5 GiB of VRAM.

## Versions

Both images use torch 2.11.0 + ROCm 7.2: torchaudio stopped at 2.11, and its binary loads only with the torch it was built for. ROCm 7.2 covers RDNA3 (gfx1100, RX 7900) and RDNA4 (gfx1201, RX 9070). Both images use Transformers 5.15.1 through `qwen-tts-hf`, the Transformers 5 build of `qwen-tts`; newer Transformers break it.

## Images

`.github/workflows/images.yml` builds `ghcr.io/myrikld/glados-tts-train` and
`ghcr.io/myrikld/glados-tts-server` on a GitHub-hosted runner and pushes them on
every change to `main` under `train/`, `server/` or the workflow. Each image is
about 17 GB, 15 GB of it the ROCm torch layer that both images share, so the job
first frees disk space and builds both images in one go. Pull requests build
the images without pushing them.
