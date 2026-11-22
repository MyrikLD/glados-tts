"""Wyoming TTS server for the fine-tuned Qwen3-TTS GLaDOS voice, with graph-captured decoding and audio streaming."""

import argparse
import asyncio
import logging
import os
import re
import threading
from functools import partial

import numpy as np
import torch
from faster_qwen3_tts import FasterQwen3TTS
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

_LOGGER = logging.getLogger("glados")
SPEAKER = "glados"
WIDTH = 2
CHANNELS = 1
# Full-width CJK stops end a sentence without a following space.
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[。！？])\s*")
MARKUP = re.compile(r"[*_#`~>|]+")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", MARKUP.sub(" ", text)).strip()


def split_sentences(text: str) -> list[str]:
    return [part for part in (p.strip() for p in SENTENCE_END.split(clean(text))) if part]


class SentenceBuffer:
    def __init__(self) -> None:
        self.text = ""

    def add(self, chunk: str) -> list[str]:
        self.text += chunk
        parts = SENTENCE_END.split(self.text)
        self.text = parts.pop()
        return [s for s in (clean(p) for p in parts) if s]

    def finish(self) -> list[str]:
        rest, self.text = clean(self.text), ""
        return [rest] if rest else []


class Voice:
    def __init__(self, checkpoint: str, chunk_size: int) -> None:
        self.tts = FasterQwen3TTS.from_pretrained(checkpoint, device="cuda", dtype=torch.bfloat16, max_seq_len=1024)
        self.tts.warmup(prefill_len=100)
        self.chunk_size = chunk_size
        self.rate = 24000
        self.lock = asyncio.Lock()
        for _, rate, _ in self.tts.generate_custom_voice_streaming(
            text="GLaDOS.", speaker=SPEAKER, language="Auto", max_new_tokens=40, chunk_size=chunk_size
        ):
            self.rate = rate

    def _produce(self, text: str, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        try:
            for audio, _, _ in self.tts.generate_custom_voice_streaming(
                text=text,
                speaker=SPEAKER,
                language="Auto",
                # Room for slow speech and for scripts with many sounds per character.
                max_new_tokens=4 * len(text) + 60,
                chunk_size=self.chunk_size,
            ):
                pcm = (np.clip(np.asarray(audio, dtype=np.float32).flatten(), -1.0, 1.0) * 32767).astype("<i2").tobytes()
                loop.call_soon_threadsafe(queue.put_nowait, pcm)
        except Exception as error:
            loop.call_soon_threadsafe(queue.put_nowait, error)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def stream(self, text: str):
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        async with self.lock:
            threading.Thread(target=self._produce, args=(text, loop, queue), daemon=True).start()
            while (item := await queue.get()) is not None:
                if isinstance(item, Exception):
                    raise item
                yield item


class Handler(AsyncEventHandler):
    def __init__(self, info: Info, voice: Voice, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.info = info
        self.voice = voice
        self.buffer: SentenceBuffer | None = None
        self.started = False

    async def _start(self) -> None:
        if not self.started:
            await self.write_event(AudioStart(rate=self.voice.rate, width=WIDTH, channels=CHANNELS).event())
            self.started = True

    async def _speak(self, sentences: list[str]) -> None:
        for sentence in sentences:
            _LOGGER.info("speak %r", sentence)
            await self._start()
            async for pcm in self.voice.stream(sentence):
                await self.write_event(AudioChunk(rate=self.voice.rate, width=WIDTH, channels=CHANNELS, audio=pcm).event())

    async def _stop(self) -> None:
        await self._start()
        await self.write_event(AudioStop().event())
        self.started = False

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True
        if SynthesizeStart.is_type(event.type):
            self.buffer = SentenceBuffer()
            return True
        if SynthesizeChunk.is_type(event.type) and self.buffer is not None:
            await self._speak(self.buffer.add(SynthesizeChunk.from_event(event).text))
            return True
        if SynthesizeStop.is_type(event.type) and self.buffer is not None:
            await self._speak(self.buffer.finish())
            await self._stop()
            await self.write_event(SynthesizeStopped().event())
            self.buffer = None
            return True
        if Synthesize.is_type(event.type):
            if self.buffer is not None:
                return True
            await self._speak(split_sentences(Synthesize.from_event(event).text))
            await self._stop()
            return True
        return True


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10200")
    parser.add_argument("--checkpoint", default=os.environ.get("CKPT"), required="CKPT" not in os.environ, help="env CKPT")
    parser.add_argument(
        "--language",
        default=os.environ.get("VOICE_LANG"),
        required="VOICE_LANG" not in os.environ,
        help="language code the voice was trained on (env VOICE_LANG)",
    )
    parser.add_argument("--chunk-size", type=int, default=int(os.environ.get("CHUNK_SIZE", "8")))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    voice = Voice(args.checkpoint, args.chunk_size)
    attribution = Attribution(name="Qwen3-TTS fine-tune", url="https://github.com/QwenLM/Qwen3-TTS")
    info = Info(
        tts=[
            TtsProgram(
                name="glados-qwen3",
                description="GLaDOS (Qwen3-TTS 0.6B fine-tune)",
                attribution=attribution,
                installed=True,
                version="1.0",
                supports_synthesize_streaming=True,
                voices=[
                    TtsVoice(
                        name=SPEAKER,
                        description=f"GLaDOS ({args.language})",
                        attribution=attribution,
                        installed=True,
                        version="1.0",
                        languages=[args.language],
                    )
                ],
            )
        ]
    )
    _LOGGER.info("ready on %s, %s Hz, chunk %s", args.uri, voice.rate, args.chunk_size)
    await AsyncServer.from_uri(args.uri).run(partial(Handler, info, voice))


if __name__ == "__main__":
    asyncio.run(main())
