"""Wyoming TTS server for the fine-tuned Qwen3-TTS GLaDOS voice on qwen3-tts.cpp (GGML, Vulkan), with audio streaming."""

import argparse
import asyncio
import ctypes
import logging
import os
import re
import struct
import threading
from functools import partial
from pathlib import Path

import numpy as np
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

_LOGGER = logging.getLogger("glados")
SPEAKER = "glados"
RATE = 24000
WIDTH = 2
CHANNELS = 1
# The fine-tune was trained without a language token, which the library selects with a negative id.
LANGUAGE_AUTO = -1
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


class Params(ctypes.Structure):
    _fields_ = [
        ("max_audio_tokens", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("repetition_penalty", ctypes.c_float),
        ("language_id", ctypes.c_int32),
        ("ref_code_path", ctypes.c_char_p),
    ]


class Audio(ctypes.Structure):
    _fields_ = [("samples", ctypes.POINTER(ctypes.c_float)), ("n_samples", ctypes.c_int32), ("sample_rate", ctypes.c_int32)]


STREAM_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_void_p)


def load_library(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)
    lib.qwen3_tts_create.restype = ctypes.c_void_p
    lib.qwen3_tts_create.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.qwen3_tts_default_params.argtypes = [ctypes.POINTER(Params)]
    lib.qwen3_tts_get_error.restype = ctypes.c_char_p
    lib.qwen3_tts_get_error.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_free_audio.argtypes = [ctypes.POINTER(Audio)]
    lib.qwen3_tts_synthesize_with_embedding_streaming.restype = ctypes.POINTER(Audio)
    lib.qwen3_tts_synthesize_with_embedding_streaming.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.POINTER(Params),
        STREAM_CALLBACK,
        ctypes.c_void_p,
    ]
    return lib


def read_speaker(path: Path) -> np.ndarray:
    data = path.read_bytes()
    (size,) = struct.unpack_from("<I", data)
    return np.frombuffer(data, dtype="<f4", count=size, offset=4).copy()


class Voice:
    def __init__(self, lib_path: str, model_dir: Path, threads: int) -> None:
        self.lib = load_library(lib_path)
        self.tts = self.lib.qwen3_tts_create(str(model_dir).encode(), threads)
        if not self.tts:
            raise RuntimeError(f"cannot load models from {model_dir}")
        self.speaker = read_speaker(model_dir / f"{SPEAKER}.spk")
        self.params = Params()
        self.lib.qwen3_tts_default_params(ctypes.byref(self.params))
        self.params.n_threads = threads
        self.params.language_id = LANGUAGE_AUTO
        self.lock = asyncio.Lock()
        # The first run compiles the Vulkan pipelines; doing it here keeps that cost off the first request.
        self._produce("GLaDOS.", lambda pcm: None)

    def _produce(self, text: str, emit) -> None:
        def on_audio(samples, n, _):
            emit((np.clip(np.ctypeslib.as_array(samples, shape=(n,)), -1.0, 1.0) * 32767).astype("<i2").tobytes())
            return 1

        self.params.max_audio_tokens = 4 * len(text) + 60
        audio = self.lib.qwen3_tts_synthesize_with_embedding_streaming(
            self.tts,
            text.encode(),
            self.speaker.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.speaker.size,
            ctypes.byref(self.params),
            STREAM_CALLBACK(on_audio),
            None,
        )
        if not audio:
            raise RuntimeError(self.lib.qwen3_tts_get_error(self.tts).decode())
        self.lib.qwen3_tts_free_audio(audio)

    async def stream(self, text: str):
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def run() -> None:
            try:
                self._produce(text, lambda pcm: loop.call_soon_threadsafe(queue.put_nowait, pcm))
            except Exception as error:
                loop.call_soon_threadsafe(queue.put_nowait, error)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        async with self.lock:
            threading.Thread(target=run, daemon=True).start()
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
            await self.write_event(AudioStart(rate=RATE, width=WIDTH, channels=CHANNELS).event())
            self.started = True

    async def _speak(self, sentences: list[str]) -> None:
        for sentence in sentences:
            _LOGGER.info("speak %r", sentence)
            await self._start()
            async for pcm in self.voice.stream(sentence):
                await self.write_event(AudioChunk(rate=RATE, width=WIDTH, channels=CHANNELS, audio=pcm).event())

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
    parser.add_argument(
        "--model-dir", type=Path, default=os.environ.get("MODEL_DIR"), required="MODEL_DIR" not in os.environ, help="env MODEL_DIR"
    )
    parser.add_argument(
        "--language",
        default=os.environ.get("VOICE_LANG"),
        required="VOICE_LANG" not in os.environ,
        help="language code the voice was trained on (env VOICE_LANG)",
    )
    parser.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", "4")))
    parser.add_argument("--library", default=os.environ.get("QWEN3_TTS_LIB", "libqwen3tts.so"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    voice = Voice(args.library, args.model_dir, args.threads)
    attribution = Attribution(name="Qwen3-TTS fine-tune", url="https://github.com/QwenLM/Qwen3-TTS")
    info = Info(
        tts=[
            TtsProgram(
                name="glados-qwen3",
                description="GLaDOS (Qwen3-TTS 0.6B fine-tune, GGUF)",
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
    _LOGGER.info("ready on %s, %s Hz", args.uri, RATE)
    await AsyncServer.from_uri(args.uri).run(partial(Handler, info, voice))


if __name__ == "__main__":
    asyncio.run(main())
