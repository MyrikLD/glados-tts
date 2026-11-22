"""Minimal client for any Wyoming speech-to-text server (https://github.com/OHF-Voice/wyoming).

Env:
    STT_URI       tcp://host:port of the server
    STT_LANGUAGE  language sent with each request, default VOICE_LANG
"""

import json
import os
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlparse

RATE, WIDTH, CHANNELS = 16000, 2, 1
BYTES_PER_SECOND = RATE * WIDTH * CHANNELS  # also the size of one audio-chunk event
FORMAT = {"rate": RATE, "width": WIDTH, "channels": CHANNELS}


def address() -> tuple[str, int]:
    uri = urlparse(os.environ["STT_URI"])
    if uri.scheme != "tcp" or not uri.hostname or not uri.port:
        raise ValueError(f"STT_URI must be tcp://host:port, got {os.environ['STT_URI']!r}")
    return uri.hostname, uri.port


def pcm16k(path: Path) -> bytes:
    """Decode any audio file to the raw 16 kHz mono s16le that the requests carry."""
    return subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", str(CHANNELS), "-ar", str(RATE), "-f", "s16le", "-"],
        check=True,
        capture_output=True,
    ).stdout


def _send(sock: socket.socket, kind: str, data: dict | None = None, payload: bytes = b"") -> None:
    body = json.dumps(data or {}).encode()
    header = {"type": kind, "data_length": len(body), "payload_length": len(payload)}
    sock.sendall(json.dumps(header).encode() + b"\n" + body + payload)


def transcribe(pcm: bytes, language: str | None = None) -> str:
    language = language or os.environ.get("STT_LANGUAGE") or os.environ["VOICE_LANG"]
    with socket.create_connection(address(), timeout=120) as sock:
        _send(sock, "transcribe", {"language": language})
        _send(sock, "audio-start", FORMAT)
        for i in range(0, len(pcm), BYTES_PER_SECOND):
            _send(sock, "audio-chunk", FORMAT, pcm[i : i + BYTES_PER_SECOND])
        _send(sock, "audio-stop")
        stream = sock.makefile("rb")
        while line := stream.readline():
            header = json.loads(line)
            # Event data may sit in the header, in the data block after it, or in both.
            data = dict(header.get("data") or {})
            if extra := stream.read(header.get("data_length") or 0):
                data.update(json.loads(extra))
            stream.read(header.get("payload_length") or 0)
            if header["type"] == "transcript":
                return data.get("text", "").strip()
    raise ConnectionError("STT server closed the connection without a transcript")
