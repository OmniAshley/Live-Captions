
from dataclasses import dataclass

import numpy as np
import soxr

from audio.loopback import stream_system_audio


SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2


@dataclass(frozen=True)
class AudioFrame:
    pcm: bytes
    start_s: float
    end_s: float
    captured_at: float


def stream_frames():
    """Yield continuous 16 kHz mono PCM frames."""

    resampler = None
    source_format = None
    buffer = bytearray()
    frame_index = 0

    for chunk in stream_system_audio():
        current_format = (
            chunk.sample_rate,
            chunk.channels,
        )

        if resampler is None:
            source_format = current_format

            resampler = soxr.ResampleStream(
                chunk.sample_rate,
                SAMPLE_RATE,
                1,
                dtype="float32",
                quality="MQ",
            )

        elif current_format != source_format:
            raise RuntimeError(
                "Audio device format changed."
            )

        samples = np.frombuffer(
            chunk.pcm,
            dtype="<i2",
        )

        # Interleaved multichannel PCM -> mono float.
        mono = (
            samples.reshape(-1, chunk.channels)
            .astype(np.float32)
            .mean(axis=1)
            / 32768.0
        )

        # Continuous sample-rate conversion.
        converted = resampler.resample_chunk(mono)

        pcm16 = (
            np.clip(
                np.rint(converted * 32768.0),
                -32768,
                32767,
            )
            .astype("<i2")
            .tobytes()
        )

        buffer.extend(pcm16)

        while len(buffer) >= FRAME_BYTES:
            pcm = bytes(buffer[:FRAME_BYTES])
            del buffer[:FRAME_BYTES]

            start_s = frame_index * FRAME_MS / 1000.0
            frame_index += 1

            yield AudioFrame(
                pcm=pcm,
                start_s=start_s,
                end_s=(
                    frame_index * FRAME_MS / 1000.0
                ),
                captured_at=chunk.captured_at,
            )