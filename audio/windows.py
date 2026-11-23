
from dataclasses import dataclass
from typing import Iterable, Iterator

from audio.loopback import AudioChunk


@dataclass(frozen=True)
class AudioWindow:
    pcm: bytes
    sample_rate: int
    channels: int
    start_s: float
    end_s: float
    captured_at: float


def fixed_windows(
    chunks: Iterable[AudioChunk],
    seconds: float = 3.0,
) -> Iterator[AudioWindow]:
    """Split an audio stream into fixed-duration windows."""

    if seconds <= 0:
        raise ValueError("Window duration must be positive.")

    buffer = bytearray()
    sample_rate = None
    channels = None
    frames_per_window = 0
    target_bytes = 0
    consumed_frames = 0

    for chunk in chunks:
        if sample_rate is None:
            sample_rate = chunk.sample_rate
            channels = chunk.channels

            frames_per_window = round(
                seconds * sample_rate
            )

            target_bytes = (
                frames_per_window * channels * 2
            )

            if target_bytes <= 0:
                raise ValueError("Invalid audio window size.")

        elif (
            chunk.sample_rate != sample_rate
            or chunk.channels != channels
        ):
            raise RuntimeError(
                "Audio device format changed during capture."
            )

        buffer.extend(chunk.pcm)

        while len(buffer) >= target_bytes:
            pcm = bytes(buffer[:target_bytes])
            del buffer[:target_bytes]

            start_s = consumed_frames / sample_rate
            consumed_frames += frames_per_window
            end_s = consumed_frames / sample_rate

            yield AudioWindow(
                pcm=pcm,
                sample_rate=sample_rate,
                channels=channels,
                start_s=start_s,
                end_s=end_s,
                captured_at=chunk.captured_at,
            )