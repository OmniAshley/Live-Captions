
from dataclasses import dataclass
from time import perf_counter

import pyaudiowpatch as pyaudio


@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes
    sample_rate: int
    channels: int
    captured_at: float


def stream_system_audio(frames_per_buffer=1024):
    """Yield PCM audio chunks from the default Windows output."""

    with pyaudio.PyAudio() as audio:
        device = audio.get_default_wasapi_loopback()

        sample_rate = int(device["defaultSampleRate"])
        channels = int(device["maxInputChannels"])

        print(f"Capturing: {device['name']}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Channels: {channels}")

        with audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=frames_per_buffer,
        ) as stream:

            while True:
                pcm = stream.read(
                    frames_per_buffer,
                    exception_on_overflow=False,
                )

                yield AudioChunk(
                    pcm=pcm,
                    sample_rate=sample_rate,
                    channels=channels,
                    captured_at=perf_counter(),
                )
