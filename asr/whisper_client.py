
from io import BytesIO
from math import gcd
import wave

import numpy as np
import requests
from scipy.signal import resample_poly

from audio.windows import AudioWindow


TARGET_RATE = 16000


def encode_wav(window: AudioWindow) -> bytes:
    """Convert Windows PCM audio into 16 kHz mono WAV."""

    samples = np.frombuffer(
        window.pcm, dtype="<i2"
    )

    # Downmix all channels to one mono signal.
    mono = samples.reshape(
        -1, window.channels
    ).astype(np.float32).mean(axis=1)

    if window.sample_rate != TARGET_RATE:
        factor = gcd(window.sample_rate, TARGET_RATE)

        mono = resample_poly(
            mono,
            TARGET_RATE // factor,
            window.sample_rate // factor,
        )

    mono = np.clip(
        np.rint(mono), -32768, 32767
    ).astype("<i2")

    output = BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(TARGET_RATE)
        wav.writeframes(mono.tobytes())

    return output.getvalue()


class WhisperClient:
    """Client for the local whisper.cpp HTTP server."""

    def __init__(
        self,
        url="http://127.0.0.1:8080/inference",
        language="auto",
    ):
        self.url = url
        self.language = language
        self.session = requests.Session()

    def transcribe(self, window: AudioWindow) -> str:
        audio = encode_wav(window)

        response = self.session.post(
            self.url,
            files={
                "file": (
                    "audio.wav",
                    audio,
                    "audio/wav",
                )
            },
            data={
                "language": self.language,
                "response_format": "json",
                "translate": "false",
                "no_timestamps": "true",
            },
            timeout=(3, 90),
        )

        response.raise_for_status()

        result = response.json()

        if not isinstance(result, dict):
            raise ValueError("Unexpected Whisper response.")

        return result.get("text", "").strip()