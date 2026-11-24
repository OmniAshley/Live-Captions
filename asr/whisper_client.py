
from io import BytesIO
import wave

import requests

from audio.frames import SAMPLE_RATE


def pcm_to_wav(pcm: bytes) -> bytes:
    """Encode 16-bit mono PCM as an in-memory WAV."""

    output = BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)

    return output.getvalue()


class WhisperClient:
    """Reusable client for our existing Whisper server."""

    def __init__(
        self,
        url="http://127.0.0.1:8080/inference",
        language="auto",
    ):
        self.url = url
        self.language = language
        self.session = requests.Session()

    def transcribe(self, clip) -> str:
        wav_bytes = pcm_to_wav(clip.pcm)

        response = self.session.post(
            self.url,
            files={
                "file": (
                    "speech.wav",
                    wav_bytes,
                    "audio/wav",
                )
            },
            data={
                "language": self.language,
                "translate": "false",
                "response_format": "json",
                "no_timestamps": "true",
                "token_timestamps": "false",
                "temperature": "0.0",
                "suppress_nst": "true",
            },
            timeout=(3, 90),
        )

        response.raise_for_status()
        result = response.json()

        if not isinstance(result, dict):
            raise ValueError(
                "Unexpected Whisper server response."
            )

        text = result.get("text", "")

        if not isinstance(text, str):
            raise ValueError(
                "Unexpected transcription format."
            )

        return text.strip()