
from collections import deque
from dataclasses import dataclass

import webrtcvad

from audio.frames import AudioFrame, SAMPLE_RATE, FRAME_MS


# Initial performance configuration.
VAD_MODE = 2

START_TRIGGER_MS = 60
PRE_ROLL_MS = 200

END_SILENCE_MS = 400
MIN_SPEECH_MS = 200

FIRST_PREVIEW_MS = 1200
PREVIEW_INTERVAL_MS = 1200

MAX_SPEECH_SECONDS = 6


@dataclass(frozen=True)
class SpeechClip:
    utterance_id: int
    stage: str
    pcm: bytes
    start_s: float
    end_s: float
    captured_at: float


class SpeechSegmenter:
    """Incremental VAD-based speech segmentation."""

    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_MODE)

        self.pre_roll = deque(
            maxlen=PRE_ROLL_MS // FRAME_MS
        )

        self.frames = []
        self.utterance_id = 0

        self.start_streak = 0
        self.voiced_frames = 0
        self.silent_frames = 0
        self.last_preview_count = 0

    def _make_clip(self, stage):
        return SpeechClip(
            utterance_id=self.utterance_id,
            stage=stage,
            pcm=b"".join(f.pcm for f in self.frames),
            start_s=self.frames[0].start_s,
            end_s=self.frames[-1].end_s,
            captured_at=self.frames[-1].captured_at,
        )

    def _reset(self):
        self.frames = []
        self.pre_roll.clear()

        self.start_streak = 0
        self.voiced_frames = 0
        self.silent_frames = 0
        self.last_preview_count = 0

    def push(self, frame: AudioFrame) -> list[SpeechClip]:
        events = []

        is_speech = self.vad.is_speech(
            frame.pcm,
            SAMPLE_RATE,
        )

        # Waiting for speech.
        if not self.frames:
            self.pre_roll.append(frame)

            if is_speech:
                self.start_streak += 1
            else:
                self.start_streak = 0

            trigger = START_TRIGGER_MS // FRAME_MS

            if self.start_streak >= trigger:
                self.utterance_id += 1

                # Preserve audio just before detection.
                self.frames = list(self.pre_roll)
                self.pre_roll.clear()

                self.voiced_frames = self.start_streak
                self.silent_frames = 0
                self.last_preview_count = 0

            return events

        # Recording an active speech segment.
        self.frames.append(frame)

        if is_speech:
            self.voiced_frames += 1
            self.silent_frames = 0
        else:
            self.silent_frames += 1

        elapsed_ms = len(self.frames) * FRAME_MS

        reached_pause = (
            self.silent_frames * FRAME_MS
            >= END_SILENCE_MS
        )

        reached_limit = (
            elapsed_ms >= MAX_SPEECH_SECONDS * 1000
        )

        # Finalise after a pause or maximum duration.
        if reached_pause or reached_limit:
            if (
                self.voiced_frames * FRAME_MS
                >= MIN_SPEECH_MS
            ):
                events.append(
                    self._make_clip("final")
                )

            self._reset()
            return events

        # Provisional recognition during speech.
        preview_due = (
            elapsed_ms >= FIRST_PREVIEW_MS
            and elapsed_ms - self.last_preview_count * FRAME_MS
            >= PREVIEW_INTERVAL_MS
            and self.silent_frames < 3
        )

        if preview_due:
            events.append(
                self._make_clip("preview")
            )

            self.last_preview_count = len(self.frames)

        return events