
import sys
from threading import Thread
from time import perf_counter

import requests

from audio.frames import stream_frames
from audio.vad import SpeechSegmenter
from asr.jobs import JobMailbox
from asr.whisper_client import WhisperClient


def clear_preview():
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()


def recognition_worker(mailbox: JobMailbox):
    recognizer = WhisperClient(language="auto")
    last_finalized_id = 0

    while True:
        clip = mailbox.receive()

        if clip is None:
            break

        if (
            clip.stage == "preview"
            and clip.utterance_id <= last_finalized_id
        ):
            continue

        started = perf_counter()

        try:
            text = recognizer.transcribe(clip)

        except (requests.RequestException, ValueError) as exc:
            clear_preview()
            print(f"\n[ASR ERROR] {exc}")
            continue

        finished = perf_counter()

        inference_s = finished - started
        audio_s = (clip.end_s - clip.start_s)

        rtf = (
            inference_s / audio_s
            if audio_s > 0 else 0.0
        )

        # Approximate lag since the final audio frame
        # was captured, not since the phrase began.
        capture_lag_s = finished - clip.captured_at

        if clip.stage == "preview":
            if not text:
                continue

            if sys.stdout.isatty():
                display = (
                    text[:90] + "..."
                    if len(text) > 90 else text
                )
                sys.stdout.write(
                    "\r\033[2K"
                    f"[PREVIEW #{clip.utterance_id}] "
                    f"{display}"
                )
                sys.stdout.flush()

            else:
                print(
                    f"[PREVIEW #{clip.utterance_id}] {text}"
                )

        else:
            last_finalized_id = max(
                last_finalized_id,
                clip.utterance_id,
            )

            clear_preview()

            if text:
                print(
                    f"\n[FINAL #{clip.utterance_id}]"
                    f" [{clip.start_s:.2f}-"
                    f"{clip.end_s:.2f}s]"
                )
                print(f"  {text}")
                print(
                    f"  Inference: {inference_s:.2f}s"
                    f" | RTF: {rtf:.2f}"
                    f" | Lag after capture:"
                    f" {capture_lag_s:.2f}s\n"
                )


def main():
    print("=" * 55)
    print("LiveCaption V0.2.1 - Low-Latency ASR")
    print("=" * 55)
    print("Environment: live-captions")
    print("ASR engine: whisper.cpp")
    print("Language: Auto")
    print("Audio: 16 kHz / mono / 20 ms")
    print("VAD: WebRTC")
    print("Press Ctrl+C to stop.\n")

    segmenter = SpeechSegmenter()
    mailbox = JobMailbox()

    worker = Thread(
        target=recognition_worker,
        args=(mailbox,),
        daemon=True,
        name="WhisperWorker",
    )

    worker.start()

    try:
        for frame in stream_frames():
            events = segmenter.push(frame)

            for clip in events:
                mailbox.submit(clip)

    except KeyboardInterrupt:
        print("\nStopping audio capture...")

    finally:
        mailbox.close()

        if mailbox.dropped_finals:
            print(
                "Warning: "
                f"{mailbox.dropped_finals} final "
                "segments were dropped."
            )

        print("LiveCaption stopped.")


if __name__ == "__main__":
    main()