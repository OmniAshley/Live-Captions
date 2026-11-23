
from queue import Queue, Empty, Full
from threading import Thread
from time import perf_counter

import numpy as np
import requests

from audio.loopback import stream_system_audio
from audio.windows import fixed_windows
from asr.whisper_client import WhisperClient


WINDOW_SECONDS = 3.0
MAX_PENDING_WINDOWS = 2
SILENCE_RMS = 0.002


def recognition_worker(jobs: Queue):
    recognizer = WhisperClient(language="auto")

    while True:
        window = jobs.get()

        if window is None:
            break

        # Skip extremely quiet audio.
        samples = np.frombuffer(
            window.pcm, dtype="<i2"
        ).astype(np.float32)

        rms = np.sqrt(
            np.mean(samples * samples)
        ) / 32768.0

        if rms < SILENCE_RMS:
            continue

        started = perf_counter()

        try:
            text = recognizer.transcribe(window)

            inference_time = perf_counter() - started
            lag = perf_counter() - window.captured_at

            if text:
                print(
                    f"\n[{window.start_s:.1f}s - "
                    f"{window.end_s:.1f}s] {text}"
                )

                print(
                    f"  Inference: {inference_time:.2f}s"
                    f" | Post-capture lag: {lag:.2f}s"
                )

        except (requests.RequestException, ValueError) as exc:
            print(f"\n[ASR ERROR] {exc}")


def main():
    print("=" * 50)
    print("LiveCaption V0.2 - Multilingual ASR")
    print("=" * 50)
    print("Engine: whisper.cpp")
    print("Language: Automatic detection")
    print(f"Audio window: {WINDOW_SECONDS}s")
    print("Press Ctrl+C to stop.\n")

    jobs = Queue(maxsize=MAX_PENDING_WINDOWS)

    worker = Thread(
        target=recognition_worker,
        args=(jobs,),
        daemon=True,
    )

    worker.start()

    try:
        chunks = stream_system_audio()
        windows = fixed_windows(
            chunks, seconds=WINDOW_SECONDS
        )

        for window in windows:
            try:
                jobs.put_nowait(window)

            except Full:
                # Keep recent audio instead of accumulating lag.
                try:
                    jobs.get_nowait()
                except Empty:
                    pass

                try:
                    jobs.put_nowait(window)
                except Full:
                    pass

                print(
                    "\n[WARNING] Dropped an audio window "
                    "to limit backlog."
                )

    except KeyboardInterrupt:
        print("\nCapture stopped.")

    finally:
        try:
            jobs.put_nowait(None)
        except Full:
            pass


if __name__ == "__main__":
    main()