
from time import perf_counter

import numpy as np

from audio.loopback import stream_system_audio


def main():
    print("LiveCaption V0.1 - Audio Test")
    print("Play a video to test capture.")
    print("Press Ctrl+C to stop.\n")

    last_update = perf_counter()

    try:
        for chunk in stream_system_audio():
            now = perf_counter()

            if now - last_update < 0.15:
                continue

            samples = np.frombuffer(
                chunk.pcm, dtype=np.int16
            ).astype(np.float32)

            rms = np.sqrt(np.mean(samples ** 2))
            level = min(30, int(rms / 32768 * 300))

            bar = "█" * level
            bar = bar.ljust(30, "░")

            print(f"\rAudio: [{bar}]", end="", flush=True)

            last_update = now

    except KeyboardInterrupt:
        print("\nAudio capture stopped.")


if __name__ == "__main__":
    main()
