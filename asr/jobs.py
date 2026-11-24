
from collections import deque
from threading import Condition


class JobMailbox:
    """Priority mailbox for speech recognition jobs."""

    def __init__(self, max_finals=8):
        self.condition = Condition()

        self.finals = deque()
        self.latest_preview = None

        self.max_finals = max_finals
        self.dropped_finals = 0
        self.closed = False

    def submit(self, clip):
        with self.condition:
            if self.closed:
                return

            if clip.stage == "final":
                if len(self.finals) >= self.max_finals:
                    self.finals.popleft()
                    self.dropped_finals += 1

                self.finals.append(clip)

                if (
                    self.latest_preview is not None
                    and self.latest_preview.utterance_id
                    == clip.utterance_id
                ):
                    self.latest_preview = None

            else:
                # Replace obsolete provisional work.
                self.latest_preview = clip

            self.condition.notify()

    def receive(self):
        with self.condition:
            while (
                not self.finals
                and self.latest_preview is None
                and not self.closed
            ):
                self.condition.wait()

            if self.finals:
                return self.finals.popleft()

            if self.latest_preview is not None:
                clip = self.latest_preview
                self.latest_preview = None
                return clip

            return None

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()