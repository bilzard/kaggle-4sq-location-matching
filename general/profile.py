import time

from humanize import naturaldelta


class SimpleTimer:
    def __init__(self):
        self.section = None
        self.start_sec = None
        self.end_sec = None

    def start(self, section):
        self.section = section
        self.start_sec = time.time()

    def end(self):
        if self.start is None:
            raise RuntimeError("Timer should be started")
        self.end_sec = time.time()

    def show(self):
        if self.start is None:
            raise RuntimeError("Timer should be started")
        if self.end is None:
            raise RuntimeError("Timer should be ended")
        print(
            f"Elapsed time on {self.section}: {naturaldelta(self.end_sec - self.start_sec)}"
        )

    def endshow(self):
        self.end()
        self.show()
