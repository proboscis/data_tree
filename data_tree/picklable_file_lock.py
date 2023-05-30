from dataclasses import dataclass

from filelock import FileLock


@dataclass
class PicklableFileLock:
    path: str

    def __post_init__(self):
        self.lock = FileLock(self.path)

    def __getstate__(self):
        return self.path

    def __setstate__(self, state):
        self.path = state
        self.lock = FileLock(state)

    def __enter__(self):
        return self.lock.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.lock.__exit__(exc_type, exc_val, exc_tb)
