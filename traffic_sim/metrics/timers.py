import time
from contextlib import contextmanager


class Timer:
    """
    Context for time mesurement.
    """

    def __enter__(self):
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start


@contextmanager
def walltime():
    """
    Alternative context manager.
    """

    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[TIMER] Wall time: {end - start:.4f} s")
