import time
from contextlib import contextmanager


@contextmanager
def timer_context(name: str = "function"):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{name} cost time: {end_time - start_time:.4f} s")
