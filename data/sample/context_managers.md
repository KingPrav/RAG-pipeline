# Python Context Managers

## The Protocol

A context manager implements `__enter__` (called on `with` entry) and `__exit__` (called on exit, always — even on exceptions).

```python
class ManagedResource:
    def __enter__(self):
        self._acquire()
        return self          # value bound to `as` target

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._release()
        return False         # False = don't suppress exceptions
```

## contextlib.contextmanager

Converts a generator function into a context manager. Code before `yield` is `__enter__`, code after `yield` is `__exit__`.

```python
from contextlib import contextmanager

@contextmanager
def timer(label: str):
    import time
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.3f}s")

with timer("processing"):
    heavy_computation()
```

The `yield` value becomes the `as` target:

```python
@contextmanager
def temp_directory():
    import tempfile, shutil
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)

with temp_directory() as tmpdir:
    write_files(tmpdir)
    # tmpdir is cleaned up automatically
```

## contextlib.asynccontextmanager

Async version for use with `async with`:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_connection(url: str):
    conn = await create_connection(url)
    try:
        yield conn
    finally:
        await conn.close()

async with managed_connection("postgresql://...") as conn:
    await conn.execute("SELECT 1")
```

## contextlib.suppress

Suppresses specified exceptions:

```python
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove("temp.txt")   # silently skipped if file doesn't exist
```

## contextlib.ExitStack

Dynamically compose an arbitrary number of context managers:

```python
from contextlib import ExitStack

files = ["a.txt", "b.txt", "c.txt"]
with ExitStack() as stack:
    handles = [stack.enter_context(open(f)) for f in files]
    process(handles)
# all three files are closed here
```

## Exception Handling in __exit__

`__exit__` receives the exception info if one was raised. Return `True` to suppress it:

```python
class IgnoreKeyboardInterrupt:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return exc_type is KeyboardInterrupt   # suppresses Ctrl-C only
```

## Reusable Context Managers with contextlib.AbstractContextManager

```python
from contextlib import AbstractContextManager

class Retry(AbstractContextManager):
    def __init__(self, attempts: int):
        self.attempts = attempts
        self.attempt = 0

    def __enter__(self):
        self.attempt += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and self.attempt < self.attempts:
            return True   # suppress and retry
        return False
```
