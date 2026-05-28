# Python Type Hints

## Basic Syntax

Type hints annotate variables and function signatures. They are not enforced at runtime but enable static analysis with mypy, pyright, and IDE tooling.

```python
def greet(name: str, times: int = 1) -> str:
    return (f"Hello, {name}!\n") * times
```

## Built-in Generic Types (Python 3.9+)

Use lowercase built-ins directly — no need to import from `typing`:

```python
def process(items: list[int]) -> dict[str, list[int]]:
    ...
```

For older Python, import from `typing`:

```python
from typing import List, Dict, Tuple
def process(items: List[int]) -> Dict[str, List[int]]:
    ...
```

## Optional and Union

`Optional[X]` is shorthand for `X | None`:

```python
from typing import Optional

def find(key: str) -> Optional[str]:   # returns str or None
    ...

# Python 3.10+ union syntax
def find(key: str) -> str | None:
    ...
```

## TypeVar and Generics

```python
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T:
    return items[0]
```

## Callable

```python
from typing import Callable

def apply(fn: Callable[[int, int], int], a: int, b: int) -> int:
    return fn(a, b)
```

## TypedDict

```python
from typing import TypedDict

class Movie(TypedDict):
    title: str
    year: int
    rating: float
```

## Protocol — Structural Subtyping

Protocols define interfaces without inheritance:

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

def render(obj: Drawable) -> None:
    obj.draw()
```

Any class that implements `draw()` satisfies `Drawable` without explicitly inheriting from it.

## Literal

```python
from typing import Literal

Mode = Literal["read", "write", "append"]

def open_file(path: str, mode: Mode) -> None: ...
```

## Final and ClassVar

```python
from typing import Final, ClassVar

MAX_SIZE: Final = 100          # cannot be reassigned
class Config:
    instances: ClassVar[int] = 0   # class-level, not per-instance
```

## Runtime Behavior

Type hints are stored in `__annotations__` and are accessible at runtime:

```python
def fn(x: int) -> str: ...
print(fn.__annotations__)  # {'x': <class 'int'>, 'return': <class 'str'>}
```

`get_type_hints()` from `typing` resolves forward references (string annotations).
