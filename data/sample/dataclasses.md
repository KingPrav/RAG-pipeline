# Python Dataclasses

## Introduction

The `@dataclass` decorator (Python 3.7+) auto-generates `__init__`, `__repr__`, and `__eq__` from class-level field annotations, eliminating boilerplate.

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p = Point(1.0, 2.5)
print(p)           # Point(x=1.0, y=2.5)
print(p == Point(1.0, 2.5))  # True
```

## Default Values

```python
@dataclass
class Config:
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
```

For mutable defaults, use `field(default_factory=...)`:

```python
from dataclasses import dataclass, field

@dataclass
class Pipeline:
    steps: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

## Frozen Dataclasses (Immutable)

`frozen=True` makes instances immutable and hashable:

```python
@dataclass(frozen=True)
class Vector:
    x: float
    y: float

v = Vector(1.0, 2.0)
# v.x = 3.0  # raises FrozenInstanceError
```

## Post-Init Processing

`__post_init__` runs after `__init__` for derived fields or validation:

```python
@dataclass
class Circle:
    radius: float
    area: float = field(init=False)

    def __post_init__(self):
        if self.radius < 0:
            raise ValueError("radius must be non-negative")
        self.area = 3.14159 * self.radius ** 2
```

## Inheritance

```python
@dataclass
class Animal:
    name: str
    species: str

@dataclass
class Pet(Animal):
    owner: str
```

Child class fields must come after parent fields in `__init__`. Fields with defaults cannot precede fields without defaults.

## `field()` Options

| Option | Description |
|---|---|
| `default` | Scalar default value |
| `default_factory` | Callable returning the default |
| `repr` | Include in `__repr__` (default True) |
| `compare` | Include in `__eq__` and ordering (default True) |
| `hash` | Include in `__hash__` |
| `init` | Include as `__init__` parameter (default True) |
| `metadata` | Arbitrary metadata dict for frameworks |

## `asdict` and `astuple`

```python
from dataclasses import asdict, astuple

@dataclass
class Point:
    x: float
    y: float

p = Point(1.0, 2.0)
print(asdict(p))    # {'x': 1.0, 'y': 2.0}
print(astuple(p))   # (1.0, 2.0)
```

## Comparison with Pydantic

| Feature | dataclass | Pydantic BaseModel |
|---|---|---|
| Runtime validation | No | Yes |
| Serialization | Manual | Built-in `.model_dump()` |
| JSON schema | No | Auto-generated |
| Performance | Faster | Slightly slower |
| Stdlib | Yes | Third-party |
