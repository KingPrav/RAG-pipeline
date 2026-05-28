# Python Async/Await

## Defining Coroutines

A coroutine function is defined with `async def`. Calling it returns a coroutine object; it does not execute until awaited.

```python
async def fetch(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()
```

## Running Coroutines

Use `asyncio.run()` as the entry point for async programs. It creates a new event loop, runs the coroutine, and closes the loop.

```python
import asyncio

async def main():
    data = await fetch("https://example.com/api")
    print(data)

asyncio.run(main())
```

## asyncio.gather — Concurrent Execution

`asyncio.gather` schedules multiple coroutines to run concurrently on the same event loop. Total time ≈ slowest coroutine, not sum of all.

```python
async def main():
    results = await asyncio.gather(
        fetch("https://api.example.com/a"),
        fetch("https://api.example.com/b"),
        fetch("https://api.example.com/c"),
    )
    # results is a list in the same order as the inputs
```

Pass `return_exceptions=True` to avoid cancelling all tasks on the first failure:

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
errors = [r for r in results if isinstance(r, Exception)]
```

## asyncio.TaskGroup (Python 3.11+)

`TaskGroup` is the preferred way to manage multiple tasks because it propagates exceptions immediately and cancels siblings automatically.

```python
async def main():
    async with asyncio.TaskGroup() as tg:
        task_a = tg.create_task(fetch("https://a.example.com"))
        task_b = tg.create_task(fetch("https://b.example.com"))
    # Both tasks are done here
    print(task_a.result(), task_b.result())
```

## Async Context Managers

Objects with `__aenter__` / `__aexit__` can be used with `async with`:

```python
async with aiofiles.open("data.json") as f:
    content = await f.read()
```

## Async Generators

```python
async def paginate(client, url: str):
    while url:
        page = await client.get(url)
        data = page.json()
        for item in data["items"]:
            yield item
        url = data.get("next")

async for item in paginate(client, "/api/items"):
    process(item)
```

## Pitfalls

- **Blocking I/O in async code** — `time.sleep`, `requests.get`, or any synchronous I/O blocks the event loop. Use `await asyncio.sleep()` and `httpx.AsyncClient` instead.
- **Missing await** — `result = fetch(url)` gives you a coroutine object, not the result.
- **Thread safety** — the event loop is single-threaded; use `asyncio.run_in_executor` for CPU-bound work.
