# Evaluation Q&A pairs — domain-matched to the ingested corpus.
# Ground truths are reference answers used by RAGAS context_precision metric.
# Extend this list after ingesting your own documents.

TEST_CASES: list[dict[str, str]] = [
    {
        "question": "How do you define an async function in Python and what keyword do you use to call it?",
        "ground_truth": (
            "An async function is defined with 'async def'. It must be called with the "
            "'await' keyword inside another async function or run via asyncio.run()."
        ),
    },
    {
        "question": "What is the difference between asyncio.gather and sequential awaits?",
        "ground_truth": (
            "asyncio.gather runs multiple coroutines concurrently within a single event loop "
            "iteration, reducing total wall-clock time when the coroutines do I/O. Sequential "
            "awaits run each coroutine to completion before starting the next."
        ),
    },
    {
        "question": "How do Python type hints work with generic containers like List and Dict?",
        "ground_truth": (
            "Use 'list[int]' and 'dict[str, int]' (Python 3.9+) or 'List[int]' / 'Dict[str, int]' "
            "from typing (older). Type hints are not enforced at runtime but tools like mypy use them."
        ),
    },
    {
        "question": "What is a dataclass and how does it differ from a regular class?",
        "ground_truth": (
            "A dataclass (decorated with @dataclass) auto-generates __init__, __repr__, and __eq__ "
            "from annotated fields, removing boilerplate. Regular classes require writing these manually."
        ),
    },
    {
        "question": "How do you implement a context manager using the contextlib module?",
        "ground_truth": (
            "Use @contextlib.contextmanager on a generator function. Yield once: code before yield "
            "is the __enter__ body, code after yield (in a finally block) is the __exit__ body."
        ),
    },
]
