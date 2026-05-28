from __future__ import annotations

from src.evaluation.test_dataset import TEST_CASES
from src.models import RAGRequest


def run_evaluation(pipeline) -> dict:  # type: ignore[type-arg]
    """Run RAGAS evaluation over TEST_CASES and return a metrics dict."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as e:
        raise ImportError(
            "Install eval extras: pip install 'rag-pipeline[eval]'"
        ) from e

    data: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print(f"Evaluating {len(TEST_CASES)} test cases...")
    for case in TEST_CASES:
        response = pipeline.query(RAGRequest(query=case["question"]))
        data["question"].append(case["question"])
        data["answer"].append(response.answer)
        data["contexts"].append([c.document.content for c in response.sources])
        data["ground_truth"].append(case["ground_truth"])

    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    scores = result.to_pandas()[["faithfulness", "answer_relevancy", "context_precision"]]
    return {
        "per_question": scores.to_dict(orient="records"),
        "mean": scores.mean().to_dict(),
    }
