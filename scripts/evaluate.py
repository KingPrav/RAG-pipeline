#!/usr/bin/env python3
"""Run RAGAS evaluation over the test dataset.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --mode hybrid   # default
  python scripts/evaluate.py --mode dense
  python scripts/evaluate.py --mode sparse
"""
from __future__ import annotations
import argparse
import sys
import os
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table

from src.evaluation.ragas_eval import run_evaluation
from src.pipeline import RAGPipeline

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "dense", "sparse"],
        default="hybrid",
    )
    args = parser.parse_args()

    console.rule(f"[bold blue]RAGAS Evaluation — mode: {args.mode}")

    pipeline = RAGPipeline()
    # Patch default retrieval mode
    original_query = pipeline.query

    def patched_query(request):
        request.retrieval_mode = args.mode  # type: ignore[assignment]
        return original_query(request)

    pipeline.query = patched_query  # type: ignore[method-assign]

    results = run_evaluation(pipeline)

    # Per-question table
    table = Table(title="Per-question scores", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Faithfulness", justify="right")
    table.add_column("Ans. Relevancy", justify="right")
    table.add_column("Ctx. Precision", justify="right")

    for i, row in enumerate(results["per_question"], 1):
        table.add_row(
            str(i),
            f"{row.get('faithfulness', 0):.3f}",
            f"{row.get('answer_relevancy', 0):.3f}",
            f"{row.get('context_precision', 0):.3f}",
        )

    console.print(table)

    # Mean scores
    mean = results["mean"]
    summary = Table(title="Mean scores", show_header=False)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Score", style="green", justify="right")
    for metric, score in mean.items():
        summary.add_row(metric.replace("_", " ").title(), f"{score:.3f}")

    console.print(summary)


if __name__ == "__main__":
    main()
