from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from residual_stream_lab.checkpointing import (
    build_checkpoints,
    retrieve_checkpoints,
    split_windows,
)
from residual_stream_lab.llm import GGUFRunner
from residual_stream_lab.synthetic import BenchmarkCase, build_benchmark_case
from residual_stream_lab.temporal import rerank_checkpoints

app = typer.Typer(add_completion=False)
console = Console()


def select_checkpoints(
    mode: str,
    benchmark: BenchmarkCase,
    query: str,
    recent_windows: int,
    top_k: int,
    runner: GGUFRunner,
) -> tuple[list[str], list[int]]:
    windows = split_windows(benchmark.document, lines_per_window=benchmark.window_lines)
    checkpoints = build_checkpoints(windows, runner.embed)
    recent = windows[-recent_windows:] if recent_windows > 0 else []
    recent_ids = {window.index for window in recent}
    recent_checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint.window.index in recent_ids]

    if mode == "full":
        selected = checkpoints
    elif mode == "recent":
        selected = recent_checkpoints
    elif mode == "retrieval":
        query_embedding = runner.embed(query)
        selected = retrieve_checkpoints(
            checkpoints,
            query_embedding=query_embedding,
            top_k=top_k,
            exclude_window_ids=recent_ids,
        )
        selected.extend(recent_checkpoints)
    elif mode == "temporal":
        query_embedding = runner.embed(query)
        selected = rerank_checkpoints(
            checkpoints,
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k,
            exclude_window_ids=recent_ids,
            recent_windows=recent_checkpoints,
        )
        selected.extend(recent_checkpoints)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode in {"full", "recent"}:
        memory_blocks = [checkpoint.window.text for checkpoint in selected]
    else:
        memory_blocks = [checkpoint.memory_packet() for checkpoint in selected[:-len(recent_checkpoints) or None]]
        memory_blocks.extend(checkpoint.window.text for checkpoint in recent_checkpoints)
    return "\n\n".join(memory_blocks), [checkpoint.window.index for checkpoint in selected]


def exact_match(prediction: str, answer: str) -> bool:
    left = prediction.strip().lower().strip(".")
    right = answer.strip().lower().strip(".")
    return left == right


@app.command()
def benchmark(
    model_path: str = typer.Option(..., help="Path to the GGUF model."),
    windows: int = typer.Option(8, min=4, help="Number of synthetic windows."),
    window_lines: int = typer.Option(6, min=2, help="Lines per synthetic window."),
    recent_windows: int = typer.Option(2, min=1, help="Exact local context horizon in windows."),
    top_k: int = typer.Option(2, min=1, help="Retrieved checkpoint count."),
    queries: int = typer.Option(8, min=1, help="Number of evaluation questions."),
    seed: int = typer.Option(7, help="Deterministic synthetic seed."),
    n_ctx: int = typer.Option(4096, min=1024, help="Inference context size."),
) -> None:
    benchmark_case = build_benchmark_case(
        windows=windows,
        window_lines=window_lines,
        queries=queries,
        seed=seed,
    )
    runner = GGUFRunner(model_path=model_path, n_ctx=n_ctx)

    modes = ["full", "recent", "retrieval", "temporal"]
    rows: list[dict[str, object]] = []

    console.print(
        "[bold]Residual stream sidecar benchmark[/bold]\n"
        "This measures exact recent context plus retrieved old-window checkpoints.\n"
        "It does not validate exact residual-layer reconstruction with this backend."
    )

    for mode in modes:
        correct = 0
        hits = 0
        predictions: list[str] = []
        for query_case in benchmark_case.queries:
            memory, selected_ids = select_checkpoints(
                mode=mode,
                benchmark=benchmark_case,
                query=query_case.question,
                recent_windows=recent_windows,
                top_k=top_k,
                runner=runner,
            )
            prediction = runner.answer_question(memory=memory, question=query_case.question)
            predictions.append(prediction)
            correct += int(exact_match(prediction, query_case.answer))
            hits += int(query_case.window_index in selected_ids)
        rows.append(
            {
                "mode": mode,
                "accuracy": correct / len(benchmark_case.queries),
                "hit_rate": hits / len(benchmark_case.queries),
                "sample_prediction": predictions[0] if predictions else "",
            }
        )

    table = Table(title="Benchmark Results")
    table.add_column("Mode")
    table.add_column("Accuracy")
    table.add_column("Target Hit Rate")
    table.add_column("Sample Prediction")
    for row in rows:
        table.add_row(
            str(row["mode"]),
            f"{row['accuracy']:.2f}",
            f"{row['hit_rate']:.2f}",
            str(row["sample_prediction"]),
        )
    console.print(table)

    delta = next(row["accuracy"] for row in rows if row["mode"] == "retrieval") - next(
        row["accuracy"] for row in rows if row["mode"] == "recent"
    )
    console.print(f"Retrieval lift over recent-only: {delta:+.2f}")
    temporal_delta = next(row["accuracy"] for row in rows if row["mode"] == "temporal") - next(
        row["accuracy"] for row in rows if row["mode"] == "retrieval"
    )
    console.print(f"Temporal rerank lift over semantic retrieval: {temporal_delta:+.2f}")


if __name__ == "__main__":
    app()
