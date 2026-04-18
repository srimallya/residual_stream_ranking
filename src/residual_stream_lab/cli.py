from __future__ import annotations

from collections import defaultdict
import re
from time import perf_counter

import typer
from rich.console import Console
from rich.table import Table

from residual_stream_lab.checkpointing import (
    build_checkpoints,
    retrieve_checkpoints,
    split_windows,
)
from residual_stream_lab.llm import GGUFRunner
from residual_stream_lab.apollo import ApolloCase, build_apollo_cases
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


def build_oracle_memory(
    benchmark: BenchmarkCase,
    target_window_index: int,
    recent_windows: int,
    runner: GGUFRunner,
) -> str:
    windows = split_windows(benchmark.document, lines_per_window=benchmark.window_lines)
    checkpoints = build_checkpoints(windows, runner.embed)
    recent = windows[-recent_windows:] if recent_windows > 0 else []
    recent_ids = {window.index for window in recent}
    target_checkpoint = next(
        checkpoint for checkpoint in checkpoints if checkpoint.window.index == target_window_index
    )
    recent_checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint.window.index in recent_ids]

    memory_blocks = [target_checkpoint.memory_packet()]
    memory_blocks.extend(checkpoint.window.text for checkpoint in recent_checkpoints)
    return "\n\n".join(memory_blocks)


def extract_atomic_answer(prediction: str, valid_answers: set[str]) -> str | None:
    lowered = prediction.strip().lower()
    if not lowered:
        return None

    lines = [line.strip(" -*`\t") for line in lowered.splitlines() if line.strip()]
    for line in reversed(lines):
        if line in valid_answers:
            return line
        for answer in valid_answers:
            if re.search(rf"\b{re.escape(answer)}\b", line):
                return answer

    for answer in valid_answers:
        if re.search(rf"\b{re.escape(answer)}\b", lowered):
            return answer
    return None


def evaluate_response(
    *,
    runner: GGUFRunner,
    memory: str,
    question: str,
    answer: str,
    valid_answers: set[str],
) -> dict[str, object]:
    started = perf_counter()
    prediction = runner.answer_question(memory=memory, question=question)
    latency_ms = (perf_counter() - started) * 1000.0
    parsed = extract_atomic_answer(prediction, valid_answers)
    return {
        "prediction": prediction,
        "parsed": parsed,
        "parse_success": parsed is not None,
        "correct": parsed == answer.lower(),
        "latency_ms": latency_ms,
        "memory_tokens": runner.token_count(memory),
    }


def exact_match(prediction: str, answer: str) -> bool:
    left = prediction.strip().lower().strip(".")
    right = answer.strip().lower().strip(".")
    return left == right


def evaluate_case_modes(
    *,
    cases: list[ApolloCase],
    runner: GGUFRunner,
    recent_windows: int,
    top_k: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, dict[str, float]]]]:
    valid_answers = {case.answer.lower() for case in cases}
    per_case: list[dict[str, object]] = []

    for case in cases:
        recent_memory, _ = select_checkpoints(
            mode="recent",
            benchmark=case,
            query=case.question,
            recent_windows=recent_windows,
            top_k=top_k,
            runner=runner,
        )
        recent_tokens = runner.token_count(recent_memory)

        full_memory, full_ids = select_checkpoints(
            mode="full",
            benchmark=case,
            query=case.question,
            recent_windows=recent_windows,
            top_k=top_k,
            runner=runner,
        )
        oracle_memory = build_oracle_memory(
            benchmark=case,
            target_window_index=case.target_window_index,
            recent_windows=recent_windows,
            runner=runner,
        )

        mode_results: dict[str, dict[str, object]] = {}
        for mode, memory, selected_ids in [
            ("full", full_memory, full_ids),
            ("recent", recent_memory, []),
            ("oracle", oracle_memory, [case.target_window_index]),
        ]:
            result = evaluate_response(
                runner=runner,
                memory=memory,
                question=case.question,
                answer=case.answer,
                valid_answers=valid_answers,
            )
            result["selected_ids"] = selected_ids
            result["hit"] = case.target_window_index in selected_ids if selected_ids else False
            result["replay_tokens"] = max(0, result["memory_tokens"] - recent_tokens)
            mode_results[mode] = result

        for mode in ["retrieval", "temporal"]:
            memory, selected_ids = select_checkpoints(
                mode=mode,
                benchmark=case,
                query=case.question,
                recent_windows=recent_windows,
                top_k=top_k,
                runner=runner,
            )
            result = evaluate_response(
                runner=runner,
                memory=memory,
                question=case.question,
                answer=case.answer,
                valid_answers=valid_answers,
            )
            result["selected_ids"] = selected_ids
            result["hit"] = case.target_window_index in selected_ids
            result["replay_tokens"] = max(0, result["memory_tokens"] - recent_tokens)
            mode_results[mode] = result

        retrieval_topk: dict[str, bool] = {}
        for k in [1, 2, 4]:
            _, topk_ids = select_checkpoints(
                mode="retrieval",
                benchmark=case,
                query=case.question,
                recent_windows=recent_windows,
                top_k=k,
                runner=runner,
            )
            retrieval_topk[f"top_{k}"] = case.target_window_index in topk_ids
        mode_results["retrieval"]["topk"] = retrieval_topk

        valid = mode_results["full"]["parse_success"] and mode_results["oracle"]["parse_success"]
        per_case.append(
            {
                "case_id": case.case_id,
                "distance_bin": case.distance_bin,
                "valid": valid,
                "results": mode_results,
            }
        )

    summary: dict[str, dict[str, dict[str, float]]] = {"overall": {}, "distance": {}}
    modes = ["full", "recent", "retrieval", "temporal", "oracle"]
    valid_cases = [entry for entry in per_case if entry["valid"]]
    valid_count = len(valid_cases)

    for mode in modes:
        parse_success = sum(entry["results"][mode]["parse_success"] for entry in per_case)
        correct = sum(entry["results"][mode]["correct"] for entry in valid_cases)
        hits = sum(entry["results"][mode]["hit"] for entry in valid_cases)
        correct_after_hit = sum(
            entry["results"][mode]["correct"] for entry in valid_cases if entry["results"][mode]["hit"]
        )
        avg_latency = sum(entry["results"][mode]["latency_ms"] for entry in per_case) / len(per_case)
        avg_replay = sum(entry["results"][mode]["replay_tokens"] for entry in per_case) / len(per_case)
        summary["overall"][mode] = {
            "valid_count": float(valid_count),
            "parse_rate": parse_success / len(per_case),
            "accuracy_valid": (correct / valid_count) if valid_count else 0.0,
            "hit_rate_valid": (hits / valid_count) if valid_count else 0.0,
            "usage_accuracy_when_hit": (correct_after_hit / hits) if hits else 0.0,
            "avg_latency_ms": avg_latency,
            "avg_replay_tokens": avg_replay,
        }

    if valid_count:
        retrieval_valid = [entry for entry in valid_cases]
        top1 = sum(entry["results"]["retrieval"]["topk"]["top_1"] for entry in retrieval_valid) / valid_count
        top2 = sum(entry["results"]["retrieval"]["topk"]["top_2"] for entry in retrieval_valid) / valid_count
        top4 = sum(entry["results"]["retrieval"]["topk"]["top_4"] for entry in retrieval_valid) / valid_count
        wrong_window = sum(not entry["results"]["retrieval"]["hit"] for entry in retrieval_valid) / valid_count
        right_unparsable = sum(
            entry["results"]["retrieval"]["hit"] and not entry["results"]["retrieval"]["parse_success"]
            for entry in retrieval_valid
        ) / valid_count
        right_parsed_wrong = sum(
            entry["results"]["retrieval"]["hit"]
            and entry["results"]["retrieval"]["parse_success"]
            and not entry["results"]["retrieval"]["correct"]
            for entry in retrieval_valid
        ) / valid_count
        summary["overall"]["retrieval_breakdown"] = {
            "top1_recall": top1,
            "top2_recall": top2,
            "top4_recall": top4,
            "wrong_window_rate": wrong_window,
            "right_unparsable_rate": right_unparsable,
            "right_parsed_wrong_rate": right_parsed_wrong,
        }
    else:
        summary["overall"]["retrieval_breakdown"] = {
            "top1_recall": 0.0,
            "top2_recall": 0.0,
            "top4_recall": 0.0,
            "wrong_window_rate": 0.0,
            "right_unparsable_rate": 0.0,
            "right_parsed_wrong_rate": 0.0,
        }

    by_distance: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in valid_cases:
        by_distance[entry["distance_bin"]].append(entry)
    for distance_bin, entries in by_distance.items():
        summary["distance"][distance_bin] = {}
        for mode in modes:
            hits = sum(entry["results"][mode]["hit"] for entry in entries)
            correct = sum(entry["results"][mode]["correct"] for entry in entries)
            summary["distance"][distance_bin][mode] = {
                "count": float(len(entries)),
                "accuracy_valid": correct / len(entries),
                "hit_rate_valid": hits / len(entries),
            }

    return per_case, summary


@app.command()
def benchmark(
    model_path: str = typer.Option(..., help="Path to the GGUF model."),
    windows: int = typer.Option(8, min=4, help="Number of synthetic windows."),
    window_lines: int = typer.Option(6, min=2, help="Lines per synthetic window."),
    recent_windows: int = typer.Option(2, min=0, help="Exact local context horizon in windows."),
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
    valid_answers = {query.answer.lower() for query in benchmark_case.queries}

    modes = ["full", "recent", "retrieval", "temporal"]
    rows: list[dict[str, object]] = []
    oracle_correct = 0
    oracle_parse_success = 0

    console.print(
        "[bold]Residual stream sidecar benchmark[/bold]\n"
        "This measures exact recent context plus retrieved old-window checkpoints.\n"
        "It does not validate exact residual-layer reconstruction with this backend."
    )
    console.print(f"Backend: {runner.backend} ({runner.backend_reason})")

    for query_case in benchmark_case.queries:
        oracle_memory = build_oracle_memory(
            benchmark=benchmark_case,
            target_window_index=query_case.window_index,
            recent_windows=recent_windows,
            runner=runner,
        )
        oracle_prediction = runner.answer_question(
            memory=oracle_memory,
            question=query_case.question,
        )
        parsed = extract_atomic_answer(oracle_prediction, valid_answers)
        oracle_parse_success += int(parsed is not None)
        oracle_correct += int(parsed == query_case.answer.lower())

    for mode in modes:
        correct = 0
        hits = 0
        parse_success = 0
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
            parsed = extract_atomic_answer(prediction, valid_answers)
            predictions.append(parsed or prediction)
            parse_success += int(parsed is not None)
            correct += int(parsed == query_case.answer.lower())
            hits += int(query_case.window_index in selected_ids)
        rows.append(
            {
                "mode": mode,
                "accuracy": correct / len(benchmark_case.queries),
                "hit_rate": hits / len(benchmark_case.queries),
                "parse_rate": parse_success / len(benchmark_case.queries),
                "sample_prediction": predictions[0] if predictions else "",
            }
        )

    table = Table(title="Benchmark Results")
    table.add_column("Mode")
    table.add_column("Accuracy")
    table.add_column("Target Hit Rate")
    table.add_column("Parse Rate")
    table.add_column("Sample Prediction")
    for row in rows:
        table.add_row(
            str(row["mode"]),
            f"{row['accuracy']:.2f}",
            f"{row['hit_rate']:.2f}",
            f"{row['parse_rate']:.2f}",
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
    oracle_accuracy = oracle_correct / len(benchmark_case.queries)
    oracle_parse_rate = oracle_parse_success / len(benchmark_case.queries)
    console.print(f"Oracle-correct memory accuracy: {oracle_accuracy:.2f}")
    console.print(f"Oracle parse success rate: {oracle_parse_rate:.2f}")
    retrieval_gap = oracle_accuracy - next(
        row["accuracy"] for row in rows if row["mode"] == "retrieval"
    )
    temporal_gap = oracle_accuracy - next(
        row["accuracy"] for row in rows if row["mode"] == "temporal"
    )
    console.print(f"Retrieval-to-oracle gap: {retrieval_gap:+.2f}")
    console.print(f"Temporal-to-oracle gap: {temporal_gap:+.2f}")

    full_accuracy = next(row["accuracy"] for row in rows if row["mode"] == "full")
    full_parse_rate = next(row["parse_rate"] for row in rows if row["mode"] == "full")
    comparison_valid = full_accuracy == 1.0 and oracle_accuracy == 1.0 and full_parse_rate == 1.0 and oracle_parse_rate == 1.0
    console.print(f"Memory comparison valid: {'yes' if comparison_valid else 'no'}")
    if not comparison_valid:
        console.print(
            "Memory mode deltas are not interpretable because the model failed the full-context "
            "or oracle answer contract."
        )


@app.command("benchmark-apollo")
def benchmark_apollo(
    model_path: str = typer.Option(..., help="Path to the GGUF model."),
    corpus_path: str = typer.Option(
        "data/apollo11_clean.txt",
        help="Path to the Apollo corpus text file.",
    ),
    case_count: int = typer.Option(12, min=3, help="Number of generated benchmark cases."),
    windows: int = typer.Option(12, min=6, help="Number of windows per case."),
    window_lines: int = typer.Option(24, min=8, help="Lines per window."),
    recent_windows: int = typer.Option(2, min=0, help="Exact local context horizon in windows."),
    top_k: int = typer.Option(2, min=1, help="Retrieved checkpoint count."),
    seed: int = typer.Option(7, help="Deterministic benchmark seed."),
    n_ctx: int = typer.Option(4096, min=1024, help="Inference context size."),
) -> None:
    runner = GGUFRunner(model_path=model_path, n_ctx=n_ctx)
    cases = build_apollo_cases(
        corpus_path=corpus_path,
        case_count=case_count,
        windows=windows,
        window_lines=window_lines,
        recent_windows=recent_windows,
        seed=seed,
    )
    per_case, summary = evaluate_case_modes(
        cases=cases,
        runner=runner,
        recent_windows=recent_windows,
        top_k=top_k,
    )

    valid_count = sum(1 for entry in per_case if entry["valid"])
    console.print(
        "[bold]Apollo Corpus Benchmark[/bold]\n"
        "Corpus-backed semantic sidecar evaluation with parse-valid full/oracle gating."
    )
    console.print(f"Backend: {runner.backend} ({runner.backend_reason})")
    console.print(f"Cases: {len(cases)} | Valid cases: {valid_count}")

    table = Table(title="Overall Metrics")
    table.add_column("Mode")
    table.add_column("Valid Count")
    table.add_column("Parse Rate")
    table.add_column("Accuracy (Valid)")
    table.add_column("Hit Rate (Valid)")
    table.add_column("Usage @ Hit")
    table.add_column("Avg Replay Tokens")
    table.add_column("Avg Latency ms")
    for mode in ["full", "recent", "retrieval", "temporal", "oracle"]:
        stats = summary["overall"][mode]
        table.add_row(
            mode,
            str(int(stats["valid_count"])),
            f"{stats['parse_rate']:.2f}",
            f"{stats['accuracy_valid']:.2f}",
            f"{stats['hit_rate_valid']:.2f}",
            f"{stats['usage_accuracy_when_hit']:.2f}",
            f"{stats['avg_replay_tokens']:.1f}",
            f"{stats['avg_latency_ms']:.1f}",
        )
    console.print(table)

    distance_table = Table(title="Distance Breakdown")
    distance_table.add_column("Distance")
    distance_table.add_column("Mode")
    distance_table.add_column("Count")
    distance_table.add_column("Accuracy")
    distance_table.add_column("Hit Rate")
    for distance_bin in ["near", "medium", "far"]:
        if distance_bin not in summary["distance"]:
            continue
        for mode in ["recent", "retrieval", "temporal", "oracle"]:
            stats = summary["distance"][distance_bin][mode]
            distance_table.add_row(
                distance_bin,
                mode,
                str(int(stats["count"])),
                f"{stats['accuracy_valid']:.2f}",
                f"{stats['hit_rate_valid']:.2f}",
            )
    console.print(distance_table)

    full_parse = summary["overall"]["full"]["parse_rate"]
    oracle_parse = summary["overall"]["oracle"]["parse_rate"]
    comparison_valid = full_parse == 1.0 and oracle_parse == 1.0
    console.print(f"Memory comparison valid: {'yes' if comparison_valid else 'no'}")
    if not comparison_valid:
        console.print(
            "At least one sample failed the full/oracle answer contract, so memory deltas must be interpreted cautiously."
        )

    breakdown = summary["overall"]["retrieval_breakdown"]
    breakdown_table = Table(title="Retrieval Breakdown")
    breakdown_table.add_column("Metric")
    breakdown_table.add_column("Value")
    for label, value in [
        ("Top-1 recall", breakdown["top1_recall"]),
        ("Top-2 recall", breakdown["top2_recall"]),
        ("Top-4 recall", breakdown["top4_recall"]),
        ("Wrong window rate", breakdown["wrong_window_rate"]),
        ("Right window, unparsable", breakdown["right_unparsable_rate"]),
        ("Right window, parsed wrong", breakdown["right_parsed_wrong_rate"]),
    ]:
        breakdown_table.add_row(label, f"{value:.2f}")
    console.print(breakdown_table)


@app.command("sweep-apollo")
def sweep_apollo(
    model_path: str = typer.Option(..., help="Path to the GGUF model."),
    corpus_path: str = typer.Option("data/apollo11_clean.txt", help="Path to the Apollo corpus text file."),
    case_count: int = typer.Option(9, min=3, help="Number of generated benchmark cases."),
    windows: int = typer.Option(12, min=6, help="Number of windows per case."),
    window_lines: int = typer.Option(24, min=8, help="Lines per window."),
    recent_windows_values: str = typer.Option("0,1,2,4", help="Comma-separated recent window values."),
    top_k_values: str = typer.Option("1,2,3,4", help="Comma-separated top-k values."),
    seed: int = typer.Option(7, help="Deterministic benchmark seed."),
    n_ctx: int = typer.Option(4096, min=1024, help="Inference context size."),
) -> None:
    runner = GGUFRunner(model_path=model_path, n_ctx=n_ctx)
    top_ks = [int(value.strip()) for value in top_k_values.split(",") if value.strip()]
    recent_values = [int(value.strip()) for value in recent_windows_values.split(",") if value.strip()]

    table = Table(title="Apollo Sweep")
    table.add_column("Recent")
    table.add_column("Top-K")
    table.add_column("Valid")
    table.add_column("Retrieval Acc")
    table.add_column("Temporal Acc")
    table.add_column("Retrieval Top-1")
    table.add_column("Retrieval Top-2")
    table.add_column("Retrieval Top-4")
    table.add_column("Usage @ Hit")

    console.print(
        "[bold]Apollo Sweep[/bold]\n"
        "Running gated retrieval sweeps over recent-window and top-k settings."
    )
    console.print(f"Backend: {runner.backend} ({runner.backend_reason})")

    for recent_windows in recent_values:
        cases = build_apollo_cases(
            corpus_path=corpus_path,
            case_count=case_count,
            windows=windows,
            window_lines=window_lines,
            recent_windows=recent_windows,
            seed=seed,
        )
        for top_k in top_ks:
            _, summary = evaluate_case_modes(
                cases=cases,
                runner=runner,
                recent_windows=recent_windows,
                top_k=top_k,
            )
            breakdown = summary["overall"]["retrieval_breakdown"]
            table.add_row(
                str(recent_windows),
                str(top_k),
                str(int(summary["overall"]["retrieval"]["valid_count"])),
                f"{summary['overall']['retrieval']['accuracy_valid']:.2f}",
                f"{summary['overall']['temporal']['accuracy_valid']:.2f}",
                f"{breakdown['top1_recall']:.2f}",
                f"{breakdown['top2_recall']:.2f}",
                f"{breakdown['top4_recall']:.2f}",
                f"{summary['overall']['retrieval']['usage_accuracy_when_hit']:.2f}",
            )
    console.print(table)


if __name__ == "__main__":
    app()
