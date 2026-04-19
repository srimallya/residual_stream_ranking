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
from residual_stream_lab.hf_trace import HFTraceRunner
from residual_stream_lab.apollo import ApolloCase, build_apollo_cases
from residual_stream_lab.synthetic import BenchmarkCase, build_benchmark_case
from residual_stream_lab.temporal import rerank_checkpoints
from residual_stream_lab.trace import verify_reconstruction

app = typer.Typer(add_completion=False)
console = Console()


def parse_layer_spec(spec: str) -> list[int]:
    values = [int(value.strip()) for value in spec.split(",") if value.strip()]
    if not values:
        raise typer.BadParameter("Layer list must contain at least one integer.")
    if values != sorted(set(values)):
        raise typer.BadParameter("Layer list must be unique and sorted ascending.")
    return values


def parse_int_list(spec: str, *, unique: bool = False, ascending: bool = False) -> list[int]:
    values = [int(value.strip()) for value in spec.split(",") if value.strip()]
    if not values:
        raise typer.BadParameter("Value list must contain at least one integer.")
    if unique and len(set(values)) != len(values):
        raise typer.BadParameter("Value list must be unique.")
    if ascending and values != sorted(values):
        raise typer.BadParameter("Value list must be sorted ascending.")
    return values


def expand_window_ids(
    selected_ids: list[int],
    *,
    max_window_index: int,
    exclude_ids: set[int],
    neighbor_radius: int,
) -> list[int]:
    if neighbor_radius <= 0:
        return selected_ids

    expanded: list[int] = []
    seen: set[int] = set()
    for window_id in selected_ids:
        for candidate in range(
            max(0, window_id - neighbor_radius),
            min(max_window_index, window_id + neighbor_radius) + 1,
        ):
            if candidate in exclude_ids or candidate in seen:
                continue
            expanded.append(candidate)
            seen.add(candidate)
    return expanded


def select_checkpoints(
    mode: str,
    benchmark: BenchmarkCase,
    query: str,
    recent_windows: int,
    top_k: int,
    runner: GGUFRunner,
    rerank_strategy: str = "hybrid",
    local_expansion_neighbors: int = 0,
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
            strategy=rerank_strategy,
        )
        selected.extend(recent_checkpoints)
    elif mode == "temporal_expanded":
        query_embedding = runner.embed(query)
        selected = rerank_checkpoints(
            checkpoints,
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k,
            exclude_window_ids=recent_ids,
            recent_windows=recent_checkpoints,
            strategy=rerank_strategy,
        )
        selected_ids = [checkpoint.window.index for checkpoint in selected]
        expanded_ids = expand_window_ids(
            selected_ids,
            max_window_index=len(windows) - 1,
            exclude_ids=recent_ids,
            neighbor_radius=local_expansion_neighbors,
        )
        selected = [checkpoint for checkpoint in checkpoints if checkpoint.window.index in expanded_ids]
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
    def normalize_candidate(candidate: str) -> str | None:
        cleaned = candidate.strip(" .,:;!?*`[](){}<>\"'").lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            return None
        if cleaned in valid_answers:
            return cleaned
        if cleaned == "unknown":
            return cleaned
        if re.fullmatch(r"[a-z][a-z\-]*", cleaned):
            return cleaned
        return None

    lowered = prediction.strip().lower()
    if not lowered:
        return None

    tag_matches = re.findall(r"<answer>\s*([^<]+?)\s*</answer>", lowered, flags=re.IGNORECASE | re.DOTALL)
    for candidate in reversed(tag_matches):
        normalized = normalize_candidate(candidate)
        if normalized is not None:
            return normalized

    lines = [line.strip(" -*`\t") for line in lowered.splitlines() if line.strip()]
    for line in reversed(lines):
        line = re.sub(r"^(final answer|answer|color)\s*[:\-]\s*", "", line).strip()
        normalized = normalize_candidate(line)
        if normalized is not None:
            return normalized
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


def format_prediction_snippet(prediction: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", prediction.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def collect_parse_failures(
    per_case: list[dict[str, object]],
    *,
    modes: tuple[str, ...] = ("full", "temporal"),
    limit: int = 8,
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for entry in per_case:
        for mode in modes:
            result = entry["results"][mode]
            if result["parse_success"]:
                continue
            failures.append(
                {
                    "case_id": str(entry["case_id"]),
                    "distance": str(entry["distance_bin"]),
                    "mode": mode,
                    "prediction": format_prediction_snippet(str(result["prediction"])),
                }
            )
            if len(failures) >= limit:
                return failures
    return failures


def collect_staged_oracle_gaps(
    per_case: list[dict[str, object]],
    *,
    staged_mode: str,
    limit: int = 8,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entry in per_case:
        staged = entry["results"].get(staged_mode)
        oracle = entry["results"]["oracle"]
        if staged is None:
            continue
        if staged["correct"] or not oracle["correct"]:
            continue
        rows.append(
            {
                "case_id": str(entry["case_id"]),
                "distance": str(entry["distance_bin"]),
                "mode": staged_mode,
                "selected": ",".join(str(value) for value in staged["selected_ids"][:4]),
                "prediction": str(staged["parsed"] or staged["prediction"]),
                "oracle": str(oracle["parsed"] or oracle["prediction"]),
            }
        )
        if len(rows) >= limit:
            return rows
    return rows


def evaluate_case_modes(
    *,
    cases: list[ApolloCase],
    runner: GGUFRunner,
    recent_windows: int,
    top_k: int,
    rerank_strategy: str = "hybrid",
    local_expansion_neighbors: int = 0,
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
            rerank_strategy=rerank_strategy,
            local_expansion_neighbors=local_expansion_neighbors,
        )
        recent_tokens = runner.token_count(recent_memory)

        full_memory, full_ids = select_checkpoints(
            mode="full",
            benchmark=case,
            query=case.question,
            recent_windows=recent_windows,
            top_k=top_k,
            runner=runner,
            rerank_strategy=rerank_strategy,
            local_expansion_neighbors=local_expansion_neighbors,
        )
        oracle_memory = build_oracle_memory(
            benchmark=case,
            target_window_index=case.target_window_index,
            recent_windows=recent_windows,
            runner=runner,
        )

        mode_results: dict[str, dict[str, object]] = {}
        checkpoints = build_checkpoints(
            split_windows(case.document, lines_per_window=case.window_lines),
            runner.embed,
        )
        recent = split_windows(case.document, lines_per_window=case.window_lines)[-recent_windows:] if recent_windows > 0 else []
        recent_ids = {window.index for window in recent}
        query_embedding = runner.embed(case.question)
        ranking_orders: dict[str, list[int]] = {}
        retrieval_candidates = retrieve_checkpoints(
            checkpoints,
            query_embedding=query_embedding,
            top_k=max(4, top_k),
            exclude_window_ids=recent_ids,
        )
        ranking_orders["retrieval"] = [checkpoint.window.index for checkpoint in retrieval_candidates]
        temporal_candidates = rerank_checkpoints(
            checkpoints,
            query_embedding=query_embedding,
            query_text=case.question,
            top_k=max(4, top_k),
            exclude_window_ids=recent_ids,
            recent_windows=[checkpoint for checkpoint in checkpoints if checkpoint.window.index in recent_ids],
            strategy=rerank_strategy,
        )
        ranking_orders["temporal"] = [checkpoint.window.index for checkpoint in temporal_candidates]

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

        active_modes = ["retrieval", "temporal"]
        if local_expansion_neighbors > 0:
            active_modes.append("temporal_expanded")

        for mode in active_modes:
            memory, selected_ids = select_checkpoints(
                mode=mode,
                benchmark=case,
                query=case.question,
                recent_windows=recent_windows,
                top_k=top_k,
                runner=runner,
                rerank_strategy=rerank_strategy,
                local_expansion_neighbors=local_expansion_neighbors,
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

        ranking_orders["temporal_expanded"] = ranking_orders["temporal"]
        for mode in active_modes:
            ranked_ids = ranking_orders[mode]
            mode_results[mode]["topk"] = {
                f"top_{k}": case.target_window_index in ranked_ids[:k] for k in [1, 2, 4]
            }
            mode_results[mode]["mrr"] = (
                1.0 / (ranked_ids.index(case.target_window_index) + 1)
                if case.target_window_index in ranked_ids
                else 0.0
            )

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
    modes = ["full", "recent", "retrieval", "temporal"]
    if local_expansion_neighbors > 0:
        modes.append("temporal_expanded")
    modes.append("oracle")
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

    breakdown_modes = ["retrieval", "temporal"]
    if local_expansion_neighbors > 0:
        breakdown_modes.append("temporal_expanded")
    for breakdown_mode in breakdown_modes:
        key = f"{breakdown_mode}_breakdown"
        if valid_count:
            mode_valid = [entry for entry in valid_cases]
            top1 = sum(entry["results"][breakdown_mode]["topk"]["top_1"] for entry in mode_valid) / valid_count
            top2 = sum(entry["results"][breakdown_mode]["topk"]["top_2"] for entry in mode_valid) / valid_count
            top4 = sum(entry["results"][breakdown_mode]["topk"]["top_4"] for entry in mode_valid) / valid_count
            wrong_window = sum(not entry["results"][breakdown_mode]["hit"] for entry in mode_valid) / valid_count
            right_unparsable = sum(
                entry["results"][breakdown_mode]["hit"] and not entry["results"][breakdown_mode]["parse_success"]
                for entry in mode_valid
            ) / valid_count
            hit_count = sum(entry["results"][breakdown_mode]["hit"] for entry in mode_valid)
            hit_and_parse_count = sum(
                entry["results"][breakdown_mode]["hit"] and entry["results"][breakdown_mode]["parse_success"]
                for entry in mode_valid
            )
            right_parsed_wrong = sum(
                entry["results"][breakdown_mode]["hit"]
                and entry["results"][breakdown_mode]["parse_success"]
                and not entry["results"][breakdown_mode]["correct"]
                for entry in mode_valid
            ) / valid_count
            mrr = sum(entry["results"][breakdown_mode]["mrr"] for entry in mode_valid) / valid_count
            summary["overall"][key] = {
                "top1_recall": top1,
                "top2_recall": top2,
                "top4_recall": top4,
                "mrr": mrr,
                "wrong_window_rate": wrong_window,
                "right_unparsable_rate": right_unparsable,
                "right_parsed_wrong_rate": right_parsed_wrong,
                "parse_given_hit": (hit_and_parse_count / hit_count) if hit_count else 0.0,
                "correct_given_hit_parse": (
                    sum(
                        entry["results"][breakdown_mode]["correct"]
                        for entry in mode_valid
                        if entry["results"][breakdown_mode]["hit"] and entry["results"][breakdown_mode]["parse_success"]
                    )
                    / hit_and_parse_count
                ) if hit_and_parse_count else 0.0,
            }
        else:
            summary["overall"][key] = {
                "top1_recall": 0.0,
                "top2_recall": 0.0,
                "top4_recall": 0.0,
                "mrr": 0.0,
                "wrong_window_rate": 0.0,
                "right_unparsable_rate": 0.0,
                "right_parsed_wrong_rate": 0.0,
                "parse_given_hit": 0.0,
                "correct_given_hit_parse": 0.0,
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


def evaluate_routing_only(
    *,
    cases: list[ApolloCase],
    runner: GGUFRunner,
    recent_windows: int,
    strategy: str,
) -> dict[str, float]:
    prepared_cases = []
    for case in cases:
        windows = split_windows(case.document, lines_per_window=case.window_lines)
        checkpoints = build_checkpoints(windows, runner.embed)
        recent = windows[-recent_windows:] if recent_windows > 0 else []
        recent_ids = {window.index for window in recent}
        recent_checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint.window.index in recent_ids]
        prepared_cases.append(
            {
                "case": case,
                "checkpoints": checkpoints,
                "recent_ids": recent_ids,
                "recent_checkpoints": recent_checkpoints,
                "query_embedding": runner.embed(case.question),
            }
        )

    top1 = 0
    top2 = 0
    top4 = 0
    reciprocal_rank_sum = 0.0

    for prepared in prepared_cases:
        case = prepared["case"]
        checkpoints = prepared["checkpoints"]
        recent_ids = prepared["recent_ids"]
        recent_checkpoints = prepared["recent_checkpoints"]
        query_embedding = prepared["query_embedding"]
        if strategy == "semantic":
            ranked = retrieve_checkpoints(
                checkpoints,
                query_embedding=query_embedding,
                top_k=4,
                exclude_window_ids=recent_ids,
            )
        else:
            ranked = rerank_checkpoints(
                checkpoints,
                query_embedding=query_embedding,
                query_text=case.question,
                top_k=4,
                exclude_window_ids=recent_ids,
                recent_windows=recent_checkpoints,
                strategy=strategy,
            )

        ranked_ids = [checkpoint.window.index for checkpoint in ranked]
        top1 += int(case.target_window_index in ranked_ids[:1])
        top2 += int(case.target_window_index in ranked_ids[:2])
        top4 += int(case.target_window_index in ranked_ids[:4])
        if case.target_window_index in ranked_ids:
            reciprocal_rank_sum += 1.0 / (ranked_ids.index(case.target_window_index) + 1)

    count = len(cases)
    return {
        "count": float(count),
        "top1": top1 / count,
        "top2": top2 / count,
        "top4": top4 / count,
        "mrr": reciprocal_rank_sum / count,
    }


def evaluate_routed_replay_bridge(
    *,
    cases: list[ApolloCase],
    routing_runner: GGUFRunner,
    replay_runner: HFTraceRunner,
    recent_windows: int,
    top_k: int,
    replay_boundary_layer: int,
    replay_layer: int,
    replay_steps: int,
    rerank_strategy: str = "staged",
    replay_top_k: int = 5,
) -> dict[str, object]:
    object_order = [
        "text@window",
        f"token@{replay_layer}",
        f"token@{replay_layer}/fp16",
        f"token@{replay_layer}/int8",
        f"delta_depth={replay_layer - replay_boundary_layer}",
    ]
    aggregates: dict[str, dict[str, float | int]] = {
        label: {
            "cases": 0,
            "token_agreement_sum": 0.0,
            "topk_full_steps_sum": 0,
            "steps_sum": 0,
            "divergence_count": 0,
            "compact_bytes": 0,
        }
        for label in object_order
    }
    per_case: list[dict[str, object]] = []
    top1_hits = 0
    topk_hits = 0

    for case in cases:
        windows = split_windows(case.document, lines_per_window=case.window_lines)
        checkpoints = build_checkpoints(windows, routing_runner.embed)
        recent = windows[-recent_windows:] if recent_windows > 0 else []
        recent_ids = {window.index for window in recent}
        recent_checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint.window.index in recent_ids]
        query_embedding = routing_runner.embed(case.question)
        ranked = rerank_checkpoints(
            checkpoints,
            query_embedding=query_embedding,
            query_text=case.question,
            top_k=top_k,
            exclude_window_ids=recent_ids,
            recent_windows=recent_checkpoints,
            strategy=rerank_strategy,
        )
        ranked_ids = [checkpoint.window.index for checkpoint in ranked]
        top1_hit = bool(ranked_ids) and ranked_ids[0] == case.target_window_index
        topk_hit = case.target_window_index in ranked_ids
        top1_hits += int(top1_hit)
        topk_hits += int(topk_hit)

        case_row: dict[str, object] = {
            "case_id": case.case_id,
            "distance_bin": case.distance_bin,
            "ranked_ids": ranked_ids,
            "top1_hit": top1_hit,
            "topk_hit": topk_hit,
            "target_window_index": case.target_window_index,
        }

        if top1_hit and ranked:
            routed_window_text = ranked[0].window.text
            replay_result = replay_runner.compare_compact_continuation_variants(
                text=routed_window_text,
                boundary_layer=replay_boundary_layer,
                replay_layer=replay_layer,
                delta_depths=[replay_layer - replay_boundary_layer],
                steps=replay_steps,
                top_k=replay_top_k,
            )
            replay_rows = {
                row["object_label"]: row
                for row in replay_result["rows"]
                if row["object_label"] in object_order
            }
            replay_rows["text@window"] = {
                "object_label": "text@window",
                "compact_bytes": len(routed_window_text.encode("utf-8")),
                "token_agreement": 1.0,
                "topk_full_steps": replay_steps,
                "steps_completed": replay_steps,
                "first_divergence_step": None,
            }
            case_row["replay"] = replay_rows
            for label in object_order:
                row = replay_rows[label]
                aggregate = aggregates[label]
                aggregate["cases"] += 1
                aggregate["token_agreement_sum"] += float(row["token_agreement"])
                aggregate["topk_full_steps_sum"] += int(row["topk_full_steps"])
                aggregate["steps_sum"] += int(row["steps_completed"])
                aggregate["divergence_count"] += int(row["first_divergence_step"] is not None)
                if aggregate["compact_bytes"] == 0:
                    aggregate["compact_bytes"] = int(row["compact_bytes"])

        per_case.append(case_row)

    summary_rows: list[dict[str, object]] = []
    for label in object_order:
        aggregate = aggregates[label]
        cases_count = int(aggregate["cases"])
        steps_sum = int(aggregate["steps_sum"])
        summary_rows.append(
            {
                "object_label": label,
                "cases": cases_count,
                "compact_bytes": int(aggregate["compact_bytes"]),
                "token_agreement": (float(aggregate["token_agreement_sum"]) / cases_count) if cases_count else 0.0,
                "topk_full_rate": (int(aggregate["topk_full_steps_sum"]) / steps_sum) if steps_sum else 0.0,
                "divergence_rate": (int(aggregate["divergence_count"]) / cases_count) if cases_count else 0.0,
            }
        )

    return {
        "case_count": len(cases),
        "top1_hit_rate": top1_hits / len(cases) if cases else 0.0,
        "topk_hit_rate": topk_hits / len(cases) if cases else 0.0,
        "rerank_strategy": rerank_strategy,
        "replay_boundary_layer": replay_boundary_layer,
        "replay_layer": replay_layer,
        "replay_steps": replay_steps,
        "summary_rows": summary_rows,
        "per_case": per_case,
    }


@app.command("trace-verify")
def trace_verify(
    model_name_or_path: str = typer.Option(..., help="Transformers model id or local path."),
    prompt: str = typer.Option(..., help="Prompt to trace."),
    layer_cutoff_b: int = typer.Option(..., help="Boundary layer used as the replay anchor."),
    delta_layers: str = typer.Option(..., help="Comma-separated logical layers to add after the boundary."),
    token_index: int = typer.Option(-1, help="Token index to trace. Defaults to the last token."),
    device: str = typer.Option("cpu", help="Torch device for the HF trace backend."),
    dtype: str = typer.Option("auto", help="Torch dtype: auto, float32, float16, or bfloat16."),
) -> None:
    parsed_delta_layers = parse_layer_spec(delta_layers)
    if parsed_delta_layers[0] <= layer_cutoff_b:
        raise typer.BadParameter("All delta layers must be greater than the boundary layer.")

    runner = HFTraceRunner(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
    )
    capture = runner.capture_trace(
        text=prompt,
        layer_cutoff_b=layer_cutoff_b,
        delta_layers=parsed_delta_layers,
        token_index=token_index,
    )
    target_layer = parsed_delta_layers[-1]
    verification = verify_reconstruction(
        capture.payload,
        capture.observed_states[target_layer],
    )

    console.print(
        "[bold]HF Trace Verification[/bold]\n"
        "Offline reconstruction from a boundary hidden state plus observed per-layer deltas."
    )
    console.print(f"Backend: {runner.backend}")
    console.print(f"Model: {runner.model_name_or_path}")
    console.print(f"Prompt tokens: {capture.token_count}")
    console.print(f"Boundary layer: {layer_cutoff_b}")
    console.print(f"Delta layers: {', '.join(str(value) for value in parsed_delta_layers)}")
    console.print(f"Target layer: {target_layer}")
    console.print(f"Token index: {capture.payload.boundary_token_index}")

    table = Table(title="Reconstruction Metrics")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("L2 error", f"{verification.l2_error:.8f}")
    table.add_row("Cosine similarity", f"{verification.cosine_similarity:.8f}")
    table.add_row("Exact trace", "yes" if verification.exact_trace else "no")
    table.add_row("Vector shape", str(capture.payload.metadata.shape if capture.payload.metadata else ()))
    console.print(table)


@app.command("trace-resume-verify")
def trace_resume_verify(
    model_name_or_path: str = typer.Option(..., help="Transformers model id or local path."),
    prompt: str = typer.Option(..., help="Prompt to trace and resume."),
    boundary_layer: int = typer.Option(..., help="Logical layer whose output becomes the injected boundary state."),
    target_token_index: int = typer.Option(-1, help="Token index to compare logits for. Defaults to the last token."),
    device: str = typer.Option("cpu", help="Torch device for the HF trace backend."),
    dtype: str = typer.Option("auto", help="Torch dtype: auto, float32, float16, or bfloat16."),
) -> None:
    runner = HFTraceRunner(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
    )
    comparison = runner.compare_resumed_logits(
        text=prompt,
        boundary_layer=boundary_layer,
        target_token_index=target_token_index,
    )

    console.print(
        "[bold]HF Resume Verification[/bold]\n"
        "Phase 2A check: inject a captured full-sequence boundary hidden state and resume the remaining layers."
    )
    console.print(f"Backend: {runner.backend}")
    console.print(f"Model: {runner.model_name_or_path}")
    console.print(f"Boundary layer: {comparison['boundary_layer']}")
    console.print(f"Start layer: {comparison['start_layer']}")
    console.print(f"Target token index: {comparison['target_token_index']}")

    table = Table(title="Resumed Logit Agreement")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("L2 error", f"{comparison['l2_error']:.8f}")
    table.add_row("Cosine similarity", f"{comparison['cosine_similarity']:.8f}")
    table.add_row("Max abs diff", f"{comparison['max_abs_diff']:.8f}")
    table.add_row("Vocab size", str(comparison["vocab_size"]))
    console.print(table)


@app.command("trace-next-token-verify")
def trace_next_token_verify(
    model_name_or_path: str = typer.Option(..., help="Transformers model id or local path."),
    prompt: str = typer.Option(..., help="Prompt to trace and resume."),
    boundary_layer: int = typer.Option(..., help="Logical layer whose output becomes the injected boundary state."),
    top_k: int = typer.Option(5, min=1, help="How many next-token candidates to display."),
    device: str = typer.Option("cpu", help="Torch device for the HF trace backend."),
    dtype: str = typer.Option("auto", help="Torch dtype: auto, float32, float16, or bfloat16."),
) -> None:
    runner = HFTraceRunner(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
    )
    comparison = runner.compare_next_token(
        text=prompt,
        boundary_layer=boundary_layer,
        top_k=top_k,
    )

    console.print(
        "[bold]HF Next-Token Verification[/bold]\n"
        "Phase 2B check: compare direct next-token logits and greedy token against resumed execution from a boundary state."
    )
    console.print(f"Backend: {runner.backend}")
    console.print(f"Model: {runner.model_name_or_path}")
    console.print(f"Boundary layer: {comparison['boundary_layer']}")
    console.print(f"Start layer: {comparison['start_layer']}")

    metrics = Table(title="Next-Token Agreement")
    metrics.add_column("Metric")
    metrics.add_column("Value")
    metrics.add_row("L2 error", f"{comparison['l2_error']:.8f}")
    metrics.add_row("Cosine similarity", f"{comparison['cosine_similarity']:.8f}")
    metrics.add_row("Max abs diff", f"{comparison['max_abs_diff']:.8f}")
    metrics.add_row("Direct greedy id", str(comparison["direct_token_id"]))
    metrics.add_row("Direct greedy text", repr(comparison["direct_token_text"]))
    metrics.add_row("Resumed greedy id", str(comparison["resumed_token_id"]))
    metrics.add_row("Resumed greedy text", repr(comparison["resumed_token_text"]))
    metrics.add_row("Greedy match", "yes" if comparison["token_match"] else "no")
    console.print(metrics)

    top_table = Table(title="Top-K Next Tokens")
    top_table.add_column("Rank")
    top_table.add_column("Direct")
    top_table.add_column("Resumed")
    direct_top_k = comparison["direct_top_k"]
    resumed_top_k = comparison["resumed_top_k"]
    for index, (direct_row, resumed_row) in enumerate(zip(direct_top_k, resumed_top_k), start=1):
        top_table.add_row(
            str(index),
            f"{direct_row[0]} {direct_row[1]!r} {direct_row[2]:.6f}",
            f"{resumed_row[0]} {resumed_row[1]!r} {resumed_row[2]:.6f}",
        )
    console.print(top_table)


@app.command("trace-generate-verify")
def trace_generate_verify(
    model_name_or_path: str = typer.Option(..., help="Transformers model id or local path."),
    prompt: str = typer.Option(..., help="Prompt to trace and continue from."),
    boundary_layer: int = typer.Option(..., help="Logical layer whose output becomes the injected boundary state."),
    steps: int = typer.Option(8, min=1, help="Greedy continuation horizon."),
    top_k: int = typer.Option(5, min=1, help="How many next-token candidates to display per step."),
    device: str = typer.Option("cpu", help="Torch device for the HF trace backend."),
    dtype: str = typer.Option("auto", help="Torch dtype: auto, float32, float16, or bfloat16."),
) -> None:
    runner = HFTraceRunner(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
    )
    result = runner.compare_greedy_continuation(
        text=prompt,
        boundary_layer=boundary_layer,
        steps=steps,
        top_k=top_k,
    )

    console.print(
        "[bold]HF Generation Verification[/bold]\n"
        "Phase 2C check: greedy short-horizon continuation from a boundary state, compared step by step against direct generation."
    )
    console.print(f"Backend: {runner.backend}")
    console.print(f"Model: {runner.model_name_or_path}")
    console.print(f"Boundary layer: {result['boundary_layer']}")
    console.print(f"Steps requested: {result['steps_requested']}")
    console.print(f"Steps completed: {result['steps_completed']}")
    console.print(f"Exact continuation match: {'yes' if result['exact_match'] else 'no'}")
    if result["first_divergence_step"] is not None:
        console.print(f"First divergence step: {result['first_divergence_step']}")
    console.print(f"Generated text: {result['generated_text']!r}")

    table = Table(title="Per-Step Agreement")
    table.add_column("Step")
    table.add_column("L2")
    table.add_column("Cos")
    table.add_column("Max Abs Diff")
    table.add_column("Greedy Match")
    table.add_column("Direct Token")
    table.add_column("Resumed Token")
    for step in result["per_step"]:
        table.add_row(
            str(step["step"]),
            f"{step['l2_error']:.8f}",
            f"{step['cosine_similarity']:.8f}",
            f"{step['max_abs_diff']:.8f}",
            "yes" if step["token_match"] else "no",
            f"{step['direct_token_id']} {step['direct_token_text']!r}",
            f"{step['resumed_token_id']} {step['resumed_token_text']!r}",
        )
    console.print(table)


@app.command("trace-generate-sweep")
def trace_generate_sweep(
    model_name_or_path: str = typer.Option(..., help="Transformers model id or local path."),
    prompt: str = typer.Option(..., help="Prompt to trace and continue from."),
    boundary_layers: str = typer.Option("0,6,10", help="Comma-separated boundary layers to test."),
    step_values: str = typer.Option("5,10,20", help="Comma-separated greedy continuation horizons to test."),
    top_k: int = typer.Option(3, min=1, help="How many next-token candidates to retain in backend comparisons."),
    device: str = typer.Option("cpu", help="Torch device for the HF trace backend."),
    dtype: str = typer.Option("auto", help="Torch dtype: auto, float32, float16, or bfloat16."),
) -> None:
    parsed_boundary_layers = parse_int_list(boundary_layers, unique=True, ascending=True)
    parsed_step_values = parse_int_list(step_values, unique=True, ascending=True)

    runner = HFTraceRunner(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
    )

    console.print(
        "[bold]HF Generation Sweep[/bold]\n"
        "Maps exact greedy continuation stability across selected boundary layers and step horizons."
    )
    console.print(f"Backend: {runner.backend}")
    console.print(f"Model: {runner.model_name_or_path}")
    console.print(f"Prompt: {prompt!r}")

    summary_table = Table(title="Continuation Stability")
    summary_table.add_column("Boundary")
    summary_table.add_column("Steps")
    summary_table.add_column("Exact")
    summary_table.add_column("Completed")
    summary_table.add_column("First Divergence")
    summary_table.add_column("Final Direct Token")
    summary_table.add_column("Final Resumed Token")

    for boundary_layer in parsed_boundary_layers:
        for step_count in parsed_step_values:
            result = runner.compare_greedy_continuation(
                text=prompt,
                boundary_layer=boundary_layer,
                steps=step_count,
                top_k=top_k,
            )
            last_step = result["per_step"][-1] if result["per_step"] else None
            summary_table.add_row(
                str(boundary_layer),
                str(step_count),
                "yes" if result["exact_match"] else "no",
                str(result["steps_completed"]),
                "-" if result["first_divergence_step"] is None else str(result["first_divergence_step"]),
                "-" if last_step is None else f"{last_step['direct_token_id']} {last_step['direct_token_text']!r}",
                "-" if last_step is None else f"{last_step['resumed_token_id']} {last_step['resumed_token_text']!r}",
            )

    console.print(summary_table)


@app.command("trace-compact-sweep")
def trace_compact_sweep(
    model_name_or_path: str = typer.Option(..., help="Transformers model id or local path."),
    prompt: str = typer.Option(..., help="Prompt to trace and compact-replay from."),
    boundary_layer: int = typer.Option(..., help="Boundary layer where the compact object starts."),
    replay_layer: int = typer.Option(..., help="Later layer reconstructed from the compact object."),
    delta_depths: str = typer.Option(
        "0,1,2,4",
        help="Comma-separated counts of trailing deltas to keep between boundary and replay layer.",
    ),
    top_k: int = typer.Option(5, min=1, help="Top-k width used for overlap scoring."),
    token_index: int = typer.Option(-1, help="Token index to reconstruct; defaults to the last prompt token."),
    device: str = typer.Option("cpu", help="Torch device for the HF trace backend."),
    dtype: str = typer.Option("auto", help="Torch dtype: auto, float32, float16, or bfloat16."),
) -> None:
    parsed_delta_depths = parse_int_list(delta_depths, unique=True, ascending=True)

    runner = HFTraceRunner(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
    )
    result = runner.compare_compact_replay_variants(
        text=prompt,
        boundary_layer=boundary_layer,
        replay_layer=replay_layer,
        delta_depths=parsed_delta_depths,
        top_k=top_k,
        token_index=token_index,
    )

    console.print(
        "[bold]HF Compact Replay Sweep[/bold]\n"
        "Measures how much target-token replay state can be dropped before next-token behavior diverges from the exact replay baseline."
    )
    console.print(f"Backend: {runner.backend}")
    console.print(f"Model: {runner.model_name_or_path}")
    console.print(f"Boundary layer: {result['boundary_layer']}")
    console.print(f"Replay layer: {result['replay_layer']}")
    console.print(f"Token index: {result['token_index']}")
    console.print(
        "Available delta layers: "
        + ", ".join(str(layer) for layer in result["available_delta_layers"])
    )

    table = Table(title="Compact Replay Variants")
    table.add_column("Object")
    table.add_column("Kept Layers")
    table.add_column("Bytes")
    table.add_column("Token Match")
    table.add_column(f"Top-{top_k}")
    table.add_column("L2")
    table.add_column("Max Abs Diff")
    table.add_column("Compact Token")
    for row in result["rows"]:
        kept_layers = ",".join(str(layer) for layer in row["kept_layers"]) or "-"
        table.add_row(
            str(row["object_label"]),
            kept_layers,
            str(row["compact_bytes"]),
            "yes" if row["token_match"] else "no",
            f"{row['topk_overlap']}/{row['topk_size']}",
            f"{row['l2_error']:.8f}",
            f"{row['max_abs_diff']:.8f}",
            f"{row['compact_token_id']} {row['compact_token_text']!r}",
        )
    console.print(table)

    if result["rows"]:
        reference = result["rows"][-1]
        metrics = Table(title="Compact Object Size Reference")
        metrics.add_column("Metric")
        metrics.add_column("Bytes")
        metrics.add_row("Exact replay token", str(reference["full_replay_token_bytes"]))
        metrics.add_row("Full boundary+delta trace", str(reference["full_trace_bytes"]))
        console.print(metrics)


@app.command("trace-compact-operational")
def trace_compact_operational(
    model_name_or_path: str = typer.Option(..., help="Transformers model id or local path."),
    prompt: str = typer.Option(..., help="Prompt to trace and compact-replay from."),
    boundary_layer: int = typer.Option(..., help="Boundary layer where the compact object starts."),
    replay_layer: int = typer.Option(..., help="Later layer reconstructed from the compact object."),
    delta_depths: str = typer.Option(
        "0,2,4",
        help="Comma-separated counts of trailing deltas to keep between boundary and replay layer.",
    ),
    steps: int = typer.Option(10, min=1, help="Greedy continuation horizon."),
    top_k: int = typer.Option(5, min=1, help="Top-k width used for overlap scoring."),
    device: str = typer.Option("cpu", help="Torch device for the HF trace backend."),
    dtype: str = typer.Option("auto", help="Torch dtype: auto, float32, float16, or bfloat16."),
) -> None:
    parsed_delta_depths = parse_int_list(delta_depths, unique=True, ascending=True)

    runner = HFTraceRunner(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
    )
    result = runner.compare_compact_continuation_variants(
        text=prompt,
        boundary_layer=boundary_layer,
        replay_layer=replay_layer,
        delta_depths=parsed_delta_depths,
        steps=steps,
        top_k=top_k,
    )

    console.print(
        "[bold]HF Compact Operational Comparison[/bold]\n"
        "Measures how reduced replay objects hold up over short greedy continuation against the exact replay baseline."
    )
    console.print(f"Backend: {runner.backend}")
    console.print(f"Model: {runner.model_name_or_path}")
    console.print(f"Boundary layer: {result['boundary_layer']}")
    console.print(f"Replay layer: {result['replay_layer']}")
    console.print(f"Steps requested: {result['steps_requested']}")

    table = Table(title="Compact Continuation Envelope")
    table.add_column("Object")
    table.add_column("Kept Layers")
    table.add_column("Token Agreement")
    table.add_column(f"Top-{top_k}")
    table.add_column("First Divergence")
    table.add_column("Final L2")
    table.add_column("Final Max Abs Diff")
    for row in result["rows"]:
        kept_layers = ",".join(str(layer) for layer in row["kept_layers"]) or "-"
        table.add_row(
            str(row["object_label"]),
            kept_layers,
            f"{row['token_agreement']:.2f}",
            f"{row['topk_full_steps']}/{row['steps_completed']}",
            "-" if row["first_divergence_step"] is None else str(row["first_divergence_step"]),
            f"{row['final_l2_error']:.8f}",
            f"{row['final_max_abs_diff']:.8f}",
        )
    console.print(table)


@app.command("trace-generate-kv-verify")
def trace_generate_kv_verify(
    model_name_or_path: str = typer.Option(..., help="Transformers model id or local path."),
    prompt: str = typer.Option(..., help="Prompt to trace and continue from."),
    boundary_layer: int = typer.Option(..., help="Logical layer whose output becomes the injected boundary state."),
    steps: int = typer.Option(8, min=1, help="Greedy continuation horizon."),
    top_k: int = typer.Option(3, min=1, help="How many next-token candidates to retain per step."),
    device: str = typer.Option("cpu", help="Torch device for the HF trace backend."),
    dtype: str = typer.Option("auto", help="Torch dtype: auto, float32, float16, or bfloat16."),
) -> None:
    runner = HFTraceRunner(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
    )
    result = runner.compare_greedy_continuation_kv(
        text=prompt,
        boundary_layer=boundary_layer,
        steps=steps,
        top_k=top_k,
    )

    console.print(
        "[bold]HF KV Generation Verification[/bold]\n"
        "Compares a narrow KV-aware upper-stack continuation path against the frozen exact replay baseline."
    )
    console.print(f"Backend: {runner.backend}")
    console.print(f"Model: {runner.model_name_or_path}")
    console.print(f"Boundary layer: {result['boundary_layer']}")
    console.print(f"Steps requested: {result['steps_requested']}")
    console.print(f"Steps completed: {result['steps_completed']}")
    console.print(f"Exact KV match: {'yes' if result['exact_match'] else 'no'}")
    if result["first_divergence_step"] is not None:
        console.print(f"First divergence step: {result['first_divergence_step']}")

    metrics = Table(title="KV Path Summary")
    metrics.add_column("Metric")
    metrics.add_column("Value")
    metrics.add_row("Baseline ms", f"{result['baseline_ms']:.2f}")
    metrics.add_row("KV path ms", f"{result['kv_ms']:.2f}")
    metrics.add_row("Cache bytes", str(result["cache_bytes"]))
    console.print(metrics)

    table = Table(title="Per-Step KV Agreement")
    table.add_column("Step")
    table.add_column("L2")
    table.add_column("Cos")
    table.add_column("Max Abs Diff")
    table.add_column("Token Match")
    table.add_column("Baseline Token")
    table.add_column("KV Token")
    for step in result["per_step"]:
        table.add_row(
            str(step["step"]),
            f"{step['l2_error']:.8f}",
            f"{step['cosine_similarity']:.8f}",
            f"{step['max_abs_diff']:.8f}",
            "yes" if step["token_match"] else "no",
            f"{step['baseline_token_id']} {step['baseline_token_text']!r}",
            f"{step['kv_token_id']} {step['kv_token_text']!r}",
        )
    console.print(table)


@app.command("trace-generate-kv-diagnose")
def trace_generate_kv_diagnose(
    model_name_or_path: str = typer.Option(..., help="Transformers model id or local path."),
    prompt: str = typer.Option(..., help="Prompt to trace and continue from."),
    boundary_layer: int = typer.Option(..., help="Logical layer whose output becomes the injected boundary state."),
    top_k: int = typer.Option(3, min=1, help="How many next-token candidates to display."),
    device: str = typer.Option("cpu", help="Torch device for the HF trace backend."),
    dtype: str = typer.Option("auto", help="Torch dtype: auto, float32, float16, or bfloat16."),
) -> None:
    runner = HFTraceRunner(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
    )
    result = runner.diagnose_kv_step_two(
        text=prompt,
        boundary_layer=boundary_layer,
        top_k=top_k,
    )

    console.print(
        "[bold]HF KV Step-2 Diagnosis[/bold]\n"
        "Localizes the first layer where the KV-aware path diverges from the frozen exact baseline at the second resumed step."
    )
    console.print(f"Backend: {runner.backend}")
    console.print(f"Model: {runner.model_name_or_path}")
    console.print(f"Boundary layer: {result['boundary_layer']}")
    console.print(f"Step-1 token: {result['step1_token_id']} {result['step1_token_text']!r}")
    console.print(
        "First divergent layer: "
        + ("none" if result["first_divergent_layer"] is None else str(result["first_divergent_layer"]))
    )

    metrics = Table(title="Step-2 Logit Divergence")
    metrics.add_column("Metric")
    metrics.add_column("Value")
    metrics.add_row("Cache bytes", str(result["cache_bytes"]))
    metrics.add_row("Logit L2", f"{result['logit_l2_error']:.8f}")
    metrics.add_row("Logit Cos", f"{result['logit_cosine_similarity']:.8f}")
    metrics.add_row("Logit Max Abs Diff", f"{result['logit_max_abs_diff']:.8f}")
    console.print(metrics)

    layer_table = Table(title="Per-Layer Step-2 State Diff")
    layer_table.add_column("Layer")
    layer_table.add_column("Exact")
    layer_table.add_column("L2")
    layer_table.add_column("Cos")
    layer_table.add_column("Max Abs Diff")
    for row in result["per_layer"]:
        layer_table.add_row(
            str(row["layer"]),
            "yes" if row["exact_match"] else "no",
            f"{row['l2_error']:.8f}",
            f"{row['cosine_similarity']:.8f}",
            f"{row['max_abs_diff']:.8f}",
        )
    console.print(layer_table)


@app.command("trace-generate-kv-three-path")
def trace_generate_kv_three_path(
    model_name_or_path: str = typer.Option(..., help="Transformers model id or local path."),
    prompt: str = typer.Option(..., help="Prompt to trace and continue from."),
    boundary_layer: int = typer.Option(..., help="Logical layer whose output becomes the injected boundary state."),
    top_k: int = typer.Option(3, min=1, help="How many next-token candidates to display."),
    device: str = typer.Option("cpu", help="Torch device for the HF trace backend."),
    dtype: str = typer.Option("auto", help="Torch dtype: auto, float32, float16, or bfloat16."),
) -> None:
    runner = HFTraceRunner(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
    )
    result = runner.diagnose_kv_step_two_three_path(
        text=prompt,
        boundary_layer=boundary_layer,
        top_k=top_k,
    )

    console.print(
        "[bold]HF KV Three-Path Diagnosis[/bold]\n"
        "Separates step-2 drift into exact resumed baseline (A), full-sequence recompute (B), and KV-aware path (C)."
    )
    console.print(f"Backend: {runner.backend}")
    console.print(f"Model: {runner.model_name_or_path}")
    console.print(f"Boundary layer: {result['boundary_layer']}")
    console.print(f"Step-1 token: {result['step1_token_id']} {result['step1_token_text']!r}")
    console.print(
        "First A vs B divergent layer: "
        + ("none" if result["first_recompute_divergent_layer"] is None else str(result["first_recompute_divergent_layer"]))
    )
    console.print(
        "First A vs C divergent layer: "
        + ("none" if result["first_kv_divergent_layer"] is None else str(result["first_kv_divergent_layer"]))
    )

    metrics = Table(title="Step-2 Logit Drift")
    metrics.add_column("Path Pair")
    metrics.add_column("L2")
    metrics.add_column("Cos")
    metrics.add_column("Max Abs Diff")
    metrics.add_row("A vs B", f"{result['ab_logits']['l2_error']:.8f}", f"{result['ab_logits']['cosine_similarity']:.8f}", f"{result['ab_logits']['max_abs_diff']:.8f}")
    metrics.add_row("A vs C", f"{result['ac_logits']['l2_error']:.8f}", f"{result['ac_logits']['cosine_similarity']:.8f}", f"{result['ac_logits']['max_abs_diff']:.8f}")
    metrics.add_row("B vs C", f"{result['bc_logits']['l2_error']:.8f}", f"{result['bc_logits']['cosine_similarity']:.8f}", f"{result['bc_logits']['max_abs_diff']:.8f}")
    console.print(metrics)

    layer_table = Table(title="Per-Layer Step-2 Drift")
    layer_table.add_column("Layer")
    layer_table.add_column("A=B")
    layer_table.add_column("A vs B L2")
    layer_table.add_column("A=C")
    layer_table.add_column("A vs C L2")
    layer_table.add_column("B=C")
    layer_table.add_column("B vs C L2")
    for row in result["per_layer"]:
        layer_table.add_row(
            str(row["layer"]),
            "yes" if row["ab_exact"] else "no",
            f"{row['ab_l2']:.8f}",
            "yes" if row["ac_exact"] else "no",
            f"{row['ac_l2']:.8f}",
            "yes" if row["bc_exact"] else "no",
            f"{row['bc_l2']:.8f}",
        )
    console.print(layer_table)


@app.command("trace-generate-kv-operational")
def trace_generate_kv_operational(
    model_name_or_path: str = typer.Option(..., help="Transformers model id or local path."),
    prompt: str = typer.Option(..., help="Prompt to trace and continue from."),
    boundary_layer: int = typer.Option(..., help="Logical layer whose output becomes the injected boundary state."),
    steps: int = typer.Option(20, min=1, help="Greedy continuation horizon."),
    top_k: int = typer.Option(5, min=1, help="Top-k width used for overlap scoring."),
    device: str = typer.Option("cpu", help="Torch device for the HF trace backend."),
    dtype: str = typer.Option("auto", help="Torch dtype: auto, float32, float16, or bfloat16."),
) -> None:
    runner = HFTraceRunner(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
    )
    result = runner.compare_greedy_continuation_kv_operational(
        text=prompt,
        boundary_layer=boundary_layer,
        steps=steps,
        top_k=top_k,
    )

    console.print(
        "[bold]HF KV Operational Comparison[/bold]\n"
        "Compares the fast KV-aware path against the frozen exact baseline on greedy continuation behavior."
    )
    console.print(f"Backend: {runner.backend}")
    console.print(f"Model: {runner.model_name_or_path}")
    console.print(f"Boundary layer: {result['boundary_layer']}")
    console.print(f"Steps requested: {result['steps_requested']}")
    console.print(f"Steps completed: {result['steps_completed']}")
    console.print(f"Token agreement: {result['token_agreement']:.2f}")
    console.print(
        "First divergence step: "
        + ("none" if result["first_divergence_step"] is None else str(result["first_divergence_step"]))
    )

    metrics = Table(title="Operational Summary")
    metrics.add_column("Metric")
    metrics.add_column("Value")
    metrics.add_row("Exact latency / step ms", f"{result['exact_latency_per_step_ms']:.2f}")
    metrics.add_row("KV latency / step ms", f"{result['kv_latency_per_step_ms']:.2f}")
    metrics.add_row("Final cache bytes", str(result["final_cache_bytes"]))
    console.print(metrics)

    step_table = Table(title="Per-Step Operational Agreement")
    step_table.add_column("Step")
    step_table.add_column("Token Match")
    step_table.add_column(f"Top-{top_k} Overlap")
    step_table.add_column("Exact Token")
    step_table.add_column("KV Token")
    for row in result["per_step"]:
        step_table.add_row(
            str(row["step"]),
            "yes" if row["token_match"] else "no",
            f"{row['topk_overlap']}/{row['topk_size']}",
            f"{row['exact_token_id']} {row['exact_token_text']!r}",
            f"{row['kv_token_id']} {row['kv_token_text']!r}",
        )
    console.print(step_table)


@app.command("trace-generate-kv-sweep")
def trace_generate_kv_sweep(
    model_name_or_path: str = typer.Option(..., help="Transformers model id or local path."),
    prompt: str = typer.Option(..., help="Prompt to trace and continue from."),
    boundary_layers: str = typer.Option("0,6,10", help="Comma-separated boundary layers to test."),
    step_values: str = typer.Option("20,50", help="Comma-separated greedy continuation horizons to test."),
    top_k: int = typer.Option(5, min=1, help="Top-k width used for overlap scoring."),
    device: str = typer.Option("cpu", help="Torch device for the HF trace backend."),
    dtype: str = typer.Option("auto", help="Torch dtype: auto, float32, float16, or bfloat16."),
) -> None:
    parsed_boundary_layers = parse_int_list(boundary_layers, unique=True, ascending=True)
    parsed_step_values = parse_int_list(step_values, unique=True, ascending=True)

    runner = HFTraceRunner(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
    )

    console.print(
        "[bold]HF KV Operational Sweep[/bold]\n"
        "Maps behavioral agreement between the frozen exact baseline and the kv_fast path across selected boundaries and horizons."
    )
    console.print(f"Backend: {runner.backend}")
    console.print(f"Model: {runner.model_name_or_path}")
    console.print(f"Prompt: {prompt!r}")

    table = Table(title="KV Fast Operational Envelope")
    table.add_column("Boundary")
    table.add_column("Steps")
    table.add_column("Token Agreement")
    table.add_column(f"Top-{top_k}")
    table.add_column("First Divergence")
    table.add_column("Exact ms/step")
    table.add_column("KV ms/step")
    table.add_column("Cache Bytes")

    for boundary_layer in parsed_boundary_layers:
        for step_count in parsed_step_values:
            result = runner.compare_greedy_continuation_kv_operational(
                text=prompt,
                boundary_layer=boundary_layer,
                steps=step_count,
                top_k=top_k,
            )
            full_topk_steps = sum(1 for row in result["per_step"] if row["topk_overlap"] == row["topk_size"])
            table.add_row(
                str(boundary_layer),
                str(step_count),
                f"{result['token_agreement']:.2f}",
                f"{full_topk_steps}/{result['steps_completed']}",
                "-" if result["first_divergence_step"] is None else str(result["first_divergence_step"]),
                f"{result['exact_latency_per_step_ms']:.2f}",
                f"{result['kv_latency_per_step_ms']:.2f}",
                str(result["final_cache_bytes"]),
            )

    console.print(table)


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
    rerank_strategy: str = typer.Option("hybrid", help="Rerank strategy for temporal mode."),
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
                rerank_strategy=rerank_strategy,
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
    rerank_strategy: str = typer.Option("hybrid", help="Rerank strategy for temporal mode."),
    local_expansion_neighbors: int = typer.Option(0, min=0, help="Neighbor radius for temporal expansion ablation."),
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
        rerank_strategy=rerank_strategy,
        local_expansion_neighbors=local_expansion_neighbors,
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
    mode_order = ["full", "recent", "retrieval", "temporal"]
    if local_expansion_neighbors > 0:
        mode_order.append("temporal_expanded")
    mode_order.append("oracle")
    for mode in mode_order:
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
        distance_modes = ["recent", "retrieval", "temporal"]
        if local_expansion_neighbors > 0:
            distance_modes.append("temporal_expanded")
        distance_modes.append("oracle")
        for mode in distance_modes:
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

    breakdown_table = Table(title="Routing Breakdown")
    breakdown_table.add_column("Metric")
    breakdown_table.add_column("Retrieval")
    breakdown_table.add_column("Temporal")
    if local_expansion_neighbors > 0:
        breakdown_table.add_column("Expanded")
    retrieval_breakdown = summary["overall"]["retrieval_breakdown"]
    temporal_breakdown = summary["overall"]["temporal_breakdown"]
    expanded_breakdown = summary["overall"].get("temporal_expanded_breakdown")
    breakdown_rows = [
        ("Top-1 recall", retrieval_breakdown["top1_recall"], temporal_breakdown["top1_recall"]),
        ("Top-2 recall", retrieval_breakdown["top2_recall"], temporal_breakdown["top2_recall"]),
        ("Top-4 recall", retrieval_breakdown["top4_recall"], temporal_breakdown["top4_recall"]),
        ("MRR", retrieval_breakdown["mrr"], temporal_breakdown["mrr"]),
        ("Parse | hit", retrieval_breakdown["parse_given_hit"], temporal_breakdown["parse_given_hit"]),
        (
            "Correct | hit & parse",
            retrieval_breakdown["correct_given_hit_parse"],
            temporal_breakdown["correct_given_hit_parse"],
        ),
        ("Wrong window rate", retrieval_breakdown["wrong_window_rate"], temporal_breakdown["wrong_window_rate"]),
        (
            "Right window, unparsable",
            retrieval_breakdown["right_unparsable_rate"],
            temporal_breakdown["right_unparsable_rate"],
        ),
        (
            "Right window, parsed wrong",
            retrieval_breakdown["right_parsed_wrong_rate"],
            temporal_breakdown["right_parsed_wrong_rate"],
        ),
    ]
    for label, retrieval_value, temporal_value in breakdown_rows:
        row = [label, f"{retrieval_value:.2f}", f"{temporal_value:.2f}"]
        if expanded_breakdown is not None:
            field_map = {
                "Top-1 recall": "top1_recall",
                "Top-2 recall": "top2_recall",
                "Top-4 recall": "top4_recall",
                "MRR": "mrr",
                "Parse | hit": "parse_given_hit",
                "Correct | hit & parse": "correct_given_hit_parse",
                "Wrong window rate": "wrong_window_rate",
                "Right window, unparsable": "right_unparsable_rate",
                "Right window, parsed wrong": "right_parsed_wrong_rate",
            }
            row.append(f"{expanded_breakdown[field_map[label]]:.2f}")
        breakdown_table.add_row(*row)
    console.print(breakdown_table)

    parse_failures = collect_parse_failures(per_case)
    if parse_failures:
        failures_table = Table(title="Parse Failures")
        failures_table.add_column("Case")
        failures_table.add_column("Distance")
        failures_table.add_column("Mode")
        failures_table.add_column("Raw Output")
        for failure in parse_failures:
            failures_table.add_row(
                failure["case_id"],
                failure["distance"],
                failure["mode"],
                failure["prediction"],
            )
        console.print(failures_table)

    staged_mode = "temporal_expanded" if local_expansion_neighbors > 0 else "temporal"
    oracle_gaps = collect_staged_oracle_gaps(per_case, staged_mode=staged_mode)
    if oracle_gaps:
        gaps_table = Table(title="Staged vs Oracle Gaps")
        gaps_table.add_column("Case")
        gaps_table.add_column("Distance")
        gaps_table.add_column("Mode")
        gaps_table.add_column("Selected")
        gaps_table.add_column("Staged")
        gaps_table.add_column("Oracle")
        for row in oracle_gaps:
            gaps_table.add_row(
                row["case_id"],
                row["distance"],
                row["mode"],
                row["selected"],
                row["prediction"],
                row["oracle"],
            )
        console.print(gaps_table)


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
    rerank_strategy: str = typer.Option("hybrid", help="Rerank strategy for temporal mode."),
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
                rerank_strategy=rerank_strategy,
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


@app.command("route-apollo")
def route_apollo(
    model_path: str = typer.Option(..., help="Path to the GGUF model."),
    corpus_path: str = typer.Option("data/apollo11_clean.txt", help="Path to the Apollo corpus text file."),
    case_count: int = typer.Option(6, min=3, help="Number of generated benchmark cases."),
    windows: int = typer.Option(12, min=6, help="Number of windows per case."),
    window_lines: int = typer.Option(24, min=8, help="Lines per window."),
    recent_windows: int = typer.Option(2, min=0, help="Exact local context horizon in windows."),
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
    strategies = ["semantic", "temporal", "adjacency", "hybrid", "staged"]

    table = Table(title="Apollo Routing Comparison")
    table.add_column("Strategy")
    table.add_column("Top-1")
    table.add_column("Top-2")
    table.add_column("Top-4")
    table.add_column("MRR")

    console.print("[bold]Apollo Routing Comparison[/bold]")
    console.print(f"Backend: {runner.backend} ({runner.backend_reason})")

    for strategy in strategies:
        summary = evaluate_routing_only(
            cases=cases,
            runner=runner,
            recent_windows=recent_windows,
            strategy=strategy,
        )
        table.add_row(
            strategy,
            f"{summary['top1']:.2f}",
            f"{summary['top2']:.2f}",
            f"{summary['top4']:.2f}",
            f"{summary['mrr']:.2f}",
        )
    console.print(table)


@app.command("bridge-apollo-replay")
def bridge_apollo_replay(
    model_path: str = typer.Option(..., help="Path to the GGUF routing model."),
    hf_model_name_or_path: str = typer.Option(..., help="Transformers model id or local path for replay evaluation."),
    corpus_path: str = typer.Option("data/apollo11_clean.txt", help="Path to the Apollo corpus text file."),
    case_count: int = typer.Option(6, min=3, help="Number of generated benchmark cases."),
    windows: int = typer.Option(12, min=6, help="Number of windows per case."),
    window_lines: int = typer.Option(24, min=8, help="Lines per window."),
    recent_windows: int = typer.Option(2, min=0, help="Exact local context horizon in windows."),
    top_k: int = typer.Option(4, min=1, help="Candidate count kept after staged reranking."),
    seed: int = typer.Option(7, help="Deterministic benchmark seed."),
    n_ctx: int = typer.Option(4096, min=1024, help="Inference context size for the GGUF routing model."),
    replay_boundary_layer: int = typer.Option(6, help="Boundary layer for the replay object."),
    replay_layer: int = typer.Option(10, help="Replay layer for tracked replay objects."),
    replay_steps: int = typer.Option(10, min=1, help="Greedy continuation horizon for replay-object checks."),
    replay_top_k: int = typer.Option(5, min=1, help="Top-k width used for replay ranking stability."),
    hf_device: str = typer.Option("cpu", help="Torch device for the HF replay backend."),
    hf_dtype: str = typer.Option("auto", help="Torch dtype: auto, float32, float16, or bfloat16."),
) -> None:
    routing_runner = GGUFRunner(model_path=model_path, n_ctx=n_ctx)
    replay_runner = HFTraceRunner(
        model_name_or_path=hf_model_name_or_path,
        device=hf_device,
        dtype=hf_dtype,
    )
    cases = build_apollo_cases(
        corpus_path=corpus_path,
        case_count=case_count,
        windows=windows,
        window_lines=window_lines,
        recent_windows=recent_windows,
        seed=seed,
    )
    summary = evaluate_routed_replay_bridge(
        cases=cases,
        routing_runner=routing_runner,
        replay_runner=replay_runner,
        recent_windows=recent_windows,
        top_k=top_k,
        replay_boundary_layer=replay_boundary_layer,
        replay_layer=replay_layer,
        replay_steps=replay_steps,
        rerank_strategy="staged",
        replay_top_k=replay_top_k,
    )

    console.print(
        "[bold]Apollo Routed Replay Bridge[/bold]\n"
        "Uses staged routing (semantic pool -> temporal/PageRank rerank -> graph-local refinement), then evaluates tracked replay objects only on top-1 routed hits."
    )
    console.print(f"Routing backend: {routing_runner.backend} ({routing_runner.backend_reason})")
    console.print(f"Replay backend: {replay_runner.backend}")
    console.print(f"HF replay model: {replay_runner.model_name_or_path}")
    console.print(f"Cases: {summary['case_count']}")
    console.print(f"Rerank strategy: {summary['rerank_strategy']}")
    console.print(f"Replay cut: boundary {summary['replay_boundary_layer']} -> layer {summary['replay_layer']}")
    console.print(f"Replay horizon: {summary['replay_steps']} steps")

    routing_table = Table(title="Routing Bridge Summary")
    routing_table.add_column("Metric")
    routing_table.add_column("Value")
    routing_table.add_row("Top-1 hit rate", f"{summary['top1_hit_rate']:.2f}")
    routing_table.add_row("Top-K hit rate", f"{summary['topk_hit_rate']:.2f}")
    console.print(routing_table)

    replay_table = Table(title="Hit-Conditioned Replay Object Summary")
    replay_table.add_column("Object")
    replay_table.add_column("Bytes")
    replay_table.add_column("Hit Cases")
    replay_table.add_column("Token Agreement")
    replay_table.add_column(f"Top-{replay_top_k} Full Rate")
    replay_table.add_column("Divergence Rate")
    for row in summary["summary_rows"]:
        replay_table.add_row(
            str(row["object_label"]),
            str(row["compact_bytes"]),
            str(row["cases"]),
            f"{row['token_agreement']:.2f}",
            f"{row['topk_full_rate']:.2f}",
            f"{row['divergence_rate']:.2f}",
        )
    console.print(replay_table)


if __name__ == "__main__":
    app()
