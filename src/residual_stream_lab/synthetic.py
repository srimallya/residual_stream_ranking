from __future__ import annotations

from dataclasses import dataclass
import random


NAMES = [
    "Aria", "Basil", "Cora", "Dane", "Etta", "Fenn", "Gia", "Hale",
    "Iris", "Juno", "Kian", "Lark", "Mira", "Nash", "Orla", "Pax",
]
THREADS = [
    "aurora", "beacon", "citadel", "drift", "ember", "fjord",
]
COLORS = [
    "crimson", "teal", "violet", "silver", "gold", "orange", "indigo", "bronze",
    "scarlet", "azure", "olive", "cobalt", "pearl", "umber", "coral", "mint",
]
ROLES = [
    "original archivist",
    "field courier",
    "signal keeper",
]


@dataclass(slots=True)
class QueryCase:
    window_index: int
    role: str
    answer: str
    question: str


@dataclass(slots=True)
class BenchmarkCase:
    document: str
    queries: list[QueryCase]
    window_lines: int


def build_benchmark_case(
    windows: int,
    window_lines: int,
    queries: int,
    seed: int,
) -> BenchmarkCase:
    rng = random.Random(seed)
    blocks: list[str] = []
    query_cases: list[QueryCase] = []
    if windows < 5:
        raise ValueError("Synthetic temporal benchmark needs at least 5 windows.")

    thread_count = max(2, min(len(THREADS), (windows - 2) // len(ROLES)))
    chosen_threads = rng.sample(THREADS, thread_count)
    chosen_names = rng.sample(NAMES, thread_count * len(ROLES))
    chosen_colors = rng.sample(COLORS, thread_count * len(ROLES))

    window_index = 0
    role_lookup: dict[str, dict[str, tuple[int, str]]] = {}
    for thread_idx, thread_name in enumerate(chosen_threads):
        role_lookup[thread_name] = {}
        for role_idx, role_name in enumerate(ROLES):
            person = chosen_names[thread_idx * len(ROLES) + role_idx]
            color = chosen_colors[thread_idx * len(ROLES) + role_idx]
            role_lookup[thread_name][role_name] = (window_index, color)
            filler = [
                (
                    f"Window {window_index} note {line_idx}: "
                    f"thread {thread_name} remains linked to {role_name} operations."
                )
                for line_idx in range(window_lines - 1)
            ]
            fact_line = (
                f"Window {window_index}: thread {thread_name} assigns {person} as the "
                f"{role_name} and the marker color is {color}."
            )
            blocks.extend(filler + [fact_line])
            window_index += 1

    active_thread = chosen_threads[0]
    while window_index < windows - 2:
        filler = [
            f"Window {window_index} note {line_idx}: background archive traffic without stable facts."
            for line_idx in range(window_lines)
        ]
        blocks.extend(filler)
        window_index += 1

    blocks.extend(
        [
            "Window recent_a note 0: current control has reopened the active archive thread.",
            f"Window recent_a note 1: the active thread is {active_thread}.",
            "Window recent_a note 2: recover the original archivist, field courier, and signal keeper details.",
            "Window recent_a note 3: only historical checkpoints contain the marker colors.",
        ]
    )
    blocks.extend(
        [
            f"Window recent_b note 0: briefing remains focused on thread {active_thread}.",
            f"Window recent_b note 1: thread {active_thread} must be reconstructed from older memory.",
            "Window recent_b note 2: answer using the color only when asked.",
            "Window recent_b note 3: avoid unrelated threads during recall.",
        ]
    )

    for query_index in range(queries):
        role_name = ROLES[query_index % len(ROLES)]
        target_index, color = role_lookup[active_thread][role_name]
        query_cases.append(
            QueryCase(
                window_index=target_index,
                role=role_name,
                answer=color,
                question=(
                    f"What marker color belongs to the {role_name} of the active thread? "
                    "Answer with the color only."
                ),
            )
        )

    return BenchmarkCase(
        document="\n".join(blocks),
        queries=query_cases,
        window_lines=window_lines,
    )
