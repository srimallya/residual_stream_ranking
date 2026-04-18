from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path


COLORS = [
    "crimson", "teal", "violet", "silver", "gold", "orange", "indigo", "bronze",
    "scarlet", "azure", "olive", "cobalt", "pearl", "umber", "coral", "mint",
]


@dataclass(slots=True)
class ApolloCase:
    document: str
    question: str
    answer: str
    target_window_index: int
    distance_bin: str
    case_id: str
    window_lines: int


def load_apollo_lines(corpus_path: str) -> list[str]:
    path = Path(corpus_path)
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return lines


def target_index_for_bin(old_window_count: int, distance_bin: str, rng: random.Random) -> int:
    if distance_bin == "far":
        candidates = list(range(0, max(1, old_window_count // 3)))
    elif distance_bin == "medium":
        start = max(0, old_window_count // 3)
        end = max(start + 1, (2 * old_window_count) // 3)
        candidates = list(range(start, end))
    elif distance_bin == "near":
        start = max(0, (2 * old_window_count) // 3)
        candidates = list(range(start, old_window_count))
    else:
        raise ValueError(f"Unsupported distance bin: {distance_bin}")
    return rng.choice(candidates)


def build_apollo_cases(
    *,
    corpus_path: str,
    case_count: int,
    windows: int,
    window_lines: int,
    recent_windows: int,
    seed: int,
) -> list[ApolloCase]:
    if recent_windows < 0:
        raise ValueError("recent_windows must be non-negative.")
    if windows <= max(recent_windows, 1):
        raise ValueError("windows must exceed recent_windows.")

    lines = load_apollo_lines(corpus_path)
    rng = random.Random(seed)
    old_window_count = windows - recent_windows
    distance_cycle = ["far", "medium", "near"]
    cases: list[ApolloCase] = []

    for case_index in range(case_count):
        distance_bin = distance_cycle[case_index % len(distance_cycle)]
        target_window_index = target_index_for_bin(old_window_count, distance_bin, rng)
        start = rng.randint(0, max(0, len(lines) - windows * window_lines - 1))

        windows_text: list[list[str]] = []
        for window_idx in range(windows):
            chunk_start = start + window_idx * window_lines
            chunk = lines[chunk_start : chunk_start + window_lines]
            windows_text.append(list(chunk))

        case_id = f"ARCHIVE-{case_index:04d}"
        answer = COLORS[case_index % len(COLORS)]
        needle_line = f"Archive case {case_id} marker color is {answer}."
        target_window = windows_text[target_window_index]
        insert_at = rng.randint(0, len(target_window))
        target_window.insert(insert_at, needle_line)

        distractor_windows = [idx for idx in range(old_window_count) if idx != target_window_index]
        rng.shuffle(distractor_windows)
        for distractor_count, distractor_window_index in enumerate(distractor_windows[:2], start=1):
            distractor_color = COLORS[(case_index + distractor_count) % len(COLORS)]
            distractor_id = f"DISTRACTOR-{case_index:04d}-{distractor_count}"
            windows_text[distractor_window_index].insert(
                min(1, len(windows_text[distractor_window_index])),
                f"Archive case {distractor_id} marker color is {distractor_color}.",
            )

        if recent_windows >= 2:
            recent_anchor = windows - recent_windows
            windows_text[recent_anchor].append(f"Current archive request concerns case {case_id}.")
            windows_text[recent_anchor].append("The requested marker appears in older notes.")
            windows_text[recent_anchor + 1].append(f"Recover the marker color for case {case_id}.")
            windows_text[recent_anchor + 1].append("Return the color only.")
        elif recent_windows == 1:
            recent_anchor = windows - 1
            windows_text[recent_anchor].append(f"Current archive request concerns case {case_id}.")
            windows_text[recent_anchor].append(f"Recover the marker color for case {case_id}.")
            windows_text[recent_anchor].append("The requested marker appears in older notes.")
            windows_text[recent_anchor].append("Return the color only.")

        document = "\n".join("\n".join(chunk) for chunk in windows_text)
        question = f"What is the marker color for case {case_id}? Answer with the color only."
        cases.append(
            ApolloCase(
                document=document,
                question=question,
                answer=answer,
                target_window_index=target_window_index,
                distance_bin=distance_bin,
                case_id=case_id,
                window_lines=window_lines,
            )
        )

    return cases
