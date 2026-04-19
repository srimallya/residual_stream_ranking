from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MemoryObject:
    object_id: str
    kind: str
    bytes: int
    tier: str = "warm"
    pinned: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    source_case_id: str | None = None
    source_region_id: str | None = None
    rank_history: list[int] = field(default_factory=list)
    retrieved_count: int = 0
    topk_entries: int = 0
    downstream_utility: float = 0.0
    replay_usage_count: int = 0
    jump_score: float = 0.0
    weak_evidence_count: int = 0
    consecutive_weak_runs: int = 0
    resurgence_count: int = 0
    last_strong_recovery_at: str | None = None
    last_token_agreement: float | None = None
    last_topk_full_rate: float | None = None
    last_first_divergence_step: int | None = None
    last_steps_completed: int | None = None
    last_distance_bin: str | None = None
    last_useful_at: str | None = None
    last_retrieved_at: str | None = None
    last_reinjected_at: str | None = None

    @property
    def topk_frequency(self) -> float:
        if self.retrieved_count == 0:
            return 0.0
        return self.topk_entries / self.retrieved_count


@dataclass(slots=True)
class TierChange:
    object_id: str
    previous_tier: str
    new_tier: str
    changed_at: str = field(default_factory=utc_now_iso)


class MemoryLedger:
    def __init__(self) -> None:
        self.objects: dict[str, MemoryObject] = {}
        self.tier_changes: list[TierChange] = []

    @classmethod
    def load(cls, path: str | Path) -> MemoryLedger:
        ledger = cls()
        file_path = Path(path)
        if not file_path.exists():
            return ledger
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        for object_row in payload.get("objects", []):
            memory_object = MemoryObject(**object_row)
            ledger.objects[memory_object.object_id] = memory_object
        for change_row in payload.get("tier_changes", []):
            ledger.tier_changes.append(TierChange(**change_row))
        return ledger

    def save(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "objects": [asdict(memory_object) for memory_object in self.objects.values()],
            "tier_changes": [asdict(tier_change) for tier_change in self.tier_changes],
        }
        file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _is_weak_evidence(
        self,
        *,
        token_agreement: float | None,
        topk_full_rate: float | None,
        first_divergence_step: int | None,
        distance_bin: str | None,
    ) -> bool:
        safe_token_agreement = token_agreement if token_agreement is not None else 1.0
        safe_topk_full_rate = topk_full_rate if topk_full_rate is not None else 1.0
        return (
            safe_token_agreement < 0.999
            or first_divergence_step is not None
            or (distance_bin in {"medium", "far"} and safe_topk_full_rate < 0.90)
        )

    def _is_strong_recovery(
        self,
        *,
        token_agreement: float | None,
        topk_full_rate: float | None,
        first_divergence_step: int | None,
    ) -> bool:
        safe_token_agreement = token_agreement if token_agreement is not None else 1.0
        safe_topk_full_rate = topk_full_rate if topk_full_rate is not None else 1.0
        return (
            safe_token_agreement >= 0.999
            and safe_topk_full_rate >= 0.99
            and first_divergence_step is None
        )

    def register_event(
        self,
        *,
        object_id: str,
        kind: str,
        bytes: int,
        source_case_id: str | None,
        source_region_id: str | None,
        rank: int | None,
        retrieved: bool,
        entered_topk: bool,
        reinjected: bool,
        behavior_helped: bool,
        behavior_score: float | None = None,
        token_agreement: float | None = None,
        topk_full_rate: float | None = None,
        first_divergence_step: int | None = None,
        steps_completed: int | None = None,
        distance_bin: str | None = None,
        pinned: bool = False,
    ) -> MemoryObject:
        memory_object = self.objects.get(object_id)
        if memory_object is None:
            memory_object = MemoryObject(
                object_id=object_id,
                kind=kind,
                bytes=bytes,
                pinned=pinned,
                source_case_id=source_case_id,
                source_region_id=source_region_id,
            )
            self.objects[object_id] = memory_object

        timestamp = utc_now_iso()
        memory_object.bytes = bytes
        memory_object.source_case_id = source_case_id
        memory_object.source_region_id = source_region_id
        memory_object.pinned = memory_object.pinned or pinned

        if retrieved:
            memory_object.retrieved_count += 1
            memory_object.last_retrieved_at = timestamp
            if rank is not None:
                previous_best = min(memory_object.rank_history) if memory_object.rank_history else None
                memory_object.rank_history.append(rank)
                if previous_best is not None and rank < previous_best:
                    memory_object.jump_score = max(
                        memory_object.jump_score,
                        float(previous_best - rank),
                    )
        if entered_topk:
            memory_object.topk_entries += 1
        if reinjected:
            memory_object.replay_usage_count += 1
            memory_object.last_reinjected_at = timestamp
        if token_agreement is not None:
            memory_object.last_token_agreement = token_agreement
        if topk_full_rate is not None:
            memory_object.last_topk_full_rate = topk_full_rate
        memory_object.last_first_divergence_step = first_divergence_step
        memory_object.last_steps_completed = steps_completed
        memory_object.last_distance_bin = distance_bin
        weak_evidence = self._is_weak_evidence(
            token_agreement=token_agreement,
            topk_full_rate=topk_full_rate,
            first_divergence_step=first_divergence_step,
            distance_bin=distance_bin,
        )
        strong_recovery = self._is_strong_recovery(
            token_agreement=token_agreement,
            topk_full_rate=topk_full_rate,
            first_divergence_step=first_divergence_step,
        )
        if weak_evidence:
            memory_object.weak_evidence_count += 1
            memory_object.consecutive_weak_runs += 1
        else:
            if strong_recovery and memory_object.consecutive_weak_runs > 0:
                memory_object.resurgence_count += 1
                memory_object.last_strong_recovery_at = timestamp
            memory_object.consecutive_weak_runs = 0
        if behavior_helped:
            memory_object.downstream_utility += behavior_score if behavior_score is not None else 1.0
            memory_object.last_useful_at = timestamp
        elif behavior_score is not None:
            memory_object.downstream_utility += behavior_score
        return memory_object

    def set_tier(self, object_id: str, new_tier: str) -> None:
        memory_object = self.objects[object_id]
        if memory_object.tier == new_tier:
            return
        self.tier_changes.append(
            TierChange(
                object_id=object_id,
                previous_tier=memory_object.tier,
                new_tier=new_tier,
            )
        )
        memory_object.tier = new_tier

    def tier_counts(self) -> dict[str, int]:
        counts = {"pinned": 0, "warm": 0, "cold": 0, "archived": 0, "pruned": 0}
        for memory_object in self.objects.values():
            counts[memory_object.tier] = counts.get(memory_object.tier, 0) + 1
        return counts

    def recent_tier_changes(self, *, limit: int = 5) -> list[TierChange]:
        return self.tier_changes[-limit:]

    def pinned_objects(self) -> list[MemoryObject]:
        return sorted(
            (memory_object for memory_object in self.objects.values() if memory_object.pinned),
            key=lambda memory_object: memory_object.object_id,
        )

    def low_utility_tail(self, *, limit: int = 5) -> list[MemoryObject]:
        return sorted(
            self.objects.values(),
            key=lambda memory_object: (
                memory_object.downstream_utility,
                memory_object.topk_frequency,
                -memory_object.replay_usage_count,
                memory_object.object_id,
            ),
        )[:limit]

    def suggest_transition(self, memory_object: MemoryObject) -> dict[str, object] | None:
        if memory_object.pinned:
            return None
        if memory_object.replay_usage_count == 0:
            return None

        reasons: list[str] = []
        confidence = 0.0
        suggested_tier: str | None = None
        token_agreement = memory_object.last_token_agreement if memory_object.last_token_agreement is not None else 1.0
        topk_full_rate = memory_object.last_topk_full_rate if memory_object.last_topk_full_rate is not None else 1.0
        distance_bin = memory_object.last_distance_bin
        first_divergence_step = memory_object.last_first_divergence_step
        steps_completed = memory_object.last_steps_completed or 1

        if token_agreement < 0.999:
            reasons.append("token drift")
            confidence += 0.25
        if topk_full_rate < 0.95:
            reasons.append("ranking softness")
            confidence += 0.20
        if first_divergence_step is not None:
            reasons.append("early divergence")
            confidence += 0.20 + 0.10 * (1.0 - ((first_divergence_step - 1) / max(steps_completed, 1)))
        if distance_bin in {"medium", "far"} and (token_agreement < 0.999 or topk_full_rate < 0.95):
            reasons.append(f"{distance_bin}-bucket failure")
            confidence += 0.15
        if memory_object.jump_score < 0.5:
            reasons.append("no meaningful jump")
            confidence += 0.05
        if memory_object.last_useful_at is None:
            reasons.append("stale usefulness")
            confidence += 0.05

        strong_continuation_weakness = token_agreement < 0.999 or first_divergence_step is not None
        harder_bucket_ranking_weakness = (
            distance_bin in {"medium", "far"} and topk_full_rate < 0.90
        )

        if (
            memory_object.tier == "warm"
            and (
                strong_continuation_weakness
                or harder_bucket_ranking_weakness
            )
        ):
            suggested_tier = "cold"

        if (
            memory_object.tier == "cold"
            and memory_object.weak_evidence_count >= 2
            and memory_object.consecutive_weak_runs >= 2
            and memory_object.resurgence_count == 0
            and (
                memory_object.downstream_utility < 0.90
                or (distance_bin in {"medium", "far"} and (token_agreement < 0.95 or topk_full_rate < 0.85))
            )
        ):
            suggested_tier = "archived"
            confidence += 0.10

        if suggested_tier is None:
            return None

        confidence = max(0.0, min(confidence, 0.99))
        return {
            "object_id": memory_object.object_id,
            "kind": memory_object.kind,
            "current_tier": memory_object.tier,
            "suggested_tier": suggested_tier,
            "bytes": memory_object.bytes,
            "replay_usage_count": memory_object.replay_usage_count,
            "topk_frequency": memory_object.topk_frequency,
            "downstream_utility": memory_object.downstream_utility,
            "jump_score": memory_object.jump_score,
            "weak_evidence_count": memory_object.weak_evidence_count,
            "consecutive_weak_runs": memory_object.consecutive_weak_runs,
            "resurgence_count": memory_object.resurgence_count,
            "last_strong_recovery_at": memory_object.last_strong_recovery_at,
            "last_useful_at": memory_object.last_useful_at,
            "token_agreement": token_agreement,
            "topk_full_rate": topk_full_rate,
            "first_divergence_step": first_divergence_step,
            "distance_bin": distance_bin,
            "suggested_reason": ", ".join(reasons),
            "confidence": confidence,
        }

    def candidate_rows(self, *, limit: int = 5) -> list[dict[str, object]]:
        candidates = [
            suggestion
            for memory_object in self.objects.values()
            if (suggestion := self.suggest_transition(memory_object)) is not None
        ]
        candidates.sort(
            key=lambda row: (
                row["suggested_tier"] != "archived",
                row["downstream_utility"],
                row["topk_frequency"],
                -row["confidence"],
                row["object_id"],
            )
        )
        return candidates[:limit]

    def apply_cold_transitions(
        self,
        *,
        confidence_threshold: float = 0.30,
        minimum_reason_count: int = 2,
    ) -> list[dict[str, object]]:
        applied: list[dict[str, object]] = []
        for memory_object in self.objects.values():
            suggestion = self.suggest_transition(memory_object)
            if suggestion is None:
                continue
            if suggestion["suggested_tier"] != "cold":
                continue
            reason_count = len([part for part in str(suggestion["suggested_reason"]).split(", ") if part])
            if suggestion["confidence"] < confidence_threshold:
                continue
            if reason_count < minimum_reason_count:
                continue
            previous_tier = memory_object.tier
            self.set_tier(memory_object.object_id, "cold")
            applied.append(
                {
                    "object_id": memory_object.object_id,
                    "kind": memory_object.kind,
                    "previous_tier": previous_tier,
                    "new_tier": memory_object.tier,
                    "confidence": suggestion["confidence"],
                    "suggested_reason": suggestion["suggested_reason"],
                    "downstream_utility": suggestion["downstream_utility"],
                    "token_agreement": suggestion["token_agreement"],
                    "topk_full_rate": suggestion["topk_full_rate"],
                    "distance_bin": suggestion["distance_bin"],
                    "consecutive_weak_runs": suggestion["consecutive_weak_runs"],
                    "resurgence_count": suggestion["resurgence_count"],
                    "changed_at": self.tier_changes[-1].changed_at,
                }
            )
        return applied

    def report(self, *, apply_cold_transitions: bool = False) -> dict[str, object]:
        before_counts = self.tier_counts()
        applied_transitions: list[dict[str, object]] = []
        if apply_cold_transitions:
            applied_transitions = self.apply_cold_transitions()
        after_counts = self.tier_counts()
        return {
            "object_count": len(self.objects),
            "tier_counts_before": before_counts,
            "tier_counts": after_counts,
            "tier_counts_after": after_counts,
            "recent_tier_changes": [
                {
                    "object_id": tier_change.object_id,
                    "previous_tier": tier_change.previous_tier,
                    "new_tier": tier_change.new_tier,
                    "changed_at": tier_change.changed_at,
                }
                for tier_change in self.recent_tier_changes()
            ],
            "applied_transitions": applied_transitions,
            "low_utility_tail": [
                {
                    "object_id": memory_object.object_id,
                    "kind": memory_object.kind,
                    "tier": memory_object.tier,
                    "bytes": memory_object.bytes,
                    "downstream_utility": memory_object.downstream_utility,
                    "topk_frequency": memory_object.topk_frequency,
                    "replay_usage_count": memory_object.replay_usage_count,
                    "weak_evidence_count": memory_object.weak_evidence_count,
                    "consecutive_weak_runs": memory_object.consecutive_weak_runs,
                    "resurgence_count": memory_object.resurgence_count,
                    "last_strong_recovery_at": memory_object.last_strong_recovery_at,
                    "last_useful_at": memory_object.last_useful_at,
                    "token_agreement": memory_object.last_token_agreement,
                    "topk_full_rate": memory_object.last_topk_full_rate,
                    "distance_bin": memory_object.last_distance_bin,
                }
                for memory_object in self.low_utility_tail()
            ],
            "archive_candidates": self.candidate_rows(),
            "pinned_objects": [
                {
                    "object_id": memory_object.object_id,
                    "kind": memory_object.kind,
                    "tier": memory_object.tier,
                    "bytes": memory_object.bytes,
                }
                for memory_object in self.pinned_objects()
            ],
        }
