from __future__ import annotations

import unittest

from residual_stream_lab.memory_ledger import MemoryLedger


class MemoryLedgerTests(unittest.TestCase):
    def test_register_event_updates_usage_fields(self) -> None:
        ledger = MemoryLedger()

        ledger.register_event(
            object_id="case-1:token@10/fp16",
            kind="token@10/fp16",
            bytes=1536,
            source_case_id="case-1",
            source_region_id="case-1:window:3",
            rank=1,
            retrieved=True,
            entered_topk=True,
            reinjected=True,
            behavior_helped=True,
            behavior_score=1.0,
            token_agreement=1.0,
            topk_full_rate=1.0,
            first_divergence_step=None,
            steps_completed=10,
            distance_bin="near",
        )

        memory_object = ledger.objects["case-1:token@10/fp16"]
        self.assertEqual(memory_object.tier, "warm")
        self.assertEqual(memory_object.rank_history, [1])
        self.assertEqual(memory_object.retrieved_count, 1)
        self.assertEqual(memory_object.topk_entries, 1)
        self.assertEqual(memory_object.replay_usage_count, 1)
        self.assertEqual(memory_object.downstream_utility, 1.0)
        self.assertEqual(memory_object.last_token_agreement, 1.0)
        self.assertEqual(memory_object.last_topk_full_rate, 1.0)
        self.assertEqual(memory_object.last_distance_bin, "near")
        self.assertIsNotNone(memory_object.last_retrieved_at)
        self.assertIsNotNone(memory_object.last_reinjected_at)
        self.assertIsNotNone(memory_object.last_useful_at)

    def test_candidate_rows_surface_zero_utility_objects(self) -> None:
        ledger = MemoryLedger()

        ledger.register_event(
            object_id="case-2:token@10/int8",
            kind="token@10/int8",
            bytes=772,
            source_case_id="case-2",
            source_region_id="case-2:window:5",
            rank=3,
            retrieved=True,
            entered_topk=False,
            reinjected=True,
            behavior_helped=False,
            behavior_score=0.78,
            token_agreement=0.83,
            topk_full_rate=0.78,
            first_divergence_step=4,
            steps_completed=10,
            distance_bin="medium",
        )

        candidates = ledger.candidate_rows()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["object_id"], "case-2:token@10/int8")
        self.assertEqual(candidates[0]["suggested_tier"], "archived")
        self.assertLess(candidates[0]["downstream_utility"], 0.9)
        self.assertIn("token drift", candidates[0]["suggested_reason"])
        self.assertIn("medium-bucket failure", candidates[0]["suggested_reason"])
        self.assertGreater(candidates[0]["confidence"], 0.0)

    def test_candidate_rows_can_suggest_archived_for_stale_objects(self) -> None:
        ledger = MemoryLedger()

        ledger.register_event(
            object_id="case-4:token@10/int8",
            kind="token@10/int8",
            bytes=772,
            source_case_id="case-4",
            source_region_id="case-4:window:7",
            rank=3,
            retrieved=True,
            entered_topk=False,
            reinjected=True,
            behavior_helped=False,
            behavior_score=0.76,
            token_agreement=0.80,
            topk_full_rate=0.74,
            first_divergence_step=3,
            steps_completed=10,
            distance_bin="far",
        )
        ledger.register_event(
            object_id="case-4:token@10/int8",
            kind="token@10/int8",
            bytes=772,
            source_case_id="case-4",
            source_region_id="case-4:window:7",
            rank=4,
            retrieved=True,
            entered_topk=False,
            reinjected=True,
            behavior_helped=False,
            behavior_score=0.70,
            token_agreement=0.75,
            topk_full_rate=0.70,
            first_divergence_step=2,
            steps_completed=10,
            distance_bin="far",
        )

        candidates = ledger.candidate_rows()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["suggested_tier"], "archived")
        self.assertIn("no meaningful jump", candidates[0]["suggested_reason"])
        self.assertIn("far-bucket failure", candidates[0]["suggested_reason"])

    def test_report_contains_expected_sections(self) -> None:
        ledger = MemoryLedger()
        ledger.register_event(
            object_id="case-3:text@window",
            kind="text@window",
            bytes=200,
            source_case_id="case-3",
            source_region_id="case-3:window:1",
            rank=1,
            retrieved=True,
            entered_topk=True,
            reinjected=True,
            behavior_helped=True,
            behavior_score=1.0,
            token_agreement=1.0,
            topk_full_rate=1.0,
            first_divergence_step=None,
            steps_completed=10,
            distance_bin="near",
            pinned=True,
        )

        report = ledger.report()
        self.assertIn("tier_counts", report)
        self.assertIn("low_utility_tail", report)
        self.assertIn("archive_candidates", report)
        self.assertIn("pinned_objects", report)
        self.assertEqual(report["object_count"], 1)
        self.assertEqual(report["tier_counts"]["warm"], 1)
        self.assertEqual(len(report["pinned_objects"]), 1)

    def test_apply_cold_transitions_moves_only_cold_candidates(self) -> None:
        ledger = MemoryLedger()
        ledger.register_event(
            object_id="case-5:token@10/int8",
            kind="token@10/int8",
            bytes=772,
            source_case_id="case-5",
            source_region_id="case-5:window:2",
            rank=2,
            retrieved=True,
            entered_topk=True,
            reinjected=True,
            behavior_helped=False,
            behavior_score=0.93,
            token_agreement=1.0,
            topk_full_rate=0.75,
            first_divergence_step=None,
            steps_completed=10,
            distance_bin="near",
        )

        report = ledger.report(apply_cold_transitions=True)
        self.assertEqual(report["tier_counts_before"]["warm"], 1)
        self.assertEqual(report["tier_counts_after"]["cold"], 1)
        self.assertEqual(len(report["applied_transitions"]), 1)
        self.assertEqual(report["applied_transitions"][0]["new_tier"], "cold")

    def test_apply_cold_transitions_does_not_apply_archived_suggestions(self) -> None:
        ledger = MemoryLedger()
        ledger.register_event(
            object_id="case-6:token@10/int8",
            kind="token@10/int8",
            bytes=772,
            source_case_id="case-6",
            source_region_id="case-6:window:8",
            rank=4,
            retrieved=True,
            entered_topk=False,
            reinjected=True,
            behavior_helped=False,
            behavior_score=0.05,
            token_agreement=0.0,
            topk_full_rate=0.05,
            first_divergence_step=1,
            steps_completed=10,
            distance_bin="far",
        )

        report = ledger.report(apply_cold_transitions=True)
        self.assertEqual(report["tier_counts_after"]["warm"], 1)
        self.assertEqual(report["tier_counts_after"]["cold"], 0)
        self.assertEqual(len(report["applied_transitions"]), 0)


if __name__ == "__main__":
    unittest.main()
