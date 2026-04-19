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
        )

        memory_object = ledger.objects["case-1:token@10/fp16"]
        self.assertEqual(memory_object.tier, "warm")
        self.assertEqual(memory_object.rank_history, [1])
        self.assertEqual(memory_object.retrieved_count, 1)
        self.assertEqual(memory_object.topk_entries, 1)
        self.assertEqual(memory_object.replay_usage_count, 1)
        self.assertEqual(memory_object.downstream_utility, 1.0)
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
            rank=1,
            retrieved=True,
            entered_topk=True,
            reinjected=True,
            behavior_helped=False,
        )

        candidates = ledger.candidate_rows()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["object_id"], "case-2:token@10/int8")
        self.assertEqual(candidates[0]["suggested_tier"], "cold")
        self.assertEqual(candidates[0]["downstream_utility"], 0.0)

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


if __name__ == "__main__":
    unittest.main()
