from __future__ import annotations

import unittest

import numpy as np

from residual_stream_lab.trace import (
    build_trace_payload_from_states,
    capture_from_layer_states,
    NullTraceProvider,
    TraceCheckpointPayload,
    TraceMetadata,
    reconstruct_boundary_state,
    verify_reconstruction,
)


class TraceReconstructionTests(unittest.TestCase):
    def test_build_trace_payload_from_states_computes_incremental_deltas(self) -> None:
        layer_states = {
            22: np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            23: np.asarray([1.5, 1.0, 4.0], dtype=np.float32),
            24: np.asarray([3.0, 2.0, 2.0], dtype=np.float32),
        }

        payload = build_trace_payload_from_states(
            layer_states=layer_states,
            layer_cutoff_b=22,
            token_index=5,
            delta_layers=[23, 24],
        )

        self.assertEqual(payload.layer_cutoff_b, 22)
        self.assertEqual(payload.boundary_token_index, 5)
        self.assertTrue(np.allclose(payload.boundary_residual, layer_states[22]))
        self.assertTrue(np.allclose(payload.late_layer_deltas[0], np.asarray([0.5, -1.0, 1.0], dtype=np.float32)))
        self.assertTrue(np.allclose(payload.late_layer_deltas[1], np.asarray([1.5, 1.0, -2.0], dtype=np.float32)))
        self.assertTrue(payload.is_exact_reconstructable)

    def test_reconstruction_matches_direct_sum(self) -> None:
        payload = TraceCheckpointPayload(
            layer_cutoff_b=22,
            boundary_token_index=128,
            boundary_residual=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            late_layer_deltas=[
                np.asarray([0.5, -0.5, 1.0], dtype=np.float32),
                np.asarray([1.5, 0.5, -2.0], dtype=np.float32),
            ],
            delta_layers=[23, 24],
            metadata=TraceMetadata(shape=(3,), dtype="float32", backend="test"),
            exact_trace=True,
        )
        observed = np.asarray([3.0, 2.0, 2.0], dtype=np.float32)

        reconstructed = reconstruct_boundary_state(payload)
        verification = verify_reconstruction(payload, observed)

        self.assertTrue(np.allclose(reconstructed, observed))
        self.assertAlmostEqual(verification.l2_error, 0.0, places=6)
        self.assertAlmostEqual(verification.cosine_similarity, 1.0, places=6)
        self.assertTrue(verification.exact_trace)

    def test_capture_from_layer_states_preserves_observed_target(self) -> None:
        layer_states = {
            10: np.asarray([0.0, 1.0], dtype=np.float32),
            11: np.asarray([1.0, 2.0], dtype=np.float32),
            12: np.asarray([3.0, 5.0], dtype=np.float32),
        }

        capture = capture_from_layer_states(
            layer_states=layer_states,
            layer_cutoff_b=10,
            token_index=0,
            delta_layers=[11, 12],
        )
        verification = verify_reconstruction(capture.payload, capture.observed_states[12])

        self.assertAlmostEqual(verification.l2_error, 0.0, places=6)
        self.assertAlmostEqual(verification.cosine_similarity, 1.0, places=6)
        self.assertIn(12, capture.observed_states)

    def test_null_trace_provider_marks_payload_unavailable(self) -> None:
        provider = NullTraceProvider()
        payload = provider.capture_boundary_residual(
            boundary_token_index=4,
            layer_cutoff_b=12,
            window_text="dummy",
        )

        self.assertFalse(payload.exact_trace)
        self.assertFalse(payload.approximate_trace)
        self.assertEqual(payload.metadata.provenance, "null-trace-provider")


if __name__ == "__main__":
    unittest.main()
