from __future__ import annotations

import unittest

import numpy as np

from residual_stream_lab.trace import (
    NullTraceProvider,
    TraceCheckpointPayload,
    TraceMetadata,
    reconstruct_boundary_state,
    verify_reconstruction,
)


class TraceReconstructionTests(unittest.TestCase):
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
