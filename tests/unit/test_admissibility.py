#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_admissibility_normalizes_nonfinite_and_composes_reasons():
    header = phx.AdmissibilityHeader(
        jnp.asarray((0.25, -0.1, jnp.nan)),
        jnp.asarray(
            (
                0,
                int(phx.AdmissibilityReason.OUTSIDE_SUPPORT),
                0,
            ),
            dtype=jnp.uint32,
        ),
        "model",
        "evidence",
    )

    np.testing.assert_array_equal(header.eligible, (True, False, False))
    assert np.isneginf(np.asarray(header.margin)[2])
    assert int(header.reason_bits[2]) & int(phx.AdmissibilityReason.NONFINITE)
    assert not bool(header.globally_eligible)


def test_admissibility_is_jittable_and_transition_is_diagnostic_only():
    @jax.jit
    def evaluate(margin):
        return phx.AdmissibilityHeader(
            margin,
            phx.reason_bits_where(
                margin >= 0.0,
                phx.AdmissibilityReason.OUTSIDE_SUPPORT,
            ),
            "fixed-model",
            "fixed-evidence",
        )

    header = evaluate(jnp.asarray((1.0, -1.0)))
    request = phx.AdmissibilityTransitionRequest(
        jnp.asarray((False, True)),
        jnp.asarray(3),
        header,
        "continuum",
        "kinetic",
    )

    np.testing.assert_array_equal(request.region_mask, (False, True))
    assert int(request.requested_epoch) == 3
    assert request.current_model_id == "continuum"
    assert request.target_model_id == "kinetic"


def test_combined_admissibility_preserves_worst_margin_and_reason_union():
    first = phx.AdmissibilityHeader(
        jnp.asarray((0.5, 0.2)),
        jnp.zeros((2,), dtype=jnp.uint32),
        "first",
        "first-evidence",
    )
    second = phx.AdmissibilityHeader(
        jnp.asarray((0.1, -0.4)),
        jnp.asarray(
            (0, int(phx.AdmissibilityReason.CAPACITY_INSUFFICIENT)),
            dtype=jnp.uint32,
        ),
        "second",
        "second-evidence",
    )

    combined = phx.combine_admissibility((first, second), "coupled")

    np.testing.assert_allclose(combined.margin, (0.1, -0.4))
    np.testing.assert_array_equal(combined.eligible, (True, False))
    assert int(combined.reason_bits[1]) & int(
        phx.AdmissibilityReason.CAPACITY_INSUFFICIENT
    )
