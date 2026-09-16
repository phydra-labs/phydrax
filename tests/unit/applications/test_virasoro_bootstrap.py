#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.applications.conformal_bootstrap import (
    BPZVirasoroBlockPlan,
    elliptic_nome,
    ising_sigma_crossing_evidence,
    IsingSigmaVirasoroPlan,
    prepare_bpz_virasoro_blocks,
    prepare_ising_sigma_virasoro_blocks,
)


def test_elliptic_nome_at_crossing_symmetric_point_is_exp_minus_pi():
    np.testing.assert_allclose(
        elliptic_nome(0.5),
        (np.exp(-np.pi),),
        rtol=2e-13,
        atol=2e-15,
    )


def test_finite_bpz_hypergeometric_compiler_matches_closed_logarithm():
    points = jnp.asarray((0.1, 0.2, 0.5))
    prepared = prepare_bpz_virasoro_blocks(
        BPZVirasoroBlockPlan(
            points,
            central_charge=0.7,
            external_weights=(0.1, 0.1, 0.1, 0.1),
            internal_weight=0.2,
            z_exponent=1.0,
            one_minus_z_exponent=0.0,
            hypergeometric_parameters=(1.0, 1.0, 2.0),
            series_order=512,
            branch_id="principal-real",
            derivation_source_id="closed-hypergeometric-control",
        )
    )
    values, _, tails = prepared.block()
    np.testing.assert_allclose(values, -jnp.log1p(-points), rtol=2e-13, atol=2e-14)
    assert float(jnp.max(tails)) < 1e-13
    evidence = prepared.evidence()
    assert bool(evidence.finite)
    assert "caller-derived" in evidence.claim


def test_exact_ising_sigma_blocks_and_channel_sum_cross_crossing():
    points = jnp.asarray((0.1, 0.2, 0.35, 0.65, 0.8, 0.9))
    identity = prepare_ising_sigma_virasoro_blocks(
        IsingSigmaVirasoroPlan(points, "identity")
    )
    energy = prepare_ising_sigma_virasoro_blocks(IsingSigmaVirasoroPlan(points, "energy"))
    identity_values = identity.block()
    energy_values = energy.block()
    assert jnp.all(identity_values > 0.0)
    assert jnp.all(energy_values > 0.0)
    correlator = identity_values**2 + energy_values**2
    expected = (points * (1.0 - points)) ** (-0.25)
    np.testing.assert_allclose(correlator, expected, rtol=2e-13, atol=2e-13)
    crossing = ising_sigma_crossing_evidence(points)
    assert bool(crossing.accepted)
    assert float(crossing.maximum_residual) < 1e-12
    np.testing.assert_allclose(crossing.direct_correlator, crossing.crossed_correlator)
    assert "not-general" in crossing.claim
