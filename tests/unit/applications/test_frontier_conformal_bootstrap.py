#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.conformal_bootstrap import (
    assemble_scalar_crossing_cone,
    compare_known_gap_bound,
    CrossingConePlan,
    exclude_scalar_gap,
    prepare_crossing_cone,
    prepare_scalar_blocks,
    ScalarBlockPlan,
    solve_crossing_cone,
)


jax.config.update("jax_enable_x64", True)


def test_frontier_sl2_scalar_block_matches_closed_delta_one_reference():
    prepared = prepare_scalar_blocks(
        ScalarBlockPlan(
            jnp.asarray((0.1, 0.2, 0.35, 0.65, 0.8, 0.9)),
            external_dimension=0.5,
            radial_order=384,
        )
    )
    block, _, tails = prepared.block(1.0)
    np.testing.assert_allclose(block, -jnp.log1p(-prepared.plan.cross_ratios), rtol=1e-12)
    assert float(jnp.max(tails)) < 1e-10
    crossing = prepared.crossing_vector(1.0)
    np.testing.assert_allclose(crossing, -jnp.flip(crossing), atol=2e-12)
    evidence = prepared.evidence(1.0)
    assert bool(evidence.finite)
    assert "reference-only" in evidence.claim


def test_frontier_crossing_conic_residual_and_gap_exclusion_evidence():
    feasible_plan = CrossingConePlan(
        jnp.asarray((-1.0, 0.0)),
        jnp.asarray(((1.0, 0.0), (0.0, 1.0))),
        jnp.asarray((1.0, 2.0)),
        maximum_iterations=256,
        residual_tolerance=1e-10,
    )
    prepared = prepare_crossing_cone(feasible_plan)
    feasible = jax.jit(solve_crossing_cone)(prepared)
    assert bool(feasible.converged)
    np.testing.assert_allclose(feasible.relative_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(feasible.coefficients, jnp.asarray((1.0, 0.0)), atol=1e-12)

    exclusion = exclude_scalar_gap(prepare_crossing_cone(feasible_plan), 2.0)
    assert bool(exclusion.excludes_candidate_gap)
    np.testing.assert_allclose(exclusion.identity_evaluation, 1.0, atol=1e-12)
    assert float(exclusion.minimum_block_evaluation) >= -1e-12
    known = compare_known_gap_bound(
        exclusion,
        2.0,
        "finite-two-component-reference-bound",
    )
    assert bool(known.reproduces_or_strengthens)
    assert "user-declared" in known.claim


def test_frontier_scalar_blocks_lower_to_fixed_crossing_matrix():
    blocks = prepare_scalar_blocks(
        ScalarBlockPlan(
            jnp.asarray((0.2, 0.35, 0.65, 0.8)),
            external_dimension=0.25,
            radial_order=64,
        )
    )
    plan = assemble_scalar_crossing_cone(
        blocks,
        (0.5, 1.0, 1.5, 2.0),
        maximum_iterations=32,
    )
    assert plan.block_vectors.shape == (4, 4)
    np.testing.assert_allclose(
        plan.block_vectors,
        -jnp.flip(plan.block_vectors, axis=0),
        atol=1e-11,
    )
