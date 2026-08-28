#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.uq._nested_diagnostics import (
    build_nested_diagnostics,
    insertion_rank_pvalue,
    rolling_insertion_rank_pvalues,
)


def test_insertion_rank_crosscheck_distinguishes_uniform_and_biased_sequences():
    uniform = jnp.tile(jnp.arange(10, dtype=jnp.int32), 40)
    biased = jnp.zeros((400,), dtype=jnp.int32)

    assert insertion_rank_pvalue(uniform, 10) > 0.99
    assert insertion_rank_pvalue(biased, 10) < 1e-12
    assert rolling_insertion_rank_pvalues(uniform, 10).shape == (4,)


def test_nested_diagnostics_reports_constraint_and_lineage_evidence():
    diagnostics = build_nested_diagnostics(
        dead_log_likelihood=jnp.asarray([0.0, 1.0, 2.0, 3.0]),
        dead_birth_log_likelihood=jnp.asarray([jnp.nan, 0.0, 1.0, 2.0]),
        insertion_ranks=jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        inner_accepted=jnp.ones((4, 2), dtype=bool),
        num_expansions=jnp.zeros((4, 2), dtype=jnp.int32),
        num_shrink=jnp.ones((4, 2), dtype=jnp.int32),
        max_expansions=10,
        max_shrinkage=100,
        initial_log_likelihood=jnp.asarray([0.0, 1.0, -jnp.inf, 2.0]),
        sample_ids=jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        posterior_log_weights=jnp.full((4,), -jnp.log(4.0)),
        num_live=4,
        quadrature_valid=jnp.asarray(True),
        final_live_positions=jnp.arange(4.0)[:, None],
    )

    assert diagnostics.passed
    assert diagnostics.constraints_satisfied
    assert diagnostics.likelihood_monotonic
    assert diagnostics.initial_finite_fraction == 0.75
    assert diagnostics.unique_lineage_count == 4
    assert diagnostics.effective_lineage_count == 4.0
    assert diagnostics.covariance_rank == 1
    assert diagnostics.covariance_condition == 1.0


def test_nested_diagnostics_preserves_failures_without_repair():
    diagnostics = build_nested_diagnostics(
        dead_log_likelihood=jnp.asarray([1.0, 0.0]),
        dead_birth_log_likelihood=jnp.asarray([0.0, 0.5]),
        insertion_ranks=jnp.asarray([0, 1], dtype=jnp.int32),
        inner_accepted=jnp.zeros((2, 2), dtype=bool),
        num_expansions=jnp.full((2, 2), 10, dtype=jnp.int32),
        num_shrink=jnp.full((2, 2), 100, dtype=jnp.int32),
        max_expansions=10,
        max_shrinkage=100,
        initial_log_likelihood=jnp.asarray([0.0, 1.0]),
        sample_ids=jnp.asarray([0, 1], dtype=jnp.int32),
        posterior_log_weights=jnp.full((2,), -jnp.log(2.0)),
        num_live=2,
        quadrature_valid=jnp.asarray(False),
        final_live_positions=jnp.zeros((2, 1)),
    )

    assert not diagnostics.passed
    assert not diagnostics.likelihood_monotonic
    assert not diagnostics.constraints_satisfied
    assert diagnostics.expansion_cap_fraction == 1.0
    assert diagnostics.shrinkage_cap_fraction == 1.0
    assert diagnostics.zero_movement_fraction == 1.0
