#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import pytest

from phydrax.integration._vegas import (
    FrozenVegasGrid,
    prepare_vegas,
    run_vegas,
    transform_frozen_vegas,
    VegasPlan,
    VegasPreparationEvidence,
    VegasStatus,
)


def test_vegas_calibrates_peaked_integral_and_freezes_production_grid():
    plan = VegasPlan(
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([1.0, 1.0]),
        bins=16,
        adaptation_iterations=4,
        adaptation_samples=2048,
        production_iterations=6,
        production_samples=4096,
    )

    def integrand(points):
        peak = jnp.exp(-120.0 * (points[:, 0] - 0.27) ** 2)
        return peak * (1.0 + points[:, 1])

    prepared = prepare_vegas(integrand, plan, jr.key(101))
    frozen_edges = prepared.grid.edges
    result = run_vegas(integrand, prepared, jr.key(102))
    reference = (
        1.5
        * jnp.sqrt(jnp.pi / 120.0)
        * 0.5
        * (
            jsp.special.erf(jnp.sqrt(120.0) * (1.0 - 0.27))
            + jsp.special.erf(jnp.sqrt(120.0) * 0.27)
        )
    )

    assert isinstance(result.grid, FrozenVegasGrid)
    assert result.grid.grid_id == prepared.grid.grid_id
    assert jnp.array_equal(result.grid.edges, frozen_edges)
    assert result.status == int(VegasStatus.CONVERGED)
    assert jnp.abs(result.value - reference) <= (6.0 * result.standard_error + 2.0e-4)
    assert result.standard_error > 0.0
    assert result.iteration_estimates.shape == (plan.production_iterations,)


def test_vegas_plan_enforces_fixed_evaluation_guard():
    with pytest.raises(ValueError, match="max_evaluations"):
        VegasPlan(
            [0.0],
            [1.0],
            bins=8,
            adaptation_iterations=4,
            adaptation_samples=100,
            production_iterations=4,
            production_samples=100,
            max_evaluations=799,
        )


def test_frozen_vegas_rejects_points_outside_the_unit_cube():
    evidence = VegasPreparationEvidence(
        iteration_estimates=jnp.zeros((1,)),
        marginal_weights=jnp.ones((1, 2)),
        finite_iterations=jnp.ones((1,), dtype=jnp.bool_),
        status=jnp.asarray(VegasStatus.CONVERGED, dtype=jnp.int32),
        num_evaluations=jnp.asarray(2, dtype=jnp.int32),
    )
    grid = FrozenVegasGrid(
        jnp.asarray([[0.0, 0.5, 1.0]]),
        evidence,
        plan_id="vegas-unit-domain-test",
    )

    for invalid in (
        jnp.asarray([[-0.1]]),
        jnp.asarray([[1.1]]),
        jnp.asarray([[jnp.inf]]),
    ):
        with pytest.raises(Exception, match=r"finite coordinates in \[0, 1\]"):
            transform_frozen_vegas(grid, invalid)
