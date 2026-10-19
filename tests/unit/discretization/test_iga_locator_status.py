# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""IGA point-location admission retains native numerical failures."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import FieldQueryStatus
from phydrax.discretization.iga import (
    BSplineGrid,
    IsogeometricFieldReconstructionKernel,
    IsogeometricPlan,
    IsogeometricQuadraturePolicy,
    NURBSGeometryState,
)


@pytest.mark.parametrize(
    "derivative", ((0, 0), (1, 0)), ids=("value", "physical-gradient")
)
@pytest.mark.parametrize(
    ("x_scale", "expected_status", "finite_conditioning"),
    (
        pytest.param(1.0, FieldQueryStatus.VALID, True, id="regular-map"),
        pytest.param(
            2.0**-60, FieldQueryStatus.ILL_CONDITIONED, False, id="refused-small-solve"
        ),
    ),
)
def test_locator_does_not_accept_a_seed_after_native_solve_refusal(
    derivative: tuple[int, ...],
    x_scale: float,
    expected_status: FieldQueryStatus,
    finite_conditioning: bool,
) -> None:
    grid = BSplineGrid(np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64), 1)
    x, y = np.meshgrid(np.asarray((0.0, 1.0)), np.asarray((0.0, 1.0)), indexing="ij")
    controls = np.stack((x, y), axis=-1)
    weights = np.ones((2, 2), dtype=np.float64)
    prepared = IsogeometricPlan.isoparametric(
        (grid, grid),
        NURBSGeometryState(controls, weights),
        axis_names=("xi", "eta"),
        quadrature_policy=IsogeometricQuadraturePolicy(2),
    ).prepare()
    runtime = prepared.prepare_runtime(
        NURBSGeometryState(controls * np.asarray((x_scale, 1.0)), weights),
        numeric_version="query-map",
    )
    kernel = IsogeometricFieldReconstructionKernel(
        prepared,
        0,
        runtime,
        newton_iterations=2,
        location_tolerance=1.0e-11,
        seeds_per_axis=1,
    )
    # The center is the seed itself, so a zero residual alone cannot distinguish
    # a usable Jacobian from the determinant-refused 2**-60 scaled map.
    _, evidence = kernel.locate(
        jnp.asarray(((0.5 * x_scale, 0.5),), dtype=jnp.float64), derivative, None
    )

    np.testing.assert_array_equal(evidence.status, (int(expected_status),))
    assert bool(evidence.valid[0]) is (expected_status is FieldQueryStatus.VALID)
    assert bool(jnp.isfinite(evidence.conditioning[0])) is finite_conditioning
