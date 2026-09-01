#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization import iga


def test_fixed_topology_geometry_residual_gradient_matches_difference():
    degree = 2
    grid = iga.BSplineGrid.open_uniform(degree, 1)
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    control_points = jnp.stack((xx, yy), axis=-1)
    weights = jnp.ones(control_points.shape[:-1])
    prepared = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        iga.NURBSGeometryState(control_points, weights),
        field_name="u",
        axis_names=("xi", "eta"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    ).prepare(numeric_version="sensitivity-base")
    source = phx.equations.coefficient(
        lambda points, args: jnp.ones(points.shape[:-1], dtype=points.dtype),
        coefficient_id="iga-sensitivity-source",
    )
    form = phx.equations.FiniteElementForm(
        "iga-sensitivity",
        "u",
        (phx.equations.SourceAction("u", source),),
    )
    constraint = prepared.homogeneous_trace_constraint("u")
    compiled = phx.equations.compile_finite_element_problem(
        form,
        prepared,
        constraint=constraint,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="sum_factorized",
        ),
    )
    reduced = constraint.reduced_space.zeros()
    direction = jnp.zeros_like(control_points).at[1, 1, 0].set(1.0)

    def objective(amplitude):
        geometry = iga.NURBSGeometryState(
            control_points + amplitude * direction,
            weights,
        )
        runtime = prepared.prepare_runtime(
            geometry,
            numeric_version="sensitivity-evaluation",
        )
        context = phx.equations.FiniteElementExecutionContext(runtime)
        residual = compiled.residual(reduced, context)
        return jnp.real(jnp.vdot(residual, residual))

    derivative = jax.grad(objective)(jnp.asarray(0.0))
    step = jnp.asarray(1.0e-4, dtype=derivative.dtype)
    difference = (objective(step) - objective(-step)) / (2.0 * step)

    assert jnp.isfinite(derivative)
    np.testing.assert_allclose(derivative, difference, rtol=2e-5, atol=2e-7)
