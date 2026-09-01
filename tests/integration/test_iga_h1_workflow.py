#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization import iga


def test_public_isogeometric_workflow_compiles_and_executes():
    grid = iga.BSplineGrid.open_uniform(2, 1)
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    geometry = iga.NURBSGeometryState(
        jnp.stack((xx, yy), axis=-1),
        jnp.ones((grid.coefficient_count, grid.coefficient_count)),
    )
    plan = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        geometry,
        field_name="temperature",
        axis_names=("xi", "eta"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    )
    prepared = plan.prepare(numeric_version="public-workflow")
    constraint = prepared.homogeneous_trace_constraint("temperature")
    form = phx.equations.FiniteElementForm(
        "iga-public-workflow",
        "temperature",
        (phx.equations.DiffusionAction("temperature", 1.0),),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        prepared,
        constraint=constraint,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="sum_factorized",
        ),
    )

    residual = compiled.residual(constraint.reduced_space.zeros())

    np.testing.assert_allclose(residual, 0.0, atol=1e-13)
    assert compiled.discretization.prepared_id == prepared.prepared_id
