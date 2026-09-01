#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax.discretization import iga


def _geometry(grid: iga.BSplineGrid):
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    return iga.NURBSGeometryState(
        jnp.stack((xx, yy), axis=-1),
        3.0 * jnp.ones((grid.coefficient_count, grid.coefficient_count)),
    )


def _prepared(*, degree: int = 2):
    grid = iga.BSplineGrid.open_uniform(
        degree,
        1,
        interval=(0.0, 1.0),
    )
    plan = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        _geometry(grid),
        field_name="u",
        axis_names=("xi", "eta"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(degree + 1),
        qualification_policy=iga.IsogeometricH1QualificationPolicy(),
    )
    return grid, plan, plan.prepare(numeric_version="unit-square")


def test_single_patch_topology_runtime_and_trace_constraint():
    grid, plan, prepared = _prepared()

    assert prepared.basis.control_shape == (
        grid.coefficient_count,
        grid.coefficient_count,
    )
    assert prepared.cell_gathers.shape == (1, 9)
    assert prepared.default_runtime.numeric_version == "unit-square"
    np.testing.assert_allclose(prepared.default_runtime.weights, 1.0)
    assert float(prepared.default_geometry_evidence.minimum_rank_ratio) > 0.0
    assert float(prepared.default_geometry_evidence.minimum_orientation_ratio) > 0.0

    moved = prepared.prepare_runtime(
        iga.NURBSGeometryState(
            2.0 * plan.geometry.control_points,
            plan.geometry.weights,
        ),
        numeric_version="scaled",
    )
    assert moved.topology_id == prepared.default_runtime.topology_id
    assert moved.geometry_layout_id == prepared.default_runtime.geometry_layout_id
    assert moved.runtime_id != prepared.default_runtime.runtime_id

    constraint = prepared.homogeneous_trace_constraint("u")
    correction = constraint.homogeneous_correction(constraint.reduced_space.zeros())
    assert correction.shape == prepared.basis.control_shape
    np.testing.assert_allclose(correction, 0.0)


def test_anisotropic_isoparametric_axes_prepare():
    quadratic = iga.BSplineGrid.open_uniform(2, 1)
    cubic = iga.BSplineGrid.open_uniform(3, 1)
    xx, yy = jnp.meshgrid(
        quadratic.greville_abscissae,
        cubic.greville_abscissae,
        indexing="ij",
    )
    plan = iga.IsogeometricPlan.isoparametric(
        (quadratic, cubic),
        iga.NURBSGeometryState(
            jnp.stack((xx, yy), axis=-1),
            jnp.ones((quadratic.coefficient_count, cubic.coefficient_count)),
        ),
        field_name="u",
        axis_names=("xi", "eta"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy((3, 4)),
    )

    prepared = plan.prepare(numeric_version="anisotropic")

    assert plan.basis.degrees == (2, 3)
    assert prepared.field_spaces[0].layout.value_shape == (
        quadratic.coefficient_count,
        cubic.coefficient_count,
    )
