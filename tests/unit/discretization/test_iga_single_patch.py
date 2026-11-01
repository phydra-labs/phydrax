#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from phydrax.discretization import iga


def _geometry(grid: iga.BSplineGrid) -> Any:
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    return iga.NURBSGeometryState(
        jnp.stack((xx, yy), axis=-1),
        3.0 * jnp.ones((grid.coefficient_count, grid.coefficient_count)),
    )


def _prepared(*, degree: int = 2) -> Any:
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


def test_iga_single_patch_scenario_1() -> None:
    grid, plan, prepared = _prepared()

    assert prepared.basis.control_shape == (
        grid.coefficient_count,
        grid.coefficient_count,
    )
    assert prepared.cell_gathers.shape == (1, 9)
    assert prepared.default_runtime.numeric_version == "unit-square"
    np.testing.assert_allclose(prepared.default_runtime.weights, 3.0)
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
    # ty: ignore[unresolved-attribute]
    assert prepared.field_spaces[0].layout.value_shape == (
        quadratic.coefficient_count,
        cubic.coefficient_count,
    )


def test_nonuniform_raw_weights_survive_runtime_refresh() -> None:
    # Two quadratic intervals carry the four control rows the 4x4 weights name.
    grid = iga.BSplineGrid.open_uniform(2, 2)
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    control_points = jnp.stack((xx, yy), axis=-1)
    raw_weights = jnp.arange(1, 17, dtype=jnp.float64).reshape((4, 4)) / 7.0
    geometry = iga.NURBSGeometryState(control_points, raw_weights)
    np.testing.assert_array_equal(np.asarray(geometry.weights), np.asarray(raw_weights))

    prepared = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        geometry,
        axis_names=("xi", "eta"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    ).prepare(numeric_version="awkward-raw-weights")
    refreshed_geometry = iga.NURBSGeometryState(control_points, raw_weights * 3.7)
    refreshed = prepared.prepare_runtime(
        refreshed_geometry, numeric_version="awkward-scaled-raw-weights"
    )
    np.testing.assert_array_equal(
        np.asarray(refreshed.weights), np.asarray(raw_weights * 3.7)
    )
    assert refreshed.runtime_id != prepared.default_runtime.runtime_id

    region = prepared.prepare_local_regions(
        prepared.cell_domain,
        field_names=("u",),
        maximum_derivative_order=1,
        kernel_mode="sum_factorized",
    )[0]
    original_metric = region.geometry_actions.realize(prepared.default_runtime)
    refreshed_metric = region.geometry_actions.realize(refreshed)
    np.testing.assert_allclose(
        np.asarray(refreshed_metric.points),
        np.asarray(original_metric.points),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(refreshed_metric.physical_weights),
        np.asarray(original_metric.physical_weights),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
