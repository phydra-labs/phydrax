#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization import iga


_FIXTURE = json.loads(
    (Path(__file__).parents[2] / "fixtures" / "iga_s1_migration.json").read_text()
)


def _unit_square(case):
    grid = iga.BSplineGrid.open_uniform(
        case["degree"],
        case["span_count"],
        interval=(0.0, 1.0),
    )
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    geometry = iga.NURBSGeometryState(
        jnp.stack((xx, yy), axis=-1),
        jnp.ones(tuple(case["control_shape"])),
    )
    return grid, geometry


def test_s1_fixture_preserves_topology_runtime_and_exact_poisson_result():
    case = _FIXTURE["cases"]["exact_quadratic_poisson"]
    grid, geometry = _unit_square(case)
    plan = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        geometry,
        field_name=case["field_name"],
        axis_names=tuple(case["axis_names"]),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(
            case["quadrature_points_per_axis"]
        ),
    )
    prepared = plan.prepare(numeric_version="s1-migration-poisson")
    constraint = prepared.homogeneous_trace_constraint(case["field_name"])
    source = phx.equations.coefficient(
        lambda points, args: (
            2.0
            * (
                points[..., 0] * (1.0 - points[..., 0])
                + points[..., 1] * (1.0 - points[..., 1])
            )
        ),
        coefficient_id="iga-s1-migration-source",
    )
    form = phx.equations.FiniteElementForm(
        "iga-s1-migration-poisson",
        case["field_name"],
        (
            phx.equations.DiffusionAction(case["field_name"], 1.0),
            phx.equations.SourceAction(case["field_name"], source),
        ),
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
    operator, right_hand_side = compiled.linear_system()
    result = phx.linalg.solve(operator, right_hand_side)
    coefficients = compiled.expand(result.value)

    assert prepared.basis.control_shape == tuple(case["control_shape"])
    assert prepared.cell_gathers.shape == (1, 9)
    assert prepared.default_runtime.numeric_version == "s1-migration-poisson"
    assert bool(jnp.all(result.successful))
    np.testing.assert_allclose(
        coefficients,
        np.asarray(case["expected_coefficients"]),
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_s1_fixture_preserves_fixed_topology_numeric_refresh():
    case = _FIXTURE["cases"]["runtime_refresh"]
    grid, geometry = _unit_square(case)
    prepared = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        geometry,
        field_name=case["field_name"],
        axis_names=tuple(case["axis_names"]),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(
            case["quadrature_points_per_axis"]
        ),
    ).prepare(numeric_version=case["initial_numeric_revision"])
    refreshed = prepared.prepare_runtime(
        iga.NURBSGeometryState(
            2.0 * geometry.control_points,
            geometry.weights,
        ),
        numeric_version=case["refreshed_numeric_revision"],
    )

    assert refreshed.numeric_version == case["refreshed_numeric_revision"]
    assert refreshed.topology_id == prepared.default_runtime.topology_id
    assert refreshed.geometry_layout_id == prepared.default_runtime.geometry_layout_id
    assert refreshed.runtime_id != prepared.default_runtime.runtime_id
