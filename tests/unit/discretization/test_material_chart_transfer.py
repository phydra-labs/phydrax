#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Material chart moments, differential companions, and transpose duality."""

from typing import Literal

from phydrax import config as pxconfig


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.discretization as D
from phydrax.discretization import prepare_sphere_chart_deformation
from phydrax.discretization.fem import (
    form_element,
    prepare_sphere_chart_compatible_transfer,
    prepare_surface_chart_compatible_transfer,
    prepare_surface_chart_field_transfer,
)
from tests._support.mapped_cells import curved_surface_chart_fv
from tests._support.sphere_material import accepted_sphere_material


# Empirical publication bounds retained without widening from the corresponding
# test_compatible_mesh_transfer tests: analytical RT0 moments (1e-14/2e-14),
# cubic moment interpolation (2e-12), and independent native sphere meshes
# with order-two projective moments (1e-11).
AFFINE_SHARED_ATOL = 1e-14
AFFINE_TRANSFER_ATOL = 2e-14
ORDER_THREE_ATOL = 2e-12
SPHERE_ATOL = 1e-11


def test_embedded_twisted_flux_transfer_preserves_moments_across_reversed_charts() -> (
    None
):

    _, _, chart, _ = curved_surface_chart_fv()
    element = form_element("triangle", 1, 1, twist="twisted", proxy="flux")
    field = D.FiniteElementFieldSpec("flux", element)
    source = D.FiniteElementPlan(
        chart.source_mesh,
        field,
        coordinate_spec=chart.source_geometry,
    ).prepare()
    target = D.FiniteElementPlan(
        chart.target_mesh,
        field,
        coordinate_spec=chart.target_geometry,
    ).prepare()
    # binding banks are sorted by scientific cell ID, not mesh/DOF row order.
    source_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in source.mesh.blocks]
    )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in target.mesh.blocks]
    )
    source_order = np.searchsorted(
        np.asarray(chart.source_witness.cell_global_ids), source_ids
    )
    target_order = np.searchsorted(
        np.asarray(chart.target_witness.cell_global_ids), target_ids
    )
    source_charts = np.asarray(chart.source_witness.charts)[source_order]
    target_charts = np.asarray(chart.target_witness.charts)[target_order]
    source_determinants = np.linalg.det(source_charts[:, 1:] - source_charts[:, :1])
    target_determinants = np.linalg.det(target_charts[:, 1:] - target_charts[:, :1])
    assert np.all(source_determinants < 0.0)
    assert np.all(target_determinants > 0.0)

    # Analytical moments of the material RT0 vector a + b * UV. A twisted flux
    # pulls back with |det A|, not det A, even when the local chart reverses.
    constant, slope = np.asarray((0.3, -0.8)), 0.7
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    edges = np.asarray(((0, 1), (0, 2), (1, 2)))
    tangents = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    midpoints = 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])

    def moments(
        discretization: D.FiniteElementDiscretization, charts: np.ndarray
    ) -> np.ndarray:
        dofs = discretization.dof_maps[0]
        coefficients = np.full(dofs.global_dof_count, np.nan)
        for cell, corners in enumerate(charts):
            frame = (corners[1:] - corners[:1]).T
            material = constant + slope * (corners[0] + midpoints @ frame.T)
            reference = abs(np.linalg.det(frame)) * np.linalg.solve(frame, material.T).T
            local = np.sum(reference * normals, axis=-1)
            canonical = np.linalg.solve(np.asarray(dofs.cell_transforms[0])[cell], local)
            routes = np.asarray(dofs.cell_dofs[0])[cell]
            shared = np.isfinite(coefficients[routes])
            np.testing.assert_allclose(
                coefficients[routes][shared], canonical[shared], atol=AFFINE_SHARED_ATOL
            )
            coefficients[routes] = canonical
        return coefficients

    prepared = prepare_surface_chart_compatible_transfer(
        source,
        target,
        chart,
        field_name="flux",
        source_geometry=chart.source_geometry,
        target_geometry=chart.target_geometry,
    )
    values = moments(source, source_charts)
    transferred = prepared.field_transfer.transfer.apply(jnp.asarray(values))
    assert prepared.field_transfer.evidence.passed
    np.testing.assert_allclose(
        transferred, moments(target, target_charts), atol=AFFINE_TRANSFER_ATOL
    )
    source_content = prepared.source_differential.mv(values)
    target_content = prepared.target_differential.mv(transferred)
    # div(a + b * UV) = 2b and chart area = |det A| / 2: the
    # integrated material divergence remains positive under chart reversal.
    np.testing.assert_allclose(
        source_content, slope * np.abs(source_determinants), atol=AFFINE_TRANSFER_ATOL
    )
    np.testing.assert_allclose(
        target_content, slope * target_determinants, atol=AFFINE_TRANSFER_ATOL
    )
    np.testing.assert_allclose(
        target_content,
        prepared.differential_companion.mv(source_content),
        atol=AFFINE_TRANSFER_ATOL,
    )
    target_load = jnp.arange(transferred.size, dtype=jnp.float64) + 0.25
    np.testing.assert_allclose(
        jnp.vdot(target_load, transferred),
        jnp.vdot(prepared.field_transfer.transfer.pullback(target_load), values),
        atol=AFFINE_TRANSFER_ATOL,
    )


@pytest.mark.parametrize(
    "proxy,twist", [("circulation", "untwisted"), ("flux", "twisted")]
)
def test_order_three_material_transfer_preserves_independent_moments_and_companion(
    proxy: Literal["circulation", "flux"],
    twist: Literal["untwisted", "twisted"],
) -> None:

    _, _, chart, _ = curved_surface_chart_fv()
    element = form_element("triangle", 1, 3, family="trimmed", twist=twist, proxy=proxy)
    basis = element.form_basis
    if basis is None:
        raise TypeError(
            "material.field must retain an order-three form basis; got no form basis."
        )
    field = D.FiniteElementFieldSpec("material", element)
    source = D.FiniteElementPlan(
        chart.source_mesh, field, coordinate_spec=chart.source_geometry
    ).prepare()
    target = D.FiniteElementPlan(
        chart.target_mesh, field, coordinate_spec=chart.target_geometry
    ).prepare()

    def coefficients(
        discretization: D.FiniteElementDiscretization, corners: np.ndarray
    ) -> np.ndarray:
        points = np.asarray(basis.functional_points)
        dofs = discretization.dof_maps[0]
        values = np.full(dofs.global_dof_count, np.nan)
        for cell, triangle in enumerate(corners):
            frame = (triangle[1:] - triangle[:1]).T
            uv = triangle[0] + points @ frame.T
            u, v = uv[:, 0], uv[:, 1]
            material = np.column_stack(
                (0.3 + 0.4 * u * u - 0.2 * u * v, -0.8 + 0.7 * v * v + 0.1 * u * v)
            )
            if proxy == "flux":
                reference = (
                    abs(np.linalg.det(frame)) * np.linalg.solve(frame, material.T).T
                )
                components = np.column_stack((-reference[:, 1], reference[:, 0]))
            else:
                components = material @ frame
            local = np.asarray(basis.interpolate(jnp.asarray(components)))
            canonical = np.linalg.solve(np.asarray(dofs.cell_transforms[0])[cell], local)
            rows = np.asarray(dofs.cell_dofs[0])[cell]
            shared = np.isfinite(values[rows])
            np.testing.assert_allclose(
                values[rows][shared], canonical[shared], atol=ORDER_THREE_ATOL
            )
            values[rows] = canonical
        return values

    def chart_rows(
        discretization: D.FiniteElementDiscretization, witness: D.SurfaceChartWitness
    ) -> np.ndarray:
        ids = np.concatenate(
            [np.asarray(block.global_ids) for block in discretization.mesh.blocks]
        )
        return np.asarray(witness.charts)[
            np.searchsorted(np.asarray(witness.cell_global_ids), ids)
        ]

    prepared = prepare_surface_chart_compatible_transfer(
        source,
        target,
        chart,
        field_name="material",
        source_geometry=chart.source_geometry,
        target_geometry=chart.target_geometry,
    )
    old = coefficients(source, chart_rows(source, chart.source_witness))
    expected = coefficients(target, chart_rows(target, chart.target_witness))
    actual = prepared.field_transfer.transfer.apply(jnp.asarray(old))
    assert prepared.field_transfer.evidence.passed
    np.testing.assert_allclose(actual, expected, atol=ORDER_THREE_ATOL)
    np.testing.assert_allclose(
        prepared.target_differential.mv(actual),
        prepared.differential_companion.mv(prepared.source_differential.mv(old)),
        atol=ORDER_THREE_ATOL,
    )
    load = jnp.linspace(-1.0, 2.0, actual.size, dtype=jnp.float64)
    np.testing.assert_allclose(
        jnp.vdot(load, actual),
        jnp.vdot(prepared.field_transfer.transfer.pullback(load), jnp.asarray(old)),
        atol=ORDER_THREE_ATOL,
    )


@pytest.mark.parametrize(
    "proxy,twist", [("circulation", "untwisted"), ("flux", "twisted")]
)
def test_order_two_sphere_material_transfer_commutes_and_is_dual_on_native_remesh(
    proxy: Literal["circulation", "flux"],
    twist: Literal["untwisted", "twisted"],
) -> None:

    # Two independently generated native sphere meshes: projective
    # material pieces, not a shared chart or an identity correspondence.
    _, _, source_atlas = accepted_sphere_material(1.2)
    _, _, target_atlas = accepted_sphere_material(1.0)
    chart = prepare_sphere_chart_deformation(
        source_atlas, target_atlas, maximum_fidelity=0.3, maximum_displacement=0.5
    )
    element = form_element("triangle", 1, 2, family="trimmed", twist=twist, proxy=proxy)
    field = D.FiniteElementFieldSpec("material", element)
    source = D.FiniteElementPlan(
        chart.source_mesh, field, coordinate_spec=chart.source_geometry
    ).prepare()
    target = D.FiniteElementPlan(
        chart.target_mesh, field, coordinate_spec=chart.target_geometry
    ).prepare()
    prepared = prepare_sphere_chart_compatible_transfer(
        source,
        target,
        chart,
        field_name="material",
        source_geometry=chart.source_geometry,
        target_geometry=chart.target_geometry,
    )
    assert prepared.field_transfer.evidence.passed
    old = jnp.sin(jnp.arange(source.dof_maps[0].global_dof_count, dtype=jnp.float64))
    new = prepared.field_transfer.transfer.apply(old)
    np.testing.assert_allclose(
        prepared.target_differential.mv(new),
        prepared.differential_companion.mv(prepared.source_differential.mv(old)),
        atol=SPHERE_ATOL,
    )
    load = jnp.cos(jnp.arange(new.size, dtype=jnp.float64))
    np.testing.assert_allclose(
        jnp.vdot(load, new),
        jnp.vdot(prepared.field_transfer.transfer.pullback(load), old),
        atol=SPHERE_ATOL,
    )


@pytest.fixture
def scalar_surface_endpoints() -> tuple[
    D.FiniteElementDiscretization,
    D.FiniteElementDiscretization,
    D.PreparedSurfaceChartDeformation,
]:
    _, _, chart, _ = curved_surface_chart_fv()
    field = D.FiniteElementFieldSpec("u", D.lagrange_element("triangle", 1))
    source = D.FiniteElementPlan(
        chart.source_mesh, field, coordinate_spec=chart.source_geometry
    ).prepare()
    target = D.FiniteElementPlan(
        chart.target_mesh, field, coordinate_spec=chart.target_geometry
    ).prepare()
    return source, target, chart


def test_surface_field_transfer_refuses_wrong_source_kind(
    scalar_surface_endpoints: tuple[
        D.FiniteElementDiscretization,
        D.FiniteElementDiscretization,
        D.PreparedSurfaceChartDeformation,
    ],
) -> None:
    _, target, chart = scalar_surface_endpoints
    with pytest.raises(TypeError, match="source"):
        prepare_surface_chart_field_transfer(
            None,  # ty: ignore[invalid-argument-type]
            target,
            chart,
            field_name="u",
            source_geometry=chart.source_geometry,
            target_geometry=chart.target_geometry,
        )


def test_surface_field_transfer_refuses_unknown_semantics(
    scalar_surface_endpoints: tuple[
        D.FiniteElementDiscretization,
        D.FiniteElementDiscretization,
        D.PreparedSurfaceChartDeformation,
    ],
) -> None:
    source, target, chart = scalar_surface_endpoints
    with pytest.raises(ValueError, match=r"chart\.semantics"):
        prepare_surface_chart_field_transfer(
            source,
            target,
            chart,
            field_name="u",
            source_geometry=chart.source_geometry,
            target_geometry=chart.target_geometry,
            semantics="unsupported",  # ty: ignore[invalid-argument-type]
        )


def test_sphere_deformation_refuses_wrong_source_atlas_kind() -> None:
    _, _, target = accepted_sphere_material(1.0)
    with pytest.raises(TypeError, match="source_atlas"):
        prepare_sphere_chart_deformation(
            None,  # ty: ignore[invalid-argument-type]
            target,
            maximum_fidelity=0.3,
            maximum_displacement=0.5,
        )
