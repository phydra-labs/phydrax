# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

from phydrax.discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometrySpec,
    coordinate_lagrange_element,
    LayerColumnCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import transition_nested_cell_geometry
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization.fem._reference import FiniteElementSpec
from phydrax.discretization.fem._topology_transfer import (
    _prepare_mapped_nested_dg_transfer,
)
from phydrax.meshing._mixed_adaptation import (
    adapt_mixed_mesh,
    MixedAdaptationOutcome,
)
from phydrax.meshing._topology_edit import assemble_topology_edit
from tests.unit.discretization.test_mapped_nested_dg import _content, _curved, _space


def _exact_chart(
    source: (
        FiniteElementSpec
        | BarycentricCellGeometryElement
        | RestrictedCellGeometryElement
        | PolynomialComposedCellGeometryElement
        | RationalComposedCellGeometryElement
        | SplineCellGeometryElement
        | LayerColumnCellGeometryElement
    ),
    chart: FiniteElementSpec,
    values: NDArray[np.float64],
) -> PolynomialComposedCellGeometryElement:
    fractions = [[Fraction(float(value)) for value in row] for row in values]
    return PolynomialComposedCellGeometryElement(
        source,
        chart,
        np.asarray(
            [[value.numerator for value in row] for row in fractions],
            dtype=object,
        ),
        np.asarray(
            [[value.denominator for value in row] for row in fractions],
            dtype=object,
        ),
    )


def _affine_children(
    scale: float,
) -> tuple[
    CellMesh,
    CellGeometrySpec,
    CellMesh,
    CellGeometrySpec,
    MixedAdaptationOutcome,
    NDArray[np.int64],
    NDArray[np.float64],
]:
    mesh, geometry = _curved("hexahedron")
    # Two genuine source banks, not a fit to target coordinate nodes. The Q2
    # root has a nonconstant volume Jacobian as well as curved physical edges.
    coordinates = np.asarray(geometry.coordinates).copy()
    root = geometry.elements[0]
    if not isinstance(root, FiniteElementSpec):
        raise TypeError("Curved root fixture requires its finite-element chart.")
    nodes = np.asarray(root.reference_nodes)
    coordinates[:, 0] += scale * nodes[:, 0] ** 2
    vertices = np.asarray(mesh.coordinates).copy()
    vertices[:, 0] += scale * vertices[:, 0] ** 2
    mesh = mesh.with_coordinates(vertices, numeric_version=f"curved-root-{scale}")
    geometry = CellGeometrySpec(
        {"volume": geometry.elements[0]},
        {"volume": geometry.geometry_dofs[0]},
        coordinates,
    )
    outcome = adapt_mixed_mesh(mesh, refine_cell_ids=np.asarray([17]))
    fine_mesh, _, _ = assemble_topology_edit(
        mesh, outcome.edit, numeric_version="composed-dg-fine"
    )
    transition = transition_nested_cell_geometry(
        mesh,
        geometry,
        fine_mesh,
        CellGeometrySpec.affine(fine_mesh),
        refinement=outcome.edit.refinement,
    )
    restricted = transition.geometry
    chart = coordinate_lagrange_element("hexahedron", 1)
    if not isinstance(chart, FiniteElementSpec):
        raise TypeError("Affine child fixture requires its Q1 chart.")
    elements = {}
    for block, element in zip(fine_mesh.blocks, restricted.elements, strict=True):
        assert isinstance(element, RestrictedCellGeometryElement)
        assert np.array_equal(np.asarray(element.matrix), 0.5 * np.eye(3))
        values = np.asarray(chart.reference_nodes) @ np.asarray(
            element.matrix
        ).T + np.asarray(element.offset)
        elements[block.name] = _exact_chart(element.source_element, chart, values)
    composed = CellGeometrySpec(
        elements,
        {
            block.name: route
            for block, route in zip(
                fine_mesh.blocks, restricted.geometry_dofs, strict=True
            )
        },
        restricted.coordinates,
        restriction_source=restricted.restriction_source,
    )
    parents = np.zeros(len(transition.target_cell_ids), dtype=np.int64)
    return (
        mesh,
        geometry,
        fine_mesh,
        composed,
        outcome,
        parents,
        np.asarray(transition.parent_reference_vertices),
    )


@pytest.mark.parametrize("scale", [1 / 8, 0.1], ids=["binary-source", "nonbinary-source"])
def test_curved_q2_affine_composed_eight_child_dg1_conservation_coarsen_and_adjoint(
    scale: float,
) -> None:
    mesh, geometry, fine_mesh, composed, outcome, parents, corners = _affine_children(
        scale
    )
    before = np.asarray(geometry.coordinates).view(np.uint64).copy()
    child_before = np.asarray(composed.coordinates).view(np.uint64).copy()
    source, fine = _space(mesh, geometry, 1), _space(fine_mesh, composed, 1)
    prepared = _prepare_mapped_nested_dg_transfer(
        source,
        fine,
        geometry,
        composed,
        field_name="u",
        parent_cells=parents,
        parent_reference_vertices=corners,
    )
    assert len(parents) == 8
    assert prepared.evidence.passed
    n = source.dof_maps[0].global_dof_count
    values: NDArray[np.float64] = np.column_stack(
        (np.arange(n) + 0.4, np.arange(n) ** 2 + 2.0)
    ).astype(np.float64)
    transferred = np.asarray(prepared.transfer.apply(values))
    np.testing.assert_allclose(
        _content(fine, composed, transferred),
        _content(source, geometry, values),
        atol=3e-11,
    )
    dual = np.sin(np.arange(transferred.size)).reshape(transferred.shape)
    np.testing.assert_allclose(
        np.vdot(transferred, dual),
        np.vdot(values, prepared.transfer.pullback(dual)),
        atol=3e-11,
    )
    tangent = jnp.cos(jnp.arange(values.size)).reshape(values.shape)
    _, derivative = jax.jvp(prepared.transfer.apply, (jnp.asarray(values),), (tangent,))
    np.testing.assert_allclose(derivative, prepared.transfer.apply(tangent), atol=2e-12)
    gradient = jax.grad(
        lambda coefficients: jnp.vdot(prepared.transfer.apply(coefficients), dual)
    )(jnp.asarray(values))
    np.testing.assert_allclose(gradient, prepared.transfer.pullback(dual), atol=2e-12)
    ids = np.concatenate([np.asarray(block.global_ids) for block in fine_mesh.blocks])
    coarsening = adapt_mixed_mesh(
        fine_mesh,
        refine_cell_ids=np.empty(0, dtype=np.int64),
        coarsen_cell_ids=ids,
        hierarchy=outcome.hierarchy,
    )
    coarse_mesh, _, _ = assemble_topology_edit(
        fine_mesh, coarsening.edit, numeric_version="composed-dg-coarse"
    )
    restoration = transition_nested_cell_geometry(
        fine_mesh,
        composed,
        coarse_mesh,
        CellGeometrySpec.affine(coarse_mesh),
        refinement=coarsening.edit.refinement,
        coarsening=coarsening.edit.coarsening,
    )
    coarse = _space(coarse_mesh, restoration.geometry, 1)
    reverse = _prepare_mapped_nested_dg_transfer(
        fine,
        coarse,
        composed,
        restoration.geometry,
        field_name="u",
        geometry_transition=restoration,
    )
    restored = np.asarray(reverse.transfer.apply(transferred))
    np.testing.assert_allclose(restored, values, atol=3e-10)
    np.testing.assert_allclose(
        _content(coarse, restoration.geometry, restored),
        _content(source, geometry, values),
        atol=3e-11,
    )
    np.testing.assert_array_equal(
        np.asarray(geometry.coordinates).view(np.uint64), before
    )
    np.testing.assert_array_equal(
        np.asarray(composed.coordinates).view(np.uint64), child_before
    )


def test_affine_composed_3d_dg_refuses_wrong_actual_parent_map_atomically() -> None:
    mesh, geometry, fine_mesh, composed, _, parents, corners = _affine_children(0.1)
    before = np.asarray(geometry.coordinates).view(np.uint64).copy()
    wrong = corners.copy()
    wrong[0, :, 0] += 1 / 16
    with pytest.raises(ValueError):
        _prepare_mapped_nested_dg_transfer(
            _space(mesh, geometry, 1),
            _space(fine_mesh, composed, 1),
            geometry,
            composed,
            field_name="u",
            parent_cells=parents,
            parent_reference_vertices=wrong,
        )
    np.testing.assert_array_equal(
        np.asarray(geometry.coordinates).view(np.uint64), before
    )


def test_genuine_nonlinear_3d_composed_dg_chart_remains_refused() -> None:
    mesh, geometry, fine_mesh, composed, _, parents, corners = _affine_children(1 / 8)
    chart = coordinate_lagrange_element("hexahedron", 2)
    if not isinstance(chart, FiniteElementSpec):
        raise TypeError("Nonlinear child fixture requires its Q2 chart.")
    elements = {}
    for block, element in zip(fine_mesh.blocks, composed.elements, strict=True):
        if not isinstance(element, PolynomialComposedCellGeometryElement):
            raise TypeError("Nonlinear fixture requires affine composed child charts.")
        nodes = np.asarray(chart.reference_nodes)
        lower = np.min(np.asarray(element.chart_coordinates), axis=0)
        values = lower + 0.5 * nodes
        values[:, 0] += nodes[:, 0] * (1 - nodes[:, 0]) / 16
        elements[block.name] = _exact_chart(element.source_element, chart, values)
    nonlinear = CellGeometrySpec(
        elements,
        {
            block.name: route
            for block, route in zip(fine_mesh.blocks, composed.geometry_dofs, strict=True)
        },
        composed.coordinates,
        restriction_source=composed.restriction_source,
    )
    before = np.asarray(geometry.coordinates).view(np.uint64).copy()
    with pytest.raises(
        ValueError,
        match="Nonlinear reference transfer requires one common declared surface coordinate space",
    ):
        _prepare_mapped_nested_dg_transfer(
            _space(mesh, geometry, 1),
            _space(fine_mesh, nonlinear, 1),
            geometry,
            nonlinear,
            field_name="u",
            parent_cells=parents,
            parent_reference_vertices=corners,
        )
    np.testing.assert_array_equal(
        np.asarray(geometry.coordinates).view(np.uint64), before
    )


@pytest.mark.parametrize("scale", [1 / 8, 0.1], ids=["binary-source", "nonbinary-source"])
def test_affine_composed_3d_actual_source_bank_volume_jvp(
    scale: float,
) -> None:
    import equinox as eqx

    from phydrax.discretization._cell_geometry import (
        _require_scalar_coordinate_element,
    )
    from phydrax.discretization.fem import FiniteElementRuntimeData
    from phydrax.discretization.fem._generic import _degree_aware_reference_rule
    from phydrax.discretization.fem._local_provider import FiniteElementGeometryActions

    mesh, geometry, fine_mesh, composed, _, _, _ = _affine_children(scale)
    before = np.asarray(geometry.coordinates).view(np.uint64).copy()
    derivatives = []
    volumes = []
    for carrier, actual in ((mesh, geometry), (fine_mesh, composed)):
        elements, routes, bank = actual.resolve(carrier)
        points, weights = _degree_aware_reference_rule("hexahedron", 8)
        actions = []
        for element, route in zip(elements, routes, strict=True):
            basis, gradients = _require_scalar_coordinate_element(
                element, "Source-bank volume JVP fixture"
            ).tabulate(points)
            actions.append(
                FiniteElementGeometryActions(
                    actual.geometry_layout_id, "cell", basis, gradients, route, weights
                )
            )
        runtime = FiniteElementRuntimeData(
            carrier,
            bank,
            numeric_version="original-bank-jvp",
            geometry_layout_id=actual.geometry_layout_id,
        )

        def volume(nodes: Array) -> Array:
            moved = eqx.tree_at(lambda data: data.coordinates, runtime, nodes)
            return jnp.sum(
                jnp.stack(
                    [
                        jnp.sum(action.realize(moved).physical_weights)
                        for action in actions
                    ]
                )
            )

        # Differentiate the original root coefficient bank, not chart samples.
        root = geometry.elements[0]
        if not isinstance(root, FiniteElementSpec):
            raise TypeError("Source-bank volume JVP requires its finite-element root.")
        direction = (
            jnp.zeros_like(runtime.coordinates)
            .at[:, 0]
            .set(jnp.asarray(root.reference_nodes)[:, 0] ** 2)
        )
        value, tangent = jax.jvp(volume, (runtime.coordinates,), (direction,))
        gradient = jax.grad(volume)(runtime.coordinates)
        np.testing.assert_allclose(jnp.vdot(gradient, direction), tangent, atol=3e-12)
        volumes.append(value)
        derivatives.append(tangent)
        np.testing.assert_array_equal(
            np.asarray(runtime.coordinates).view(np.uint64), before
        )
    np.testing.assert_allclose(volumes, [1 + scale, 1 + scale], atol=3e-12)
    np.testing.assert_allclose(derivatives, [1.0, 1.0], atol=3e-12)
    np.testing.assert_array_equal(
        np.asarray(geometry.coordinates).view(np.uint64), before
    )
