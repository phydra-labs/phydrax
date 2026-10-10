# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Original-bank affine reference restrictions of curved tensor form sources."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from phydrax.discretization._cell_geometry import (
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import transition_nested_cell_geometry
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import (
    FiniteElementDiscretization,
    form_element,
    prepare_nested_field_transfer,
)
from phydrax.discretization.fem._reference import FiniteElementSpec
from phydrax.meshing._mixed_adaptation import adapt_mixed_mesh
from phydrax.meshing._topology_edit import assemble_topology_edit
from tests.unit.discretization.test_mapped_nested_dg import _curved


jax.config.update("jax_enable_x64", True)


def _charts(
    amplitude: float,
) -> tuple[CellMesh, CellGeometrySpec, CellMesh, CellGeometrySpec, np.ndarray]:
    mesh, original = _curved("hexahedron")
    root_element = original.elements[0]
    if not isinstance(root_element, FiniteElementSpec):
        raise TypeError("Affine form fixture requires its finite-element root.")
    nodes = np.asarray(root_element.reference_nodes).copy()
    nodes[:, 1] += amplitude * nodes[:, 0] * nodes[:, 2]
    vertices = np.asarray(mesh.coordinates).copy()
    vertices[:, 1] += (amplitude - 0.05) * vertices[:, 0] * vertices[:, 2]
    mesh = mesh.with_coordinates(vertices, numeric_version=f"form-root-{amplitude}")
    root = CellGeometrySpec(
        {"volume": root_element}, {"volume": original.geometry_dofs[0]}, nodes
    )
    outcome = adapt_mixed_mesh(mesh, refine_cell_ids=np.array([17]))
    child, _, _ = assemble_topology_edit(mesh, outcome.edit, numeric_version="form-child")
    transition = transition_nested_cell_geometry(
        mesh,
        root,
        child,
        CellGeometrySpec.affine(child),
        refinement=outcome.edit.refinement,
    )
    child = child.with_coordinates(
        transition.vertex_coordinates, numeric_version="form-child-curved"
    )
    chart = coordinate_lagrange_element("hexahedron", 1)
    if not isinstance(chart, FiniteElementSpec):
        raise TypeError("Affine form fixture requires its Q1 chart.")
    elements, routes, references = {}, {}, []
    for block, restricted, route in zip(
        child.blocks,
        transition.geometry.elements,
        transition.geometry.geometry_dofs,
        strict=True,
    ):
        if not isinstance(restricted, RestrictedCellGeometryElement):
            raise TypeError("Affine form fixture requires exact restrictions.")
        points = np.asarray(chart.reference_nodes) @ np.asarray(
            restricted.matrix
        ).T + np.asarray(restricted.offset)
        fractions = [[Fraction(float(value)) for value in row] for row in points]
        elements[block.name] = PolynomialComposedCellGeometryElement(
            root_element,
            chart,
            np.array([[v.numerator for v in row] for row in fractions], dtype=object),
            np.array([[v.denominator for v in row] for row in fractions], dtype=object),
        )
        routes[block.name] = route
        corners = np.asarray(reference_cell_topology("hexahedron").vertices)
        references.extend(
            [corners @ np.asarray(restricted.matrix).T + np.asarray(restricted.offset)]
            * len(route)
        )
    composed = CellGeometrySpec(
        elements,
        routes,
        root.coordinates,
        restriction_source=transition.geometry.restriction_source,
    )
    assert sum(len(block.vertices) for block in child.blocks) == 8
    return mesh, root, child, composed, np.asarray(references)


def _spaces(
    mesh: CellMesh,
    root: CellGeometrySpec,
    child: CellMesh,
    composed: CellGeometrySpec,
    degree: int,
) -> tuple[FiniteElementDiscretization, FiniteElementDiscretization]:
    def plan(
        carrier: CellMesh, geometry: CellGeometrySpec
    ) -> FiniteElementDiscretization:
        fields = tuple(
            FiniteElementFieldSpec(
                name,
                {
                    block.name: form_element(
                        "hexahedron",
                        degree,
                        2,
                        family="tensor-trimmed",
                        twist="twisted" if degree == 2 else "untwisted",
                        proxy="flux" if degree == 2 else "circulation",
                    )
                    for block in carrier.blocks
                },
            )
            for name in ("current", "history")
        )
        return FiniteElementPlan(carrier, fields, coordinate_spec=geometry).prepare()

    return plan(mesh, root), plan(child, composed)


@pytest.mark.parametrize("degree", (1, 2), ids=("Hcurl", "Hdiv"))
@pytest.mark.parametrize("amplitude", (0.05, 0.125))
def test_affine_composed_3d_named_forms_pullback_jvp_and_dual(
    degree: int, amplitude: float
) -> None:
    mesh, root, child, composed, references = _charts(amplitude)
    before = np.asarray(root.coordinates).copy()
    source, target = _spaces(mesh, root, child, composed, degree)
    for field_index, name in enumerate(("current", "history")):
        prepared = prepare_nested_field_transfer(
            source,
            target,
            np.zeros(8, dtype=np.int64),
            field_name=name,
            source_geometry=root,
            target_geometry=composed,
            parent_reference_vertices=references,
        )
        count = source.dof_maps[field_index].global_dof_count
        values = np.random.default_rng(71 + field_index).normal(size=count)
        tangent = np.random.default_rng(81 + field_index).normal(size=values.shape)
        actual, derivative = jax.jvp(
            prepared.transfer.apply, (jnp.asarray(values),), (jnp.asarray(tangent),)
        )
        np.testing.assert_allclose(
            derivative, prepared.transfer.apply(tangent), atol=2e-12, rtol=2e-12
        )
        dual = np.random.default_rng(91).normal(size=actual.shape)
        np.testing.assert_allclose(
            np.vdot(actual, dual),
            np.vdot(values, prepared.transfer.pullback(dual)),
            atol=2e-11,
            rtol=2e-11,
        )
        # Independently tabulate the source form on each actual parent chart;
        # covariant and contravariant pullbacks use different exterior powers.
        points = np.array([[0.17, 0.31, 0.43], [0.73, 0.59, 0.21]])
        coarse_local = (
            np.asarray(source.dof_maps[field_index].cell_transforms[0])[0]
            @ values[np.asarray(source.dof_maps[field_index].cell_dofs[0])[0]]
        )
        offset = 0
        for element, coordinate, routes, transforms in zip(
            target.elements[field_index],
            composed.elements,
            target.dof_maps[field_index].cell_dofs,
            target.dof_maps[field_index].cell_transforms,
            strict=True,
        ):
            if not isinstance(coordinate, PolynomialComposedCellGeometryElement):
                raise TypeError("Affine form fixture requires composed child charts.")
            chart_values, chart_gradients = coordinate.chart_element.tabulate(points)
            parent_points = np.asarray(chart_values) @ np.asarray(
                coordinate.chart_coordinates
            )
            matrix = np.einsum(
                "md,qma->qda",
                np.asarray(coordinate.chart_coordinates),
                np.asarray(chart_gradients),
            )
            coarse_basis, _ = source.elements[field_index][0].tabulate(parent_points)
            parent_values = np.einsum("qnc,n->qc", np.asarray(coarse_basis), coarse_local)
            expected = (
                np.einsum("qda,qd->qa", matrix, parent_values)
                if degree == 1
                else np.linalg.det(matrix)[:, None]
                * np.einsum("qad,qd->qa", np.linalg.inv(matrix), parent_values)
            )
            basis, _ = element.tabulate(points)
            for route, transform in zip(
                np.asarray(routes), np.asarray(transforms), strict=True
            ):
                local = transform @ np.asarray(actual)[route]
                np.testing.assert_allclose(
                    np.einsum("qnc,n->qc", np.asarray(basis), local),
                    expected,
                    atol=3e-11,
                    rtol=3e-11,
                )
                offset += 1
        assert offset == 8
        assert prepared.evidence.passed
    np.testing.assert_array_equal(root.coordinates, before)
    np.testing.assert_array_equal(composed.coordinates, before)


@pytest.mark.parametrize("resource", ("work", "storage"))
def test_affine_composed_3d_form_resource_refusal_preserves_bank(
    resource: str,
) -> None:
    mesh, root, child, composed, references = _charts(0.125)
    source, target = _spaces(mesh, root, child, composed, 1)
    before = np.asarray(root.coordinates).copy()
    with pytest.raises(ValueError, match="budget"):
        prepare_nested_field_transfer(
            source,
            target,
            np.zeros(8, dtype=np.int64),
            field_name="history",
            source_geometry=root,
            target_geometry=composed,
            parent_reference_vertices=references,
            maximum_work=1 if resource == "work" else 100_000_000,
            maximum_storage_bytes=1 if resource == "storage" else 256_000_000,
        )
    np.testing.assert_array_equal(root.coordinates, before)


def test_affine_composed_3d_form_wrong_parent_map_preserves_bank() -> None:
    mesh, root, child, composed, references = _charts(0.125)
    source, target = _spaces(mesh, root, child, composed, 2)
    before = np.asarray(root.coordinates).copy()
    wrong = references.copy()
    wrong[0, :, 0] += 0.125
    with pytest.raises(ValueError):
        prepare_nested_field_transfer(
            source,
            target,
            np.zeros(8, dtype=np.int64),
            field_name="current",
            source_geometry=root,
            target_geometry=composed,
            parent_reference_vertices=wrong,
        )
    np.testing.assert_array_equal(root.coordinates, before)
