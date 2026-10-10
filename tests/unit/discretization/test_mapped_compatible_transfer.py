# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Actual nonlinear source restrictions and complete compatible source patches."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from phydrax.discretization._cell_geometry_transfer import (
    CellGeometryTransition,
    transition_nested_cell_geometry,
)
from phydrax.discretization._nested_reference import (
    _NestedReferencePair,
    _PolynomialReferencePair,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementSpec,
    form_element,
    prepare_nested_field_transfer,
)
from phydrax.discretization.fem._exact_form_moments import (
    ExactFormMoments,
    MomentIntegralMatrix,
)
from phydrax.discretization.fem._form_elements import FormBasis
from phydrax.discretization.fem._generic import (
    _degree_aware_reference_rule,
    FiniteElementTransferDiscretization,
)
from phydrax.exterior._form_type import FormProxy, FormTwist
from phydrax.meshing._mixed_adaptation import adapt_mixed_mesh
from phydrax.meshing._topology_edit import assemble_topology_edit
from tests.unit.meshing.test_mixed_adaptation import _periodic_template_source


jax.config.update("jax_enable_x64", True)


def _elements(
    carrier: CellMesh,
    form_degree: int,
    degree: int,
    twist: FormTwist,
    proxy: FormProxy,
) -> dict[str, FiniteElementSpec]:
    return {
        block.name: form_element(
            block.cell_kind,
            form_degree,
            degree,
            family="tensor-trimmed" if block.cell_kind == "hexahedron" else "trimmed",
            twist=twist,
            proxy=proxy,
        )
        for block in carrier.blocks
    }


def _tetrahedral_hdiv_refinement() -> tuple[
    FiniteElementTransferDiscretization,
    FiniteElementTransferDiscretization,
    CellGeometryTransition,
]:
    mesh, geometry = _periodic_template_source("tetrahedron", True)
    outcome = adapt_mixed_mesh(mesh, refine_cell_ids=np.asarray([17], dtype=np.int64))
    fine_mesh, _, _ = assemble_topology_edit(
        mesh, outcome.edit, numeric_version="hdiv-orientation-fine"
    )
    restriction = transition_nested_cell_geometry(
        mesh,
        geometry,
        fine_mesh,
        CellGeometrySpec.affine(fine_mesh),
        refinement=outcome.edit.refinement,
    )
    fine_mesh = fine_mesh.with_coordinates(
        restriction.vertex_coordinates, numeric_version="hdiv-orientation-fine"
    )
    source_field = FiniteElementFieldSpec("u", _elements(mesh, 2, 2, "twisted", "flux"))
    target_field = FiniteElementFieldSpec(
        "u", _elements(fine_mesh, 2, 2, "twisted", "flux")
    )
    source = FiniteElementPlan(
        mesh, source_field, coordinate_spec=geometry
    ).prepare_transfer()
    target = FiniteElementPlan(
        fine_mesh, target_field, coordinate_spec=restriction.geometry
    ).prepare_transfer()
    return source, target, restriction


def _fault_shared_hdiv_face(
    space: FiniteElementTransferDiscretization, /
) -> FiniteElementTransferDiscretization:
    dof_map = space.dof_maps[0]
    occurrences: dict[int, list[tuple[int, int, int]]] = {}
    for block_index, (element, routes) in enumerate(
        zip(space.elements[0], dof_map.cell_dofs, strict=True)
    ):
        face_slots = {slot for dofs in element.entity_dofs[2] for slot in dofs}
        for cell_index, route in enumerate(np.asarray(routes)):
            for slot in face_slots:
                occurrences.setdefault(int(route[slot]), []).append(
                    (block_index, cell_index, slot)
                )
    shared = next(values for values in occurrences.values() if len(values) == 2)
    transforms = [
        np.asarray(value, dtype=np.float64).copy() for value in dof_map.cell_transforms
    ]
    block_index, cell_index, slot = shared[1]
    transforms[block_index][cell_index, slot] *= -1.0
    faulted_map = eqx.tree_at(
        lambda value: value.cell_transforms,
        dof_map,
        tuple(jnp.asarray(value) for value in transforms),
    )
    return eqx.tree_at(lambda value: value.dof_maps[0], space, faulted_map)


def _integrated_divergence(
    space: FiniteElementDiscretization,
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    total = np.zeros(values.shape[1:], dtype=np.float64)
    for element, routes, transforms in zip(
        space.elements[0],
        space.dof_maps[0].cell_dofs,
        space.dof_maps[0].cell_transforms,
        strict=True,
    ):
        points, weights = _degree_aware_reference_rule(element.cell_kind, 8)
        _, gradients = element.tabulate(points)
        divergence = np.trace(np.asarray(gradients), axis1=-2, axis2=-1)
        for route, transform in zip(
            np.asarray(routes), np.asarray(transforms), strict=True
        ):
            total += np.einsum(
                "q,qn,nk->k", np.asarray(weights), divergence, transform @ values[route]
            )
    return total


def _line_integral(
    element: FiniteElementSpec,
    values: NDArray[np.float64],
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> NDArray[np.float64]:
    q, weights = np.polynomial.legendre.leggauss(6)
    points = first + ((q + 1) / 2)[:, None] * (second - first)
    basis, _ = element.tabulate(points)
    return np.einsum(
        "q,qnc,nk,c->k", weights / 2, np.asarray(basis), values, second - first
    )


def _patch_edge_circulations(
    fine: FiniteElementDiscretization,
    coarse: FiniteElementDiscretization,
    fine_values: NDArray[np.float64],
    coarse_values: NDArray[np.float64],
    references: NDArray[np.float64],
) -> None:
    topology = reference_cell_topology(coarse.mesh.blocks[0].cell_kind)
    vertices = np.asarray(topology.vertices, dtype=np.float64)
    coarse_local = (
        np.asarray(coarse.dof_maps[0].cell_transforms[0])[0]
        @ coarse_values[np.asarray(coarse.dof_maps[0].cell_dofs[0])[0]]
    )
    coarse_element = coarse.elements[0][0]
    for first, second in topology.entities[1]:
        a, tangent = vertices[first], vertices[second] - vertices[first]
        expected = np.zeros(coarse_values.shape[1:], dtype=np.float64)
        seen: set[tuple[tuple[float, ...], ...]] = set()
        cell_offset = 0
        for element, routes, transforms in zip(
            fine.elements[0],
            fine.dof_maps[0].cell_dofs,
            fine.dof_maps[0].cell_transforms,
            strict=True,
        ):
            local_topology = reference_cell_topology(element.cell_kind)
            local_vertices = np.asarray(local_topology.vertices, dtype=np.float64)
            for cell, (route, transform) in enumerate(
                zip(np.asarray(routes), np.asarray(transforms), strict=True)
            ):
                for start, stop in local_topology.entities[1]:
                    image = references[cell_offset + cell, [start, stop]]
                    parameter = (image - a) @ tangent / (tangent @ tangent)
                    if (
                        np.any(parameter < 0)
                        or np.any(parameter > 1)
                        or not np.array_equal(a + parameter[:, None] * tangent, image)
                    ):
                        continue
                    key = tuple(sorted(tuple(row) for row in image))
                    if key in seen:
                        continue
                    seen.add(key)
                    sign = np.sign(parameter[1] - parameter[0])
                    expected += sign * _line_integral(
                        element,
                        transform @ fine_values[route],
                        local_vertices[start],
                        local_vertices[stop],
                    )
            cell_offset += len(routes)
        actual = _line_integral(coarse_element, coarse_local, a, a + tangent)
        np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("kind", ("tetrahedron", "hexahedron", "prism", "pyramid"))
@pytest.mark.parametrize("degree", (1, 2))
def test_nonlinear_compatible_complete_patch_and_dual_payload(
    kind: str, degree: int
) -> None:
    mesh, geometry = _periodic_template_source(kind, True)
    outcome = adapt_mixed_mesh(mesh, refine_cell_ids=np.asarray([17], dtype=np.int64))
    fine_mesh, _, _ = assemble_topology_edit(
        mesh, outcome.edit, numeric_version="compatible-fine"
    )
    restriction = transition_nested_cell_geometry(
        mesh,
        geometry,
        fine_mesh,
        CellGeometrySpec.affine(fine_mesh),
        refinement=outcome.edit.refinement,
    )
    fine_mesh = fine_mesh.with_coordinates(
        restriction.vertex_coordinates, numeric_version="compatible-fine"
    )
    fine_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in fine_mesh.blocks]
    )
    coarsening = adapt_mixed_mesh(
        fine_mesh,
        refine_cell_ids=np.empty(0, dtype=np.int64),
        coarsen_cell_ids=fine_ids,
        hierarchy=outcome.hierarchy,
    )
    coarse_mesh, _, _ = assemble_topology_edit(
        fine_mesh, coarsening.edit, numeric_version="compatible-coarse"
    )
    restoration = transition_nested_cell_geometry(
        fine_mesh,
        restriction.geometry,
        coarse_mesh,
        CellGeometrySpec.affine(coarse_mesh),
        refinement=coarsening.edit.refinement,
        coarsening=coarsening.edit.coarsening,
    )
    identities: tuple[tuple[int, FormTwist, FormProxy], ...] = (
        (1, "untwisted", "circulation"),
        (2, "twisted", "flux"),
    )
    for form_degree, twist, proxy in identities:
        source = FiniteElementPlan(
            mesh,
            FiniteElementFieldSpec(
                "u", _elements(mesh, form_degree, degree, twist, proxy)
            ),
            coordinate_spec=geometry,
        ).prepare()
        fine = FiniteElementPlan(
            fine_mesh,
            FiniteElementFieldSpec(
                "u", _elements(fine_mesh, form_degree, degree, twist, proxy)
            ),
            coordinate_spec=restriction.geometry,
        ).prepare()
        coarse = FiniteElementPlan(
            coarse_mesh,
            FiniteElementFieldSpec(
                "u", _elements(coarse_mesh, form_degree, degree, twist, proxy)
            ),
            coordinate_spec=restoration.geometry,
        ).prepare()
        forward = prepare_nested_field_transfer(
            source,
            fine,
            None,
            field_name="u",
            source_geometry=geometry,
            target_geometry=restriction.geometry,
            geometry_transition=restriction,
        )
        reverse = prepare_nested_field_transfer(
            fine,
            coarse,
            None,
            field_name="u",
            source_geometry=restriction.geometry,
            target_geometry=restoration.geometry,
            geometry_transition=restoration,
        )
        values: NDArray[np.float64] = np.random.default_rng(19).normal(
            size=(source.dof_maps[0].global_dof_count, 2)
        )
        fine_values = forward.transfer.apply(values)
        restored = reverse.transfer.apply(fine_values)
        np.testing.assert_allclose(restored, values, atol=2e-11, rtol=2e-11)
        dual = np.random.default_rng(29).normal(size=fine_values.shape)
        np.testing.assert_allclose(
            np.vdot(fine_values, dual),
            np.vdot(values, forward.transfer.pullback(dual)),
            atol=2e-11,
        )
        compiled = eqx.filter_jit(forward.transfer.apply)(jnp.asarray(values))
        np.testing.assert_allclose(compiled, fine_values, atol=2e-13)
        assert forward.evidence.passed and reverse.evidence.passed
        assert reverse.evidence.defect("commuting") <= reverse.evidence.tolerance
        np.testing.assert_array_equal(
            source.dof_maps[0].cell_dofs[0], coarse.dof_maps[0].cell_dofs[0]
        )
        np.testing.assert_array_equal(
            source.dof_maps[0].cell_transforms[0], coarse.dof_maps[0].cell_transforms[0]
        )
        arbitrary_fine = np.random.default_rng(47).normal(size=fine_values.shape)
        projected = np.asarray(reverse.transfer.apply(arbitrary_fine))
        if proxy == "flux":
            np.testing.assert_allclose(
                _integrated_divergence(coarse, projected),
                _integrated_divergence(fine, arbitrary_fine),
                atol=3e-11,
                rtol=3e-11,
            )
        else:
            witnesses = coarsening.edit.coarsening
            if witnesses is None:
                raise AssertionError(
                    "Complete coarsening must retain its exact reference witnesses."
                )
            indices = {
                int(identifier): row
                for row, identifier in enumerate(witnesses.fine_cell_ids)
            }
            ordered_reference = np.stack(
                [
                    witnesses.fine_reference_vertices[indices[int(identifier)]]
                    for identifier in fine_ids
                ]
            )
            _patch_edge_circulations(
                fine, coarse, arbitrary_fine, projected, ordered_reference
            )


def test_hdiv_shared_face_coorientation_fault_is_certified() -> None:
    source, target, transition = _tetrahedral_hdiv_refinement()
    faulted = _fault_shared_hdiv_face(target)
    with pytest.raises(ValueError, match="continuity"):
        prepare_nested_field_transfer(
            source,
            faulted,
            None,
            field_name="u",
            source_geometry=source.coordinate_spec,
            target_geometry=faulted.coordinate_spec,
            geometry_transition=transition,
        )


def test_hdiv_raw_commuting_moment_fault_is_certified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, transition = _tetrahedral_hdiv_refinement()
    original = ExactFormMoments.refine

    def faulted_refine(
        self: ExactFormMoments,
        source_basis: FormBasis,
        target_basis: FormBasis,
        pair: _NestedReferencePair | _PolynomialReferencePair,
    ) -> MomentIntegralMatrix:
        result = original(self, source_basis, target_basis, pair)
        if source_basis.form_degree == 2 and target_basis.form_degree == 2:
            values = result.value.copy()
            values[0, 0] += 0.25
            return MomentIntegralMatrix(values, result.error)
        return result

    monkeypatch.setattr(ExactFormMoments, "refine", faulted_refine)
    with pytest.raises(ValueError, match="commuting"):
        prepare_nested_field_transfer(
            source,
            target,
            None,
            field_name="u",
            source_geometry=source.coordinate_spec,
            target_geometry=target.coordinate_spec,
            geometry_transition=transition,
        )


@pytest.mark.parametrize("resource", ("work", "storage"))
def test_mapped_compatible_resource_refusal_preserves_source(resource: str) -> None:
    mesh, geometry = _periodic_template_source("tetrahedron", True)
    field = FiniteElementFieldSpec(
        "u", form_element("tetrahedron", 1, 2, proxy="circulation")
    )
    source = FiniteElementPlan(mesh, field, coordinate_spec=geometry).prepare()
    before = np.asarray(geometry.coordinates).copy()
    with pytest.raises(ValueError, match="budget"):
        prepare_nested_field_transfer(
            source,
            source,
            np.asarray([0], dtype=np.int64),
            field_name="u",
            source_geometry=geometry,
            target_geometry=geometry,
            parent_reference_vertices=np.asarray(
                reference_cell_topology("tetrahedron").vertices
            )[None],
            maximum_work=1 if resource == "work" else 100_000_000,
            maximum_storage_bytes=1 if resource == "storage" else 256_000_000,
        )
    np.testing.assert_array_equal(geometry.coordinates, before)
