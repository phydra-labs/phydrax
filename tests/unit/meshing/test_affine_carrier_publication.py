#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Affine bisection preserves accepted roots and rounds ideal descendants once."""

from fractions import Fraction

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometrySpec,
    coordinate_lagrange_element,
)
from phydrax.meshing import (
    BisectionCompatibility,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from tests._support.cell_meshes import proposal_source


def _affine_source(*, transformed: bool) -> phx.meshing.CellMeshingResult:
    coefficients = np.asarray(
        ((0.0, 0.0), (1.2345678901234567, 0.0), (0.0, 0.9876543210987654)),
        dtype=np.float64,
    )
    root = coordinate_lagrange_element("triangle", 1)
    if transformed:
        weights = np.asarray(
            ((0.75, 0.125, 0.125), (0.125, 0.75, 0.125), (0.125, 0.125, 0.75)),
            dtype=np.float64,
        )
        element = BarycentricCellGeometryElement(root, weights)
        points = np.asarray(
            [
                [
                    float(
                        sum(
                            (
                                Fraction(float(weights[row, column]))
                                * Fraction(float(coefficients[column, axis]))
                                for column in range(3)
                            ),
                            Fraction(),
                        )
                    )
                    for axis in range(2)
                ]
                for row in range(3)
            ],
            dtype=np.float64,
        )
    else:
        element = root
        points = coefficients.copy()
        points[0, 0] = -0.0
        coefficients = points.copy()
    mesh = phx.discretization.CellMesh.from_triangles(
        points, np.asarray(((0, 1, 2),), dtype=np.int32)
    )
    geometry = CellGeometrySpec(
        {mesh.blocks[0].name: element},
        {mesh.blocks[0].name: np.asarray(((0, 1, 2),), dtype=np.int32)},
        coefficients,
    )
    return phx.meshing.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )


@pytest.mark.parametrize(
    "transformed", (False, True), ids=("signed-root", "affine-basis")
)
def test_affine_carrier_chain_uses_published_roots_without_roundoff_accumulation(
    transformed: bool,
) -> None:
    source = _affine_source(transformed=transformed)
    source_ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    source_points = np.asarray(source.mesh.coordinates, dtype=np.float64)
    ideals = {
        int(identifier): tuple(Fraction(float(value)) for value in source_points[row])
        for row, identifier in enumerate(source_ids)
    }
    root_bits = source_points.view(np.uint64).copy()
    current = source
    hierarchy = None
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=BisectionCompatibility.CONFORMING_CLOSURE,
    )
    for _ in range(3):
        marks = np.concatenate(
            [np.asarray(block.global_ids) for block in current.mesh.blocks]
        )
        update = execute_mesh_adaptation(
            prepare_mesh_adaptation(
                current, MarkedMeshAdaptation(marks, hierarchy=hierarchy), policy=policy
            )
        )
        assert update.transition is not None
        stencil = update.transition.vertex_stencil
        assert stencil is not None
        identifiers = np.asarray(stencil.target_global_ids, dtype=np.int64)
        parents = np.asarray(stencil.source_global_ids, dtype=np.int64)
        weights = np.asarray(stencil.weights, dtype=np.float64)
        valid = np.asarray(stencil.valid, dtype=np.bool_)
        expected = {}
        for row, identifier in enumerate(identifiers):
            expected[int(identifier)] = tuple(
                sum(
                    (
                        Fraction(float(weights[row, slot]))
                        * ideals[int(parents[row, slot])][axis]
                        for slot in range(weights.shape[1])
                        if valid[row, slot]
                    ),
                    Fraction(),
                )
                for axis in range(2)
            )
        target_ids = np.asarray(update.target.mesh.vertex_global_ids, dtype=np.int64)
        target = np.asarray(update.target.mesh.coordinates, dtype=np.float64)
        for row, identifier in enumerate(target_ids):
            if int(identifier) in set(source_ids.tolist()):
                old = int(np.flatnonzero(source_ids == identifier)[0])
                np.testing.assert_array_equal(target[row].view(np.uint64), root_bits[old])
            else:
                rounded = np.asarray(
                    tuple(float(value) for value in expected[int(identifier)]),
                    dtype=np.float64,
                )
                np.testing.assert_array_equal(
                    target[row].view(np.uint64), rounded.view(np.uint64)
                )
        ideals = expected
        current, hierarchy = update.target, update.hierarchy


def test_affine_carrier_proof_refuses_changed_publication_map() -> None:
    source = _affine_source(transformed=False)
    update = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(np.asarray([0], dtype=np.int64)),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=BisectionCompatibility.CONFORMING_CLOSURE,
            ),
        )
    )
    restriction = update.target.geometry.restriction_source
    assert restriction is not None and restriction.carrier_geometry is not None
    changed = eqx.tree_at(
        lambda value: value.restriction_source.carrier_geometry.coordinates,
        update.target.geometry,
        restriction.carrier_geometry.coordinates + 0.25,
    )
    with pytest.raises(ValueError, match="carrier_geometry must reproduce"):
        changed.resolve(update.target.mesh)


@pytest.mark.parametrize(
    "compatibility",
    (
        BisectionCompatibility.UNIFORM_REFINEMENT,
        BisectionCompatibility.CONFORMING_CLOSURE,
    ),
    ids=("uniform", "conforming"),
)
def test_affine_carrier_preserves_non_storage_order_vertex_identity(
    compatibility: BisectionCompatibility,
) -> None:
    source = proposal_source()
    source_points = {
        int(identifier): np.asarray(source.mesh.coordinates)[row]
        for row, identifier in enumerate(np.asarray(source.mesh.vertex_global_ids))
    }
    update = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(
                np.concatenate(
                    [np.asarray(block.global_ids) for block in source.mesh.blocks]
                )
            ),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_BISECTION, compatibility=compatibility
            ),
        )
    )
    update.target.geometry.resolve(update.target.mesh)
    assert update.transition is not None
    stencil = update.transition.vertex_stencil
    assert stencil is not None
    parents = np.asarray(stencil.source_global_ids)
    weights = np.asarray(stencil.weights)
    valid = np.asarray(stencil.valid)
    expected = {}
    for row, identifier in enumerate(np.asarray(stencil.target_global_ids)):
        expected[int(identifier)] = np.asarray(
            [
                float(
                    sum(
                        (
                            Fraction(float(weights[row, slot]))
                            * Fraction(float(source_points[int(parent)][axis]))
                            for slot, parent in enumerate(parents[row])
                            if valid[row, slot]
                        ),
                        Fraction(),
                    )
                )
                for axis in range(2)
            ],
            dtype=np.float64,
        )
    for row, identifier in enumerate(np.asarray(update.target.mesh.vertex_global_ids)):
        reference = (
            source_points[int(identifier)]
            if int(identifier) in source_points
            else expected[int(identifier)]
        )
        np.testing.assert_array_equal(
            np.asarray(update.target.mesh.coordinates)[row].view(np.uint64),
            reference.view(np.uint64),
        )


def test_affine_carrier_refuses_swapped_vertex_coordinate_association() -> None:
    source = proposal_source()
    update = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(np.asarray((10, 20, 30, 40), dtype=np.int64)),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
            ),
        )
    )
    coordinates = np.array(update.target.mesh.coordinates, dtype=np.float64, copy=True)
    coordinates[[0, 1]] = coordinates[[1, 0]]
    swapped = eqx.tree_at(
        lambda value: value.coordinates, update.target.mesh, coordinates
    )
    with pytest.raises(ValueError, match="carrier_geometry must reproduce"):
        update.target.geometry.resolve(swapped)
