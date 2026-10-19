# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Coordinate joins reuse numerical compilation across source and mesh epochs."""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    coordinate_lagrange_element,
)
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from phydrax.meshing._quality import (
    _boundary_rule,
    _mapped_boundary_samples,
    _mapped_boundary_values,
    _mapped_quality_rule,
    _mapped_sample_values,
    _mapped_sample_worksets,
    _vertex_geometry_coordinates,
)
from tools.compile_counter import CompileCounter


def _epoch_mesh(cell_kind: str, epoch: int) -> CellMesh:
    match cell_kind:
        case "triangle":
            points = np.asarray(
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
            )
            cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
        case "polygon":
            points = np.asarray(
                (
                    (-1.0, 0.0),
                    (-0.0, 0.0),
                    (-0.0, 1.0),
                    (-0.5, 1.5),
                    (-1.0, 1.0),
                    (1.0, 0.0),
                    (1.0, 1.0),
                    (0.5, 1.5),
                ),
                dtype=np.float64,
            )
            cells = np.asarray(((0, 1, 2, 3, 4), (1, 5, 6, 7, 2)), dtype=np.int32)
        case _:
            raise ValueError(
                f"quality_compile.cell_kind must name a test fixture; got {cell_kind!r}"
            )
    return CellMesh(
        points,
        (
            CellBlock(
                "cells",
                cell_kind,
                cells,
                global_ids=np.asarray((11 + epoch, 23 + epoch), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(points.shape[0], dtype=np.int64) + 101 + epoch,
        numeric_version=str(epoch),
    )


def test_scalar_coordinate_join_compiles_nothing_across_source_metadata() -> None:
    meshes = (_epoch_mesh("triangle", 0), _epoch_mesh("triangle", 2))
    reference = coordinate_lagrange_element("triangle", 1)
    geometries = tuple(
        CellGeometrySpec(
            {
                "cells": FiniteElementSpec(
                    f"source-epoch-{epoch}",
                    reference.cell_kind,
                    reference.degree,
                    reference.reference_nodes,
                    reference.entity_dofs,
                    value_spec=reference.value_spec,
                    continuity=reference.continuity,
                    representation=reference.representation,
                    tabulator=reference.tabulator,
                    tabulator_id=reference.tabulator_id,
                    form_basis=reference.form_basis,
                )
            },
            {"cells": mesh.blocks[0].vertices},
            mesh.coordinates,
        )
        for epoch, mesh in zip((0, 2), meshes, strict=True)
    )
    resolved = tuple(
        geometry.resolve(mesh) for mesh, geometry in zip(meshes, geometries, strict=True)
    )
    identities = tuple(
        (mesh.mesh_id, mesh.numeric_version, geometry.elements[0].element_id)
        for mesh, geometry in zip(meshes, geometries, strict=True)
    )
    assert identities[0][0] != identities[1][0]
    assert identities[0][2] != identities[1][2]
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(
            _vertex_geometry_coordinates(meshes[0], *resolved[0])
        )
    with CompileCounter() as warm:
        second = jax.block_until_ready(
            _vertex_geometry_coordinates(meshes[1], *resolved[1])
        )
    assert cold.count == 0
    assert warm.count == 0, [event.to_record() for event in warm.events]
    assert first is meshes[0].coordinates
    assert second is meshes[1].coordinates
    np.testing.assert_array_equal(
        np.asarray(first).view(np.uint64), np.asarray(second).view(np.uint64)
    )
    assert identities == tuple(
        (mesh.mesh_id, mesh.numeric_version, geometry.elements[0].element_id)
        for mesh, geometry in zip(meshes, geometries, strict=True)
    )


def test_vertex_coordinate_join_reuses_epoch_compilation_and_shared_value_refusal() -> (
    None
):
    meshes = (_epoch_mesh("polygon", 0), _epoch_mesh("polygon", 2))
    bank = np.asarray(meshes[0].coordinates)[meshes[0].blocks[0].vertices].reshape(
        (-1, 2)
    )
    routes = np.arange(10, dtype=np.int32).reshape((2, 5))
    geometries = tuple(
        CellGeometrySpec(
            {"cells": CellGeometrySpec.affine(mesh).elements[0]},
            {"cells": route},
            values,
        )
        for mesh, route, values in zip(
            meshes, (routes, 9 - routes), (bank, bank[::-1]), strict=True
        )
    )
    resolved = tuple(
        geometry.resolve(mesh) for mesh, geometry in zip(meshes, geometries, strict=True)
    )
    identities = tuple(
        (mesh.mesh_id, mesh.numeric_version, geometry.geometry_layout_id)
        for mesh, geometry in zip(meshes, geometries, strict=True)
    )
    assert identities[0][0] != identities[1][0]
    assert identities[0][2] != identities[1][2]
    inconsistent_bank = bank.copy()
    inconsistent_bank[5, 0] = 0.125
    inconsistent = CellGeometrySpec(
        {"cells": geometries[0].elements[0]}, {"cells": routes}, inconsistent_bank
    ).resolve(meshes[0])
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(
            _vertex_geometry_coordinates(meshes[0], *resolved[0])
        )
    with CompileCounter() as warm:
        second = jax.block_until_ready(
            _vertex_geometry_coordinates(meshes[1], *resolved[1])
        )
    assert cold.count > 0
    assert warm.count == 0, [event.to_record() for event in warm.events]
    expected = bank[[0, 1, 2, 3, 4, 6, 7, 8]]
    for points in (first, second):
        np.testing.assert_array_equal(
            np.asarray(points).view(np.uint64), expected.view(np.uint64)
        )
    with CompileCounter() as rejected:
        with pytest.raises(
            eqx.EquinoxRuntimeError,
            match=(
                r"quality\.geometry\.coordinates must agree at shared vertices; "
                "got different source values"
            ),
        ):
            jax.block_until_ready(_vertex_geometry_coordinates(meshes[0], *inconsistent))
    assert rejected.count == 0, [event.to_record() for event in rejected.events]
    assert identities == tuple(
        (mesh.mesh_id, mesh.numeric_version, geometry.geometry_layout_id)
        for mesh, geometry in zip(meshes, geometries, strict=True)
    )


@pytest.mark.parametrize("count", (1, 3))
@pytest.mark.parametrize("depth", (1, 2))
def test_mapped_quality_keeps_barycentric_source_actions_dynamic(
    count: int, depth: int
) -> None:
    root = coordinate_lagrange_element("triangle", 1)
    identity = np.eye(3, dtype=np.float64)
    changed = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.5, 1.0)), dtype=np.float64
    )
    sources = tuple(
        BarycentricCellGeometryElement(root, weights) for weights in (identity, changed)
    )
    if depth == 2:
        sources = tuple(
            BarycentricCellGeometryElement(source, changed) for source in sources
        )
    coordinates = jax.device_put(
        np.broadcast_to(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
            (count, 3, 2),
        ).copy()
    )
    rule = _mapped_quality_rule("triangle", root)
    boundary = _boundary_rule("triangle")
    assert boundary is not None
    probes, directions = boundary
    identities = tuple(source.element_id for source in sources)
    assert identities[0] != identities[1]
    # Passing the scientific source directly retains the pre-projection fused
    # tabulation program as the bitwise reference for each numerical action.
    expected = tuple(
        jax.block_until_ready(
            (
                _mapped_sample_values("triangle", source, 2, coordinates, None, rule),
                _mapped_boundary_values(source, 2, coordinates, probes, directions),
            )
        )
        for source in sources
    )
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(
            (
                _mapped_sample_worksets("triangle", sources[0], coordinates, None, rule),
                _mapped_boundary_samples(sources[0], coordinates, probes, directions),
            )
        )
    with CompileCounter() as warm:
        second = jax.block_until_ready(
            (
                _mapped_sample_worksets("triangle", sources[1], coordinates, None, rule),
                _mapped_boundary_samples(sources[1], coordinates, probes, directions),
            )
        )
    assert cold.count > 0
    assert warm.count == 0, [event.to_record() for event in warm.events]
    for actual, reference in zip((first, second), expected, strict=True):
        for values, baseline in zip(
            jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True
        ):
            np.testing.assert_array_equal(
                np.asarray(values).view(np.uint8), np.asarray(baseline).view(np.uint8)
            )
    assert not np.array_equal(first[0].measure, second[0].measure)
    assert identities == tuple(source.element_id for source in sources)
