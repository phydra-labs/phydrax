# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Original mixed/polyhedral cell inventories, distinct from point interpolation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.meshing import CellMeshingResult, MeshPart
from phydrax.meshing._overset import (
    OversetPartSpec,
    prepare_overset_connectivity,
    prepare_overset_conservative_remap,
    prepare_overset_conservative_state_transport,
    prepare_overset_motion_rebind,
)


jax.config.update("jax_enable_x64", True)


def _carrier(part: MeshPart) -> CellMeshingResult:
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise TypeError(
            "The conservative fixture requires a certified cell-mesh producer."
        )
    return carrier


def test_mixed_polyhedral_common_refinement_and_selected_scientific_cells() -> None:
    points = np.asarray(
        [(x, y, z) for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0, 2.0)]
    )

    def vertex(x: float, y: float, z: float) -> int:
        return int(x + 3 * y + 6 * z)

    hexahedron = np.asarray(
        [[vertex(*ref) for ref in reference_cell_topology("hexahedron").vertices]],
        dtype=np.int32,
    )
    prisms = np.asarray(
        [
            [
                vertex(1, 0, 0),
                vertex(2, 0, 0),
                vertex(2, 1, 0),
                vertex(1, 0, 1),
                vertex(2, 0, 1),
                vertex(2, 1, 1),
            ],
            [
                vertex(1, 0, 0),
                vertex(2, 1, 0),
                vertex(1, 1, 0),
                vertex(1, 0, 1),
                vertex(2, 1, 1),
                vertex(1, 1, 1),
            ],
        ],
        dtype=np.int32,
    )
    old_mesh = phx.discretization.CellMesh.from_mixed_3d(
        points,
        (
            phx.discretization.CellBlock(
                "hex", "hexahedron", hexahedron, global_ids=np.asarray([11])
            ),
            phx.discretization.CellBlock(
                "prisms", "prism", prisms, global_ids=np.asarray([21, 22])
            ),
        ),
        polyhedra={},
    )
    faces = (
        (0, 4, 6, 2),
        (1, 3, 7, 5),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 2, 3, 1),
        (4, 5, 7, 6),
    )
    cells = []
    for origin in (0, 1):
        ids = [
            vertex(origin + (index & 1), (index >> 1) & 1, (index >> 2) & 1)
            for index in range(8)
        ]
        cells.append([np.asarray([ids[index] for index in face]) for face in faces])
    new_mesh = phx.discretization.CellMesh.from_polyhedra(
        points, cells, cell_global_ids=np.asarray([101, 102])
    )
    contract = phx.SpatialCoordinateContract.si()
    old = phx.meshing.MeshPart("old", phx.meshing.certify_cell_mesh(old_mesh, contract))
    new = phx.meshing.MeshPart("new", phx.meshing.certify_cell_mesh(new_mesh, contract))
    source = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        _carrier(old).mesh
    ).prepare()
    # Affine cell averages equal the value at each actual polyhedral centroid.
    values = 2 + source.cell_centers @ jnp.asarray([1.0, 2.0, 3.0])
    prepared = prepare_overset_conservative_remap(old, new)
    assert prepared.succeeded, prepared.reason
    plan = prepared.plan
    if plan is None:
        raise ValueError(
            f"Successful conservative remap omitted its plan: {prepared.reason}"
        )
    transferred = plan.apply(values)
    np.testing.assert_allclose(transferred, [5.0, 6.0], atol=1e-11)
    np.testing.assert_allclose(
        plan.conservation_defect(values, transferred), 0.0, atol=1e-11
    )
    selected = prepare_overset_conservative_remap(
        old,
        new,
        source_cells=old.scope(3, np.asarray([21, 22])),
        target_cells=new.scope(3, np.asarray([102])),
    )
    assert selected.succeeded, selected.reason
    selected_plan = selected.plan
    if selected_plan is None:
        raise ValueError(
            f"Successful selected-cell remap omitted its plan: {selected.reason}"
        )
    rows = [
        int(np.flatnonzero(np.asarray(source.cell_global_ids) == identifier)[0])
        for identifier in np.asarray(selected_plan.source_cell_global_ids)
    ]
    result = selected_plan.apply(values[jnp.asarray(rows)])
    np.testing.assert_allclose(result, [6.0], atol=1e-11)
    np.testing.assert_array_equal(selected_plan.source_cell_global_ids, [21, 22])
    np.testing.assert_array_equal(selected_plan.target_cell_global_ids, [102])
    np.testing.assert_allclose(
        selected_plan.conservation_defect(values[jnp.asarray(rows)], result),
        0.0,
        atol=1e-11,
    )


def _moving_inventory() -> tuple[
    phx.meshing.OversetConnectivity,
    phx.meshing.OversetConnectivity,
    phx.lifecycle.Composition,
    tuple[phx.lifecycle.CompositionEntry, ...],
]:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    cells = np.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32)
    mesh = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("cells", "triangle", cells),)
    )
    part = MeshPart(
        "moving", phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    )
    fixed_mesh = phx.discretization.CellMesh(
        np.asarray([[-1.0, -1.0], [4.0, -1.0], [-1.0, 4.0]]),
        (
            phx.discretization.CellBlock(
                "background", "triangle", np.asarray([[0, 1, 2]], dtype=np.int32)
            ),
        ),
    )
    fixed = MeshPart(
        "reference",
        phx.meshing.certify_cell_mesh(fixed_mesh, phx.SpatialCoordinateContract.si()),
    )
    incidence = mesh.connectivity
    if not isinstance(incidence, PolygonalConnectivity):
        raise TypeError("Moving inventory requires planar incidence.")
    boundary = part.scope(
        1, np.asarray(mesh.entity_set(1).entity_ids)[np.asarray(incidence.boundary_edges)]
    )
    previous = prepare_overset_connectivity(
        phx.meshing.MeshAssembly((part, fixed)),
        (OversetPartSpec(part.name, boundary=boundary), OversetPartSpec(fixed.name)),
    )
    moved = points.copy()
    moved[4] = [0.65, 0.4]
    candidate = previous.moved({part.name: moved})
    owner = previous.composition_entries()[0]
    payloads = (
        jnp.asarray([1.0, 2.0, 4.0, 3.0]),
        jnp.asarray([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.3, 0.7]]),
        jnp.asarray([[1.0, 2.0], [2.0, 5.0], [4.0, 17.0], [3.0, 10.0]]),
    )
    states = tuple(
        phx.lifecycle.CompositionEntry(
            value,
            entry_id=name,
            role=role,
            owner_id="moving-physics",
            structure_id=owner.structure_id,
            revision_id=f"accepted:{name}",
            semantics_id=f"cell-average:{name}",
            dependencies=(owner.binding("revision"),),
        )
        for name, role, value in zip(
            ("density", "materials", "history"),
            ("physical-state", "model-state", "history"),
            payloads,
            strict=True,
        )
    )
    composition = phx.lifecycle.Composition(
        (*previous.composition_entries(), *states), boundary_id="accepted-state"
    )
    return previous, candidate, composition, states


def test_moving_mesh_conserves_every_field_material_and_history_component_atomically() -> (
    None
):
    previous, candidate, composition, states = _moving_inventory()
    old, new = previous.composition_entries()[0], candidate.composition_entries()[0]
    prepared, transports = prepare_overset_conservative_state_transport(
        old, new, states, content_tolerance=1e-12
    )
    remap = prepared.plan
    if remap is None:
        raise ValueError("Complete moving common refinement must publish its route.")
    assert all(transport.conservative for transport in transports)
    assert old.revision_id != new.revision_id
    assert not np.array_equal(remap.source_volumes, remap.target_volumes)

    # Independent oriented triangle areas, not the route's own inventory echo.
    def areas(part: MeshPart) -> jax.Array:
        mesh = _carrier(part).mesh
        corners = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
        first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        return jnp.asarray(
            0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
        )

    source_areas, target_areas = areas(old.value), areas(new.value)
    targets = tuple(transport.targets[0] for transport in transports)
    for source, target in zip(states, targets, strict=True):
        trailing = (1,) * (source.value.ndim - 1)
        np.testing.assert_allclose(
            jnp.sum(source.value * source_areas.reshape((-1,) + trailing), axis=0),
            jnp.sum(target.value * target_areas.reshape((-1,) + trailing), axis=0),
            atol=1e-12,
        )
        assert target.semantics_id == source.semantics_id
        assert target.dependencies == (new.binding("revision"),)
    assert not np.array_equal(targets[0].value, states[0].value)
    material = targets[1].value
    np.testing.assert_allclose(material.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(np.asarray(material) >= 0.0)
    motion = prepare_overset_motion_rebind(
        composition, previous, candidate, transports=transports
    )
    refused = motion.commit(accepted_boundary=False)
    assert not refused.published and refused.composition is composition
    accepted = motion.commit(accepted_boundary=True)
    assert accepted.published
    np.testing.assert_allclose(
        accepted.composition.entry("history").value, targets[2].value
    )
    np.testing.assert_allclose(composition.entry("density").value, [1.0, 2.0, 4.0, 3.0])


def test_moving_conservative_state_refuses_unbound_history_and_uncovered_physical_domain() -> (
    None
):
    previous, candidate, composition, states = _moving_inventory()
    old, new = previous.composition_entries()[0], candidate.composition_entries()[0]
    unbound = phx.lifecycle.CompositionEntry(
        states[2].value,
        entry_id="history",
        role="history",
        owner_id="moving-physics",
        structure_id=old.structure_id,
        revision_id="unbound",
        semantics_id=states[2].semantics_id,
    )
    with pytest.raises(ValueError, match="exact source part revision"):
        prepare_overset_conservative_state_transport(
            old, new, (*states[:2], unbound), content_tolerance=1e-12
        )
    shifted = previous.moved(
        {"moving": _carrier(old.value).mesh.coordinates + jnp.asarray([0.1, 0.0])}
    )
    with pytest.raises(ValueError, match="overlap refused"):
        prepare_overset_conservative_state_transport(
            old, shifted.composition_entries()[0], states, content_tolerance=1e-12
        )
    assert composition.entry("history") is states[2]


def test_rigid_native_motion_preserves_sparse_authority_scope_and_refreshes_source_revision() -> (
    None
):
    from tools.native_moving_overset import native_planar_part, rigid_planar_successor

    moving, plan = native_planar_part("moving", 1.0, size=0.7)
    background_mesh = phx.discretization.CellMesh(
        np.asarray([[-4.0, -4.0], [8.0, -4.0], [-4.0, 8.0]]),
        (
            phx.discretization.CellBlock(
                "background", "triangle", np.asarray([[0, 1, 2]], dtype=np.int32)
            ),
        ),
    )
    background = MeshPart(
        "background",
        phx.meshing.certify_cell_mesh(
            background_mesh, phx.SpatialCoordinateContract.si()
        ),
    )
    mesh = _carrier(moving).mesh
    incidence = mesh.connectivity
    if not isinstance(incidence, PolygonalConnectivity):
        raise TypeError("Rigid native planar motion requires polygonal incidence.")
    boundary = moving.scope(
        1, np.asarray(mesh.entity_set(1).entity_ids)[np.asarray(incidence.boundary_edges)]
    )
    previous = prepare_overset_connectivity(
        phx.meshing.MeshAssembly((background, moving)),
        (
            OversetPartSpec(background.name),
            OversetPartSpec(moving.name, boundary=boundary),
        ),
    )
    previous.require_complete()
    successor, successor_plan = rigid_planar_successor(
        moving,
        np.asarray([0.55, 0.0]),
        motion_id="native-rigid-motion-sparse-authority",
        plan=plan,
    )
    candidate = previous.reregister({moving.name: successor})
    candidate.require_complete()
    old_edge = next(
        a
        for a in _carrier(moving).associations
        if a.target_entity_set_id == mesh.entity_set(1).entity_set_id
    )
    new_edge = next(
        a
        for a in _carrier(successor).associations
        if a.target_entity_set_id == mesh.entity_set(1).entity_set_id
    )
    assert old_edge.target_global_ids.size < mesh.entity_set(1).count
    np.testing.assert_array_equal(new_edge.target_global_ids, old_edge.target_global_ids)
    assert new_edge.source_occurrence_paths == old_edge.source_occurrence_paths
    assert new_edge.source_entity_roles == old_edge.source_entity_roles
    np.testing.assert_array_equal(new_edge.source_dimensions, old_edge.source_dimensions)
    np.testing.assert_array_equal(new_edge.source_indices, old_edge.source_indices)
    authority = successor_plan.source
    if not isinstance(authority, phx.meshing.NativePlanarSource):
        raise TypeError("Rigid motion must retain its canonical native planar source.")
    assert new_edge.source_revision == authority.source_revision
    assert new_edge.source_revision != old_edge.source_revision
    assert new_edge.source_entity_ids != old_edge.source_entity_ids
    assert new_edge.parent_association_id == old_edge.association_id
    np.testing.assert_array_equal(new_edge.parent_ids, old_edge.target_global_ids)
