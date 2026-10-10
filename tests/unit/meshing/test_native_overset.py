# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import ArrayLike

from phydrax import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from phydrax.discretization._hexahedral import HexahedralConnectivity
from phydrax.discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    FiniteElementSpec,
    lagrange_element,
)
from phydrax.lifecycle import Composition, CompositionEntry, CompositionTransport
from phydrax.meshing import CellMeshingResult, certify_cell_mesh, MeshingScope
from phydrax.meshing._assembly import MeshAssembly, MeshPart
from phydrax.meshing._contracts import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._coupling import CouplingSearchStatus
from phydrax.meshing._overset import (
    OversetCellStatus,
    OversetConnectivity,
    OversetConnectivityError,
    OversetPartSpec,
    OversetPolicy,
    OversetVertexStatus,
    prepare_overset_connectivity,
    prepare_overset_conservative_remap,
    prepare_overset_field_transfer,
    prepare_overset_motion_rebind,
)


jax.config.update("jax_enable_x64", True)


def _carrier(part: MeshPart) -> CellMeshingResult:
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise TypeError("The overset fixture requires a certified cell-mesh producer.")
    return carrier


def _part(
    name: str,
    points: ArrayLike,
    cells: ArrayLike,
    *,
    ids: ArrayLike | None = None,
) -> MeshPart:
    points = np.asarray(points, dtype=np.float64)
    kind = "triangle" if points.shape[1] == 2 else "tetrahedron"
    mesh = CellMesh(
        points,
        (CellBlock("cells", kind, np.asarray(cells, dtype=np.int32)),),
        vertex_global_ids=None if ids is None else np.asarray(ids),
    )
    return MeshPart(name, certify_cell_mesh(mesh, SpatialCoordinateContract.si()))


def _boundary(part: MeshPart) -> MeshingScope:
    mesh = _carrier(part).mesh
    dimension = mesh.ambient_dimension
    connectivity = mesh.connectivity
    if isinstance(connectivity, PolygonalConnectivity):
        mask = np.asarray(connectivity.boundary_edges)
    elif isinstance(
        connectivity,
        (TetrahedralConnectivity, HexahedralConnectivity, PolyhedralConnectivity),
    ):
        mask = np.asarray(connectivity.boundary_faces)
    else:
        raise TypeError(
            "The overset boundary requires a two- or three-dimensional cell mesh."
        )
    return part.scope(
        dimension - 1, np.asarray(mesh.entity_set(dimension - 1).entity_ids)[mask]
    )


def _simple_registration(dimension: int, *, third: bool = False) -> OversetConnectivity:
    source_points = np.vstack((np.zeros(dimension), 4 * np.eye(dimension)))
    cells = [list(range(dimension + 1))]
    source = _part("background", source_points, cells)
    receptor = _part("moving", 0.3 + 0.2 * source_points, cells)
    parts = [source, receptor]
    specs = [
        OversetPartSpec(source.name),
        OversetPartSpec(receptor.name, boundary=_boundary(receptor)),
    ]
    if third:
        other = _part("preferred", source_points, cells)
        parts.append(other)
        specs.append(OversetPartSpec(other.name, priority=7))
    return prepare_overset_connectivity(
        MeshAssembly(tuple(parts)), tuple(specs), policy=OversetPolicy(fringe_layers=1)
    )


def _field(part: MeshPart, degree: int = 1) -> FiniteElementDiscretization:
    return FiniteElementPlan(
        _carrier(part).mesh,
        FiniteElementFieldSpec(
            "u", lagrange_element(_carrier(part).mesh.blocks[0].cell_kind, degree)
        ),
        coordinate_spec=_carrier(part).geometry,
    ).prepare()


@pytest.mark.parametrize("dimension", [2, 3])
def test_moving_overlap_affine_order_and_transpose(dimension: int) -> None:
    connectivity = _simple_registration(dimension)
    connectivity.require_complete()
    for epoch in range(2):
        source = connectivity.assembly.part("background")
        field = _field(source)
        route = prepare_overset_field_transfer(connectivity, {source.name: field}, "u")
        coordinates = np.asarray(field.dof_maps[0].dof_coordinates)
        slope = np.arange(1, dimension + 1)
        coefficients = {source.name: jnp.asarray(2 + coordinates @ slope)}
        values = route.apply(coefficients)
        target = connectivity.assembly.part("moving")
        evidence = connectivity.receptors_of(target.name)
        sites = np.asarray(_carrier(target).mesh.coordinates)[
            np.asarray(evidence.receptor_rows)
        ]
        np.testing.assert_allclose(values[target.name], 2 + sites @ slope, atol=1e-11)
        ones = {source.name: jnp.ones(coordinates.shape[0])}
        np.testing.assert_allclose(route.apply(ones)[target.name], 1, atol=1e-12)
        cotangents = {
            name: jnp.arange(value.size, dtype=jnp.float64).reshape(value.shape) + 1
            for name, value in values.items()
        }
        assert bool(route.duality_evidence(coefficients, cotangents).valid)
        previous = connectivity
        connectivity = connectivity.moved(
            {target.name: _carrier(target).mesh.coordinates + 0.05}
        )
        assert connectivity.epoch == previous.epoch + 1
        for old, new in zip(
            previous.assembly.parts, connectivity.assembly.parts, strict=True
        ):
            assert _carrier(old).mesh.topology_id == _carrier(new).mesh.topology_id
            np.testing.assert_array_equal(
                _carrier(old).mesh.vertex_global_ids, _carrier(new).mesh.vertex_global_ids
            )
        with pytest.raises(ValueError, match="stale"):
            connectivity.assembly.part("moving").require_scope(_boundary(target))


@pytest.mark.parametrize("dimension", [2, 3])
def test_quadratic_field_is_not_downgraded_to_vertex_interpolation(
    dimension: int,
) -> None:
    connectivity = _simple_registration(dimension)
    donor = connectivity.assembly.part("background")
    field = _field(donor, 2)
    nodes = np.asarray(field.dof_maps[0].dof_coordinates)
    polynomial = lambda points: (
        points[:, 0] ** 2 + points[:, 0] * points[:, 1] - 2 * points[:, -1] + 3
    )
    coefficients = {donor.name: jnp.asarray(polynomial(nodes))}
    route = prepare_overset_field_transfer(connectivity, {donor.name: field}, "u")
    receptor = connectivity.assembly.part("moving")
    rows = np.asarray(connectivity.receptors_of(receptor.name).receptor_rows)
    nodes = np.asarray(_carrier(receptor).mesh.coordinates)[rows]
    np.testing.assert_allclose(
        route.apply(coefficients)[receptor.name], polynomial(nodes), atol=1e-11
    )
    dual = {
        name: jnp.ones(value.shape) for name, value in route.apply(coefficients).items()
    }
    assert bool(route.duality_evidence(coefficients, dual).valid)


def test_overlap_ambiguity_uses_declared_priority_and_reports_candidates() -> None:
    connectivity = _simple_registration(2, third=True)
    evidence = connectivity.receptors_of("moving")
    np.testing.assert_array_equal(evidence.candidate_parts, 2)
    selected = connectivity.part_index("preferred")
    np.testing.assert_array_equal(evidence.donor_parts, selected)


def test_missing_and_excluded_donors_keep_pointwise_orphan_reasons() -> None:
    previous = _simple_registration(2)
    moved = previous.moved(
        {"moving": _carrier(previous.assembly.part("moving")).mesh.coordinates + 10}
    )
    assert not moved.complete
    np.testing.assert_array_equal(
        moved.receptors_of("moving").status, int(CouplingSearchStatus.OUTSIDE)
    )
    with pytest.raises(OversetConnectivityError):
        moved.require_complete()
    source, target = (
        previous.assembly.part("background"),
        previous.assembly.part("moving"),
    )
    excluded = source.scope(2, np.asarray(_carrier(source).mesh.blocks[0].global_ids))
    connectivity = prepare_overset_connectivity(
        MeshAssembly((source, target)),
        (
            OversetPartSpec(source.name, excluded=excluded),
            OversetPartSpec(target.name, boundary=_boundary(target)),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    np.testing.assert_array_equal(
        connectivity.receptors_of(target.name).status,
        int(CouplingSearchStatus.EXCLUDED_DONOR),
    )


def _square_grid(name: str, axis: ArrayLike, *, cavity: float | None = None) -> MeshPart:
    axis = np.asarray(axis)
    points = np.asarray([(x, y) for y in axis for x in axis])
    n = len(axis)
    cells = []
    for j in range(n - 1):
        for i in range(n - 1):
            center = 0.5 * (points[j * n + i] + points[(j + 1) * n + i + 1])
            if cavity is not None and np.all(np.abs(center) < cavity):
                continue
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i
            cells.extend(((a, b, c), (a, c, d)))
    used = np.unique(cells)
    renumber = np.full(len(points), -1, dtype=np.int32)
    renumber[used] = np.arange(used.size)
    return _part(name, points[used], renumber[np.asarray(cells)])


def _annular_registration() -> OversetConnectivity:
    background = _square_grid("background", np.arange(-3.0, 4.0))
    body = _square_grid(
        "moving", [-2.25, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, 2.25], cavity=0.25
    )
    mesh = _carrier(body).mesh
    if not isinstance(mesh.connectivity, PolygonalConnectivity):
        raise TypeError("The annular fixture requires polygonal connectivity.")
    edges = np.asarray(mesh.connectivity.edges)
    boundary = np.asarray(mesh.connectivity.boundary_edges)
    corners = np.asarray(mesh.coordinates)[edges]
    inner = boundary & np.all(np.abs(corners) <= 0.25, axis=(1, 2))
    ids = np.asarray(mesh.entity_set(1).entity_ids)
    wall = body.scope(1, ids[inner])
    outer = body.scope(1, ids[boundary & ~inner])
    return prepare_overset_connectivity(
        MeshAssembly((background, body)),
        (
            OversetPartSpec(background.name),
            OversetPartSpec(body.name, wall=wall, boundary=outer),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )


def test_hole_cutting_moving_annulus_preserves_protected_walls() -> None:
    connectivity = _annular_registration()
    background = connectivity.assembly.part("background")
    body = connectivity.assembly.part("moving")
    for _ in range(2):
        connectivity.require_complete()
        background_blanking = connectivity.blanking_of(background.name)
        assert background_blanking.vertex_ids_with(OversetVertexStatus.HOLE).size == 1
        protected = connectivity.blanking_of(body.name)
        current = connectivity.assembly.part(body.name)
        mesh = _carrier(current).mesh
        if not isinstance(mesh.connectivity, PolygonalConnectivity):
            raise TypeError("The moving annular wall requires polygonal connectivity.")
        wall = connectivity.specs[connectivity.part_index(body.name)].wall
        if wall is None:
            raise ValueError("The moving annulus must retain its protected wall scope.")
        wall_rows = np.unique(
            np.asarray(mesh.connectivity.edges)[
                np.isin(
                    np.asarray(mesh.entity_set(1).entity_ids), np.asarray(wall.entity_ids)
                )
            ]
        )
        np.testing.assert_array_equal(
            np.asarray(protected.vertex_status)[wall_rows],
            int(OversetVertexStatus.ACTIVE),
        )
        assert not np.any(np.asarray(protected.protected_conflict))
        connectivity = connectivity.moved(
            {
                body.name: _carrier(current).mesh.coordinates
                + jnp.asarray([0.05, 0.0], dtype=jnp.float64)
            }
        )


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("image", [False, True])
def test_closed_wall_cuts_contained_cells_and_keeps_own_wall_active(
    dimension: int, image: bool
) -> None:
    shell = _part(
        "solid",
        np.vstack((np.zeros(dimension), 4 * np.eye(dimension))),
        [list(range(dimension + 1))],
    )
    rotation = np.eye(dimension)
    rotation[:2, :2] = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    translation = np.asarray([5.0, -2.0] if dimension == 2 else [5.0, -2.0, 1.0])
    target_rotation = rotation.T
    target_translation = np.asarray([0.2, 0.5] if dimension == 2 else [0.2, 0.5, 0.1])
    inside_points = 0.3 + 0.1 * np.vstack((np.zeros(dimension), np.eye(dimension)))
    if image:
        inside_points = (
            inside_points @ rotation.T + translation - target_translation
        ) @ target_rotation
    inside = _part("background", inside_points, [list(range(dimension + 1))])
    specs = (
        OversetPartSpec(
            shell.name,
            wall=_boundary(shell),
            image_rotation=rotation if image else None,
            image_translation=translation if image else None,
        ),
        OversetPartSpec(
            inside.name,
            image_rotation=target_rotation if image else None,
            image_translation=target_translation if image else None,
        ),
    )
    connectivity = prepare_overset_connectivity(
        MeshAssembly((shell, inside)),
        specs,
        policy=OversetPolicy(fringe_layers=1),
    )
    connectivity.require_complete()
    np.testing.assert_array_equal(
        connectivity.blanking_of(inside.name).cell_status, int(OversetCellStatus.HOLE)
    )
    np.testing.assert_array_equal(
        connectivity.blanking_of(shell.name).vertex_status,
        int(OversetVertexStatus.ACTIVE),
    )
    with pytest.raises(MeshingFailure) as error:
        prepare_overset_connectivity(
            MeshAssembly((shell, inside)),
            specs,
            policy=OversetPolicy(maximum_wall_candidate_pairs=1),
        )
    assert error.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED


def test_motion_rebind_has_exact_rollback_and_refuses_untransported_history() -> None:
    previous = _simple_registration(2)
    own = previous.composition_entries()
    part_entry = next(item for item in own if item.entry_id == "overset:part:moving")
    donor = previous.assembly.part("background")
    field = _field(donor)
    coefficients = {donor.name: 2 + jnp.sum(field.dof_maps[0].dof_coordinates, axis=1)}
    old_route = prepare_overset_field_transfer(previous, {donor.name: field}, "u")
    donor_part_entry = next(
        item for item in own if item.entry_id == "overset:part:background"
    )
    donor_state = CompositionEntry(
        coefficients[donor.name],
        entry_id="donor-solution",
        role="physical-state",
        owner_id="solver",
        structure_id=field.dof_maps[0].dof_map_id,
        revision_id="donor-solution:old",
        semantics_id="temperature",
        dependencies=(donor_part_entry.binding("revision"),),
    )
    state = CompositionEntry(
        old_route.apply(coefficients)["moving"],
        entry_id="solution",
        role="physical-state",
        owner_id="solver",
        structure_id=part_entry.structure_id,
        revision_id="solution:old",
        semantics_id="temperature",
        dependencies=(part_entry.binding("revision"),),
    )
    history = CompositionEntry(
        jnp.asarray([0.5]),
        entry_id="history",
        role="history",
        owner_id="solver",
        structure_id="history-layout",
        revision_id="history:old",
        semantics_id="time-history",
    )
    source = Composition((*own, state, donor_state, history), boundary_id="accepted-step")
    moving = previous.assembly.part("moving")
    candidate = previous.moved({moving.name: _carrier(moving).mesh.coordinates + 0.05})
    new_part = next(
        item
        for item in candidate.composition_entries()
        if item.entry_id == part_entry.entry_id
    )
    route = prepare_overset_field_transfer(candidate, {donor.name: field}, "u")
    target = CompositionEntry(
        route.apply(coefficients)["moving"],
        entry_id=state.entry_id,
        role=state.role,
        owner_id=state.owner_id,
        structure_id=new_part.structure_id,
        revision_id="solution:new",
        semantics_id=state.semantics_id,
        dependencies=(new_part.binding("revision"),),
    )
    transport = CompositionTransport(
        "physical-remap",
        (state.entry_id, donor_state.entry_id),
        (target, donor_state),
        source_structure_ids=(state.structure_id, donor_state.structure_id),
        route_id=route.transfer_id,
        successful=True,
    )
    with pytest.raises(ValueError, match="lack an explicit"):
        prepare_overset_motion_rebind(
            source, previous, candidate, transports=(transport,)
        )
    staged = prepare_overset_motion_rebind(
        source, previous, candidate, transports=(transport,), retain=(history.entry_id,)
    )
    refused = staged.commit(accepted_boundary=False)
    assert not refused.published and refused.composition is source
    accepted = staged.commit(accepted_boundary=True)
    assert accepted.published
    receptor_rows = np.asarray(candidate.receptors_of("moving").receptor_rows)
    expected = 2 + np.sum(
        np.asarray(_carrier(candidate.assembly.part("moving")).mesh.coordinates)[
            receptor_rows
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        accepted.composition.value("solution"), expected, atol=1e-11
    )
    assert accepted.composition.value("history") is history.value
    nonconservative_claim = CompositionTransport(
        "physical-remap",
        (state.entry_id, donor_state.entry_id),
        (target, donor_state),
        source_structure_ids=(state.structure_id, donor_state.structure_id),
        route_id=route.transfer_id,
        successful=True,
        source_content=8 * jnp.mean(donor_state.value) + 0.32 * jnp.mean(state.value),
        target_content=8 * jnp.mean(donor_state.value) + 0.32 * jnp.mean(target.value),
        content_tolerance=jnp.asarray(1e-12),
    )
    rejected = prepare_overset_motion_rebind(
        source,
        previous,
        candidate,
        transports=(nonconservative_claim,),
        retain=(history.entry_id,),
    ).commit(accepted_boundary=True)
    assert not rejected.published and rejected.composition is source
    missing = previous.moved({moving.name: _carrier(moving).mesh.coordinates + 10.0})
    no_state = Composition(previous.composition_entries(), boundary_id="accepted-step")
    orphaned = prepare_overset_motion_rebind(no_state, previous, missing).commit(
        accepted_boundary=True
    )
    assert not orphaned.published and orphaned.composition is no_state


def test_conservative_overlap_preserves_inventory_not_just_constants() -> None:
    points = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    source = _part("old", points, ((0, 1, 2), (0, 2, 3)))
    target = _part("new", points, ((0, 1, 3), (1, 2, 3)))
    prepared = prepare_overset_conservative_remap(source, target)
    assert prepared.succeeded, prepared.reason
    plan = prepared.plan
    if plan is None:
        raise ValueError(
            f"Successful conservative remap omitted its plan: {prepared.reason}"
        )
    values = jnp.asarray([2.0, 7.0])
    remapped = plan.apply(values)
    np.testing.assert_allclose(remapped, [4.5, 4.5], atol=1e-12)
    np.testing.assert_allclose(
        plan.conservation_defect(values, remapped), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(plan.apply(jnp.ones(2)), 1.0, atol=1e-12)
    # Same-domain interpolation preserves constants but not the P1 inventory:
    # a changed diagonal changes the vertex's integrated hat function.
    connectivity = prepare_overset_connectivity(
        MeshAssembly((source, target)),
        (
            OversetPartSpec(source.name),
            OversetPartSpec(target.name, boundary=_boundary(target)),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    field = _field(source)
    transfer = prepare_overset_field_transfer(connectivity, {source.name: field}, "u")
    nodes = np.asarray(field.dof_maps[0].dof_coordinates)
    coefficients = {source.name: jnp.asarray(nodes[:, 0] * nodes[:, 1])}
    interpolated = transfer.apply(coefficients)[target.name]
    constant = transfer.apply({source.name: jnp.ones(4)})[target.name]
    np.testing.assert_allclose(constant, 1.0)
    old_averages = jnp.mean(
        coefficients[source.name][_carrier(source).mesh.blocks[0].vertices], axis=1
    )
    new_averages = jnp.mean(
        interpolated[_carrier(target).mesh.blocks[0].vertices], axis=1
    )
    old_inventory = jnp.sum(old_averages * plan.source_volumes)
    new_inventory = jnp.sum(new_averages * plan.target_volumes)
    np.testing.assert_allclose(old_inventory, 1 / 3, atol=1e-12)
    np.testing.assert_allclose(new_inventory, 1 / 6, atol=1e-12)
    conservative = plan.apply(old_averages)
    np.testing.assert_allclose(
        jnp.sum(conservative * plan.target_volumes), old_inventory, atol=1e-12
    )


def test_nonaffine_motion_never_discards_the_coordinate_map() -> None:
    part = _part("curved", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0, 1, 2),))
    element = lagrange_element("triangle", 2)
    coordinates = np.asarray(element.reference_nodes).copy()
    coordinates[:, 1] += (
        0.1 * coordinates[:, 0] * (1 - coordinates[:, 0] - coordinates[:, 1])
    )
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(element.local_dof_count)[None, :]},
        coordinates,
    )
    curved = MeshPart(
        part.name,
        certify_cell_mesh(
            _carrier(part).mesh, SpatialCoordinateContract.si(), geometry=geometry
        ),
    )
    with pytest.raises(MeshingFailure, match="complete successor coordinate map"):
        curved.with_coordinates(
            _carrier(curved).mesh.coordinates + 0.1, motion_id="translate"
        )
    successor_geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(element.local_dof_count)[None, :]},
        coordinates + 0.1,
    )
    moved = curved.with_coordinates(
        _carrier(curved).mesh.coordinates + 0.1,
        motion_id="translate",
        geometry=successor_geometry,
    )
    np.testing.assert_array_equal(_carrier(moved).geometry.coordinates, coordinates + 0.1)
    moved_element = _carrier(moved).geometry.elements[0]
    if not isinstance(moved_element, FiniteElementSpec):
        raise TypeError(
            "The translated quadratic source must retain its finite-element coordinates."
        )
    assert moved_element.degree == 2


def test_cut_cell_crossing_without_any_enclosed_vertex_is_reported_and_blanked() -> None:
    wall = _part(
        "wall", ((-1.0, 0.9), (3.0, 0.9), (3.0, 1.1), (-1.0, 1.1)), ((0, 1, 2), (0, 2, 3))
    )
    background = _part("background", ((0.0, 0.0), (2.0, 0.0), (1.0, 2.0)), ((0, 1, 2),))
    connectivity = prepare_overset_connectivity(
        MeshAssembly((wall, background)),
        (
            OversetPartSpec(wall.name, wall=_boundary(wall)),
            OversetPartSpec(background.name),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    blanking = connectivity.blanking_of(background.name)
    np.testing.assert_array_equal(blanking.ambiguous_cells, True)
    np.testing.assert_array_equal(blanking.cell_status, int(OversetCellStatus.HOLE))
    assert blanking.vertex_ids_with(OversetVertexStatus.HOLE).size == 0
    assert not connectivity.complete


def test_registration_rebuilds_moved_donors_without_live_spatial_state() -> None:
    previous = _simple_registration(3)
    moved = previous.moved(
        {"moving": _carrier(previous.assembly.part("moving")).mesh.coordinates + 0.1}
    )
    restored = moved.registration().prepare()
    field = _field(restored.assembly.part("background"))
    values = 1 + jnp.sum(field.dof_maps[0].dof_coordinates, axis=1)
    route = prepare_overset_field_transfer(restored, {"background": field}, "u")
    receptor = restored.assembly.part("moving")
    rows = restored.receptors_of("moving").receptor_rows
    np.testing.assert_allclose(
        route.apply({"background": values})["moving"],
        1 + jnp.sum(_carrier(receptor).mesh.coordinates[rows], axis=1),
        atol=1e-11,
    )
    assert restored.connectivity_id == moved.connectivity_id


def test_motion_transfer_fills_uncovered_nodes_with_valid_material_donors() -> None:
    previous = _annular_registration()
    body = previous.assembly.part("moving")
    candidate = previous.moved(
        {
            body.name: _carrier(body).mesh.coordinates
            + jnp.asarray([0.55, 0.0], dtype=jnp.float64)
        }
    )
    candidate.require_complete()
    fields = {part.name: _field(part) for part in candidate.assembly.parts}
    coefficients = {
        name: 3
        + jnp.sum(
            _field(previous.assembly.part(name)).dof_maps[0].dof_coordinates, axis=1
        )
        for name in fields
    }
    # Deliberately invalid old hole content must never become a donor.
    old_holes = np.asarray(previous.blanking_of("background").vertex_status) == int(
        OversetVertexStatus.HOLE
    )
    coefficients["background"] = (
        coefficients["background"].at[jnp.asarray(old_holes)].set(-1000.0)
    )
    route = prepare_overset_field_transfer(candidate, fields, "u", previous=previous)
    evidence = next(item for item in route.receptors if item.part_name == "background")
    center_row = int(np.flatnonzero(old_holes)[0])
    slot = int(np.flatnonzero(np.asarray(evidence.receptor_rows) == center_row)[0])
    values = route.apply(coefficients)
    np.testing.assert_allclose(values["background"][slot], 3 - 0.55, atol=1e-11)
    dual = {name: jnp.ones(value.shape) for name, value in values.items()}
    assert bool(route.duality_evidence(coefficients, dual).valid)


@pytest.mark.parametrize("image", [False, True])
def test_authoritative_native_brep_membership_cuts_the_registered_solid(
    image: bool,
) -> None:
    from itertools import permutations

    from phydrax.geometry import brep_box, prepare_brep_query

    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.float64
    )
    cells = []
    for order in permutations(range(3)):
        row = (0, 1 << order[0], (1 << order[0]) | (1 << order[1]), 7)
        parity = sum(
            order[first] > order[second]
            for first in range(3)
            for second in range(first + 1, 3)
        )
        cells.append(row if parity % 2 == 0 else (row[0], row[2], row[1], row[3]))
    body = _part("solid", points, cells)
    inside_points = np.asarray(
        ((0.2, 0.2, 0.2), (0.3, 0.2, 0.2), (0.2, 0.3, 0.2), (0.2, 0.2, 0.3))
    )
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([5.0, -2.0, 1.0])
    target_rotation = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    target_translation = np.asarray([0.2, 0.5, 0.1])
    if image:
        inside_points = (
            inside_points @ rotation.T + translation - target_translation
        ) @ target_rotation
    background = _part("background", inside_points, ((0, 1, 2, 3),))
    query = prepare_brep_query(
        brep_box((0, 0, 0), (1, 1, 1), coordinate_contract=SpatialCoordinateContract.si())
    )
    connectivity = prepare_overset_connectivity(
        MeshAssembly((body, background)),
        (
            OversetPartSpec(
                body.name,
                wall=_boundary(body),
                solid_query=query,
                image_rotation=rotation if image else None,
                image_translation=translation if image else None,
            ),
            OversetPartSpec(
                background.name,
                image_rotation=target_rotation if image else None,
                image_translation=target_translation if image else None,
            ),
        ),
    )
    connectivity.require_complete()
    np.testing.assert_array_equal(
        connectivity.blanking_of(background.name).vertex_status,
        int(OversetVertexStatus.HOLE),
    )
    with pytest.raises(ValueError, match="successor B-Rep query"):
        connectivity.moved({body.name: _carrier(body).mesh.coordinates + 0.1})


@pytest.mark.parametrize("authority", ["plc", "brep"])
def test_near_isometry_large_source_wall_boundary_uses_encoded_map_pullback(
    authority: str,
) -> None:
    from itertools import permutations

    from phydrax.geometry import brep_box, prepare_brep_query

    upper_x = float(2**30)
    lower = np.asarray([upper_x - 1.0, 0.0, 0.0], dtype=np.float64)
    points = lower + np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)],
        dtype=np.float64,
    )
    cells = []
    for order in permutations(range(3)):
        row = (0, 1 << order[0], (1 << order[0]) | (1 << order[1]), 7)
        parity = sum(
            order[first] > order[second]
            for first in range(3)
            for second in range(first + 1, 3)
        )
        cells.append(row if parity % 2 == 0 else (row[0], row[2], row[1], row[3]))
    body = _part("solid", points, cells)
    rotation = (1.0 + 2.0**-44) * np.eye(3, dtype=np.float64)
    source_sites = lower + np.asarray(
        [[1.0, 0.25, 0.25], [0.75, 0.25, 0.25], [1.0, 0.375, 0.25], [1.0, 0.25, 0.375]],
        dtype=np.float64,
    )
    background = _part("background", source_sites @ rotation.T, ((0, 2, 1, 3),))
    query = (
        prepare_brep_query(
            brep_box(
                lower, lower + 1.0, coordinate_contract=SpatialCoordinateContract.si()
            )
        )
        if authority == "brep"
        else None
    )
    registration = prepare_overset_connectivity(
        MeshAssembly((body, background)),
        (
            OversetPartSpec(
                body.name,
                wall=_boundary(body),
                solid_query=query,
                image_rotation=rotation,
                image_translation=np.zeros(3, dtype=np.float64),
            ),
            OversetPartSpec(background.name),
        ),
    )
    registration.require_complete()
    np.testing.assert_array_equal(
        registration.blanking_of(background.name).vertex_status,
        int(OversetVertexStatus.HOLE),
    )
    np.testing.assert_array_equal(
        registration.blanking_of(background.name).wall_contact, [True, False, True, True]
    )
    np.testing.assert_array_equal(
        registration.blanking_of(body.name).vertex_status, int(OversetVertexStatus.ACTIVE)
    )


def test_native_brep_wall_work_allowance_is_shared_across_parts() -> None:
    from itertools import permutations

    from phydrax._meshcore import NativeExecutionBudget
    from phydrax.geometry import brep_box, prepare_brep_query
    from phydrax.geometry.brep._query import BRepQueryBudget, BRepQueryResourceError

    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.float64
    )
    cells = []
    for order in permutations(range(3)):
        row = (0, 1 << order[0], (1 << order[0]) | (1 << order[1]), 7)
        parity = sum(
            order[first] > order[second]
            for first in range(3)
            for second in range(first + 1, 3)
        )
        cells.append(row if parity % 2 == 0 else (row[0], row[2], row[1], row[3]))
    body = _part("solid", points, cells)
    first_points = np.asarray(
        [[0.2, 0.2, 0.2], [0.3, 0.2, 0.2], [0.2, 0.3, 0.2], [0.2, 0.2, 0.3]],
        dtype=np.float64,
    )
    first = _part("first", first_points, ((0, 1, 2, 3),))
    second = _part("second", first_points + 0.4, ((0, 1, 2, 3),))
    query = prepare_brep_query(
        brep_box((0, 0, 0), (1, 1, 1), coordinate_contract=SpatialCoordinateContract.si())
    )
    rejected = BRepQueryBudget(0)
    with pytest.raises(BRepQueryResourceError) as admission:
        query.contains(first_points, budget=rejected)
    assert admission.value.resource == "operations"
    assert rejected.operations == 0
    # One original wall box adds one cell/wall candidate to the source owner's
    # admitted operation envelope; neither allowance is renewed for another part.
    policy = OversetPolicy(maximum_wall_candidate_pairs=admission.value.requested + 1)
    body_spec = OversetPartSpec(body.name, wall=_boundary(body), solid_query=query)
    single = prepare_overset_connectivity(
        MeshAssembly((body, first)),
        (body_spec, OversetPartSpec(first.name)),
        policy=policy,
    )
    single.require_complete()
    np.testing.assert_array_equal(
        single.blanking_of(first.name).cell_status, int(OversetCellStatus.HOLE)
    )
    with pytest.raises(MeshingFailure) as exhausted:
        prepare_overset_connectivity(
            MeshAssembly((body, first, second)),
            (body_spec, OversetPartSpec(first.name), OversetPartSpec(second.name)),
            policy=policy,
        )
    assert exhausted.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert exhausted.value.stage == "hole-cut"
    outer = NativeExecutionBudget(
        max_work=10000,
        max_geometry_queries=4,
        max_cavity_cells=100,
        max_scratch_bytes=32 * 1024 * 1024,
        max_wall_seconds=120.0,
    )
    inner = NativeExecutionBudget(
        max_work=20000,
        max_geometry_queries=100,
        max_cavity_cells=100,
        max_scratch_bytes=64 * 1024 * 1024,
        max_wall_seconds=180.0,
    )
    with outer, inner:
        accepted = prepare_overset_connectivity(
            MeshAssembly((body, first)),
            (body_spec, OversetPartSpec(first.name)),
        )
        accepted.require_complete()
        assert inner.remaining().remaining_geometry_queries == 0
        with pytest.raises(MeshingFailure) as inherited:
            prepare_overset_connectivity(
                MeshAssembly((body, second)),
                (body_spec, OversetPartSpec(second.name)),
            )
        assert inherited.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    if outer.evidence is None:
        raise RuntimeError(
            "The native source-query budget must publish its actual evidence."
        )
    assert outer.evidence.externally_charged_geometry_queries == 4


def test_disconnected_closed_walls_preserve_both_holes() -> None:
    left = _part("left", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0, 1, 2),))
    right = _part("right", ((3.0, 0.0), (4.0, 0.0), (3.0, 1.0)), ((0, 1, 2),))
    background = _part(
        "background",
        ((0.1, 0.1), (0.2, 0.1), (0.1, 0.2), (3.1, 0.1), (3.2, 0.1), (3.1, 0.2)),
        ((0, 1, 2), (3, 4, 5)),
    )
    connectivity = prepare_overset_connectivity(
        MeshAssembly((background, left, right)),
        (
            OversetPartSpec(background.name),
            OversetPartSpec(left.name, wall=_boundary(left)),
            OversetPartSpec(right.name, wall=_boundary(right)),
        ),
    )
    connectivity.require_complete()
    np.testing.assert_array_equal(
        connectivity.blanking_of(background.name).cell_status, int(OversetCellStatus.HOLE)
    )


def test_curved_inverse_map_exhaustion_reports_orphans_without_p1_flattening() -> None:
    affine = _part("background", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0, 1, 2),))
    element = lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes).copy()
    x, y = nodes[:, 0].copy(), nodes[:, 1].copy()
    nodes[:, 0] += 0.8 * x * y
    nodes[:, 1] += 0.6 * x * (1 - x - y)
    geometry = CellGeometrySpec(
        {"cells": element}, {"cells": np.arange(element.local_dof_count)[None, :]}, nodes
    )
    source = MeshPart(
        affine.name,
        certify_cell_mesh(
            _carrier(affine).mesh, SpatialCoordinateContract.si(), geometry=geometry
        ),
    )
    target = _part("moving", ((0.2, 0.2), (0.22, 0.2), (0.2, 0.22)), ((0, 1, 2),))
    from phydrax.discretization import SimplicialLocationPolicy

    connectivity = prepare_overset_connectivity(
        MeshAssembly((source, target)),
        (
            OversetPartSpec(source.name),
            OversetPartSpec(target.name, boundary=_boundary(target)),
        ),
        policy=OversetPolicy(
            fringe_layers=1,
            location_policy=SimplicialLocationPolicy(1, 1, 1, residual_tolerance=1e-14),
        ),
    )
    assert not connectivity.complete
    np.testing.assert_array_equal(
        connectivity.receptors_of(target.name).status,
        int(CouplingSearchStatus.UNRESOLVED),
    )


def test_donor_work_capacity_is_refused_before_route_publication() -> None:
    previous = _simple_registration(3)
    source, target = (
        previous.assembly.part("background"),
        previous.assembly.part("moving"),
    )
    with pytest.raises(MeshingFailure) as error:
        prepare_overset_connectivity(
            MeshAssembly((source, target)),
            (
                OversetPartSpec(source.name),
                OversetPartSpec(target.name, boundary=_boundary(target)),
            ),
            policy=OversetPolicy(fringe_layers=1, maximum_donor_candidate_pairs=3),
        )
    assert error.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED


def test_sliver_donor_affine_evaluation_retains_inverse_map_evidence() -> None:
    source = _part(
        "background",
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 1e-5)),
        ((0, 1, 2, 3),),
    )
    points = 0.1 * np.asarray(_carrier(source).mesh.coordinates) + [0.2, 0.2, 1e-6]
    target = _part("moving", points, ((0, 1, 2, 3),))
    connectivity = prepare_overset_connectivity(
        MeshAssembly((source, target)),
        (
            OversetPartSpec(source.name),
            OversetPartSpec(target.name, boundary=_boundary(target)),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    connectivity.require_complete()
    field = _field(source)
    coefficients = {
        source.name: 2 + field.dof_maps[0].dof_coordinates @ jnp.asarray([1.0, 2.0, 3.0])
    }
    route = prepare_overset_field_transfer(connectivity, {source.name: field}, "u")
    rows = connectivity.receptors_of(target.name).receptor_rows
    np.testing.assert_allclose(
        route.apply(coefficients)[target.name],
        2 + _carrier(target).mesh.coordinates[rows] @ jnp.asarray([1.0, 2.0, 3.0]),
        atol=1e-10,
    )
    assert np.all(np.asarray(connectivity.receptors_of(target.name).residuals) <= 1e-10)


def test_tetrahedral_conservative_overlap_preserves_inventory() -> None:
    from itertools import permutations

    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.float64
    )
    cells = np.asarray(
        [
            (0, 1 << order[0], (1 << order[0]) | (1 << order[1]), 7)
            for order in permutations(range(3))
        ],
        dtype=np.int32,
    )
    # The permutation construction alternates orientation; a valid tetrahedral
    # carrier requires every shared face to have opposite induced orientations.
    negative = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]]) < 0.0
    cells[negative] = cells[negative][:, (1, 0, 2, 3)]
    target_cells = cells ^ 1
    negative = (
        np.linalg.det(points[target_cells[:, 1:]] - points[target_cells[:, :1]]) < 0.0
    )
    target_cells[negative] = target_cells[negative][:, (1, 0, 2, 3)]
    source = _part("old", points, cells)
    target = _part("new", points, target_cells)
    prepared = prepare_overset_conservative_remap(source, target)
    assert prepared.succeeded, prepared.reason
    plan = prepared.plan
    if plan is None:
        raise ValueError(
            f"Successful conservative remap omitted its plan: {prepared.reason}"
        )
    values = jnp.arange(1.0, 7.0)
    remapped = plan.apply(values)
    np.testing.assert_allclose(jnp.sum(values * plan.source_volumes), 3.5, atol=1e-12)
    np.testing.assert_allclose(jnp.sum(remapped * plan.target_volumes), 3.5, atol=1e-12)
    np.testing.assert_allclose(
        plan.conservation_defect(values, remapped), 0.0, atol=1e-12
    )
