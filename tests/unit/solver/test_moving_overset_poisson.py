from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._execution_runtime import ExecutionGroup, ExecutionGroupSpec
from phydrax._meshcore import meshcore_available
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.discretization._view_support import OwnerLocalCellGeometry
from phydrax.discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)
from phydrax.lifecycle import Composition, CompositionEntry
from phydrax.meshing._assembly import MeshAssembly, MeshPart, MeshPartManifest
from phydrax.meshing._distribution import ShardedOwnerDirectoryWitness
from phydrax.meshing._motion import (
    advance_mesh_motion,
    advance_owner_local_mesh_motion,
    MeshMotionAdvance,
    MeshMotionMonitor,
)
from phydrax.meshing._overset import (
    OversetConnectivity,
    OversetConnectivityError,
    OversetPartSpec,
    OversetPolicy,
    prepare_overset_connectivity,
    prepare_overset_conservative_remap,
    prepare_overset_field_transfer,
)
from phydrax.meshing._result import CellMeshingResult
from phydrax.solver._moving_overset_poisson import (
    _owner_local_coefficient_ids,
    _owner_local_reconstruction,
    prepare_moving_overset_poisson_update,
    PreparedMovingOversetPoissonUpdate,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="moving overset overlap requires meshcore"
)


def _square_part(name: str, triangles: tuple[tuple[int, int, int], ...]) -> MeshPart:
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                f"{name}-triangles",
                "triangle",
                np.asarray(triangles, dtype=np.int64),
            ),
        ),
        numeric_version=f"{name}-source",
    )
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    return MeshPart(name, result)


def _p1_cell_content(
    discretization: FiniteElementDiscretization, coefficients: Array, /
) -> np.ndarray:
    dof_map = discretization.dof_maps[discretization._field_index("u")]
    measures = np.asarray(discretization.measures[0].weights, dtype=np.float64)
    values = np.asarray(coefficients, dtype=np.float64)
    content = np.empty((measures.size,), dtype=np.float64)
    offset = 0
    for block_index, block in enumerate(discretization.mesh.blocks):
        cell_dofs = np.asarray(dof_map.cell_dofs[block_index], dtype=np.int64)
        stop = offset + block.cell_count
        content[offset:stop] = measures[offset:stop] * np.mean(values[cell_dofs], axis=1)
        offset = stop
    return content


def _part(name: str, points: tuple[tuple[float, float], ...]) -> MeshPart:
    coordinates = np.asarray(points, dtype=np.float64)
    cells = np.asarray(((0, 1, 3), (1, 2, 3), (2, 0, 3)), dtype=np.int64)
    mesh = CellMesh(
        coordinates,
        (CellBlock(f"{name}-triangles", "triangle", cells),),
        numeric_version=f"{name}-source",
    )
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    return MeshPart(name, result)


def _cell_result(part: MeshPart, /) -> CellMeshingResult:
    if not isinstance(part.carrier, CellMeshingResult):
        raise TypeError("This focused FE test requires a certified cell mesh.")
    return part.carrier


def _cell_geometry(result: CellMeshingResult, /) -> CellGeometrySpec:
    if not isinstance(result.geometry, CellGeometrySpec):
        raise TypeError("This focused FE test requires affine cell geometry.")
    return result.geometry


def _owner_view(part: MeshPart, /) -> OwnerLocalCellGeometry:
    if not isinstance(part.carrier, OwnerLocalCellGeometry):
        raise TypeError("The three-process fixture requires owner-local views.")
    return part.carrier


def _boundary_ids(mesh: CellMesh, facet_dimension: int, /) -> np.ndarray:
    entity_ids = np.asarray(mesh.entity_set(facet_dimension).entity_ids, dtype=np.int64)
    if isinstance(mesh.connectivity, SimplicialConnectivity):
        boundary = mesh.connectivity.boundary_masks[facet_dimension]
    elif isinstance(mesh.connectivity, PolygonalConnectivity) and facet_dimension == 1:
        boundary = mesh.connectivity.boundary_edges
    else:
        raise TypeError("The focused FE test requires simplex or polygon connectivity.")
    return entity_ids[np.asarray(boundary, dtype=np.bool_)]


def _owned_boundary_ids(view: OwnerLocalCellGeometry, mesh: CellMesh, /) -> np.ndarray:
    owner_by_cell = {
        int(identifier): int(owner)
        for identifier, owner in zip(
            np.asarray(view.cell_ids, dtype=np.int64).tolist(),
            np.asarray(view.cell_owner_ranks, dtype=np.int32).tolist(),
            strict=True,
        )
    }
    cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    owned_cells = np.asarray(
        [owner_by_cell[int(identifier)] == view.owner_rank for identifier in cell_ids],
        dtype=np.bool_,
    )
    connectivity = mesh.connectivity
    if isinstance(connectivity, SimplicialConnectivity):
        cell_facets = np.asarray(connectivity.cell_entities[1], dtype=np.int64)
        facet_valid = np.ones(cell_facets.shape, dtype=np.bool_)
        boundary = np.asarray(connectivity.boundary_masks[1], dtype=np.bool_)
    elif isinstance(connectivity, PolygonalConnectivity):
        cell_facets = np.asarray(connectivity.cell_edges, dtype=np.int64)
        facet_valid = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
        boundary = np.asarray(connectivity.boundary_edges, dtype=np.bool_)
    else:
        raise TypeError("The owner-local fixture requires simplex or polygon cells.")
    entity_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    owned_facets = np.zeros((entity_ids.size,), dtype=np.bool_)
    owned_facets[cell_facets[owned_cells][facet_valid[owned_cells]]] = True
    return entity_ids[boundary & owned_facets]


def _case() -> tuple[
    OversetConnectivity,
    dict[str, FiniteElementPlan],
    Composition,
    MeshPart,
]:
    donor = _part(
        "donor",
        ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.5, 0.5)),
    )
    receptor = _part(
        "receptor",
        ((0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.36, 0.36)),
    )
    receptor_result = _cell_result(receptor)
    receptor_mesh = receptor_result.mesh
    boundary_ids = _boundary_ids(receptor_mesh, 1)
    connectivity = prepare_overset_connectivity(
        MeshAssembly((donor, receptor)),
        (
            OversetPartSpec("donor"),
            OversetPartSpec("receptor", boundary=receptor.scope(1, boundary_ids)),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    plans: dict[str, FiniteElementPlan] = {}
    entries = list(connectivity.composition_entries())
    part_entries = {entry.entry_id: entry for entry in entries}
    for part in (donor, receptor):
        result = _cell_result(part)
        geometry = _cell_geometry(result)
        plan = FiniteElementPlan(
            result.mesh,
            FiniteElementFieldSpec("u", lagrange_element("triangle", 1)),
            coordinate_spec=geometry,
        )
        discretization = plan.prepare()
        plans[part.name] = plan
        field_entry = CompositionEntry(
            discretization,
            entry_id=f"fe:{part.name}",
            role="discretization",
            owner_id="moving-overset-test",
            structure_id=discretization.dof_maps[0].dof_map_id,
            revision_id=discretization.prepared_id,
            semantics_id=f"poisson-space:{part.name}",
            dependencies=(part_entries[f"overset:part:{part.name}"].binding("revision"),),
        )
        state = jnp.zeros(
            (discretization.dof_maps[0].global_dof_count,), dtype=jnp.float64
        )
        state_entry = CompositionEntry(
            state,
            entry_id=f"poisson:{part.name}",
            role="physical-state",
            owner_id="moving-overset-test",
            structure_id=field_entry.structure_id,
            revision_id=f"poisson:{part.name}:0",
            semantics_id="scalar-poisson-potential",
            dependencies=(field_entry.binding("revision"),),
        )
        entries.extend((field_entry, state_entry))
    return connectivity, plans, Composition(entries, boundary_id="accepted-0"), receptor


def _advance(
    part: MeshPart,
    shift: tuple[float, float],
    /,
    *,
    move_all: bool = False,
) -> MeshMotionAdvance:
    result = _cell_result(part)
    monitor = MeshMotionMonitor(result.mesh)
    coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64).copy()
    displacement = np.asarray(shift, dtype=np.float64)
    if move_all:
        coordinates += displacement
    else:
        coordinates[-1] += displacement
    return advance_mesh_motion(
        monitor,
        result,
        coordinates,
        boundary_residual=0.0,
        numeric_version="moving-overset-test",
    )


def test_moving_overset_poisson_commits_all_parts_with_overlap_flux_balance() -> None:
    connectivity, plans, composition, receptor = _case()
    advance = _advance(receptor, (0.01, 0.0))

    update = prepare_moving_overset_poisson_update(
        composition,
        connectivity,
        {"receptor": advance},
        plans,
        "u",
    )

    assert isinstance(update, PreparedMovingOversetPoissonUpdate)
    assert update.accepted
    expected_content = {"donor": 2.0, "receptor": 0.125}
    assert {part.part_name for part in update.parts} == set(expected_content)
    for part in update.parts:
        assert part.conservative_remap.complete
        assert part.source_content == pytest.approx(expected_content[part.part_name])
        assert part.target_content == pytest.approx(expected_content[part.part_name])
        assert part.overlap_defect <= 1e-9
        assert part.linear_residual <= 1e-11
        assert part.boundary_flux + part.overset_flux == pytest.approx(
            -expected_content[part.part_name], abs=1e-9
        )
        assert part.flux_defect <= 1e-9

    rejected = update.commit(accepted_boundary=False)
    assert not rejected.published
    assert rejected.composition is composition
    accepted = update.commit(accepted_boundary=True)
    assert accepted.published
    assert accepted.composition.composition_id != composition.composition_id
    assert accepted.composition.entry("poisson:donor").value is not None
    assert accepted.composition.entry("poisson:receptor").value is not None


def test_moving_overset_orphan_refuses_before_composition_publication() -> None:
    connectivity, plans, composition, receptor = _case()
    advance = _advance(receptor, (5.0, 0.0), move_all=True)
    accepted_id = composition.composition_id

    with pytest.raises(OversetConnectivityError, match="incomplete"):
        prepare_moving_overset_poisson_update(
            composition,
            connectivity,
            {"receptor": advance},
            plans,
            "u",
        )

    assert composition.composition_id == accepted_id


def _square_part(name: str, triangles: tuple[tuple[int, int, int], ...]) -> MeshPart:
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                f"{name}-triangles",
                "triangle",
                np.asarray(triangles, dtype=np.int64),
            ),
        ),
        numeric_version=f"{name}-source",
    )
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    return MeshPart(name, result)


def _p1_cell_content(
    discretization: FiniteElementDiscretization, coefficients: Array, /
) -> np.ndarray:
    dof_map = discretization.dof_maps[discretization._field_index("u")]
    measures = np.asarray(discretization.measures[0].weights, dtype=np.float64)
    values = np.asarray(coefficients, dtype=np.float64)
    content = np.empty((measures.size,), dtype=np.float64)
    offset = 0
    for block_index, block in enumerate(discretization.mesh.blocks):
        cell_dofs = np.asarray(dof_map.cell_dofs[block_index], dtype=np.int64)
        stop = offset + block.cell_count
        content[offset:stop] = measures[offset:stop] * np.mean(values[cell_dofs], axis=1)
        offset = stop
    return content


def test_overset_interpolation_is_distinct_from_conservative_overlap() -> None:
    donor = _square_part("square-donor", ((0, 1, 2), (0, 2, 3)))
    receptor = _square_part("square-receptor", ((0, 1, 3), (1, 2, 3)))
    receptor_result = _cell_result(receptor)
    receptor_mesh = receptor_result.mesh
    boundary_ids = _boundary_ids(receptor_mesh, 1)
    connectivity = prepare_overset_connectivity(
        MeshAssembly((donor, receptor)),
        (
            OversetPartSpec(donor.name),
            OversetPartSpec(receptor.name, boundary=receptor.scope(1, boundary_ids)),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    field = FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    donor_result = _cell_result(donor)
    donor_discretization = FiniteElementPlan(
        donor_result.mesh,
        field,
        coordinate_spec=_cell_geometry(donor_result),
    ).prepare()
    receptor_discretization = FiniteElementPlan(
        receptor_mesh,
        field,
        coordinate_spec=_cell_geometry(receptor_result),
    ).prepare()

    def bilinear(points: Array, _args: object) -> Array:
        return points[..., 0] * points[..., 1]

    source_coefficients = donor_discretization.project("u", bilinear)
    interpolation = prepare_overset_field_transfer(
        connectivity, {donor.name: donor_discretization}, "u"
    )
    receptor_values = interpolation.apply({donor.name: source_coefficients})[
        receptor.name
    ]
    receptor_evidence = connectivity.receptors_of(receptor.name)
    receptor_dofs: dict[int, int] = {}
    receptor_dof_map = receptor_discretization.dof_maps[0]
    for block_index, block in enumerate(receptor_mesh.blocks):
        for vertex_row, dof_row in zip(
            np.asarray(block.vertices, dtype=np.int64).reshape(-1),
            np.asarray(receptor_dof_map.cell_dofs[block_index], dtype=np.int64).reshape(
                -1
            ),
            strict=True,
        ):
            identifier = int(receptor_mesh.vertex_global_ids[vertex_row])
            receptor_dofs.setdefault(identifier, int(dof_row))
    interpolated_coefficients = jnp.zeros(
        (receptor_dof_map.global_dof_count,), dtype=jnp.float64
    )
    for identifier, value in zip(
        np.asarray(receptor_evidence.receptor_ids, dtype=np.int64),
        np.asarray(receptor_values).reshape((-1,)),
        strict=True,
    ):
        interpolated_coefficients = interpolated_coefficients.at[
            receptor_dofs[int(identifier)]
        ].set(value)

    remap = prepare_overset_conservative_remap(donor, receptor)
    source_cell_content = _p1_cell_content(donor_discretization, source_coefficients)
    conservative_target = remap.apply_extensive(source_cell_content)
    interpolated_target_content = float(
        np.sum(_p1_cell_content(receptor_discretization, interpolated_coefficients))
    )

    assert not interpolation.conservative
    assert remap.complete
    assert float(np.sum(source_cell_content)) == pytest.approx(1.0 / 3.0)
    assert float(np.sum(np.asarray(conservative_target))) == pytest.approx(1.0 / 3.0)
    assert interpolated_target_content == pytest.approx(1.0 / 6.0)
    assert not np.isclose(
        interpolated_target_content,
        float(np.sum(source_cell_content)),
        rtol=0.0,
        atol=1e-12,
    )


def _owner_local_source_part(
    name: str,
    points: tuple[tuple[float, float], ...],
    execution_group: ExecutionGroup,
    rank: int,
) -> tuple[MeshPart, MeshPartManifest]:
    part = _part(name, points)
    result = _cell_result(part)
    mesh = result.mesh
    geometry = _cell_geometry(result)
    cell_owners = np.asarray((0, 1, 2), dtype=np.int32)
    distribution = phx.meshing.MeshDistribution(
        part,
        phx.discretization.CellPartition(cell_owners, 3),
        halo_width=3,
    )
    local = distribution.local_cell_geometry(rank, include_halo=True)
    cell_ids = np.asarray(local["cell_ids"], dtype=np.int64)
    cell_rows = np.asarray(local["cell_rows"], dtype=np.int64)
    cell_owner_ranks = cell_owners[cell_rows]
    vertex_owner_by_id: dict[int, int] = {}
    coordinate_owner_by_row: dict[int, int] = {}
    for cell_row, owner in enumerate(cell_owners):
        for vertex_row in np.asarray(mesh.blocks[0].vertices[cell_row], dtype=np.int64):
            identifier = int(mesh.vertex_global_ids[vertex_row])
            vertex_owner_by_id[identifier] = min(
                int(owner), vertex_owner_by_id.get(identifier, int(owner))
            )
            coordinate_owner_by_row[int(vertex_row)] = min(
                int(owner),
                coordinate_owner_by_row.get(int(vertex_row), int(owner)),
            )
    vertex_ids = np.asarray(local["vertex_ids"], dtype=np.int64)
    vertex_owners = np.asarray(
        [vertex_owner_by_id[int(identifier)] for identifier in vertex_ids],
        dtype=np.int32,
    )
    coordinate_rows = np.asarray(local["coordinate_rows"], dtype=np.int64)
    source_owners = np.asarray(
        [coordinate_owner_by_row[int(identifier)] for identifier in coordinate_rows],
        dtype=np.int32,
    )
    manifest = MeshPartManifest.from_part(part)
    directory_id = f"{name}/owner-directory"
    base_directory = ShardedOwnerDirectoryWitness(
        execution_group,
        rank,
        manifest.manifest_id,
        directory_id,
        coordinate_rows,
        source_owners,
        source_owners == rank,
        cell_ids,
        cell_owner_ranks,
        vertex_ids,
        vertex_owners,
        source_map_id=geometry.source_map_id,
        axis_name=f"moving_owner_local_{name}_directory",
    )
    source_entry = CompositionEntry(
        result,
        entry_id=f"{name}/source",
        role="topology",
        owner_id="moving-owner-local-test-source",
        structure_id=mesh.mesh_id,
        revision_id=part.part_id,
        semantics_id=manifest.source_geometry_id or "",
    )
    part_authority = CompositionEntry(
        part,
        entry_id=f"{name}/part-authority",
        role="topology",
        owner_id="moving-owner-local-test",
        structure_id=mesh.topology_id,
        revision_id=part.part_id,
        semantics_id=manifest.source_geometry_id or "",
        dependencies=(
            source_entry.binding("structure"),
            source_entry.binding("revision"),
            source_entry.binding("semantics"),
            base_directory.composition_entry().binding("structure"),
        ),
    )
    owner_directory = base_directory.bind_source_authority(
        part_authority.record_id,
        execution_group,
        axis_name=f"moving_owner_local_{name}_directory",
    )
    accepted_composition = Composition(
        (
            source_entry,
            owner_directory.composition_entry(),
            part_authority,
        ),
        boundary_id=manifest.manifest_id,
    )
    record = distribution.owner_local_geometry_record(
        rank,
        part_authority,
        accepted_composition,
        owner_directory,
        source_id=name,
        source_revision=part.part_id,
        include_halo=True,
    )
    view = OwnerLocalCellGeometry.from_owner_local_result(record)
    local_part = MeshPart(name, view, coordinate_contract=part.coordinate_contract)
    return local_part, MeshPartManifest.from_part(local_part)


def _owner_local_poisson_case(
    execution_group: ExecutionGroup, rank: int
) -> tuple[
    OversetConnectivity,
    dict[str, FiniteElementPlan],
    Composition,
    dict[str, MeshPart],
]:
    donor, donor_manifest = _owner_local_source_part(
        "owner-poisson-donor",
        ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.5, 0.5)),
        execution_group,
        rank,
    )
    receptor, receptor_manifest = _owner_local_source_part(
        "owner-poisson-receptor",
        ((0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.36, 0.36)),
        execution_group,
        rank,
    )
    parts = {part.name: part for part in (donor, receptor)}
    assembly = MeshAssembly(
        tuple(parts.values()),
        global_parts=(donor_manifest, receptor_manifest),
        local_rank=rank,
    )
    receptor_view = _owner_view(receptor)
    receptor_mesh, _ = receptor_view.reconstruct_local_geometry()
    if receptor_mesh is None:
        raise ValueError("Owner-local receptor mesh must be non-empty.")
    # Owner-local scopes declare each global face once: on the owner of the
    # cell carrying that boundary face.
    boundary_ids = _owned_boundary_ids(receptor_view, receptor_mesh)
    specifications = (
        OversetPartSpec("owner-poisson-donor", owner_rank=0),
        OversetPartSpec(
            "owner-poisson-receptor",
            boundary=receptor.scope(1, boundary_ids),
            owner_rank=1,
        ),
    )
    prepared = prepare_overset_connectivity(
        assembly,
        specifications,
        policy=OversetPolicy(fringe_layers=1),
        execution_group=execution_group,
        axis_name="moving_owner_local_poisson_initial",
    )
    field_spec = FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    plans: dict[str, FiniteElementPlan] = {}
    discretizations: dict[str, FiniteElementDiscretization] = {}
    reconstructions: dict[str, object] = {}
    coefficient_ids: dict[str, np.ndarray] = {}
    for part in parts.values():
        view = _owner_view(part)
        mesh, geometry = view.reconstruct_local_geometry()
        if mesh is None or geometry is None:
            raise ValueError(f"Owner-local part {part.name!r} must be non-empty.")
        plan = FiniteElementPlan(mesh, field_spec, coordinate_spec=geometry)
        discretization = plan.prepare()
        plans[part.name] = plan
        discretizations[part.name] = discretization
        reconstructions[part.name] = _owner_local_reconstruction(
            view, discretization, "u"
        )
        coefficient_ids[part.name] = _owner_local_coefficient_ids(discretization, "u")
    # Owner-local donor coverage is proved by the collective donor search of the
    # field exchange; only the resolved connectivity publishes entries.
    connectivity = (
        prepare_overset_field_transfer(prepared, reconstructions, "u")
        .prepare_owner_local_exchange(
            coefficient_ids,
            execution_group,
            {name: _owner_view(part) for name, part in parts.items()},
            axis_name="moving_owner_local_poisson_initial_field",
        )
        .connectivity
    )
    entries = list(connectivity.composition_entries())
    part_entries = {entry.entry_id: entry for entry in entries}
    for name, discretization in discretizations.items():
        fe_entry = CompositionEntry(
            discretization,
            entry_id=f"fe:{name}",
            role="discretization",
            owner_id="moving-owner-local-test",
            structure_id=discretization.dof_maps[0].dof_map_id,
            revision_id=discretization.prepared_id,
            semantics_id=f"moving-owner-local-space:{name}",
            dependencies=(part_entries[f"overset:part:{name}"].binding("revision"),),
        )
        state_entry = CompositionEntry(
            jnp.zeros((discretization.dof_maps[0].global_dof_count,), dtype=jnp.float64),
            entry_id=f"poisson:{name}",
            role="physical-state",
            owner_id="moving-owner-local-test",
            structure_id=fe_entry.structure_id,
            revision_id=f"moving-owner-local-state:{name}:0",
            semantics_id="scalar-poisson-potential",
            dependencies=(fe_entry.binding("revision"),),
        )
        entries.extend((fe_entry, state_entry))
    return (
        connectivity,
        plans,
        Composition(entries, boundary_id="moving-owner-local-poisson-accepted"),
        parts,
    )


def test_owner_local_moving_overset_poisson_commits_collectively() -> None:
    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    process_ranks = tuple(device.process_index for device in devices)
    if (
        jax.process_count() != 3
        or len(devices) != 3
        or process_ranks != (0, 1, 2)
        or len(jax.local_devices()) != 1
    ):
        pytest.skip("This route requires three native processes with one device each.")
    execution_group = ExecutionGroup(
        ExecutionGroupSpec(
            "moving-owner-local-poisson",
            process_ranks,
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("overset", 3),),
        ),
        devices,
    )
    rank = process_ranks.index(jax.process_index())
    previous, plans, composition, source_parts = _owner_local_poisson_case(
        execution_group, rank
    )
    advances: dict[str, MeshMotionAdvance] = {}
    target_parts: list[MeshPart] = []
    for index, (name, part) in enumerate(sorted(source_parts.items())):
        view = _owner_view(part)
        coordinates = np.asarray(view.coordinates, dtype=np.float64).copy()
        center_row = int(np.flatnonzero(np.asarray(view.coordinate_rows) == 3)[0])
        coordinates[center_row, 0] += 0.002 * (index + 1)
        fixed_vertices = np.asarray(view.vertex_ids, dtype=np.int64) != 3
        advance = advance_owner_local_mesh_motion(
            view,
            coordinates,
            execution_group,
            boundary_residual=0.0,
            fixed_vertices=fixed_vertices,
            driver_id=f"moving-owner-local-poisson-{name}",
            axis_name=f"moving_owner_local_poisson_motion_{index}",
        )
        assert advance.accepted and advance.local_result is not None
        advances[name] = advance
        target_parts.append(
            MeshPart(
                name,
                advance.local_result,
                coordinate_contract=part.coordinate_contract,
            )
        )
    successor_assembly = MeshAssembly(
        tuple(target_parts),
        global_parts=tuple(MeshPartManifest.from_part(part) for part in target_parts),
        local_rank=rank,
    )

    update = prepare_moving_overset_poisson_update(
        composition,
        previous,
        advances,
        plans,
        "u",
        execution_group=execution_group,
        successor_assembly=successor_assembly,
    )

    assert update.accepted
    assert update.candidate.complete
    assert {part.part_name for part in update.parts} == set(previous.global_part_names)
    for part in update.parts:
        assert part.conservative_remap.complete
        assert part.overlap_defect <= 1e-9
        assert part.boundary_flux + part.overset_flux == pytest.approx(
            -part.target_content, abs=1e-9
        )
        assert part.flux_defect <= 1e-9

    rejected = update.commit(accepted_boundary=False)
    assert not rejected.published
    assert rejected.composition is composition
    accepted = update.commit(accepted_boundary=True)
    assert accepted.published
    assert accepted.composition.composition_id != composition.composition_id
