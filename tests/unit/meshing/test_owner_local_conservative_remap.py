#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils

import phydrax as phx
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransition,
    CellGeometryTransitionEvidence,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._nested_reference_types import (
    exact_reference_cell_vertices,
    ReferenceCellMapWitnesses,
)
from phydrax.discretization._partition import CellPartition
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._view_support import OwnerLocalCellGeometry
from phydrax.discretization.finite_volume._owner_local_remap import (
    _clean_clipped_polyhedron,
    _merge_coplanar_clipped_faces,
    _root_owner_overlap,
    _validate_transition,
)
from phydrax.geometry._common_root_volume_overlap import (
    CurvedOverlapIntegrationPolicy,
)
from phydrax.lifecycle import Composition, CompositionEntry
from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshingResult,
    CellMeshingTarget,
    MeshAssembly,
    MeshingEntityKind,
    MeshingScope,
    NativeMeshingOptions,
    NativeMeshingProvider,
    NativePeriodicSource,
    PeriodicPointOrbits,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._assembly import MeshPartManifest
from phydrax.meshing._distribution import (
    MeshDistribution,
    ShardedOwnerDirectoryWitness,
)
from tests._support.volume_material_map_fixtures import (
    make_curved_material_refinement,
    make_curved_root,
    make_root_restriction,
    reference_map_witness,
)


def test_coplanar_clipped_face_patches_rebuild_one_exact_shell() -> None:
    # The extra top triangle overlaps the top facet on the same support plane,
    # as in the captured native clip output, but has a distinct plane label.
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.75, 0.25, 1.0),
        ),
        dtype=np.float64,
    )
    faces = (
        np.asarray((3, 4, 5), dtype=np.int32),
        np.asarray((5, 3, 6), dtype=np.int32),
        np.asarray((0, 2, 1), dtype=np.int32),
        np.asarray((0, 1, 4, 3), dtype=np.int32),
        np.asarray((1, 2, 5, 4), dtype=np.int32),
        np.asarray((2, 0, 3, 5), dtype=np.int32),
    )
    labels = np.asarray((4, 5, 0, 1, 2, 3), dtype=np.int32)
    normals = np.asarray(
        (
            (0.0, 0.0, -1.0),
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    offsets = np.asarray((0.0, 0.0, 1.0, 0.0, 1.0, 1.0), dtype=np.float64)
    clean_vertices, clean_faces, clean_labels = _clean_clipped_polyhedron(
        vertices, faces, labels
    )
    exact_planes = tuple(
        (
            tuple(Fraction.from_float(float(value)) for value in normal),
            Fraction.from_float(float(offset)),
        )
        for normal, offset in zip(normals, offsets, strict=True)
    )
    shell_vertices, shell_faces, shell_measure, shell_error = (
        _merge_coplanar_clipped_faces(
            clean_vertices,
            clean_faces,
            clean_labels,
            exact_planes,
            np.zeros((3,), dtype=np.float64),
            np.ones((3,), dtype=np.float64),
            0.5,
        )
    )
    assert shell_measure == 0.5
    assert shell_error >= 0.0
    shell_face_rows = tuple(tuple(int(vertex) for vertex in face) for face in shell_faces)
    mesh = CellMesh.from_polyhedra(shell_vertices, (shell_face_rows,))
    assert len(shell_faces) == 5
    assert shell_vertices.shape == (6, 3)
    assert mesh.blocks[0].cell_count == 1


def _group(owner_count: int) -> tuple[ExecutionGroup, int] | None:
    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    if (
        jax.process_count() != owner_count
        or len(devices) != owner_count
        or tuple(device.process_index for device in devices) != tuple(range(owner_count))
        or len(jax.local_devices()) != 1
    ):
        return None
    ranks = tuple(range(owner_count))
    spec = ExecutionGroupSpec(
        f"native-owner-local-conservative-remap-{owner_count}",
        ranks,
        tuple((device.process_index, device.id) for device in devices),
        mesh_axes=(("owner_local_remap", owner_count),),
    )
    group = ExecutionGroup(spec, devices)
    return group, ranks.index(jax.process_index())


def _run_in_owner_group(function_name: str, owner_count: int, /, *arguments: int) -> None:
    """Run one module test in ``owner_count`` real native process owners.

    Outside a live group the coordinator launches the same test function in
    supervised child processes; a child that cannot bind its group fails rather
    than skipping, so a launched smoke is never reported as a pass by skip.
    """
    import os
    import socket
    import sys
    from pathlib import Path

    from phydrax import config as pxconfig
    from phydrax._execution_launch import (
        launch_local_processes,
        LocalProcessLaunchError,
        LocalProcessLaunchPlan,
    )

    if os.environ.get("PHYDRAX_PROCESS_ID") is not None:
        raise RuntimeError(
            f"A launched owner process could not bind its {owner_count}-owner group."
        )
    if os.name != "posix" or os.environ.get("PYTEST_XDIST_WORKER") is not None:
        pytest.skip("Native owner groups require POSIX supervision outside xdist.")
    repository = Path(__file__).resolve().parents[3]
    base_environment = dict(os.environ)
    for name in (
        "PHYDRAX_COORDINATOR_ADDRESS",
        "PHYDRAX_NUM_PROCESSES",
        "PHYDRAX_PROCESS_ID",
        "PHYDRAX_LOCAL_DEVICE_IDS",
        "PHYDRAX_COORDINATOR_BIND_ADDRESS",
    ):
        base_environment.pop(name, None)
    library = base_environment.get("PHYDRAX_MESHCORE_LIBRARY")
    if library:
        base_environment["PHYDRAX_MESHCORE_LIBRARY"] = str(
            Path(library).resolve(strict=True)
        )
    environment = pxconfig.environment(
        pxconfig.RuntimeConfig(
            enable_x64=True,
            platforms=("cpu",),
            cpu_device_count=1,
            cpu_collectives_implementation="gloo",
        ),
        base=base_environment,
    )
    environment.update(
        PYTHONPATH=str(repository),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    script = (
        "import runpy, sys; "
        "runpy.run_path(sys.argv[1])[sys.argv[2]](*map(int, sys.argv[3:]))"
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    try:
        launch_local_processes(
            (
                sys.executable,
                "-c",
                script,
                str(Path(__file__).resolve()),
                function_name,
                *(str(value) for value in arguments),
            ),
            LocalProcessLaunchPlan(
                owner_count, f"127.0.0.1:{port}", devices_per_process=1
            ),
            environment=environment,
            cwd=str(repository),
            timeout_seconds=1800.0,
        )
    except LocalProcessLaunchError as error:
        details = "\n".join(
            f"rank {result.process_id} exited {result.returncode}; stderr:\n"
            + result.stderr.decode("utf-8", errors="replace")
            for result in error.results
        )
        raise AssertionError(
            f"{error.reason or 'owner-group launch failed'}\n{details}"
        ) from error


def _owner_view(assembly: MeshAssembly, name: str, /) -> OwnerLocalCellGeometry:
    matches = tuple(part for part in assembly.parts if part.name == name)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one accepted owner-local part named {name!r}."
        )
    carrier = matches[0].carrier
    if not isinstance(carrier, OwnerLocalCellGeometry):
        raise TypeError(
            "Native distributed assembly did not retain an owner-local geometry."
        )
    return carrier


def _curved_distributed_owner_view(
    name: str,
    result: CellMeshingResult,
    cell_owners: np.ndarray,
    group: ExecutionGroup,
    rank: int,
    source_map: Any,
    source_mesh_id: str,
    /,
) -> OwnerLocalCellGeometry:
    from phydrax.discretization._cell_geometry import (
        PerCellRestrictedCellGeometryElement,
    )
    from phydrax.discretization._periodic_topology import _entity_vertex_rows
    from phydrax.discretization._view_support import (
        _reconstruct_owner_local_geometry,
        OwnerLocalCellBlock,
    )
    from phydrax.discretization.finite_volume._owner_local_remap import (
        _ExactCoordinateSourceProjection,
    )

    part = phx.meshing.MeshPart(name, result)
    distribution = MeshDistribution(
        part,
        CellPartition(cell_owners, len(group.spec.process_indices)),
        halo_width=0,
    )
    local = distribution.local_cell_geometry(rank, include_halo=False)
    mesh = result.mesh
    geometry = result.geometry
    cell_rows = np.asarray(local["cell_rows"], dtype=np.int64)
    cell_ids = np.asarray(local["cell_ids"], dtype=np.int64)
    owned = np.asarray(local["owned"], dtype=np.bool_)
    cell_owner_ranks = np.asarray(cell_owners, dtype=np.int32)[cell_rows]
    cell_measures_all = np.asarray(result.quality.evaluation.measures, dtype=np.float64)
    if cell_measures_all.shape != np.asarray(cell_owners).shape:
        raise ValueError("Curved source measures lost their accepted cell rows.")
    vertex_rows = np.asarray(local["vertex_rows"], dtype=np.int64)
    vertex_ids = np.asarray(local["vertex_ids"], dtype=np.int64)
    vertex_coordinates = np.asarray(local["vertex_coordinates"], dtype=np.float64)
    coordinate_rows = np.asarray(local["coordinate_rows"], dtype=np.int64)
    coordinates = np.asarray(local["coordinates"], dtype=np.float64)
    exact_coordinates = tuple(
        tuple(value for value in row)
        for row in source_map.source_coordinates(coordinate_rows)
    )
    source_map_projection = _ExactCoordinateSourceProjection(
        source_map, exact_coordinates
    )

    block_counts = np.asarray([block.cell_count for block in mesh.blocks], dtype=np.int64)
    block_offsets = np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(block_counts))
    )
    block_index_by_name = {block.name: row for row, block in enumerate(mesh.blocks)}
    elements = dict(zip(geometry.block_names, geometry.elements, strict=True))
    local_block_records = []
    for block in local["blocks"]:
        name_ = str(block["name"])
        block_index = block_index_by_name[name_]
        block_rows = np.asarray(block["cell_rows"], dtype=np.int64)
        block_ids = np.asarray(block["cell_ids"], dtype=np.int64)
        global_rows = int(block_offsets[block_index]) + block_rows
        element = elements[name_]
        if isinstance(element, PerCellRestrictedCellGeometryElement):
            source_cell_ids = np.asarray(element.source_cell_ids, dtype=np.int64)[
                block_rows
            ]
            chart_arguments = tuple(
                element.chart_arguments(int(row)) for row in block_rows
            )
            local_element = PerCellRestrictedCellGeometryElement(
                element.source_element,
                element.cell_kind,
                block_ids,
                source_cell_ids,
                chart_arguments,
            )
            element_rows = np.arange(block_ids.size, dtype=np.int32)
        else:
            restriction = geometry.restriction_source
            if restriction is None:
                source_cell_ids = block_ids.copy()
            else:
                restriction_index = restriction.block_names.index(name_)
                source_cell_ids = np.asarray(
                    restriction.parent_cell_ids[restriction_index],
                    dtype=np.int64,
                )[block_rows]
            chart_arguments = None
            local_element = element
            element_rows = block_rows.astype(np.int32, copy=False)
        local_block_records.append(
            {
                **block,
                "element": local_element,
                "element_rows": element_rows,
                "source_cell_ids": source_cell_ids,
                "chart_arguments": chart_arguments,
                "cell_measures": cell_measures_all[global_rows],
                "cell_vertex_shifts": None,
            }
        )
    local_blocks = tuple(OwnerLocalCellBlock(record) for record in local_block_records)
    local_mesh, local_geometry = _reconstruct_owner_local_geometry(
        jnp.asarray(vertex_coordinates),
        jnp.asarray(vertex_ids),
        jnp.asarray(coordinates),
        local_blocks,
        source_map_projection,
        None,
        None,
    )
    if local_mesh is None or local_geometry is None:
        raise ValueError("The curved owner-local fixture has no resident cells.")

    dimension = mesh.topological_dimension
    entity_vertices = _entity_vertex_rows(mesh)
    selected_entity_rows: list[np.ndarray | None] = [None for _ in range(dimension + 1)]
    selected_entity_rows[dimension] = cell_rows
    for degree in range(dimension, 0, -1):
        incidence = mesh.topology.incidences[degree - 1].relation
        source_rows = np.asarray(incidence.source_indices, dtype=np.int64)
        target_rows = np.asarray(incidence.target_indices, dtype=np.int64)
        valid = np.asarray(incidence.valid, dtype=np.bool_)
        parent_rows = selected_entity_rows[degree]
        if parent_rows is None:
            raise ValueError("Curved local entity closure lost its parent rows.")
        selected_entity_rows[degree - 1] = np.unique(
            source_rows[valid & np.isin(target_rows, parent_rows, assume_unique=False)]
        )
    entity_set_ids: dict[int, str] = {}
    local_entity_ids: dict[int, np.ndarray] = {}
    entity_vertex_offsets: dict[int, np.ndarray] = {}
    entity_vertex_values: dict[int, np.ndarray] = {}
    for degree, selected_rows in enumerate(selected_entity_rows):
        if selected_rows is None:
            raise ValueError("Curved local entity closure is incomplete.")
        entity_set = mesh.entity_set(degree)
        global_ids = np.asarray(entity_set.entity_ids, dtype=np.int64)
        if degree == 0:
            ordered_rows = vertex_rows
        elif degree == dimension:
            ordered_rows = cell_rows
        else:
            ordered_rows = selected_rows[
                np.argsort(global_ids[selected_rows], kind="stable")
            ]
        local_entity_ids[degree] = global_ids[ordered_rows]
        entity_set_ids[degree] = entity_set.entity_set_id
        offsets = np.zeros((ordered_rows.size + 1,), dtype=np.int64)
        values: list[np.ndarray] = []
        for local_row, entity_row in enumerate(ordered_rows):
            vertex_route = np.asarray(
                entity_vertices[degree][int(entity_row)], dtype=np.int64
            )
            local_route = np.searchsorted(vertex_rows, vertex_route)
            if np.any(local_route >= vertex_rows.size) or not np.array_equal(
                vertex_rows[local_route], vertex_route
            ):
                raise ValueError("Curved local entity escaped its compact vertices.")
            values.append(local_route.astype(np.int32, copy=False))
            offsets[local_row + 1] = offsets[local_row] + local_route.size
        entity_vertex_offsets[degree] = offsets
        entity_vertex_values[degree] = (
            np.concatenate(values).astype(np.int32, copy=False)
            if values
            else np.zeros((0,), dtype=np.int32)
        )

    vertex_owner_by_id: dict[int, int] = {}
    coordinate_owner_by_row: dict[int, int] = {}
    global_rows_offset = 0
    for block_index, block in enumerate(mesh.blocks):
        block_owners = np.asarray(cell_owners, dtype=np.int32)[
            global_rows_offset : global_rows_offset + block.cell_count
        ]
        block_vertices = np.asarray(block.vertices, dtype=np.int64)
        block_valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        routes = np.asarray(geometry.geometry_dofs[block_index], dtype=np.int64)
        for row, owner in enumerate(block_owners):
            for vertex_row in block_vertices[row, block_valid[row]]:
                identifier = int(mesh.vertex_global_ids[int(vertex_row)])
                vertex_owner_by_id[identifier] = min(
                    int(owner), vertex_owner_by_id.get(identifier, int(owner))
                )
            for coordinate_row in routes[row]:
                if coordinate_row >= 0:
                    coordinate_owner_by_row[int(coordinate_row)] = min(
                        int(owner),
                        coordinate_owner_by_row.get(int(coordinate_row), int(owner)),
                    )
        global_rows_offset += block.cell_count
    vertex_owner_ranks = np.asarray(
        [vertex_owner_by_id[int(identifier)] for identifier in vertex_ids],
        dtype=np.int32,
    )
    source_owner_ranks = np.asarray(
        [coordinate_owner_by_row[int(identifier)] for identifier in coordinate_rows],
        dtype=np.int32,
    )
    source_owned = source_owner_ranks == rank
    source_manifest_id = MeshPartManifest.from_part(part).manifest_id
    directory_entry_id = f"{name}/owner-directory"
    base_directory = ShardedOwnerDirectoryWitness(
        group,
        rank,
        source_manifest_id,
        directory_entry_id,
        coordinate_rows,
        source_owner_ranks,
        source_owned,
        cell_ids,
        cell_owner_ranks,
        vertex_ids,
        vertex_owner_ranks,
        source_map_id=source_map.source_map_id,
        axis_name="owner_local_remap",
    )
    source_entry = CompositionEntry(
        source_map,
        entry_id=f"{name}/source",
        role="topology",
        owner_id="phydrax.test.owner-local-curved-root.source",
        structure_id=source_mesh_id,
        revision_id=source_map.source_revision,
        semantics_id=source_map.source_map_id,
    )
    part_authority = CompositionEntry(
        source_map_projection,
        entry_id=f"{name}/part",
        role="topology",
        owner_id="phydrax.test.owner-local-curved-root",
        structure_id=mesh.topology_id,
        revision_id=part.part_id,
        semantics_id=cell_geometry_id(geometry),
        dependencies=(
            source_entry.binding("structure"),
            source_entry.binding("revision"),
            source_entry.binding("semantics"),
            base_directory.composition_entry().binding("structure"),
        ),
    )
    directory = base_directory.bind_source_authority(
        part_authority.record_id, group, axis_name="owner_local_remap"
    )
    accepted_composition = Composition(
        (source_entry, directory.composition_entry(), part_authority),
        boundary_id=part.part_id,
    )
    coordinate_contract = result.coordinate_contract
    record = {
        "part_name": name,
        "part_id": part.part_id,
        "distribution_id": directory.directory_id,
        "source_id": source_map.source_id,
        "source_revision": source_map.source_revision,
        "source_mesh_id": source_mesh_id,
        "source_topology_id": mesh.topology_id,
        "source_geometry_id": cell_geometry_id(geometry),
        "source_map_id": source_map.source_map_id,
        "source_context_id": None,
        "geometry_layout_id": geometry.geometry_layout_id,
        "coordinate_contract_id": coordinate_contract.spatial_id,
        "coordinate_contract": coordinate_contract,
        "topological_dimension": dimension,
        "part_authority": part_authority,
        "accepted_composition": accepted_composition,
        "acceptance_receipt": None,
        "owner_directory_witness": directory,
        "rank": rank,
        "serial_source_view": False,
        "cell_rows": cell_rows,
        "cell_ids": cell_ids,
        "owned": owned,
        "cell_measures": cell_measures_all[cell_rows],
        "cell_owner_ranks": cell_owner_ranks,
        "support_cell_mask": np.zeros(cell_ids.shape, dtype=np.bool_),
        "vertex_rows": vertex_rows,
        "vertex_ids": vertex_ids,
        "vertex_coordinates": vertex_coordinates,
        "exact_vertex_coordinates": None,
        "coordinate_rows": coordinate_rows,
        "coordinates": coordinates,
        "exact_coordinates": exact_coordinates,
        "source_coordinate_stencil": None,
        "exact_vertex_coordinate_offsets": None,
        "entity_set_ids": entity_set_ids,
        "local_entity_ids": local_entity_ids,
        "entity_vertex_offsets": entity_vertex_offsets,
        "entity_vertex_values": entity_vertex_values,
        "cell_key_offsets": np.arange(cell_ids.size + 1, dtype=np.int64),
        "cell_key_source_ids": cell_ids.copy(),
        "cell_key_relative_windings": np.empty((cell_ids.size, 0), dtype=np.int64),
        "vertex_source_ids": vertex_ids.copy(),
        "source_point_ids": coordinate_rows,
        "source_owner_ranks": source_owner_ranks,
        "source_owned": source_owned,
        "vertex_owner_ranks": vertex_owner_ranks,
        "vertex_absolute_windings": np.zeros((vertex_ids.size, 0), dtype=np.int64),
        # A non-serial owner-local view names the candidate revision it was cut
        # from; the fixture's candidate is the accepted partition itself.
        "candidate_revision_id": part.part_id,
        "certified_periodic_margin": None,
        "periodic_shell_certificate": None,
        "adaptive_result": None,
        "graph_migration_evidence": None,
        "periodic_face_coverage_id": None,
        "periodic_topology_id": None,
        "periodic_cell_ids": None,
        "vertex_representative_ids": None,
        "vertex_windings": None,
        "cell_windings": None,
        "periodic_lift_id": None,
        "periodic_lift_matrix": None,
        "coordinates_are_lifted": True,
        "support_fractional_lower": None,
        "support_fractional_upper": None,
        "blocks": tuple(local_block_records),
    }
    return OwnerLocalCellGeometry.from_owner_local_result(record)


def test_owner_local_transition_accepts_exact_global_root_without_local_source() -> None:
    (
        root_mesh,
        root_geometry,
        _,
        target_mesh,
        target_geometry,
        target_map,
        _,
    ) = make_curved_material_refinement()
    target_ids = np.asarray(target_mesh.blocks[0].global_ids, dtype=np.int64)
    map_witness = ReferenceCellMapWitnesses(
        target_ids,
        target_map.source_cell_ids,
        target_map.target_cell_kind,
        target_map.source_cell_kind,
        target_map.source_reference_vertices,
        exact_reference_cell_vertices(target_map),
        target_map.source_reference_chart_arguments,
    )
    transition = CellGeometryTransition(
        root_mesh,
        root_geometry,
        target_mesh,
        target_geometry,
        evidence=CellGeometryTransitionEvidence(
            "nested_restriction", reference_map_count=target_ids.size
        ),
        reference_map=map_witness,
    )

    def accepted_view(
        name: str,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        local: bool,
        accepted_geometry_id: str | None = None,
    ) -> object:
        topology_id = mesh.topology_id
        geometry_id = (
            cell_geometry_id(geometry)
            if accepted_geometry_id is None
            else accepted_geometry_id
        )
        part_id = f"{name}-part-r1"
        return SimpleNamespace(
            part_name=name,
            part_id=part_id,
            distribution_id=f"{name}-distribution",
            source_id=name,
            source_revision=f"{name}-source-r1",
            source_mesh_id=mesh.mesh_id,
            source_topology_id=topology_id,
            source_geometry_id=geometry_id,
            source_map_id=geometry.source_map_id,
            source_context_id=None,
            source_value=None,
            source_geometry_authority=geometry.source_map,
            local_mesh_id=mesh.mesh_id if local else None,
            local_topology_id=mesh.topology_id if local else None,
            local_geometry_id=cell_geometry_id(geometry) if local else None,
            reconstruct_local_geometry=lambda: (
                (mesh, geometry) if local else (None, None)
            ),
        )

    source_view = accepted_view(
        "empty-source-owner", root_mesh, root_geometry, local=False
    )
    target_view = accepted_view("target-owner", target_mesh, target_geometry, local=True)
    assert (
        _validate_transition(transition, source_view, target_view)
        == transition.transition_id
    )

    stale_source_view = accepted_view(
        "stale-source-owner",
        root_mesh,
        root_geometry,
        local=False,
        accepted_geometry_id="stale-geometry",
    )
    with pytest.raises(ValueError, match="does not bind"):
        _validate_transition(transition, stale_source_view, target_view)


def test_curved_root_owner_integrates_exact_remote_reference_maps() -> None:
    (
        root_mesh,
        root_geometry,
        _,
        target_mesh,
        _,
        target_map,
        _,
    ) = make_curved_material_refinement(exact_chart_arguments=True)
    block = target_mesh.blocks[0]
    target_ids = np.asarray(block.global_ids, dtype=np.int64)
    vertex_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    chart_arguments = target_map.source_reference_chart_arguments
    if chart_arguments is None:
        raise ValueError("The exact-chart curved-root fixture omitted chart arguments.")
    exact_rows = exact_reference_cell_vertices(target_map)
    map_rows = tuple(
        (
            ReferenceCellMapWitnesses(
                target_ids[row : row + 1],
                np.asarray(target_map.source_cell_ids, dtype=np.int64)[row : row + 1],
                target_map.target_cell_kind,
                target_map.source_cell_kind,
                np.asarray(target_map.source_reference_vertices)[row : row + 1],
                (exact_rows[row],),
                (chart_arguments[row : row + 1]),
            ),
            vertex_ids[np.asarray(block.vertices, dtype=np.int64)[row]],
        )
        for row in range(block.cell_count)
    )
    root_view = SimpleNamespace(
        reconstruct_local_geometry=lambda: (root_mesh, root_geometry)
    )
    refinement, integration = _root_owner_overlap(
        root_view,
        map_rows,
        "source",
        1.0e-10,
        True,
        CurvedOverlapIntegrationPolicy(),
    )
    assert refinement.succeeded
    assert integration.exact_polynomial_integral_count > 0
    np.testing.assert_allclose(
        np.sum(np.asarray(refinement.volumes, dtype=np.float64)),
        np.sum(np.asarray(refinement.target_measures, dtype=np.float64)),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(refinement.target_measures, dtype=np.float64)),
        np.asarray(refinement.source_measures)[0],
        rtol=1.0e-12,
        atol=1.0e-14,
    )


def test_owner_local_curved_root_remap_routes_exact_remote_maps() -> None:
    bound = _group(2)
    if bound is None:
        _run_in_owner_group(
            "test_owner_local_curved_root_remap_routes_exact_remote_maps", 2
        )
        return
    group, rank = bound
    base_mesh, base_geometry, _ = make_curved_root()
    source_mesh, source_geometry, _ = make_root_restriction(
        base_mesh,
        base_geometry,
        cell_ids=(10, 11),
        block_name="curved-source",
    )
    target_mesh, target_geometry, _ = make_root_restriction(
        base_mesh,
        base_geometry,
        cell_ids=(20, 21),
        block_name="curved-target",
    )
    from phydrax.meshing._curving import _exact_coordinate_bank_source_map

    coordinate_contract = phx.SpatialCoordinateContract.si()
    root_result = phx.meshing.certify_cell_mesh(
        base_mesh, coordinate_contract, geometry=base_geometry
    )
    root_part = phx.meshing.MeshPart("curved-root-source", root_result)
    root_coordinate_values = np.asarray(base_geometry.coordinates, dtype=np.float64)
    root_exact_coordinates = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in root_coordinate_values
    )

    def with_root_coordinate_bank(
        mesh: CellMesh, geometry: CellGeometrySpec, /
    ) -> CellGeometrySpec:
        # Each restriction routes into the root's coordinate DOF bank, so its
        # exact map carries that bank while binding the restricted mesh epoch.
        if not np.array_equal(
            np.asarray(geometry.coordinates, dtype=np.float64), root_coordinate_values
        ):
            raise ValueError("A root restriction must keep the root coordinate bank.")
        source_map = _exact_coordinate_bank_source_map(
            mesh,
            root_part.name,
            root_part.part_id,
            (
                "curved-root-coordinate-bank",
                root_part.part_id,
                cell_geometry_id(base_geometry),
            ),
            root_exact_coordinates,
            root_coordinate_values,
        )
        return CellGeometrySpec(
            dict(zip(geometry.block_names, geometry.elements, strict=True)),
            {
                name: np.asarray(routes, dtype=np.int32)
                for name, routes in zip(
                    geometry.block_names, geometry.geometry_dofs, strict=True
                )
            },
            root_coordinate_values,
            source_map=source_map,
            source_mesh=mesh,
            restriction_source=geometry.restriction_source,
        )

    source_geometry = with_root_coordinate_bank(source_mesh, source_geometry)
    target_geometry = with_root_coordinate_bank(target_mesh, target_geometry)
    source_result = phx.meshing.certify_cell_mesh(
        source_mesh, coordinate_contract, geometry=source_geometry
    )
    target_result = phx.meshing.certify_cell_mesh(
        target_mesh, coordinate_contract, geometry=target_geometry
    )
    source_ids = tuple(
        int(identifier)
        for block in source_result.mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    )
    target_ids = tuple(
        int(identifier)
        for block in target_result.mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    )
    reference_tetrahedron = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    transition = CellGeometryTransition(
        source_result.mesh,
        source_result.geometry,
        target_result.mesh,
        target_result.geometry,
        evidence=CellGeometryTransitionEvidence(
            "nested_restriction", reference_map_count=len(target_ids)
        ),
        reference_map=reference_map_witness(
            target_ids,
            source_ids,
            tuple(reference_tetrahedron for _ in target_ids),
        ),
    )
    source_view = _curved_distributed_owner_view(
        "curved-remap-source",
        source_result,
        np.asarray((0, 1), dtype=np.int32),
        group,
        rank,
        source_geometry.source_map,
        source_mesh.mesh_id,
    )
    target_view = _curved_distributed_owner_view(
        "curved-remap-target",
        target_result,
        np.asarray((1, 0), dtype=np.int32),
        group,
        rank,
        target_geometry.source_map,
        target_mesh.mesh_id,
    )
    plan = phx.discretization.finite_volume.prepare_owner_local_conservative_remap(
        source_view, target_view, group, transition=transition
    )
    assert plan.evidence.coverage_complete
    assert int(plan.status) == 0
    assert source_view.cell_ids.size == target_view.cell_ids.size == 1
    source_id_for_target = dict(zip(target_ids, source_ids, strict=True))
    local_target_id = int(np.asarray(plan.target_cell_ids, dtype=np.int64)[0])
    assert source_id_for_target[local_target_id] not in set(
        np.asarray(source_view.cell_ids, dtype=np.int64).tolist()
    )
    assert np.all(np.asarray(plan.overlap_source_owner_ranks, dtype=np.int32) != rank)
    np.testing.assert_allclose(
        np.asarray(plan.evidence.source_coverage_defects), 0.0, atol=2.0e-12
    )
    np.testing.assert_allclose(
        np.asarray(plan.evidence.target_coverage_defects), 0.0, atol=2.0e-12
    )
    domain_measure = float(np.asarray(plan.evidence.source_measure))
    source_average = jnp.full(source_view.cell_ids.shape, 2.5, dtype=jnp.float64)
    average_result = plan.apply_averages(source_average, source_view, target_view, group)
    assert average_result.quantity == "average"
    np.testing.assert_allclose(
        np.asarray(jax.device_get(average_result.local_values)),
        2.5,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(average_result.source_integral)),
        2.5 * domain_measure,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(average_result.target_integral)),
        2.5 * domain_measure,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    content_result = plan.apply_content(
        2.5 * jnp.asarray(source_view.cell_measures, dtype=jnp.float64),
        source_view,
        target_view,
        group,
    )
    assert content_result.quantity == "extensive"
    np.testing.assert_allclose(
        np.asarray(jax.device_get(content_result.local_values)),
        2.5 * np.asarray(target_view.cell_measures, dtype=np.float64),
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    source_probe = jnp.asarray(
        0.5 + np.asarray(source_view.cell_ids, dtype=np.int64) * 0.0625,
        dtype=jnp.float64,
    )
    target_cotangent = jnp.asarray(
        0.75 + np.asarray(plan.target_cell_ids, dtype=np.int64) * 0.125,
        dtype=jnp.float64,
    )
    probe_result = plan.apply_averages(source_probe, source_view, target_view, group)
    raw = plan.transpose(target_cotangent, source_view, target_view, group)
    weighted = plan.weighted_adjoint(target_cotangent, source_view, target_view, group)
    source_owned = np.asarray(source_view.owned, dtype=np.bool_)
    source_values = np.asarray(jax.device_get(source_probe), dtype=np.float64)
    source_measures = np.asarray(source_view.cell_measures, dtype=np.float64)
    target_values = np.asarray(jax.device_get(target_cotangent), dtype=np.float64)
    target_measures = np.asarray(plan.target_volumes, dtype=np.float64)
    average_values = np.asarray(
        jax.device_get(probe_result.owned_values), dtype=np.float64
    )

    def group_sum(value: float, /) -> float:
        return float(
            np.sum(
                np.asarray(
                    multihost_utils.process_allgather(
                        np.asarray(value, dtype=np.float64)
                    ),
                    dtype=np.float64,
                )
            )
        )

    np.testing.assert_allclose(
        group_sum(float(np.vdot(average_values, target_values))),
        group_sum(
            float(
                np.vdot(
                    source_values[source_owned],
                    np.asarray(jax.device_get(raw), dtype=np.float64)[source_owned],
                )
            )
        ),
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        group_sum(float(np.vdot(average_values * target_measures, target_values))),
        group_sum(
            float(
                np.vdot(
                    source_values[source_owned] * source_measures[source_owned],
                    np.asarray(jax.device_get(weighted), dtype=np.float64)[source_owned],
                )
            )
        ),
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    replayed = phx.discretization.finite_volume.PreparedOwnerLocalConservativeRemap.from_stored_fields(
        plan, source_view, target_view, group, transition=transition
    )
    assert replayed.remap_id == plan.remap_id


def _owner_local_periodic_power_assembly(
    name: str,
    fractional_points: np.ndarray,
    point_owners: np.ndarray,
    execution_group: ExecutionGroup,
    /,
    *,
    origin: np.ndarray | None = None,
) -> MeshAssembly:
    revision = f"{name}-revision-1"
    lattice = np.asarray(
        ((1.0, 0.125, 0.0), (0.0, 1.0, 0.25), (0.125, 0.0, 1.0)),
        dtype=np.float64,
    )
    cell = PeriodicCell(lattice, origin=origin)
    rank = execution_group.spec.process_indices.index(jax.process_index())
    all_ids = np.asarray((91, 4, 53, 17, 62, 31), dtype=np.int64)
    all_weights = np.asarray((0.0, 0.015, -0.01, 0.02, 0.005, -0.015), dtype=np.float64)
    owners = np.asarray(point_owners, dtype=np.int32)
    if owners.shape != (fractional_points.shape[0],):
        raise ValueError("Periodic test owner ranks must align with all source sites.")
    owned = owners == rank
    source = NativePeriodicSource(
        PeriodicPointOrbits(
            cell,
            np.asarray(cell.cartesian(fractional_points[owned]), dtype=np.float64),
            all_ids[owned],
        ),
        name,
        revision,
        site_weights=all_weights[owned],
        point_owner_ranks=np.full((int(np.sum(owned)),), rank, dtype=np.int32),
    )
    scope = MeshingScope(
        name,
        revision,
        MeshingEntityKind.GEOMETRY,
        3,
        source.scope_entity_set_id(3),
        source.scope_entity_ids(3),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("polyhedron",)),
        ),
        scope,
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(),
    )
    return (
        NativeMeshingProvider(NativeMeshingOptions("periodic_power"))
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute(execution_group=execution_group)
    )


@pytest.mark.parametrize("owner_count", (1, 2, 3))
def test_owner_local_physical_remap_has_independent_periodic_mass_oracle(
    owner_count: int,
) -> None:
    bound = _group(owner_count)
    if bound is None:
        _run_in_owner_group(
            "test_owner_local_physical_remap_has_independent_periodic_mass_oracle",
            owner_count,
            owner_count,
        )
        return
    group, rank = bound
    source_name, target_name = "remap-source", "remap-target"
    lattice = np.asarray(
        ((1.0, 0.125, 0.0), (0.0, 1.0, 0.25), (0.125, 0.0, 1.0)),
        dtype=np.float64,
    )
    fractional = np.asarray(
        (
            (0.125, 0.1875, 0.25),
            (0.625, 0.125, 0.125),
            (0.1875, 0.625, 0.3125),
            (0.3125, 0.3125, 0.75),
            (0.75, 0.8125, 0.5625),
            (0.875, 0.4375, 0.9375),
        ),
        dtype=np.float64,
    )
    source_owners = (
        np.zeros((6,), dtype=np.int32)
        if owner_count == 1
        else np.asarray((0, 1, 0, 1, 0, 1), dtype=np.int32)
    )
    target_owners = source_owners if owner_count == 1 else 1 - source_owners
    source = _owner_local_periodic_power_assembly(
        source_name, fractional, source_owners, group
    )
    target_fractional = np.mod(
        fractional + np.asarray((0.125, 0.0625, 0.125), dtype=np.float64), 1.0
    )
    target = _owner_local_periodic_power_assembly(
        target_name, target_fractional, target_owners, group
    )
    source_view = _owner_view(source, source_name)
    target_view = _owner_view(target, target_name)
    if owner_count == 3 and rank == 2:
        assert source_view.local_mesh_id is None
        assert target_view.local_mesh_id is None
        assert source_view.reconstruct_local_geometry() == (None, None)
        assert target_view.reconstruct_local_geometry() == (None, None)
    plan = phx.discretization.finite_volume.prepare_owner_local_conservative_remap(
        source_view, target_view, group
    )
    replayed = phx.discretization.finite_volume.PreparedOwnerLocalConservativeRemap.from_stored_fields(
        plan, source_view, target_view, group
    )
    assert replayed.remap_id == plan.remap_id

    assert plan.evidence.coverage_complete
    assert int(plan.status) == 0
    domain_measure = abs(float(np.linalg.det(lattice)))
    np.testing.assert_allclose(
        np.asarray(jax.device_get(plan.evidence.source_measure)),
        domain_measure,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(plan.evidence.target_measure)),
        domain_measure,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    source_defects = np.asarray(plan.evidence.source_coverage_defects)
    source_limits = np.asarray(plan.evidence.source_coverage_tolerances)
    target_defects = np.asarray(plan.evidence.target_coverage_defects)
    target_limits = np.asarray(plan.evidence.target_coverage_tolerances)
    assert np.all(np.abs(source_defects) <= source_limits)
    assert np.all(np.abs(target_defects) <= target_limits)

    source_average = jnp.full((source_view.cell_ids.size,), 2.5, dtype=jnp.float64)
    result = plan.apply_averages(source_average, source_view, target_view, group)
    np.testing.assert_allclose(
        np.asarray(jax.device_get(result.local_values)),
        2.5,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    expected_mass = 2.5 * domain_measure
    np.testing.assert_allclose(
        np.asarray(jax.device_get(result.source_integral)),
        expected_mass,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(result.target_integral)),
        expected_mass,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(result.conservation_residual)),
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert result.quantity == "average"
    source_content = 2.5 * jnp.asarray(source_view.cell_measures, dtype=jnp.float64)
    content_result = plan.apply_content(source_content, source_view, target_view, group)
    assert content_result.quantity == "extensive"
    np.testing.assert_allclose(
        np.asarray(jax.device_get(content_result.local_values)),
        2.5 * np.asarray(target_view.cell_measures, dtype=np.float64),
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(content_result.source_integral)),
        expected_mass,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(content_result.target_integral)),
        expected_mass,
        rtol=2.0e-10,
        atol=2.0e-12,
    )

    local_overlap_measure = np.asarray(
        np.sum(np.asarray(plan.overlap_measures, dtype=np.float64)),
        dtype=np.float64,
    )
    gathered_overlap = np.asarray(
        multihost_utils.process_allgather(local_overlap_measure)
    ).reshape((-1,))
    np.testing.assert_allclose(
        np.sum(gathered_overlap),
        domain_measure,
        rtol=2.0e-10,
        atol=2.0e-12,
    )

    source_ids = np.asarray(source_view.cell_ids, dtype=np.int64)
    source_probe = jnp.asarray(0.5 + (source_ids % 13) * 0.0625, dtype=jnp.float64)
    source_owned_rows = np.flatnonzero(np.asarray(source_view.owned, dtype=np.bool_))
    target_cotangent = jnp.asarray(
        0.75 + (np.asarray(plan.target_cell_ids, dtype=np.int64) % 11) * 0.125,
        dtype=jnp.float64,
    )
    probe_result = plan.apply_averages(source_probe, source_view, target_view, group)
    raw = plan.transpose(target_cotangent, source_view, target_view, group)
    weighted = plan.weighted_adjoint(target_cotangent, source_view, target_view, group)
    raw_values = np.asarray(jax.device_get(raw), dtype=np.float64)
    weighted_values = np.asarray(jax.device_get(weighted), dtype=np.float64)
    source_probe_values = np.asarray(jax.device_get(source_probe), dtype=np.float64)
    source_volume_values = np.asarray(source_view.cell_measures, dtype=np.float64)
    target_volume_values = np.asarray(plan.target_volumes, dtype=np.float64)
    target_cotangent_values = np.asarray(
        jax.device_get(target_cotangent), dtype=np.float64
    )
    target_owned_ids = np.asarray(plan.target_cell_ids, dtype=np.int64)
    target_owned_values = np.asarray(
        jax.device_get(probe_result.owned_values), dtype=np.float64
    )
    owner_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(target_owned_ids.size, dtype=np.int32)
        )
    ).reshape((-1,))
    target_capacity = max(1, int(np.max(owner_counts)))
    id_packet = np.full((target_capacity,), -1, dtype=np.int64)
    value_packet = np.zeros((target_capacity,), dtype=np.float64)
    valid_packet = np.zeros((target_capacity,), dtype=np.bool_)
    id_packet[: target_owned_ids.size] = target_owned_ids
    value_packet[: target_owned_values.size] = target_owned_values
    valid_packet[: target_owned_ids.size] = True
    gathered_ids = np.asarray(
        multihost_utils.process_allgather(id_packet), dtype=np.int64
    )
    gathered_values = np.asarray(
        multihost_utils.process_allgather(value_packet), dtype=np.float64
    )
    gathered_valid = np.asarray(
        multihost_utils.process_allgather(valid_packet), dtype=np.bool_
    )
    values_by_id = {
        int(identifier): float(value)
        for identifiers, values, valid in zip(
            gathered_ids, gathered_values, gathered_valid, strict=True
        )
        for identifier, value, present in zip(identifiers, values, valid, strict=True)
        if present
    }
    assert np.array_equal(np.sum(gathered_valid, axis=1), owner_counts)
    assert len(values_by_id) == int(np.sum(owner_counts))
    # Laguerre cells are resident only on their site owner: power views carry no
    # ghost rows, so the owned target cells above partition the group exactly.
    assert np.all(np.asarray(target_view.owned, dtype=np.bool_))

    def group_sum(value: float, /) -> float:
        gathered = multihost_utils.process_allgather(np.asarray(value, dtype=np.float64))
        return float(np.sum(np.asarray(gathered, dtype=np.float64)))

    raw_lhs = np.vdot(
        np.asarray(jax.device_get(probe_result.owned_values), dtype=np.float64),
        target_cotangent_values,
    )
    raw_rhs = np.vdot(
        source_probe_values[source_owned_rows], raw_values[source_owned_rows]
    )
    np.testing.assert_allclose(
        group_sum(float(raw_lhs)),
        group_sum(float(raw_rhs)),
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    weighted_lhs = np.vdot(
        np.asarray(jax.device_get(probe_result.owned_values), dtype=np.float64)
        * target_volume_values,
        target_cotangent_values,
    )
    weighted_rhs = np.vdot(
        source_probe_values[source_owned_rows] * source_volume_values[source_owned_rows],
        weighted_values[source_owned_rows],
    )
    np.testing.assert_allclose(
        group_sum(float(weighted_lhs)),
        group_sum(float(weighted_rhs)),
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    differs = np.asarray(
        np.any(
            np.abs(raw_values[source_owned_rows] - weighted_values[source_owned_rows])
            > 1.0e-8
        ),
        dtype=np.int32,
    )
    assert np.any(np.asarray(multihost_utils.process_allgather(differs)).reshape((-1,)))

    if owner_count > 1:
        local_remote = np.asarray(
            np.any(np.asarray(plan.overlap_source_owner_ranks) != rank),
            dtype=np.int32,
        )
        remote = np.asarray(multihost_utils.process_allgather(local_remote)).reshape(
            (-1,)
        )
        assert np.any(remote), (
            "The physical overlap route did not exercise a remote source owner."
        )
    if owner_count == 3:
        local_transition = None
        if source_view.local_mesh_id is not None:
            local_mesh, local_geometry = source_view.reconstruct_local_geometry()
            assert local_mesh is not None and local_geometry is not None
            local_transition = CellGeometryTransition(
                local_mesh,
                local_geometry,
                local_mesh,
                local_geometry,
                evidence=CellGeometryTransitionEvidence("source_realization"),
            )
        identity_plan = (
            phx.discretization.finite_volume.prepare_owner_local_conservative_remap(
                source_view,
                source_view,
                group,
                transition=local_transition,
                advance_id="owner-local-identity-transition",
            )
        )
        identity_replay = phx.discretization.finite_volume.PreparedOwnerLocalConservativeRemap.from_stored_fields(
            identity_plan,
            source_view,
            source_view,
            group,
            transition=local_transition,
            advance_id="owner-local-identity-transition",
        )
        assert identity_replay.remap_id == identity_plan.remap_id


def test_owner_local_uncovered_domain_is_diagnostic_and_not_transportable() -> None:
    bound = _group(1)
    if bound is None:
        pytest.skip("This refusal smoke requires one native process owner.")
    group, _ = bound
    lattice = np.asarray(
        ((1.0, 0.125, 0.0), (0.0, 1.0, 0.25), (0.125, 0.0, 1.0)),
        dtype=np.float64,
    )
    fractional = np.asarray(
        (
            (0.125, 0.1875, 0.25),
            (0.625, 0.125, 0.125),
            (0.1875, 0.625, 0.3125),
            (0.3125, 0.3125, 0.75),
            (0.75, 0.8125, 0.5625),
            (0.875, 0.4375, 0.9375),
        ),
        dtype=np.float64,
    )
    owners = np.zeros((6,), dtype=np.int32)
    source_name, target_name = "remap-full-periodic", "remap-shifted-periodic"
    source = _owner_local_periodic_power_assembly(source_name, fractional, owners, group)
    target = _owner_local_periodic_power_assembly(
        target_name,
        fractional,
        owners,
        group,
        origin=0.5 * lattice[0],
    )
    source_view = _owner_view(source, source_name)
    target_view = _owner_view(target, target_name)
    plan = phx.discretization.finite_volume.prepare_owner_local_conservative_remap(
        source_view, target_view, group
    )

    assert not plan.evidence.coverage_complete
    assert int(plan.status) == int(phx.geometry.CommonRefinementStatus.COVERAGE_GAP)
    half_volume = 0.5 * abs(float(np.linalg.det(lattice)))
    np.testing.assert_allclose(
        float(np.asarray(plan.evidence.uncovered_source_measure)),
        half_volume,
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        float(np.asarray(plan.evidence.uncovered_target_measure)),
        half_volume,
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        float(np.sum(np.asarray(plan.overlap_measures, dtype=np.float64))),
        half_volume,
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    with pytest.raises(ValueError, match="incomplete source/target coverage"):
        plan.apply_averages(
            jnp.ones((source_view.cell_ids.size,), dtype=jnp.float64),
            source_view,
            target_view,
            group,
        )
