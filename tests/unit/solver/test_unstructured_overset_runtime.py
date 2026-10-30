#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization.finite_volume._unstructured_overset import (
    PeriodicSlidingInterfacePlan,
)
from phydrax.geometry._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    prepare_common_refinement,
)
from phydrax.meshing._assembly import MeshAssembly, MeshPart
from phydrax.meshing._overset import (
    OversetPartSpec,
    OversetPolicy,
    prepare_overset_connectivity,
)


if TYPE_CHECKING:
    from phydrax.meshing._contracts import SurfaceMeshingSpec
    from phydrax.meshing.providers._native_periodic import NativePeriodicSource


def _mesh(system: Any) -> Any:
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    )
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(((0, 1, 4, 3), (1, 2, 5, 4)), dtype=np.int32),
        vertex_global_ids=np.arange(10, 16),
        cell_global_ids=np.asarray((101, 103)),
        component_names=system.component_names,
    )


def _compiled(
    *,
    hole_mask: Any = None,
    epoch_id: Any = "overset-epoch",
    donor_cell: Any = 0,
    receptor_cell: Any = 1,
) -> Any:
    system = phx.equations.EulerSystem(2)
    plan = _mesh(system)
    discretization = plan.prepare()
    internal_faces = np.where(np.asarray(discretization.neighbor_cells) >= 0)[0]
    face_id = int(internal_faces[0])
    face_points = np.asarray(discretization.face_quadrature_points)[face_id : face_id + 1]
    unit_normal = np.asarray(discretization.area_vectors)[face_id] / float(
        np.asarray(discretization.face_measures)[face_id]
    )
    if int(np.asarray(discretization.owner_cells)[face_id]) != receptor_cell:
        unit_normal = -unit_normal
    face_normals = np.broadcast_to(unit_normal, face_points.shape)
    face_measures = np.asarray(discretization.face_quadrature_weights)[
        face_id : face_id + 1
    ]
    overset = phx.discretization.UnstructuredOversetPlan(
        discretization,
        discretization,
        np.asarray((receptor_cell,), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((donor_cell,), dtype=np.int32),
        np.asarray((1.0,)),
        hole_mask=hole_mask,
        epoch_id=epoch_id,
        receptor_face_ids=np.asarray((face_id,), dtype=np.int32),
        receptor_face_points=face_points,
        receptor_face_normals=face_normals,
        receptor_face_measures=face_measures,
        receptor_face_cells=np.asarray((receptor_cell,), dtype=np.int32),
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    coupling = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(overset=overset)
    problem = phx.equations.ConservationProblemIR(
        "overset-runtime", "state", system, boundaries
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method, coupling=coupling
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        # ty: ignore[invalid-argument-type]
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(maximum_retries=2),
    )
    return system, discretization, overset, dynamics, runtime


def _nonuniform_state(
    system: Any, discretization: Any, *, donor_cell: Any = 0, receptor_cell: Any = 1
) -> Any:
    assert donor_cell != receptor_cell
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0, 1.0)),
        discretization.state_shape,
    )
    primitive = primitive.at[receptor_cell, 0].set(2.0)
    return system.primitive_to_conserved(primitive)


def _quad_grid(
    x_coordinates: Any,
    y_coordinates: Any,
    /,
    *,
    vertex_id_offset: int,
    cell_id_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_coordinates, dtype=np.float64)
    y = np.asarray(y_coordinates, dtype=np.float64)
    points = np.asarray([(xv, yv) for yv in y for xv in x], dtype=np.float64)
    cells = np.asarray(
        tuple(
            (
                row * x.size + column,
                row * x.size + column + 1,
                (row + 1) * x.size + column + 1,
                (row + 1) * x.size + column,
            )
            for row in range(y.size - 1)
            for column in range(x.size - 1)
        ),
        dtype=np.int32,
    )
    return (
        points,
        cells,
        np.arange(vertex_id_offset, vertex_id_offset + points.shape[0]),
        np.arange(cell_id_offset, cell_id_offset + cells.shape[0]),
    )


def _triangulate_quad_grid(
    quadrilaterals: Any,
    cell_id_offset: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    quads = np.asarray(quadrilaterals, dtype=np.int32)
    if quads.ndim != 2 or quads.shape[1] != 4:
        raise ValueError("The moving test grid requires quadrilateral rows.")
    triangles = np.stack(
        (quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]),
        axis=1,
    ).reshape((-1, 3))
    cell_ids = np.arange(
        cell_id_offset,
        cell_id_offset + triangles.shape[0],
        dtype=np.int64,
    )
    return triangles, cell_ids


def _native_triangle_part(
    name: str,
    points: Any,
    cells: Any,
    vertex_ids: Any,
    cell_ids: Any,
    /,
) -> Any:
    mesh = CellMesh(
        points,
        (CellBlock("triangles", "triangle", cells, global_ids=cell_ids),),
        vertex_global_ids=vertex_ids,
    )
    result = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
    )
    return MeshPart(name, result)


def _native_boundary(part: Any) -> Any:
    mesh = part.carrier.mesh
    boundary = np.asarray(mesh.connectivity.boundary_edges, dtype=np.bool_)
    identifiers = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    return part.scope(1, identifiers[boundary])


def _unique_overset_receptor_faces(
    geometry: Any,
    route_receptor_cells: Any,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    receptor_cells = np.asarray(route_receptor_cells, dtype=np.int32)
    neighbor_cells = np.asarray(geometry.neighbor_cells, dtype=np.int32)
    owner_cells = np.asarray(geometry.owner_cells, dtype=np.int32)
    if (
        receptor_cells.ndim != 1
        or np.any(receptor_cells < 0)
        or np.any(receptor_cells >= geometry.cell_count)
        or neighbor_cells.shape != owner_cells.shape
    ):
        raise ValueError("Overset test receptor face routes are malformed.")
    internal_faces = np.flatnonzero(neighbor_cells >= 0)
    boundary_faces = np.flatnonzero(neighbor_cells < 0)
    face_order = np.concatenate((internal_faces, boundary_faces))
    faces_by_cell: dict[int, list[int]] = {}
    for face in face_order:
        owner = int(owner_cells[face])
        faces_by_cell.setdefault(owner, []).append(int(face))
        neighbor = int(neighbor_cells[face])
        if neighbor >= 0:
            faces_by_cell.setdefault(neighbor, []).append(int(face))
    route_face_ids = np.full(receptor_cells.shape, -1, dtype=np.int32)
    route_by_face: dict[int, int] = {}

    def assign_unique_face(route: int, visited: set[int]) -> bool:
        cell = int(receptor_cells[route])
        for face in faces_by_cell.get(cell, ()):
            if face in visited:
                continue
            visited.add(face)
            previous = route_by_face.get(face)
            if previous is None or assign_unique_face(previous, visited):
                route_by_face[face] = route
                route_face_ids[route] = face
                return True
        return False

    for route in range(receptor_cells.size):
        if not assign_unique_face(route, set()):
            raise ValueError(
                "Test receptor geometry needs a unique physical face per overlap route."
            )
    face_points = np.asarray(geometry.face_quadrature_points)[route_face_ids]
    face_area_vectors = np.asarray(geometry.area_vectors)[route_face_ids]
    face_measures = np.asarray(geometry.face_quadrature_weights)[route_face_ids]
    route_owners = owner_cells[route_face_ids]
    orientation = np.where(route_owners == receptor_cells, 1.0, -1.0)
    face_normals = np.broadcast_to(
        orientation[:, None, None]
        * (face_area_vectors / np.asarray(geometry.face_measures)[route_face_ids, None])[
            :, None, :
        ],
        face_points.shape,
    )
    return route_face_ids, face_points, face_normals, face_measures


def _owner_local_fv_overset_map(
    donor_view: Any,
    receptor_view: Any,
    donor_geometry: Any,
    receptor_geometry: Any,
    connectivity: Any,
    donor_name: str,
    receptor_name: str,
    /,
) -> Any:
    from phydrax.meshing._overset import OversetCellStatus

    donor_ids = np.asarray(donor_geometry.cell_global_ids, dtype=np.int64)
    receptor_ids = np.asarray(receptor_geometry.cell_global_ids, dtype=np.int64)

    def statuses(part_name: str, geometry: Any, /) -> np.ndarray:
        blanking = connectivity.blanking_of(part_name)
        by_id = {
            int(identifier): int(status)
            for identifier, status in zip(
                np.asarray(blanking.cell_ids, dtype=np.int64),
                np.asarray(blanking.cell_status, dtype=np.int32),
                strict=True,
            )
        }
        ids = np.asarray(geometry.cell_global_ids, dtype=np.int64)
        missing = set(int(identifier) for identifier in ids).difference(by_id)
        if missing:
            raise ValueError("Native blanking omits a local FV cell.")
        return np.asarray([by_id[int(identifier)] for identifier in ids], dtype=np.int32)

    donor_status = statuses(donor_name, donor_geometry)
    receptor_status = statuses(receptor_name, receptor_geometry)
    donor_holes = donor_status == int(OversetCellStatus.HOLE)
    receptor_holes = receptor_status == int(OversetCellStatus.HOLE)
    donor_active = ~donor_holes
    receptor_active = ~receptor_holes
    donor_fringe = donor_status == int(OversetCellStatus.FRINGE)
    receptor_fringe = receptor_status == int(OversetCellStatus.FRINGE)
    donor_cell_index = {int(identifier): row for row, identifier in enumerate(donor_ids)}
    receptor_cell_index = {
        int(identifier): row for row, identifier in enumerate(receptor_ids)
    }
    receptor_evidence = connectivity.receptors_of(receptor_name)
    native_donors = {
        int(receptor): int(donor)
        for receptor, donor, found, donor_part in zip(
            np.asarray(receptor_evidence.receptor_ids, dtype=np.int64),
            np.asarray(receptor_evidence.donor_cells, dtype=np.int64),
            np.asarray(receptor_evidence.found, dtype=np.bool_),
            np.asarray(receptor_evidence.donor_parts, dtype=np.int32),
            strict=True,
        )
        if found and int(donor_part) == connectivity.part_index(donor_name)
    }
    routed_ids = tuple(
        int(identifier)
        for identifier in receptor_ids
        if receptor_fringe[receptor_cell_index[int(identifier)]]
        and int(identifier) in native_donors
    )
    if routed_ids and not donor_ids.size:
        raise ValueError("Owner-local receptor routes lack actual donor carrier cells.")
    route_receptor_cells: list[int] = []
    receptor_cells: list[int] = []
    receptor_offsets = [0]
    donor_indices: list[int] = []
    overlap_measures: list[float] = []
    if routed_ids:
        donor_mesh, _ = donor_view.reconstruct_local_geometry()
        receptor_mesh, _ = receptor_view.reconstruct_local_geometry()
        if donor_mesh is None or receptor_mesh is None:
            raise ValueError("Nonempty owner-local FV geometry has no carrier mesh.")
        refinement = prepare_common_refinement(
            donor_mesh,
            receptor_mesh,
            policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.TARGET),
        )
        if refinement.status is not CommonRefinementStatus.SUCCESS:
            raise ValueError(
                f"Owner-local FV overlap is incomplete: {refinement.status!r}."
            )
        source_ids = np.asarray(refinement.source_cell_global_ids, dtype=np.int64)
        target_ids = np.asarray(refinement.target_cell_global_ids, dtype=np.int64)
        target_row = {int(identifier): row for row, identifier in enumerate(target_ids)}
        offsets = np.asarray(refinement.target_offsets, dtype=np.int64)
        source_routes = np.asarray(refinement.source_cells, dtype=np.int64)
        measures = np.asarray(refinement.volumes, dtype=np.float64)
        for receptor_id in routed_ids:
            target = target_row[receptor_id]
            start, stop = (int(value) for value in offsets[target : target + 2])
            overlap_source_ids = source_ids[source_routes[start:stop]]
            if (
                not overlap_source_ids.size
                or native_donors[receptor_id] not in overlap_source_ids
            ):
                raise ValueError(
                    "Native donor evidence is outside its physical local overlap."
                )
            receptor_cells.append(receptor_cell_index[receptor_id])
            donor_indices.extend(
                donor_cell_index[int(identifier)] for identifier in overlap_source_ids
            )
            overlap_measures.extend(measures[start:stop].tolist())
            route_receptor_cells.extend(
                [receptor_cell_index[receptor_id]] * (stop - start)
            )
            receptor_offsets.append(len(donor_indices))
    receptor_cells_array = np.asarray(receptor_cells, dtype=np.int32)
    offsets_array = np.asarray(receptor_offsets, dtype=np.int32)
    donors_array = np.asarray(donor_indices, dtype=np.int32)
    measures_array = np.asarray(overlap_measures, dtype=np.float64)
    face_ids = face_points = face_normals = face_measures = face_cells = None
    if route_receptor_cells:
        (
            face_ids,
            face_points,
            face_normals,
            face_measures,
        ) = _unique_overset_receptor_faces(
            receptor_geometry,
            np.asarray(route_receptor_cells, dtype=np.int32),
        )
        face_cells = np.asarray(route_receptor_cells, dtype=np.int32)
    return phx.discretization.UnstructuredOversetPlan(
        donor_geometry,
        receptor_geometry,
        receptor_cells_array,
        offsets_array,
        donors_array,
        measures_array,
        native_connectivity_id=connectivity.connectivity_id,
        donor_active_mask=donor_active,
        donor_hole_mask=donor_holes,
        donor_fringe_mask=donor_fringe,
        receptor_active_mask=receptor_active,
        receptor_hole_mask=receptor_holes,
        receptor_fringe_mask=receptor_fringe,
        receptor_face_ids=face_ids,
        receptor_face_points=face_points,
        receptor_face_normals=face_normals,
        receptor_face_measures=face_measures,
        receptor_face_cells=face_cells,
    )


def _native_quad_part(
    name: str,
    points: Any,
    cells: Any,
    vertex_ids: Any,
    cell_ids: Any,
    /,
) -> Any:
    mesh = CellMesh(
        points,
        (CellBlock("quadrilaterals", "quadrilateral", cells, global_ids=cell_ids),),
        vertex_global_ids=vertex_ids,
    )
    result = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
    )
    return MeshPart(name, result)


def _moving_overset_case(
    *,
    affine_triangle_cells: bool = False,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    system = phx.equations.EulerSystem(2)
    donor_points, donor_cells, donor_vertices, donor_ids = _quad_grid(
        (-2.0, 3.0, 4.0),
        (-2.0, 3.0, 4.0),
        vertex_id_offset=1000,
        cell_id_offset=100,
    )
    receptor_points, receptor_cells, receptor_vertices, receptor_ids = _quad_grid(
        (0.1, 1.1, 2.1),
        (0.2, 1.2, 2.2),
        vertex_id_offset=2000,
        cell_id_offset=200,
    )
    if affine_triangle_cells:
        donor_cells, donor_ids = _triangulate_quad_grid(donor_cells, 100)
        receptor_cells, receptor_ids = _triangulate_quad_grid(receptor_cells, 200)
        donor_part = _native_triangle_part(
            "donor", donor_points, donor_cells, donor_vertices, donor_ids
        )
        receptor_part = _native_triangle_part(
            "receptor", receptor_points, receptor_cells, receptor_vertices, receptor_ids
        )
        donor_geometry = phx.discretization.UnstructuredFiniteVolumePlan(
            donor_points,
            triangles=donor_cells,
            vertex_global_ids=donor_vertices,
            cell_global_ids=donor_ids,
            component_names=system.component_names,
        ).prepare()
        source_geometry = phx.discretization.UnstructuredFiniteVolumePlan(
            receptor_points,
            triangles=receptor_cells,
            vertex_global_ids=receptor_vertices,
            cell_global_ids=receptor_ids,
            component_names=system.component_names,
        ).prepare()
    else:
        donor_part = _native_quad_part(
            "donor", donor_points, donor_cells, donor_vertices, donor_ids
        )
        receptor_part = _native_quad_part(
            "receptor", receptor_points, receptor_cells, receptor_vertices, receptor_ids
        )
        donor_geometry = phx.discretization.UnstructuredFiniteVolumePlan(
            donor_points,
            quadrilaterals=donor_cells,
            vertex_global_ids=donor_vertices,
            cell_global_ids=donor_ids,
            component_names=system.component_names,
        ).prepare()
        source_geometry = phx.discretization.UnstructuredFiniteVolumePlan(
            receptor_points,
            quadrilaterals=receptor_cells,
            vertex_global_ids=receptor_vertices,
            cell_global_ids=receptor_ids,
            component_names=system.component_names,
        ).prepare()
    receptor_part_default = receptor_part
    connectivity = prepare_overset_connectivity(
        MeshAssembly((donor_part, receptor_part)),
        (
            OversetPartSpec("donor"),
            OversetPartSpec(
                "receptor",
                boundary=_native_boundary(receptor_part),
            ),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    assert connectivity.complete

    def prepared(
        geometry: Any,
        connectivity_id: str,
        /,
        *,
        receptor_part: Any | None = None,
        motion: Any = None,
        sliding: Any = None,
        moving_wall_speed: float | None = None,
    ) -> tuple[Any, Any]:
        target_part = receptor_part_default if receptor_part is None else receptor_part
        refinement = prepare_common_refinement(
            donor_part.carrier.mesh,
            target_part.carrier.mesh,
            policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.TARGET),
        )
        if refinement.status is not CommonRefinementStatus.SUCCESS:
            raise ValueError(f"Test FV overset overlap failed: {refinement.status!r}.")
        route_offsets = np.asarray(refinement.target_offsets, dtype=np.int64)
        route_receptor_cells = np.repeat(
            np.arange(geometry.cell_count, dtype=np.int32),
            np.diff(route_offsets),
        )
        if route_receptor_cells.size != np.asarray(
            refinement.source_cells
        ).size or not np.array_equal(
            np.unique(route_receptor_cells),
            np.arange(geometry.cell_count, dtype=np.int32),
        ):
            raise ValueError(
                "Test overlap CSR must cover every routed receptor cell exactly."
            )
        route_face_ids, face_points, face_normals, face_measures = (
            _unique_overset_receptor_faces(geometry, route_receptor_cells)
        )
        overset = phx.discretization.UnstructuredOversetPlan(
            donor_geometry,
            geometry,
            np.arange(geometry.cell_count, dtype=np.int32),
            route_offsets,
            refinement.source_cells,
            refinement.volumes,
            native_connectivity_id=connectivity_id,
            receptor_face_ids=route_face_ids,
            receptor_face_points=face_points,
            receptor_face_normals=face_normals,
            receptor_face_measures=face_measures,
            receptor_face_cells=route_receptor_cells,
        )

        def wall_velocity(time: Any, points: Any, normals: Any, args: Any) -> Any:
            del time, points, normals, args
            return jnp.asarray((moving_wall_speed, 0.0))

        boundary_factory = (
            phx.discretization.ExtrapolationBoundary
            if moving_wall_speed is None
            else lambda: phx.discretization.MovingSlipWallBoundary(
                wall_velocity,
                wall_velocity_provider_id="moving-native-overset-wall",
            )
        )
        boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
            geometry.boundary_patch_names,
            {name: boundary_factory() for name in geometry.boundary_patch_names},
        )

        def density_source(time: Any, state: Any, points: Any, args: Any) -> Any:
            del time, points, args
            return jnp.zeros_like(state).at[..., 0].set(0.05)

        problem = phx.equations.ConservationProblemIR(
            "moving-native-overset",
            "state",
            system,
            boundaries,
            source=density_source,
            source_id="uniform-density-injection",
        )
        dynamics = phx.equations.compile_conservation_problem(
            problem,
            geometry,
            phx.discretization.UnstructuredFiniteVolumeMethodPlan(
                phx.discretization.PiecewiseConstantReconstruction(),
                phx.discretization.RusanovFluxPlan(),
            ),
            coupling=phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
                motion=motion,
                overset=overset,
                sliding=sliding,
                topology_event_capacity=8,
                topology_event_policy="accepted_step",
            ),
        ).dynamics
        return overset, dynamics

    source_overset, source_dynamics = prepared(
        source_geometry, connectivity.connectivity_id
    )
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        source_dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(maximum_retries=2),
    )
    return (
        system,
        source_geometry,
        connectivity,
        source_overset,
        runtime,
        prepared,
    )


def _moved_receptor(connectivity: Any) -> tuple[Any, Any, np.ndarray]:
    receptor = connectivity.assembly.part("receptor")
    mesh = receptor.carrier.mesh
    coordinates = np.asarray(mesh.coordinates).copy()
    center = np.flatnonzero(
        np.all(np.isclose(coordinates, np.asarray((1.1, 1.2))), axis=1)
    )
    assert center.shape == (1,)
    coordinates[center[0]] += np.asarray((0.03, -0.02))
    monitor = phx.meshing.MeshMotionMonitor(mesh)
    advance = phx.meshing.advance_mesh_motion(
        monitor,
        receptor.carrier,
        coordinates,
        boundary_residual=0.0,
    )
    return advance, monitor, coordinates


def test_unstructured_overset_runtime_scenario_1() -> None:
    system, discretization, overset, dynamics, runtime = _compiled()
    constant = jnp.ones((discretization.cell_count, system.component_count))
    np.testing.assert_allclose(eqx.filter_jit(overset.remap_to_receptors)(constant), 1.0)
    linear = jnp.asarray(
        ((1.0,) * system.component_count, (3.0,) * system.component_count)
    )
    np.testing.assert_allclose(overset.remap_to_receptors(linear), 1.0)

    initial = runtime.initialize_state(
        _nonuniform_state(system, discretization),
        0.0,
        1e-3,
    )
    result = runtime.advance(initial)
    assert bool(np.asarray(result.accepted))
    block = result.accepted_flux_integrals.blocks[-1]
    assert block.block_kind == "overset-correction"
    np.testing.assert_array_equal(np.asarray(block.owner_cells), np.asarray((0,)))
    np.testing.assert_array_equal(np.asarray(block.neighbor_cells), np.asarray((1,)))
    scattered = np.zeros(
        (discretization.cell_count, system.component_count),
        dtype=np.asarray(block.flux_integral).dtype,
    )
    np.add.at(scattered, np.asarray(block.owner_cells), -np.asarray(block.flux_integral))
    np.add.at(
        scattered, np.asarray(block.neighbor_cells), np.asarray(block.flux_integral)
    )
    assert scattered[0, 0] > 0.0
    assert scattered[1, 0] < 0.0
    np.testing.assert_allclose(scattered.sum(axis=0), 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="ineligible|hole"):
        _compiled(hole_mask=np.asarray((True, False), dtype="bool"))
    system, discretization, overset, dynamics, runtime = _compiled(epoch_id="epoch-a")
    assert dynamics.coupling.overset_epoch_id == "epoch-a"
    assert dynamics.overset_mapping_id == overset.identity
    assert dynamics.overset_policy_id
    assert runtime.runtime_id
    _, _, stale_map, _, _ = _compiled(epoch_id="stale-epoch")
    prepared = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
        overset=stale_map
    ).prepare(discretization)
    assert prepared.overset_epoch_id == "stale-epoch"


def test_canonical_route_sign_reverses_receptor_normal_and_preserves_cfl() -> None:
    route_rates = []
    route_speeds = []
    route_measures = []
    for donor_cell, receptor_cell in ((0, 1), (1, 0)):
        system, discretization, _, dynamics, _ = _compiled(
            donor_cell=donor_cell,
            receptor_cell=receptor_cell,
        )
        metrics = phx.discretization.lower_static_unstructured_stage_metrics(
            discretization
        )
        state = _nonuniform_state(
            system,
            discretization,
            donor_cell=donor_cell,
            receptor_cell=receptor_cell,
        )

        def correction_flux(value: Any) -> Any:
            block, _, _ = dynamics._overset_correction(value, metrics, None)
            assert block is not None
            return block.flux_rate

        flux_rate = eqx.filter_jit(correction_flux)(state)
        gradient = jax.grad(lambda value: jnp.sum(correction_flux(value) ** 2))(state)
        block, speed, measures = dynamics._overset_correction(state, metrics, None)
        assert block is not None
        np.testing.assert_allclose(flux_rate, block.flux_rate)
        assert bool(np.asarray(jnp.all(jnp.isfinite(gradient))))
        assert bool(np.asarray(jnp.any(jnp.abs(gradient) > 0.0)))

        scattered = np.zeros(
            discretization.state_shape,
            dtype=np.asarray(flux_rate).dtype,
        )
        np.add.at(scattered, np.asarray(block.owner_cells), -np.asarray(flux_rate))
        np.add.at(
            scattered,
            np.asarray(block.neighbor_cells),
            np.asarray(flux_rate),
        )
        assert scattered[donor_cell, 0] > 0.0
        assert scattered[receptor_cell, 0] < 0.0
        np.testing.assert_allclose(scattered.sum(axis=0), 0.0, atol=1e-12)
        route_rates.append(np.asarray(flux_rate))
        route_speeds.append(np.asarray(speed))
        route_measures.append(np.asarray(measures))

    np.testing.assert_allclose(
        route_rates[1],
        route_rates[0] * np.asarray((1.0, -1.0, 1.0, 1.0)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(route_speeds[1], route_speeds[0], rtol=1e-12)
    np.testing.assert_allclose(route_measures[1], route_measures[0], rtol=1e-12)


def test_vof_overset_uses_one_donor_aperture_for_partial_mass_and_alpha_fluxes() -> None:
    eos = phx.equations.TwoMaterialEOSClosure(
        phx.equations.IdealGasMaterial(1.4),
        phx.equations.StiffenedGasMaterial(4.4, 2.0, 1.0),
    )
    system = phx.equations.TwoMaterialVOFSystem(2, eos=eos)
    vertices = np.asarray([(i, j) for j in range(3) for i in range(3)], dtype="float64")
    plan = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(
            ((0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7)),
            dtype=np.int32,
        ),
        component_names=system.component_names,
    )
    discretization = plan.prepare()
    internal_face = int(np.where(np.asarray(discretization.neighbor_cells) >= 0)[0][0])
    points = np.asarray(discretization.face_quadrature_points)[
        internal_face : internal_face + 1
    ]
    normal = np.asarray(discretization.area_vectors)[internal_face] / float(
        np.asarray(discretization.face_measures)[internal_face]
    )
    receptor_cell = 1
    if int(np.asarray(discretization.owner_cells)[internal_face]) != receptor_cell:
        normal = -normal
    normals = np.broadcast_to(normal, points.shape)
    measures = np.asarray(discretization.face_quadrature_weights)[
        internal_face : internal_face + 1
    ]
    overset = phx.discretization.UnstructuredOversetPlan(
        discretization,
        discretization,
        np.asarray((receptor_cell,), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((0,), dtype=np.int32),
        np.asarray((1.0,)),
        epoch_id="vof-overset-epoch",
        receptor_face_ids=np.asarray((internal_face,), dtype=np.int32),
        receptor_face_points=points,
        receptor_face_normals=normals,
        receptor_face_measures=measures,
        receptor_face_cells=np.asarray((receptor_cell,), dtype=np.int32),
    )
    gradient = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(
        discretization
    )
    vof = phx.discretization.UnstructuredVOFPlan(discretization, gradient)
    coupling = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
        vof=vof, overset=overset
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    dynamics = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR(
            "vof-overset-runtime", "state", system, boundaries
        ),
        discretization,
        phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
        coupling=coupling,
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        # ty: ignore[invalid-argument-type]
        dynamics,
        phx.discretization.FluxPositivityPlan(
            fallback_flux=phx.discretization.RusanovFluxPlan()
        ),
    )
    alpha = jnp.asarray((0.8, 0.8, 0.2, 0.2))
    primitive = jnp.stack(
        (
            jnp.full_like(alpha, 1.2),
            jnp.full_like(alpha, 0.7),
            jnp.full_like(alpha, 0.5),
            jnp.zeros_like(alpha),
            jnp.full_like(alpha, 2.5),
            alpha,
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    result = runtime.advance(runtime.initialize_state(state, 0.0, 1.0e-4))

    assert result.accepted
    block = result.accepted_flux_integrals.blocks[-1]
    assert block.block_kind == "overset-correction"
    flux = block.flux_integral[0]
    assert jnp.isfinite(flux).all()
    assert jnp.abs(flux[0]) > 0.0
    assert jnp.abs(flux[system.alpha_index]) > 0.0
    np.testing.assert_allclose(
        flux[0], 1.2 * flux[system.alpha_index], rtol=1.0e-4, atol=1.0e-10
    )
    scattered = np.zeros(discretization.state_shape)
    np.add.at(scattered, np.asarray(block.owner_cells), -np.asarray(block.flux_integral))
    np.add.at(
        scattered,
        np.asarray(block.neighbor_cells),
        np.asarray(block.flux_integral),
    )
    np.testing.assert_allclose(scattered.sum(axis=0), 0.0, atol=1.0e-12)


def test_native_moving_overset_rezone_commits_accepted_pde_state_and_advances() -> None:
    (
        system,
        source_geometry,
        connectivity,
        _,
        runtime,
        prepare_dynamics,
    ) = _moving_overset_case(affine_triangle_cells=True)
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0, 1.0)),
        source_geometry.state_shape,
    )
    initial_average = system.primitive_to_conserved(primitive)
    controller = jnp.asarray((0.75, 0.25))
    integrator = jnp.asarray((11.0, 17.0, 23.0))
    initial = runtime.initialize_state(
        initial_average,
        0.0,
        1.0e-4,
        accepted_step=5,
        controller_state=controller,
        integrator_state=integrator,
        output_cursor=9,
    )
    initial_total = initial.content_state.volume_integral()
    accepted = runtime.advance(initial)

    assert bool(np.asarray(accepted.accepted))
    accepted_state = accepted.runtime_state
    accepted_dt = float(np.asarray(accepted.accepted_step_size))
    assert float(np.asarray(accepted_state.time)) == pytest.approx(accepted_dt)
    accepted_total = accepted_state.content_state.volume_integral()
    source_total, boundary_outward, net_total = (
        accepted.accepted_flux_integrals.conservation_sums()
    )
    np.testing.assert_allclose(
        source_total[0],
        0.05 * accepted_dt * 4.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        net_total[0],
        source_total[0] - boundary_outward[0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        accepted_total[0] - initial_total[0],
        net_total[0],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(accepted_state.controller_state, controller)
    np.testing.assert_array_equal(accepted_state.integrator_state, integrator)
    assert int(np.asarray(accepted_state.output_cursor)) == 9

    motion_advance, _, _ = _moved_receptor(connectivity)
    assert motion_advance.accepted and motion_advance.result is not None
    candidate = connectivity.moved_from_advances({"receptor": motion_advance})
    assert candidate.complete
    target_mesh = motion_advance.result.mesh
    target_block = target_mesh.blocks[0]
    target_geometry = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(target_mesh.coordinates),
        triangles=np.asarray(target_block.vertices),
        vertex_global_ids=np.asarray(target_mesh.vertex_global_ids),
        cell_global_ids=np.asarray(target_block.global_ids),
        component_names=system.component_names,
    ).prepare()
    _, successor_dynamics = prepare_dynamics(
        target_geometry,
        candidate.connectivity_id,
        receptor_part=MeshPart("receptor", motion_advance.result),
    )

    committed = runtime.commit_moving_overset_update(
        accepted,
        connectivity,
        candidate,
        {"receptor": motion_advance},
        successor_dynamics,
    )

    assert committed.successor_runtime is not None, (
        "Native motion commit did not publish a successor runtime: "
        f"same_result={committed is accepted}, "
        f"accepted={bool(np.asarray(accepted.accepted))}, "
        f"statuses={np.asarray(committed.runtime_state.topology_journal.statuses)}, "
        f"count={int(np.asarray(committed.runtime_state.topology_journal.count))}"
    )
    assert committed.overset_motion_receipt is not None
    assert committed.overset_motion_receipt.published
    committed_state = committed.runtime_state
    assert float(np.asarray(committed_state.time)) == pytest.approx(
        float(np.asarray(accepted_state.time))
    )
    assert int(np.asarray(committed_state.accepted_step)) == int(
        np.asarray(accepted_state.accepted_step)
    )
    assert int(np.asarray(committed_state.output_cursor)) == 9
    np.testing.assert_array_equal(committed_state.controller_state, controller)
    np.testing.assert_array_equal(committed_state.integrator_state, integrator)
    np.testing.assert_allclose(
        committed_state.content_state.volume_integral(),
        accepted_total,
        rtol=1.0e-10,
        atol=1.0e-11,
    )
    assert committed_state.topology_journal.current_epoch_id != (
        accepted_state.topology_journal.current_epoch_id
    )
    assert int(np.asarray(committed_state.topology_journal.count)) == 1

    continued = committed.successor_runtime.advance(committed_state)
    assert bool(np.asarray(continued.accepted))
    continued_state = continued.runtime_state
    continued_dt = float(np.asarray(continued.accepted_step_size))
    assert float(np.asarray(continued_state.time)) == pytest.approx(
        float(np.asarray(committed_state.time)) + continued_dt
    )
    assert (
        int(np.asarray(continued_state.accepted_step))
        == int(np.asarray(committed_state.accepted_step)) + 1
    )
    continuation_source, continuation_boundary, continuation_net = (
        continued.accepted_flux_integrals.conservation_sums()
    )
    np.testing.assert_allclose(
        continuation_source[0],
        0.05 * continued_dt * 4.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        continuation_net[0],
        continuation_source[0] - continuation_boundary[0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        continued_state.content_state.volume_integral()[0]
        - committed_state.content_state.volume_integral()[0],
        continuation_net[0],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(continued_state.controller_state, controller)
    np.testing.assert_array_equal(continued_state.integrator_state, integrator)
    assert int(np.asarray(continued_state.output_cursor)) == 9


def test_native_moving_overset_failure_keeps_accepted_state_runnable() -> None:
    (
        system,
        source_geometry,
        connectivity,
        _,
        runtime,
        prepare_dynamics,
    ) = _moving_overset_case(affine_triangle_cells=True)
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0, 1.0)),
        source_geometry.state_shape,
    )
    accepted = runtime.advance(
        runtime.initialize_state(
            system.primitive_to_conserved(primitive),
            0.0,
            1.0e-4,
        )
    )
    assert bool(np.asarray(accepted.accepted))
    _, monitor, moved_coordinates = _moved_receptor(connectivity)
    receptor = connectivity.assembly.part("receptor")
    rejected = phx.meshing.advance_mesh_motion(
        monitor,
        receptor.carrier,
        moved_coordinates,
        boundary_residual=1.0,
    )
    assert not rejected.accepted

    rolled_back = runtime.commit_moving_overset_update(
        accepted,
        connectivity,
        None,
        {"receptor": rejected},
        None,
    )

    assert bool(np.asarray(rolled_back.accepted))
    assert rolled_back.successor_runtime is None
    np.testing.assert_array_equal(
        rolled_back.runtime_state.content_state.conservative_content,
        accepted.runtime_state.content_state.conservative_content,
    )
    assert float(np.asarray(rolled_back.runtime_state.time)) == pytest.approx(
        float(np.asarray(accepted.runtime_state.time))
    )
    assert rolled_back.runtime_state.topology_journal.current_epoch_id == (
        accepted.runtime_state.topology_journal.current_epoch_id
    )
    physical_coordinates = np.asarray(receptor.carrier.mesh.coordinates).copy()
    physical_coordinates += np.asarray((0.01, 0.0))
    physical_motion = phx.meshing.advance_mesh_motion(
        monitor,
        receptor.carrier,
        physical_coordinates,
        boundary_residual=0.0,
        fixed_vertices=np.zeros((physical_coordinates.shape[0],), dtype=np.bool_),
    )
    assert physical_motion.accepted and physical_motion.result is not None
    physical_candidate = connectivity.moved_from_advances({"receptor": physical_motion})
    physical_target_mesh = physical_motion.result.mesh
    physical_target_block = physical_target_mesh.blocks[0]
    physical_target_geometry = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(physical_target_mesh.coordinates),
        triangles=np.asarray(physical_target_block.vertices),
        vertex_global_ids=np.asarray(physical_target_mesh.vertex_global_ids),
        cell_global_ids=np.asarray(physical_target_block.global_ids),
        component_names=system.component_names,
    ).prepare()
    _, physical_successor_dynamics = prepare_dynamics(
        physical_target_geometry,
        physical_candidate.connectivity_id,
        receptor_part=MeshPart("receptor", physical_motion.result),
    )
    boundary_rejected = runtime.commit_moving_overset_update(
        accepted,
        connectivity,
        physical_candidate,
        {"receptor": physical_motion},
        physical_successor_dynamics,
    )
    assert bool(np.asarray(boundary_rejected.accepted))
    assert boundary_rejected.successor_runtime is None
    np.testing.assert_array_equal(
        boundary_rejected.runtime_state.content_state.conservative_content,
        accepted.runtime_state.content_state.conservative_content,
    )
    assert boundary_rejected.runtime_state.topology_journal.current_epoch_id == (
        accepted.runtime_state.topology_journal.current_epoch_id
    )
    resumed = runtime.advance(boundary_rejected.runtime_state)
    assert bool(np.asarray(resumed.accepted))
    assert float(np.asarray(resumed.runtime_state.time)) > float(
        np.asarray(boundary_rejected.runtime_state.time)
    )


def test_native_overset_continuous_ale_rebind_preserves_gcl_and_next_step() -> None:
    (
        system,
        source_geometry,
        connectivity,
        _,
        _,
        prepare_dynamics,
    ) = _moving_overset_case()
    velocity = 0.2

    def translating(time: Any, vertices: Any, args: Any) -> Any:
        del args
        return vertices.at[:, 0].add(velocity * time)

    source_plan = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(source_geometry.vertices),
        quadrilaterals=np.asarray(source_geometry.quadrilaterals),
        vertex_global_ids=np.asarray(source_geometry.vertex_global_ids),
        cell_global_ids=np.asarray(source_geometry.cell_global_ids),
        component_names=system.component_names,
    )
    source_motion = phx.discretization.FixedConnectivityMotionPlan(
        source_plan,
        translating,
        mapping_id="native-overset-ale-source",
    )
    sliding = PeriodicSlidingInterfacePlan(
        np.asarray((0.0, 0.5, 1.0, 1.5, 2.0)),
        np.asarray((0.0, 0.5, 1.0, 1.5, 2.0)),
        2.0,
        interface_id="native-overset-ale-interface",
    )
    _, source_dynamics = prepare_dynamics(
        source_geometry,
        connectivity.connectivity_id,
        motion=source_motion,
        sliding=sliding,
        moving_wall_speed=velocity,
    )
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        source_dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(maximum_retries=2),
    )
    assert runtime.initial_topology_artifacts.topology_artifact_id == (
        connectivity.connectivity_id
    )
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, velocity, 0.0, 1.0)),
        source_geometry.state_shape,
    )
    initial = runtime.initialize_state(
        system.primitive_to_conserved(primitive),
        0.0,
        1.0e-4,
        accepted_step=3,
        controller_state=jnp.asarray((0.4, 0.8)),
        integrator_state=jnp.asarray((17.0, 19.0)),
        output_cursor=6,
    )
    initial_total = initial.content_state.volume_integral()
    accepted = runtime.advance(initial)

    assert accepted.ale is not None
    assert accepted.runtime_state.topology_journal.artifact_table[
        -1
    ].topology_artifact_id == (connectivity.connectivity_id)
    assert bool(np.asarray(accepted.accepted))
    assert bool(np.asarray(accepted.ale.geometry.passed))
    for gcl_rate in (
        accepted.ale.geometry.g1,
        accepted.ale.geometry.g2,
        accepted.ale.geometry.g3,
    ):
        np.testing.assert_allclose(gcl_rate, 0.0, rtol=0.0, atol=1.0e-12)
    accepted_state = accepted.runtime_state
    accepted_dt = float(np.asarray(accepted.accepted_step_size))
    np.testing.assert_allclose(
        accepted_state.content_state.volume_integral()[0] - initial_total[0],
        0.05 * accepted_dt * 4.0,
        rtol=1.0e-6,
        atol=1.0e-12,
    )

    receptor = connectivity.assembly.part("receptor")
    monitor = phx.meshing.MeshMotionMonitor(receptor.carrier.mesh)
    ale_end_vertices = np.asarray(accepted.ale.geometry.end_geometry.vertices)
    motion_advance = phx.meshing.advance_mesh_motion(
        monitor,
        receptor.carrier,
        ale_end_vertices,
        boundary_residual=0.0,
        fixed_vertices=np.zeros((ale_end_vertices.shape[0],), dtype=np.bool_),
    )
    assert motion_advance.accepted and motion_advance.result is not None
    candidate = connectivity.moved_from_advances({"receptor": motion_advance})
    assert candidate.complete
    target_mesh = motion_advance.result.mesh
    target_block = target_mesh.blocks[0]
    target_geometry = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(target_mesh.coordinates),
        quadrilaterals=np.asarray(target_block.vertices),
        vertex_global_ids=np.asarray(target_mesh.vertex_global_ids),
        cell_global_ids=np.asarray(target_block.global_ids),
        component_names=system.component_names,
    ).prepare()
    event_time = float(np.asarray(accepted_state.time))

    def continuing_translation(time: Any, vertices: Any, args: Any) -> Any:
        del args
        return vertices.at[:, 0].add(velocity * (time - event_time))

    target_plan = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(target_geometry.vertices),
        quadrilaterals=np.asarray(target_geometry.quadrilaterals),
        vertex_global_ids=np.asarray(target_geometry.vertex_global_ids),
        cell_global_ids=np.asarray(target_geometry.cell_global_ids),
        component_names=system.component_names,
    )
    target_motion = phx.discretization.FixedConnectivityMotionPlan(
        target_plan,
        continuing_translation,
        mapping_id="native-overset-ale-successor",
    )
    _, successor_dynamics = prepare_dynamics(
        target_geometry,
        candidate.connectivity_id,
        receptor_part=MeshPart("receptor", motion_advance.result),
        motion=target_motion,
        sliding=sliding,
        moving_wall_speed=velocity,
    )

    rebound = runtime.commit_moving_overset_update(
        accepted,
        connectivity,
        candidate,
        {"receptor": motion_advance},
        successor_dynamics,
    )

    assert rebound.successor_runtime is not None
    np.testing.assert_array_equal(
        rebound.runtime_state.content_state.conservative_content,
        accepted_state.content_state.conservative_content,
    )
    assert float(np.asarray(rebound.runtime_state.time)) == pytest.approx(event_time)
    assert int(np.asarray(rebound.runtime_state.accepted_step)) == int(
        np.asarray(accepted_state.accepted_step)
    )
    np.testing.assert_array_equal(
        rebound.runtime_state.controller_state,
        accepted_state.controller_state,
    )
    np.testing.assert_array_equal(
        rebound.runtime_state.integrator_state,
        accepted_state.integrator_state,
    )
    assert int(np.asarray(rebound.runtime_state.output_cursor)) == 6

    continued = rebound.successor_runtime.advance(rebound.runtime_state)
    assert continued.ale is not None
    assert bool(np.asarray(continued.accepted))
    assert bool(np.asarray(continued.ale.geometry.passed))
    np.testing.assert_array_equal(
        np.asarray(continued.ale.geometry.start_geometry.vertices),
        np.asarray(target_geometry.vertices),
    )
    assert float(np.asarray(continued.runtime_state.time)) > event_time
    assert (
        int(np.asarray(continued.runtime_state.accepted_step))
        == int(np.asarray(rebound.runtime_state.accepted_step)) + 1
    )
    assert int(np.asarray(continued.runtime_state.output_cursor)) == 6


def test_native_owner_local_fv_motion_event_restarts_and_advances(
    tmp_path: Any,
) -> None:
    """Commit moving and repartition events, cold-restore, and continue a real
    three-process finite-volume solve.
    """
    from jax.experimental import multihost_utils

    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._assembly import MeshPartManifest
    from phydrax.meshing._motion import advance_owner_local_mesh_motion
    from phydrax.meshing._overset import prepare_overset_field_transfer
    from phydrax.solver._finite_volume_runtime import FiniteVolumeRuntimeState
    from phydrax.solver._finite_volume_topology_events import TopologyEventStatus
    from tests.unit.meshing.test_native_overset import (
        _owner_local_periodic_assembly,
    )

    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    ranks = tuple(device.process_index for device in devices)
    if (
        jax.process_count() != 3
        or len(devices) != 3
        or ranks != (0, 1, 2)
        or len(jax.local_devices()) != 1
    ):
        pytest.skip("This fixture requires three real process owners, one device each.")
    rank = jax.process_index()
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "native-owner-local-fv-motion-event",
            ranks,
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("owner_local_fv_event", 3),),
        ),
        devices,
    )
    donor_name, receptor_name = "fv-event-donor", "fv-event-receptor"
    donor_points = np.asarray(((0.11, 0.13), (0.58, 0.17), (0.31, 0.62), (0.84, 0.77)))
    receptor_points = np.asarray(((0.19, 0.21), (0.63, 0.11), (0.23, 0.70), (0.77, 0.83)))
    point_owners = np.asarray((1, 1, 0, 0), dtype=np.int32)
    donor_assembly = _owner_local_periodic_assembly(
        donor_name, donor_points, point_owners, group
    )
    receptor_assembly = _owner_local_periodic_assembly(
        receptor_name, receptor_points, point_owners, group
    )
    local_parts = (*donor_assembly.parts, *receptor_assembly.parts)
    global_parts = (*donor_assembly.part_manifest, *receptor_assembly.part_manifest)
    source_assembly = MeshAssembly(
        local_parts,
        global_parts=global_parts,
        local_rank=rank,
    )
    parts_by_name = {part.name: part for part in source_assembly.parts}
    source_views: dict[str, OwnerLocalCellGeometry] = {}
    for name, part in parts_by_name.items():
        carrier = part.carrier
        if not isinstance(carrier, OwnerLocalCellGeometry):
            raise TypeError("Native owner-local assembly returned a non-owner cell view.")
        source_views[name] = carrier
    donor_view = source_views[donor_name]
    receptor_view = source_views[receptor_name]
    empty_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                (
                    int(
                        np.asarray(donor_view.cell_ids).size == 0
                        and np.asarray(receptor_view.cell_ids).size == 0
                    ),
                ),
                dtype=np.int32,
            )
        )
    ).reshape((-1,))
    assert empty_flags[2] == 1
    receptor_ghost_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                (np.count_nonzero(~np.asarray(receptor_view.owned, dtype=np.bool_)),),
                dtype=np.int32,
            )
        )
    ).reshape((-1,))
    assert np.sum(receptor_ghost_counts) > 0

    boundary_ids: list[int] = []
    boundary = None
    if rank == 1:
        facet_dimension = receptor_view.topological_dimension - 1
        _, facet_ids = receptor_view.entity_binding(facet_dimension)
        vertex_ids = np.asarray(receptor_view.vertex_ids, dtype=np.int64)
        vertex_owners = np.asarray(receptor_view.vertex_owner_ranks, dtype=np.int32)
        owner_by_vertex = {
            int(identifier): int(owner)
            for identifier, owner in zip(vertex_ids, vertex_owners, strict=True)
        }
        for identifier, loop in zip(
            facet_ids,
            receptor_view.entity_vertex_loops(facet_dimension, facet_ids),
            strict=True,
        ):
            facet_vertices = vertex_ids[np.asarray(loop, dtype=np.int64)]
            if any(owner_by_vertex[int(vertex)] == rank for vertex in facet_vertices):
                boundary_ids.append(int(identifier))
        if boundary_ids:
            boundary = parts_by_name[receptor_name].scope(
                facet_dimension, np.asarray(boundary_ids, dtype=np.int64)
            )
    boundary_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((len(boundary_ids),), dtype=np.int32)
        )
    ).reshape((-1,))
    assert boundary_counts[1] > 0

    candidate = prepare_overset_connectivity(
        source_assembly,
        (
            OversetPartSpec(donor_name, priority=2, owner_rank=0),
            OversetPartSpec(
                receptor_name,
                boundary=boundary,
                priority=1,
                owner_rank=1,
            ),
        ),
        policy=OversetPolicy(fringe_layers=1, donor_priority="part-priority"),
        execution_group=group,
    )
    local_reconstructions: dict[str, Any] = {}
    coefficient_ids: dict[str, np.ndarray] = {}
    donor_layout_error = False
    donor_blocks = tuple(
        block for block in donor_view.blocks if np.asarray(block.cell_ids).size
    )
    if len(donor_blocks) > 1:
        donor_layout_error = True
    elif donor_blocks:
        block = donor_blocks[0]
        local_reconstructions[donor_name] = (
            phx.discretization.prepare_owner_local_finite_volume_field_reconstruction(
                donor_view, block.name, "state"
            )
        )
        coefficient_ids[donor_name] = np.asarray(block.cell_ids, dtype=np.int64)
    layout_errors = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(donor_layout_error),), dtype=np.int32)
        )
    ).reshape((-1,))
    assert not np.any(layout_errors)
    transfer = prepare_overset_field_transfer(candidate, local_reconstructions, "state")
    exchange = transfer.prepare_owner_local_exchange(coefficient_ids, group, source_views)
    connectivity = exchange.connectivity
    assert connectivity.owner_local and connectivity.complete
    assert connectivity.orphan_count == 0
    connectivity_digest = np.frombuffer(
        bytes.fromhex(connectivity.connectivity_id), dtype=np.uint8
    )
    connectivity_digests = np.asarray(
        multihost_utils.process_allgather(connectivity_digest)
    )
    assert np.all(connectivity_digests == connectivity_digests[0])
    found_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                (
                    np.count_nonzero(
                        np.asarray(
                            connectivity.receptors_of(receptor_name).found,
                            dtype=np.bool_,
                        )
                    ),
                ),
                dtype=np.int32,
            )
        )
    ).reshape((-1,))
    assert np.sum(found_counts) > 0

    system = phx.equations.EulerSystem(2)
    donor_geometry = (
        phx.discretization.UnstructuredFiniteVolumePlan.from_owner_local_geometry(
            donor_view, component_names=system.component_names
        ).prepare()
    )
    receptor_plan = (
        phx.discretization.UnstructuredFiniteVolumePlan.from_owner_local_geometry(
            receptor_view, component_names=system.component_names
        )
    )
    receptor_geometry = receptor_plan.prepare()
    receptor_overset = _owner_local_fv_overset_map(
        donor_view,
        receptor_view,
        donor_geometry,
        receptor_geometry,
        connectivity,
        donor_name,
        receptor_name,
    )
    mapped_route_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((receptor_overset.receptor_cell_ids.size,), dtype=np.int32)
        )
    ).reshape((-1,))
    assert np.sum(mapped_route_counts) > 0

    def density_source(time: Any, state: Any, points: Any, args: Any) -> Any:
        del time, points, args
        return jnp.zeros_like(state).at[..., 0].set(0.05)

    def dynamics_for(geometry: Any, overset: Any) -> Any:
        boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
            geometry.boundary_patch_names,
            {
                name: phx.discretization.ExtrapolationBoundary()
                for name in geometry.boundary_patch_names
            },
        )
        problem = phx.equations.ConservationProblemIR(
            "owner-local-moving-overset",
            "state",
            system,
            boundaries,
            source=density_source,
            source_id="owner-local-uniform-density-injection",
        )
        return phx.equations.compile_conservation_problem(
            problem,
            geometry,
            phx.discretization.UnstructuredFiniteVolumeMethodPlan(
                phx.discretization.PiecewiseConstantReconstruction(),
                phx.discretization.RusanovFluxPlan(),
            ),
            coupling=phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
                overset=overset,
                topology_event_capacity=8,
                topology_event_policy="accepted_step",
            ),
        ).dynamics

    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics_for(receptor_geometry, receptor_overset),
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(maximum_retries=2),
        execution_group=group,
        owner_local_geometry=receptor_view,
        owner_local_axis_name="owner_local_fv_event_runtime",
    )
    cell_ids = np.asarray(receptor_geometry.cell_global_ids, dtype=np.int64)
    expected_primitive = np.zeros((cell_ids.size, 4), dtype=np.float64)
    expected_primitive[:, 0] = 1.0 + (cell_ids % 19) * 0.01
    expected_primitive[:, 3] = 1.0
    input_primitive = expected_primitive.copy()
    owned = np.asarray(receptor_view.owned, dtype=np.bool_)
    input_primitive[~owned, 0] = 3.0
    input_primitive[~owned, 3] = 3.0
    initial = runtime.initialize_state(
        system.primitive_to_conserved(jnp.asarray(input_primitive)),
        0.0,
        1.0e-4,
    )
    initial_owned = np.asarray(initial.content_state.owned_cell_mask, dtype=np.bool_)
    np.testing.assert_array_equal(initial_owned, owned)
    ghost_rows = np.flatnonzero(~owned)
    expected_conserved = np.asarray(
        system.primitive_to_conserved(jnp.asarray(expected_primitive))
    )
    initialized_average = np.asarray(initial.content_state.cell_average())
    np.testing.assert_allclose(
        initialized_average[ghost_rows],
        expected_conserved[ghost_rows],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    refreshed_ghost_counts = np.asarray(
        multihost_utils.process_allgather(np.asarray((ghost_rows.size,), dtype=np.int32))
    ).reshape((-1,))
    assert np.sum(refreshed_ghost_counts) > 0

    def global_sum(values: Any) -> np.ndarray:
        local = np.asarray(values, dtype=np.float64)
        return np.sum(
            np.asarray(multihost_utils.process_allgather(local), dtype=np.float64),
            axis=0,
        )

    def global_integral(state: Any) -> np.ndarray:
        return global_sum(state.content_state.volume_integral())

    initial_total = global_integral(initial)
    accepted = runtime.advance(initial)
    accepted_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(bool(np.asarray(accepted.accepted))),), dtype=np.int32)
        )
    ).reshape((-1,))
    assert np.all(accepted_flags == 1)
    accepted_state = accepted.runtime_state
    accepted_total = global_integral(accepted_state)
    assert accepted_total[0] > initial_total[0]
    _, _, accepted_net = accepted.accepted_flux_integrals.conservation_sums()
    np.testing.assert_allclose(
        accepted_total - initial_total,
        global_sum(accepted_net),
        rtol=1.0e-10,
        atol=1.0e-11,
    )

    connectivity_entries = connectivity.composition_entries()
    previous_route = next(
        entry for entry in connectivity_entries if entry.role == "interface-route"
    )

    def with_geometry_roots(
        assembly: Any, entries: tuple[CompositionEntry, ...], /
    ) -> tuple[CompositionEntry, ...]:
        source_entries = list(entries)
        entries_by_id = {entry.entry_id: entry for entry in entries}
        for part in assembly.parts:
            view = part.carrier
            if not isinstance(view, OwnerLocalCellGeometry) or view.source_value is None:
                raise TypeError(
                    "Motion fixture requires each actual owner-local source root."
                )
            manifest = MeshPartManifest.from_part(part)
            topology_entry = entries_by_id[f"overset:part:{part.name}"]
            root_entry = CompositionEntry(
                view.source_value,
                entry_id=f"mesh/source-root:{part.name}",
                role="topology",
                owner_id="phydrax.test.owner-local-fv-motion",
                structure_id=view.source_mesh_id,
                revision_id=view.source_revision,
                semantics_id=view.source_geometry_id,
            )
            geometry_entry = CompositionEntry(
                manifest,
                entry_id=f"mesh/geometry:{part.name}",
                role="discretization",
                owner_id="phydrax.test.owner-local-fv-motion",
                structure_id=view.source_geometry_id,
                revision_id=manifest.manifest_id,
                semantics_id="compact-owner-local-geometry",
                dependencies=(
                    topology_entry.binding("structure"),
                    root_entry.binding("structure"),
                    root_entry.binding("revision"),
                    root_entry.binding("semantics"),
                ),
            )
            source_entries.extend((root_entry, geometry_entry))
        return tuple(source_entries)

    content = accepted_state.content_state
    source_state_revision = canonical_fingerprint(
        {
            "kind": "finite-volume-accepted-state",
            "runtime": runtime.runtime_id,
            "time": float(np.asarray(accepted_state.time)).hex(),
            "accepted_step": int(np.asarray(accepted_state.accepted_step)),
            "content": array_tree_fingerprint(content.conservative_content),
            "controller": array_tree_fingerprint(accepted_state.controller_state),
            "integrator": array_tree_fingerprint(accepted_state.integrator_state),
            "output_cursor": int(np.asarray(accepted_state.output_cursor)),
        }
    )
    source_state_entry = CompositionEntry(
        accepted_state,
        entry_id="finite-volume/state",
        role="physical-state",
        owner_id="phydrax.solver.PreparedFiniteVolumeRuntime",
        structure_id=content.geometry_layout_id,
        revision_id=source_state_revision,
        semantics_id=system.system_id,
        dependencies=(previous_route.binding("revision"),),
    )
    source_epoch = accepted_state.topology_journal.epoch_table[-1]
    accepted_composition = Composition(
        (
            *with_geometry_roots(connectivity.assembly, connectivity_entries),
            source_state_entry,
        ),
        boundary_id=canonical_fingerprint(
            {
                "kind": "finite-volume-accepted-boundary",
                "epoch": source_epoch.epoch_id,
                "accepted_step": int(np.asarray(accepted_state.accepted_step)),
            }
        ),
    )

    source_authority = receptor_view.source_value
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        SurfaceMeshingSpec,
    )
    from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
    from phydrax.meshing._sizing import SizeCompliancePolicy, UniformSizeControl

    def current_surface_specification(source: Any) -> SurfaceMeshingSpec:
        scope = MeshingScope(
            source.source_id,
            source.source_revision,
            MeshingEntityKind.GEOMETRY,
            2,
            source.scope_entity_set_id(2),
            source.scope_entity_ids(2),
        )
        return SurfaceMeshingSpec(
            CellMeshingTarget(
                2,
                2,
                CellFamilyPolicy(required=("triangle",)),
            ),
            scope,
            size_controls=(UniformSizeControl(scope, 0.35),),
            size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
        )

    coordinate_ids = np.asarray(source_authority.domain.point_ids, dtype=np.int64)
    coordinate_owners = np.asarray(source_authority.point_owner_ranks, dtype=np.int32)
    authored_ids = np.asarray((91, 4, 53, 17), dtype=np.int64)
    local_owned_sources = np.asarray(
        [
            np.any((coordinate_ids == identifier) & (coordinate_owners == rank))
            for identifier in authored_ids
        ],
        dtype=np.int32,
    )
    source_presence = np.asarray(multihost_utils.process_allgather(local_owned_sources))
    source_available = np.any(source_presence > 0, axis=0)
    assert np.all(source_available)
    moved_source_id = int(authored_ids[np.flatnonzero(source_available)[0]])
    proposal = np.asarray(source_authority.domain.points, dtype=np.float64).copy()
    proposal_rows = np.flatnonzero(coordinate_ids == moved_source_id)
    if proposal_rows.size:
        assert proposal_rows.size == 1
        proposal[proposal_rows[0]] += np.asarray((1.0e-3, -5.0e-4))
    motion = advance_owner_local_mesh_motion(
        receptor_view,
        proposal,
        group,
        boundary_residual=0.0,
        fixed_vertices=np.zeros(
            np.asarray(receptor_view.vertex_ids).shape, dtype=np.bool_
        ),
        driver_id="owner-local-fv-event-motion",
        periodic_specification=current_surface_specification(source_authority),
        axis_name="owner_local_fv_event_motion",
    )
    motion_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(motion.accepted),), dtype=np.int32)
        )
    ).reshape((-1,))
    assert np.all(motion_flags == 1)
    assert motion.local_result is not None
    target_owned_ids = np.asarray(motion.local_result.owned_cell_ids, dtype=np.int64)
    target_owned_work = np.asarray(motion.local_result.owned_cell_weights)
    assert target_owned_work.dtype == np.int64
    np.testing.assert_array_equal(
        target_owned_work, np.ones(target_owned_ids.shape, dtype=np.int64)
    )
    for block in motion.local_result.blocks:
        target_measures = np.asarray(block.cell_measures)
        assert target_measures.dtype == np.float64
        assert np.all(target_measures > 0.0)

    local_motion_extent = float(
        np.max(
            np.abs(
                np.asarray(motion.local_result.vertex_coordinates)
                - np.asarray(receptor_view.vertex_coordinates)
            ),
            initial=0.0,
        )
    )
    global_motion_extents = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((local_motion_extent,), dtype=np.float64)
        )
    ).reshape((-1,))
    assert np.max(global_motion_extents) > 0.0

    target_receptor_part = MeshPart(
        receptor_name,
        motion.local_result,
        coordinate_contract=parts_by_name[receptor_name].coordinate_contract,
    )
    target_parts = tuple(
        target_receptor_part if part.name == receptor_name else part
        for part in source_assembly.parts
    )
    target_manifests = tuple(MeshPartManifest.from_part(part) for part in target_parts)
    successor_assembly = MeshAssembly(
        target_parts,
        global_parts=target_manifests,
        local_rank=rank,
    )
    successor_boundary = None
    if rank == 1 and boundary_ids:
        successor_boundary = target_receptor_part.scope(
            receptor_view.topological_dimension - 1,
            np.asarray(boundary_ids, dtype=np.int64),
        )
    successor_specs = {
        receptor_name: OversetPartSpec(
            receptor_name,
            boundary=successor_boundary,
            priority=1,
            owner_rank=1,
        )
    }
    successor_connectivity = connectivity.moved_from_advances(
        {receptor_name: motion},
        successor_specs=successor_specs,
        owner_local_source_views=source_views,
        successor_assembly=successor_assembly,
        execution_group=group,
        owner_local_field_reconstructions=local_reconstructions,
        owner_local_coefficient_ids=coefficient_ids,
        axis_name="finite_volume_motion",
        owner_local_field_name="state",
    )
    assert successor_connectivity.complete
    successor_carrier = successor_assembly.part(receptor_name).carrier
    if not isinstance(successor_carrier, OwnerLocalCellGeometry):
        raise TypeError("Owner-local motion successor lacks its typed cell view.")
    successor_view = successor_carrier
    successor_geometry = (
        phx.discretization.UnstructuredFiniteVolumePlan.from_owner_local_geometry(
            successor_view, component_names=system.component_names
        ).prepare()
    )
    successor_overset = _owner_local_fv_overset_map(
        donor_view,
        successor_view,
        donor_geometry,
        successor_geometry,
        successor_connectivity,
        donor_name,
        receptor_name,
    )
    successor_dynamics = dynamics_for(successor_geometry, successor_overset)
    committed = runtime.commit_moving_overset_update(
        accepted,
        connectivity,
        successor_connectivity,
        {receptor_name: motion},
        successor_dynamics,
        accepted_composition=accepted_composition,
        execution_group=group,
        owner_local_source_views=source_views,
        owner_local_successor_assembly=successor_assembly,
        owner_local_successor_specs=successor_specs,
        owner_local_field_reconstructions=local_reconstructions,
        owner_local_coefficient_ids=coefficient_ids,
        owner_local_field_name="state",
        owner_local_part_name=receptor_name,
    )
    assert committed.accepted
    assert committed.successor_runtime is not None, (
        "Native owner-local event did not publish a successor runtime: "
        f"statuses={np.asarray(committed.runtime_state.topology_journal.statuses)}, "
        f"count={int(np.asarray(committed.runtime_state.topology_journal.count))}"
    )
    assert committed.overset_motion_receipt is not None
    assert committed.overset_motion_receipt.published
    committed_state = committed.runtime_state
    assert committed_state.topology_journal.epoch_table[-1].epoch_id != (
        source_epoch.epoch_id
    )
    committed_total = global_integral(committed_state)
    np.testing.assert_allclose(
        committed_total,
        accepted_total,
        rtol=1.0e-10,
        atol=1.0e-11,
    )
    target_epoch = committed_state.topology_journal.epoch_table[-1]
    assert target_epoch.topology_id == source_epoch.topology_id
    assert target_epoch.geometry_id != source_epoch.geometry_id
    target_identity = canonical_fingerprint(
        {
            "epoch": target_epoch.epoch_id,
            "topology": target_epoch.topology_id,
            "geometry": target_epoch.geometry_id,
        }
    )
    gathered_target_identity = np.asarray(
        multihost_utils.process_allgather(
            np.frombuffer(bytes.fromhex(target_identity), dtype=np.uint8)
        )
    )
    assert np.all(gathered_target_identity == gathered_target_identity[0])
    np.testing.assert_array_equal(
        committed_state.content_state.owned_cell_mask,
        committed.successor_runtime.owner_local_owned_cell_mask,
    )

    case = phx.solver.FiniteVolumeCaseSpec(
        "native-owner-local-fv-motion-event",
        committed.successor_runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 1000),
    )
    checkpoint_plan = phx.solver.FiniteVolumeCheckpointPlan(case)
    checkpoint_path = tmp_path / f"owner-local-{rank}.fvckpt"
    phx.solver.write_finite_volume_checkpoint(
        checkpoint_path, checkpoint_plan, committed_state
    )
    restored = phx.solver.read_finite_volume_checkpoint(
        checkpoint_path, checkpoint_plan
    ).runtime_state
    if not isinstance(restored, FiniteVolumeRuntimeState):
        raise TypeError(
            "Finite-volume checkpoint restored an incompatible runtime state."
        )
    assert restored.content_state.owned_cell_mask is not None
    assert restored.topology_journal.current_epoch_id == target_epoch.epoch_id
    np.testing.assert_array_equal(
        restored.content_state.owned_cell_mask,
        committed_state.content_state.owned_cell_mask,
    )
    np.testing.assert_array_equal(
        restored.content_state.conservative_content,
        committed_state.content_state.conservative_content,
    )
    continued = committed.successor_runtime.advance(restored)
    continued_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(bool(np.asarray(continued.accepted))),), dtype=np.int32)
        )
    ).reshape((-1,))
    assert np.all(continued_flags == 1)
    assert (
        int(np.asarray(continued.runtime_state.accepted_step))
        == int(np.asarray(committed_state.accepted_step)) + 1
    )
    continued_total = global_integral(continued.runtime_state)
    assert continued_total[0] > committed_total[0]
    _, _, continued_net = continued.accepted_flux_integrals.conservation_sums()
    np.testing.assert_allclose(
        continued_total - committed_total,
        global_sum(continued_net),
        rtol=1.0e-10,
        atol=1.0e-11,
    )

    continued_state = continued.runtime_state
    continued_entries = successor_connectivity.composition_entries()
    continued_route = next(
        entry for entry in continued_entries if entry.role == "interface-route"
    )
    continued_content = continued_state.content_state
    continued_state_entry = CompositionEntry(
        continued_state,
        entry_id="finite-volume/state",
        role="physical-state",
        owner_id="phydrax.solver.PreparedFiniteVolumeRuntime",
        structure_id=continued_content.geometry_layout_id,
        revision_id=canonical_fingerprint(
            {
                "kind": "finite-volume-accepted-state",
                "runtime": committed.successor_runtime.runtime_id,
                "time": float(np.asarray(continued_state.time)).hex(),
                "accepted_step": int(np.asarray(continued_state.accepted_step)),
                "content": array_tree_fingerprint(continued_content.conservative_content),
                "controller": array_tree_fingerprint(continued_state.controller_state),
                "integrator": array_tree_fingerprint(continued_state.integrator_state),
                "output_cursor": int(np.asarray(continued_state.output_cursor)),
            }
        ),
        semantics_id=system.system_id,
        dependencies=(continued_route.binding("revision"),),
    )
    continued_epoch = continued_state.topology_journal.epoch_table[-1]
    continued_composition = Composition(
        (
            *with_geometry_roots(successor_assembly, continued_entries),
            continued_state_entry,
        ),
        boundary_id=canonical_fingerprint(
            {
                "kind": "finite-volume-accepted-boundary",
                "epoch": continued_epoch.epoch_id,
                "accepted_step": int(np.asarray(continued_state.accepted_step)),
            }
        ),
    )
    current_source_views: dict[str, OwnerLocalCellGeometry] = {}
    for part in successor_assembly.parts:
        carrier = part.carrier
        if not isinstance(carrier, OwnerLocalCellGeometry):
            raise TypeError("Current successor assembly lacks an owner-local cell view.")
        current_source_views[part.name] = carrier
    current_receptor_view = current_source_views[receptor_name]
    current_source_authority = current_receptor_view.source_value
    current_coordinate_ids = np.asarray(
        current_source_authority.domain.point_ids, dtype=np.int64
    )
    rejected_proposal = np.asarray(
        current_source_authority.domain.points, dtype=np.float64
    ).copy()
    rejected_rows = np.flatnonzero(current_coordinate_ids == moved_source_id)
    if rejected_rows.size:
        assert rejected_rows.size == 1
        rejected_proposal[rejected_rows[0]] += np.asarray((1.0e-3, -5.0e-4))
    rejected_fixed = np.zeros(
        np.asarray(current_receptor_view.vertex_ids).shape, dtype=np.bool_
    )
    if rank == 1:
        moved_vertex_rows = np.flatnonzero(
            np.asarray(current_receptor_view.vertex_source_ids, dtype=np.int64)
            == moved_source_id
        )
        assert moved_vertex_rows.size > 0
        rejected_fixed[moved_vertex_rows[0]] = True
    rejected_motion = advance_owner_local_mesh_motion(
        current_receptor_view,
        rejected_proposal,
        group,
        boundary_residual=0.0,
        fixed_vertices=rejected_fixed,
        driver_id="owner-local-fv-invalid-motion",
        periodic_specification=current_surface_specification(current_source_authority),
        axis_name="owner_local_fv_invalid_motion",
    )
    rejected_motion_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(rejected_motion.accepted),), dtype=np.int32)
        )
    ).reshape((-1,))
    assert np.all(rejected_motion_flags == 0)
    rolled_back = committed.successor_runtime.commit_moving_overset_update(
        continued,
        successor_connectivity,
        None,
        {receptor_name: rejected_motion},
        None,
        accepted_composition=continued_composition,
        execution_group=group,
        owner_local_source_views=current_source_views,
        owner_local_successor_assembly=successor_assembly,
        owner_local_successor_specs=successor_specs,
        owner_local_field_reconstructions=local_reconstructions,
        owner_local_coefficient_ids=coefficient_ids,
        owner_local_field_name="state",
        owner_local_part_name=receptor_name,
    )
    assert rolled_back.accepted
    assert rolled_back.successor_runtime is None
    np.testing.assert_array_equal(
        rolled_back.runtime_state.content_state.conservative_content,
        continued_state.content_state.conservative_content,
    )
    assert float(np.asarray(rolled_back.runtime_state.time)) == pytest.approx(
        float(np.asarray(continued_state.time))
    )
    assert int(np.asarray(rolled_back.runtime_state.accepted_step)) == int(
        np.asarray(continued_state.accepted_step)
    )
    np.testing.assert_array_equal(
        rolled_back.runtime_state.content_state.owned_cell_mask,
        continued_state.content_state.owned_cell_mask,
    )
    assert (
        rolled_back.runtime_state.topology_journal.current_epoch_id
        == continued_state.topology_journal.current_epoch_id
    )
    failed_event_count = int(np.asarray(rolled_back.runtime_state.topology_journal.count))
    failure_status = int(
        np.asarray(rolled_back.runtime_state.topology_journal.statuses)[
            failed_event_count - 1
        ]
    )
    failure_statuses = np.asarray(
        multihost_utils.process_allgather(np.asarray((failure_status,), dtype=np.int32))
    ).reshape((-1,))
    assert np.all(failure_statuses == int(TopologyEventStatus.FAILED))

    after_rollback = committed.successor_runtime.advance(rolled_back.runtime_state)
    after_rollback_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(bool(np.asarray(after_rollback.accepted))),), dtype=np.int32)
        )
    ).reshape((-1,))
    assert np.all(after_rollback_flags == 1)

    from phydrax.graph import GraphPartitionPlan
    from phydrax.meshing._distribution import partition_owner_local_graph
    from phydrax.meshing.providers._native_periodic import (
        _repartition_owner_local_periodic_assembly,
    )

    source_runtime = committed.successor_runtime
    if source_runtime is None:
        raise TypeError("Accepted owner-local source runtime is missing.")
    source_cell_ids = source_runtime.owner_local_cell_ids()
    repartition_state = after_rollback.runtime_state
    current_receptor_view = current_source_views[receptor_name]
    current_native_source = current_receptor_view.source_value
    source_scope = MeshingScope(
        current_native_source.source_id,
        current_native_source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        current_native_source.scope_entity_set_id(2),
        current_native_source.scope_entity_ids(2),
    )
    current_specification = SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            2,
            CellFamilyPolicy(required=("triangle",)),
        ),
        source_scope,
        size_controls=(UniformSizeControl(source_scope, 0.35),),
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
    )
    target_graph_plan = GraphPartitionPlan(
        len(ranks),
        target_shares=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        maximum_imbalance=1.0,
        require_nonempty=False,
    )
    current_receptor_part = successor_connectivity.assembly.part(receptor_name)
    current_receptor_manifest = next(
        manifest
        for manifest in successor_connectivity.assembly.part_manifest
        if manifest.name == receptor_name
    )
    receptor_assembly = MeshAssembly(
        (current_receptor_part,),
        global_parts=(current_receptor_manifest,),
        local_rank=rank,
    )
    partition_evidence = partition_owner_local_graph(
        receptor_assembly,
        target_graph_plan,
        group,
        axis_name="owner_local_fv_repartition_labels",
    )
    repartitioned_owner_assembly = _repartition_owner_local_periodic_assembly(
        receptor_assembly,
        partition_evidence,
        group,
        current_specification=current_specification,
        graph_partition_plan=target_graph_plan,
        axis_name="owner_local_fv_repartition_geometry",
    )
    target_connectivity_candidate = successor_connectivity.reregister(
        {receptor_name: repartitioned_owner_assembly.part(receptor_name)},
        successor_specs=successor_specs,
    )
    target_field_transfer = prepare_overset_field_transfer(
        target_connectivity_candidate,
        local_reconstructions,
        "state",
    )
    target_connectivity = target_field_transfer.prepare_owner_local_exchange(
        coefficient_ids,
        group,
        current_source_views,
    ).connectivity
    assert target_connectivity.owner_local and target_connectivity.complete
    target_receptor_carrier = target_connectivity.assembly.part(receptor_name).carrier
    if not isinstance(target_receptor_carrier, OwnerLocalCellGeometry):
        raise TypeError("Graph-migrated connectivity lacks its owner-local cell view.")
    target_receptor_view = target_receptor_carrier
    target_geometry = (
        phx.discretization.UnstructuredFiniteVolumePlan.from_owner_local_geometry(
            target_receptor_view,
            component_names=system.component_names,
        ).prepare()
    )
    target_overset = _owner_local_fv_overset_map(
        current_source_views[donor_name],
        target_receptor_view,
        donor_geometry,
        target_geometry,
        target_connectivity,
        donor_name,
        receptor_name,
    )
    target_dynamics = dynamics_for(target_geometry, target_overset)

    repartition_entries = successor_connectivity.composition_entries()
    repartition_route = next(
        entry for entry in repartition_entries if entry.role == "interface-route"
    )

    def repartition_state_entry_for(state: Any) -> Any:
        return CompositionEntry(
            state,
            entry_id="finite-volume/state",
            role="physical-state",
            owner_id="phydrax.solver.PreparedFiniteVolumeRuntime",
            structure_id=state.content_state.geometry_layout_id,
            revision_id=source_runtime.accepted_state_id(state, source_cell_ids),
            semantics_id=system.system_id,
            dependencies=(repartition_route.binding("revision"),),
        )

    def accepted_repartition_composition_for(state: Any, history: Any, /) -> Any:
        epoch = state.topology_journal.epoch_table[-1]
        return Composition(
            (*repartition_entries, repartition_state_entry_for(state), history),
            boundary_id=canonical_fingerprint(
                {
                    "kind": "finite-volume-accepted-boundary",
                    "epoch": epoch.epoch_id,
                    "accepted_step": int(np.asarray(state.accepted_step)),
                }
            ),
        )

    history_cell_ids = np.sort(
        np.asarray(current_receptor_view.owned_cell_ids, dtype=np.int64)
    )
    history_values = (history_cell_ids * 3 + 7).reshape((-1, 1))
    cell_state_bank_spec = phx.solver.FiniteVolumeOwnerLocalCellStateBankSpec(
        "test/owner-local-cell-history",
        history_cell_ids,
        0,
    )
    history_entry = CompositionEntry(
        history_values,
        entry_id=cell_state_bank_spec.entry_id,
        role="history",
        owner_id="test.native-owner-local-fv",
        structure_id=current_receptor_view.geometry_layout_id,
        revision_id=_finite_volume_cell_state_bank_revision(
            cell_state_bank_spec, history_values
        ),
        semantics_id="stable-cell-id-history",
        dependencies=(repartition_route.binding("revision"),),
    )
    repartition_composition = accepted_repartition_composition_for(
        repartition_state, history_entry
    )
    source_row_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((history_cell_ids.size,), dtype=np.int32)
        )
    ).reshape((-1,))
    assert np.sum(source_row_counts) > 0
    invalid_rank = int(np.flatnonzero(source_row_counts > 0)[0])
    invalid_history_ids = (
        history_cell_ids[:-1] if rank == invalid_rank else history_cell_ids
    )
    invalid_history_spec = phx.solver.FiniteVolumeOwnerLocalCellStateBankSpec(
        history_entry.entry_id,
        np.asarray(invalid_history_ids, dtype=np.int64),
        0,
    )
    failed_source_state_id = source_runtime.accepted_state_id(
        repartition_state, source_cell_ids
    )
    failed_source_state = source_runtime.validate_restored_accepted_state(
        repartition_state,
        source_cell_ids,
        expected_runtime_id=source_runtime.runtime_id,
        expected_epoch_id=source_runtime.topology_epoch_id,
        expected_state_id=failed_source_state_id,
    )
    assert isinstance(failed_source_state, phx.solver.FiniteVolumeValidatedRuntimeState)
    failed_repartition = source_runtime.commit_owner_local_repartition(
        failed_source_state,
        successor_connectivity,
        target_connectivity,
        target_dynamics,
        accepted_composition=repartition_composition,
        execution_group=group,
        owner_local_part_name=receptor_name,
        cell_state_banks=(invalid_history_spec,),
    )
    assert isinstance(
        failed_repartition, phx.solver.FiniteVolumeOwnerLocalRepartitionResult
    )
    assert not bool(np.asarray(failed_repartition.accepted))
    assert failed_repartition.successor_runtime is None
    assert failed_repartition.repartition_receipt is None
    failed_repartition_state = failed_repartition.runtime_state
    assert (
        failed_repartition_state.topology_journal.current_epoch_id
        == repartition_state.topology_journal.current_epoch_id
    )
    np.testing.assert_array_equal(
        failed_repartition_state.content_state.conservative_content,
        repartition_state.content_state.conservative_content,
    )
    assert float(np.asarray(failed_repartition_state.time)) == float(
        np.asarray(repartition_state.time)
    )
    assert int(np.asarray(failed_repartition_state.accepted_step)) == int(
        np.asarray(repartition_state.accepted_step)
    )
    np.testing.assert_array_equal(
        failed_repartition_state.content_state.owned_cell_mask,
        repartition_state.content_state.owned_cell_mask,
    )
    failed_repartition_count = int(
        np.asarray(failed_repartition_state.topology_journal.count)
    )
    failed_repartition_status = int(
        np.asarray(failed_repartition_state.topology_journal.statuses)[
            failed_repartition_count - 1
        ]
    )
    failed_repartition_statuses = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((failed_repartition_status,), dtype=np.int32)
        )
    ).reshape((-1,))
    assert np.all(
        failed_repartition_statuses == int(TopologyEventStatus.FAILED_MISSING_ARTIFACT)
    )

    repartition_state = failed_repartition_state
    repartition_epoch = repartition_state.topology_journal.epoch_table[-1]
    repartition_composition = accepted_repartition_composition_for(
        repartition_state, history_entry
    )
    source_integral_before_repartition = global_integral(repartition_state)
    retry_source_state_id = source_runtime.accepted_state_id(
        failed_repartition.runtime_state, source_cell_ids
    )
    retry_source_state = source_runtime.validate_restored_accepted_state(
        failed_repartition.runtime_state,
        source_cell_ids,
        expected_runtime_id=source_runtime.runtime_id,
        expected_epoch_id=source_runtime.topology_epoch_id,
        expected_state_id=retry_source_state_id,
    )
    assert isinstance(retry_source_state, phx.solver.FiniteVolumeValidatedRuntimeState)
    repartitioned = source_runtime.commit_owner_local_repartition(
        retry_source_state,
        successor_connectivity,
        target_connectivity,
        target_dynamics,
        accepted_composition=repartition_composition,
        execution_group=group,
        owner_local_part_name=receptor_name,
        cell_state_banks=(cell_state_bank_spec,),
    )
    assert isinstance(repartitioned, phx.solver.FiniteVolumeOwnerLocalRepartitionResult)
    repartition_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(bool(np.asarray(repartitioned.accepted))),), dtype=np.int32)
        )
    ).reshape((-1,))
    assert np.all(repartition_flags == 1)
    assert repartitioned.successor_runtime is not None
    assert repartitioned.repartition_receipt is not None
    assert repartitioned.repartition_receipt.published
    migrated_history = np.asarray(
        repartitioned.repartition_receipt.composition.entry(history_entry.entry_id).value
    )
    target_history_ids = np.asarray(target_receptor_view.owned_cell_ids, dtype=np.int64)
    np.testing.assert_array_equal(
        migrated_history,
        (target_history_ids * 3 + 7).reshape((-1, 1)),
    )
    repartitioned_state = repartitioned.runtime_state
    repartitioned_epoch = repartitioned_state.topology_journal.epoch_table[-1]
    assert repartitioned_epoch.topology_id == repartition_epoch.topology_id
    assert repartitioned_epoch.geometry_id == repartition_epoch.geometry_id
    assert repartitioned_epoch.partition_id != repartition_epoch.partition_id
    assert float(np.asarray(repartitioned_state.time)) == float(
        np.asarray(repartition_state.time)
    )
    assert int(np.asarray(repartitioned_state.accepted_step)) == int(
        np.asarray(repartition_state.accepted_step)
    )
    np.testing.assert_array_equal(
        repartitioned_state.controller_state, repartition_state.controller_state
    )
    np.testing.assert_array_equal(
        repartitioned_state.integrator_state, repartition_state.integrator_state
    )
    assert int(np.asarray(repartitioned_state.output_cursor)) == int(
        np.asarray(repartition_state.output_cursor)
    )
    repartition_event_count = int(np.asarray(repartitioned_state.topology_journal.count))
    repartition_status = int(
        np.asarray(repartitioned_state.topology_journal.statuses)[
            repartition_event_count - 1
        ]
    )
    repartition_statuses = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((repartition_status,), dtype=np.int32)
        )
    ).reshape((-1,))
    assert np.all(repartition_statuses == int(TopologyEventStatus.SUCCESS))
    np.testing.assert_allclose(
        global_integral(repartitioned_state),
        source_integral_before_repartition,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    repartition_identity = canonical_fingerprint(
        {
            "epoch": repartitioned_epoch.epoch_id,
            "topology": repartitioned_epoch.topology_id,
            "geometry": repartitioned_epoch.geometry_id,
            "partition": repartitioned_epoch.partition_id,
        }
    )
    repartition_identities = np.asarray(
        multihost_utils.process_allgather(
            np.frombuffer(bytes.fromhex(repartition_identity), dtype=np.uint8)
        )
    )
    assert np.all(repartition_identities == repartition_identities[0])
    target_migration = target_receptor_view.graph_migration_evidence
    assert target_migration is not None
    assert target_migration.result.migration_count > 0
    target_owned_count = int(
        np.count_nonzero(
            np.asarray(
                repartitioned.successor_runtime.owner_local_owned_cell_mask,
                dtype=np.bool_,
            )
        )
    )
    target_owned_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((target_owned_count,), dtype=np.int32)
        )
    ).reshape((-1,))
    source_owned_count = int(
        np.count_nonzero(
            np.asarray(repartition_state.content_state.owned_cell_mask, dtype=np.bool_)
        )
    )
    source_owned_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((source_owned_count,), dtype=np.int32)
        )
    ).reshape((-1,))
    assert int(np.sum(target_owned_counts)) == int(np.sum(source_owned_counts))
    assert target_owned_counts[2] > 0
    assert np.all(target_owned_counts[:2] == 0)

    repartition_case = phx.solver.FiniteVolumeCaseSpec(
        "native-owner-local-fv-repartition",
        repartitioned.successor_runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 1000),
    )
    repartition_checkpoint_plan = phx.solver.FiniteVolumeCheckpointPlan(repartition_case)
    repartition_checkpoint_path = tmp_path / f"owner-local-repartition-{rank}.fvckpt"
    phx.solver.write_finite_volume_checkpoint(
        repartition_checkpoint_path,
        repartition_checkpoint_plan,
        repartitioned_state,
    )
    repartition_restored = phx.solver.read_finite_volume_checkpoint(
        repartition_checkpoint_path,
        repartition_checkpoint_plan,
    ).runtime_state
    if not isinstance(repartition_restored, FiniteVolumeRuntimeState):
        raise TypeError("Repartition checkpoint restored an incompatible FV state.")
    assert repartition_restored.topology_journal.current_epoch_id == (
        repartitioned_epoch.epoch_id
    )
    assert repartition_restored.content_state.owned_cell_mask is not None
    np.testing.assert_array_equal(
        repartition_restored.content_state.owned_cell_mask,
        repartitioned_state.content_state.owned_cell_mask,
    )
    np.testing.assert_array_equal(
        repartition_restored.content_state.conservative_content,
        repartitioned_state.content_state.conservative_content,
    )
    repartition_step = repartitioned.successor_runtime.advance(repartition_restored)
    repartition_step_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                (int(bool(np.asarray(repartition_step.accepted))),), dtype=np.int32
            )
        )
    ).reshape((-1,))
    assert np.all(repartition_step_flags == 1)
    assert (
        int(np.asarray(repartition_step.runtime_state.accepted_step))
        == int(np.asarray(repartitioned_state.accepted_step)) + 1
    )
    assert float(np.asarray(repartition_step.runtime_state.time)) > float(
        np.asarray(repartitioned_state.time)
    )


def _finite_volume_cell_state_bank_revision(bank_spec: Any, values: Any, /) -> str:
    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint

    return canonical_fingerprint(
        {
            "kind": "finite-volume-owner-local-cell-state-bank",
            "entry_id": bank_spec.entry_id,
            "cell_ids": tuple(bank_spec.cell_ids),
            "cell_axis": bank_spec.cell_axis,
            "values": array_tree_fingerprint(values),
        }
    )


def _finite_volume_archive_surface_spec(
    source: "NativePeriodicSource", /
) -> "SurfaceMeshingSpec":
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        SurfaceMeshingSpec,
    )
    from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
    from phydrax.meshing._sizing import SizeCompliancePolicy, UniformSizeControl

    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    return SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            2,
            CellFamilyPolicy(required=("triangle",)),
        ),
        scope,
        size_controls=(UniformSizeControl(scope, 0.35),),
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
    )


def _finite_volume_archive_boundary_scope(
    view: Any,
    local_rank: int,
    /,
) -> Any | None:
    from phydrax.discretization._view_support import OwnerLocalCellGeometry

    if not isinstance(view, OwnerLocalCellGeometry):
        raise TypeError("FV boundary scope requires its current compact local view.")
    if local_rank != 1:
        return None
    facet_dimension = view.topological_dimension - 1
    _, facet_ids = view.entity_binding(facet_dimension)
    vertex_ids = np.asarray(view.vertex_ids, dtype=np.int64)
    vertex_owners = np.asarray(view.vertex_owner_ranks, dtype=np.int32)
    owner_by_vertex = {
        int(identifier): int(owner)
        for identifier, owner in zip(vertex_ids, vertex_owners, strict=True)
    }
    boundary_ids = []
    for identifier, loop in zip(
        facet_ids,
        view.entity_vertex_loops(facet_dimension, facet_ids),
        strict=True,
    ):
        facet_vertices = vertex_ids[np.asarray(loop, dtype=np.int64)]
        if any(owner_by_vertex[int(vertex)] == local_rank for vertex in facet_vertices):
            boundary_ids.append(int(identifier))
    if not boundary_ids:
        return None
    return view.scope(facet_dimension, np.asarray(boundary_ids, dtype=np.int64))


def _finite_volume_archive_overset_part_spec(value: Any, /) -> Any:
    from phydrax._model._structure import FiniteVolumeOversetPartSpecArchiveRecipe
    from phydrax.meshing._overset import OversetPartSpec

    if type(value) is FiniteVolumeOversetPartSpecArchiveRecipe:
        rebuilt = OversetPartSpec(
            value.part_name,
            wall=value.wall,
            boundary=value.boundary,
            excluded=value.excluded,
            solid_index=value.solid_index,
            priority=value.priority,
            owner_rank=value.owner_rank,
            image_rotation=value.image_rotation,
            image_translation=value.image_translation,
        )
        if rebuilt.spec_id != value.expected_spec_id:
            raise ValueError("Stored FV overset part spec identity changed.")
        return rebuilt
    if type(value) is OversetPartSpec:
        if value.solid_query is not None:
            raise TypeError("Cold FV context cannot persist an unregistered solid query.")
        return OversetPartSpec(
            value.part_name,
            wall=value.wall,
            boundary=value.boundary,
            excluded=value.excluded,
            solid_index=value.solid_index,
            priority=value.priority,
            owner_rank=value.owner_rank,
            image_rotation=value.image_rotation,
            image_translation=value.image_translation,
        )
    raise TypeError("FV overset context requires its canonical part-spec record.")


def _finite_volume_archive_overset_policy(value: Any, /) -> Any:
    from phydrax._model._structure import FiniteVolumeOversetPolicyArchiveRecipe
    from phydrax.meshing._overset import OversetPolicy

    if type(value) is FiniteVolumeOversetPolicyArchiveRecipe:
        rebuilt = OversetPolicy(
            fringe_layers=value.fringe_layers,
            donor_priority=value.donor_priority,
        )
        if rebuilt.policy_id != value.expected_policy_id:
            raise ValueError("Stored FV overset policy identity changed.")
        return rebuilt
    if type(value) is OversetPolicy:
        return OversetPolicy(
            fringe_layers=value.fringe_layers,
            donor_priority=value.donor_priority,
        )
    raise TypeError("FV overset context requires its canonical policy record.")


def _finite_volume_archive_execution_group(context: Any, /) -> Any:
    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup

    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    process_indices = tuple(device.process_index for device in devices)
    if process_indices != context.process_indices or len(devices) != len(
        context.process_indices
    ):
        raise ValueError(
            "Fresh FV worker placement differs from the archived group ranks."
        )
    specification = ExecutionGroupSpec(
        context.group_id,
        context.process_indices,
        context.device_keys,
        mesh_axes=context.mesh_axes,
        parent_group_id=context.parent_group_id,
        process_host_ids=context.process_host_ids,
    )
    return ExecutionGroup(specification, devices)


def _finite_volume_archive_rebuild_context_views(
    context: Any,
    execution_group: Any,
    /,
) -> Any:
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.graph._partition import GraphPartitionPlan
    from phydrax.meshing._contracts import SurfaceMeshingSpec

    if len(context.source_views) != len(context.part_names) or len(
        context.source_specifications
    ) != len(context.part_names):
        raise ValueError("FV archive source views/specifications are not aligned.")
    source_specifications = tuple(
        SurfaceMeshingSpec.from_stored_fields(value)
        for value in context.source_specifications
    )
    source_views = tuple(
        OwnerLocalCellGeometry.from_stored_fields(
            value,
            execution_group,
            current_specification=specification,
            axis_name=f"{context.owner_local_axis_name}_archive_source_{name}",
        )
        for name, value, specification in zip(
            context.part_names,
            context.source_views,
            source_specifications,
            strict=True,
        )
    )
    target_specification = SurfaceMeshingSpec.from_stored_fields(
        context.target_specification
    )
    target_plan = GraphPartitionPlan.from_stored_fields(
        context.target_graph_partition_plan
    )
    target_view = OwnerLocalCellGeometry.from_stored_fields(
        context.target_view,
        execution_group,
        current_specification=target_specification,
        graph_partition_plan=target_plan,
        axis_name=f"{context.owner_local_axis_name}_archive_target",
    )
    return context._replace(
        source_views=source_views,
        source_specifications=source_specifications,
        source_overset_specs=tuple(
            _finite_volume_archive_overset_part_spec(value)
            for value in context.source_overset_specs
        ),
        source_overset_policy=_finite_volume_archive_overset_policy(
            context.source_overset_policy
        ),
        target_overset_specs=tuple(
            _finite_volume_archive_overset_part_spec(value)
            for value in context.target_overset_specs
        ),
        target_graph_partition_plan=target_plan,
        target_specification=target_specification,
        target_view=target_view,
    )


def _finite_volume_archive_build_context(
    context: Any,
    execution_group: Any,
    /,
    *,
    validate_expected: bool,
) -> dict[str, Any]:
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax.discretization._topology_epoch import TopologyEpoch
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.discretization.finite_volume._coupling import (
        UnstructuredFiniteVolumeCouplingPlan,
    )
    from phydrax.discretization.finite_volume._unstructured import (
        UnstructuredFiniteVolumePlan,
    )
    from phydrax.discretization.finite_volume._unstructured_dynamics import (
        UnstructuredFiniteVolumeBoundarySet,
        UnstructuredFiniteVolumeMethodPlan,
    )
    from phydrax.equations._conservation import (
        compile_conservation_problem,
        ConservationProblemIR,
    )
    from phydrax.equations._hyperbolic_systems import EulerSystem
    from phydrax.equations._materials import IdealGasMaterial
    from phydrax.graph._partition import GraphPartitionPlan
    from phydrax.meshing._assembly import MeshAssembly, MeshPart, MeshPartManifest
    from phydrax.meshing._distribution import partition_owner_local_graph
    from phydrax.meshing._overset import (
        prepare_overset_connectivity,
        prepare_overset_field_transfer,
    )
    from phydrax.meshing.providers._native_periodic import (
        _repartition_owner_local_periodic_assembly,
    )
    from phydrax.solver._finite_volume_runtime import (
        FiniteVolumeStepPolicy,
        PreparedFiniteVolumeRuntime,
    )
    from phydrax.solver._finite_volume_topology_events import (
        FiniteVolumeTopologyArtifacts,
    )
    from tests.unit.interchange._native_cad_archive_worker import (
        _rebuild_finite_volume_precision_policy,
    )

    if not isinstance(validate_expected, bool):
        raise TypeError("validate_expected must be bool.")
    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("FV archive replay requires its explicit live ExecutionGroup.")
    if (
        len(context.part_names) != len(context.source_views)
        or len(context.part_names) != len(context.source_specifications)
        or len(context.process_indices) != len(execution_group.spec.process_indices)
    ):
        raise ValueError("FV archive source or group declarations are incomplete.")
    if (
        context.system_kind != "euler"
        or context.source_id is not None
        or context.reconstruction_kind != "piecewise-constant"
        or context.flux_kind != "rusanov"
        or context.boundary_condition_kind != "extrapolation"
        or context.positivity_fallback_kind != "einfeldt-hll"
    ):
        raise ValueError(
            "FV archive runtime recipe selects an unregistered callable path."
        )
    donor_name = context.donor_part_name
    receptor_name = context.receptor_part_name
    if (
        donor_name not in context.part_names
        or receptor_name not in context.part_names
        or donor_name == receptor_name
    ):
        raise ValueError("FV archive runtime recipe has invalid donor/receptor roles.")

    source_views: dict[str, OwnerLocalCellGeometry] = {}
    for name, stored_view in zip(context.part_names, context.source_views, strict=True):
        if type(stored_view) is not OwnerLocalCellGeometry:
            raise TypeError(
                "FV archive source views must be canonical owner-local geometries."
            )
        source_views[name] = stored_view
    if any(source_views[name].part_name != name for name in context.part_names):
        raise ValueError(
            "FV archive source view names differ from their typed authority."
        )
    if any(
        view.specification_id is not None
        and specification.specification_id != view.specification_id
        for view, specification in zip(
            (source_views[name] for name in context.part_names),
            context.source_specifications,
            strict=True,
        )
    ):
        raise ValueError("FV archive view/specification binding changed.")
    source_parts = tuple(
        MeshPart(
            name,
            source_views[name],
            coordinate_contract=source_views[name].coordinate_contract,
        )
        for name in context.part_names
    )
    source_manifests = tuple(MeshPartManifest.from_part(part) for part in source_parts)
    local_rank = execution_group.spec.process_indices.index(jax.process_index())
    source_assembly = MeshAssembly(
        source_parts,
        global_parts=source_manifests,
        local_rank=local_rank,
    )
    source_overset_specs = tuple(
        _finite_volume_archive_overset_part_spec(value)
        for value in context.source_overset_specs
    )
    source_policy = _finite_volume_archive_overset_policy(context.source_overset_policy)
    if tuple(spec.part_name for spec in source_overset_specs) != (
        source_assembly.global_part_names
    ):
        raise ValueError("FV archive source overset specs differ from part order.")
    source_receptor_spec = next(
        spec for spec in source_overset_specs if spec.part_name == receptor_name
    )
    source_candidate = prepare_overset_connectivity(
        source_assembly,
        source_overset_specs,
        policy=source_policy,
        execution_group=execution_group,
    )
    donor_view = source_views[donor_name]
    receptor_view = source_views[receptor_name]
    system = EulerSystem(
        context.spatial_dimension,
        material=IdealGasMaterial(
            context.material_gamma,
            context.material_gas_constant,
            density_floor=context.material_density_floor,
            pressure_floor=context.material_pressure_floor,
        ),
    )

    def require_neighbor_carriers(
        view: OwnerLocalCellGeometry,
        part_name: str,
        /,
    ) -> None:
        from jax.experimental import multihost_utils

        cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
        owned = np.asarray(view.owned, dtype=np.bool_)
        graph_ids = np.asarray(view.owned_cell_ids, dtype=np.int64)
        offsets = np.asarray(view.cell_neighbor_offsets, dtype=np.int64)
        neighbor_ids = np.asarray(view.cell_neighbor_ids, dtype=np.int64)
        neighbor_owner_ranks = np.asarray(
            view.cell_neighbor_owner_ranks,
            dtype=np.int32,
        )
        if (
            owned.shape != cell_ids.shape
            or offsets.shape != (graph_ids.size + 1,)
            or offsets[-1] != neighbor_ids.size
            or neighbor_owner_ranks.shape != neighbor_ids.shape
            or set(graph_ids.tolist()) != set(cell_ids[owned].tolist())
        ):
            raise ValueError("FV archive source neighbor metadata is malformed.")
        local_cells = set(int(identifier) for identifier in cell_ids.tolist())
        missing_records: list[tuple[int, int, int]] = []
        for row, owned_cell_id in enumerate(graph_ids.tolist()):
            for neighbor_row in range(int(offsets[row]), int(offsets[row + 1])):
                neighbor_id = int(neighbor_ids[neighbor_row])
                if neighbor_id not in local_cells:
                    missing_records.append(
                        (
                            int(owned_cell_id),
                            neighbor_id,
                            int(neighbor_owner_ranks[neighbor_row]),
                        )
                    )
        process_indices = execution_group.spec.process_indices
        rank_counts = np.asarray(
            multihost_utils.process_allgather(
                np.asarray((len(missing_records),), dtype=np.int32)
            ),
            dtype=np.int64,
        ).reshape((-1,))
        if not np.any(rank_counts):
            return
        record_width = max(1, int(np.max(rank_counts)))
        padded_records = np.full((record_width, 3), -1, dtype=np.int64)
        if missing_records:
            padded_records[: len(missing_records)] = np.asarray(
                missing_records,
                dtype=np.int64,
            )
        gathered_records = np.asarray(
            multihost_utils.process_allgather(padded_records, tiled=False),
            dtype=np.int64,
        ).reshape((len(process_indices), record_width, 3))
        cell_counts = np.asarray(
            multihost_utils.process_allgather(
                np.asarray((cell_ids.size,), dtype=np.int32)
            ),
            dtype=np.int64,
        ).reshape((-1,))
        cell_width = max(1, int(np.max(cell_counts)))
        padded_cell_ids = np.full((cell_width,), -1, dtype=np.int64)
        padded_cell_ids[: cell_ids.size] = cell_ids
        gathered_cell_ids = np.asarray(
            multihost_utils.process_allgather(padded_cell_ids, tiled=False),
            dtype=np.int64,
        ).reshape((len(process_indices), cell_width))
        missing_summary: list[tuple[int, int, int, int, tuple[int, ...]]] = []
        for owner_position, count in enumerate(rank_counts):
            process_index = process_indices[owner_position]
            for record in gathered_records[owner_position, : int(count)]:
                owned_cell_id, neighbor_id, expected_owner = (
                    int(value) for value in record
                )
                present_ranks = tuple(
                    process_indices[position]
                    for position, local_count in enumerate(cell_counts)
                    if neighbor_id in gathered_cell_ids[position, : int(local_count)]
                )
                missing_summary.append(
                    (
                        process_index,
                        owned_cell_id,
                        neighbor_id,
                        expected_owner,
                        present_ranks,
                    )
                )
        raise ValueError(
            f"FV archive source {part_name!r} lacks owned-cell neighbor carriers "
            f"(process, owned cell, missing neighbor, expected owner, local carrier "
            f"processes): {missing_summary!r}"
        )

    require_neighbor_carriers(donor_view, donor_name)
    require_neighbor_carriers(receptor_view, receptor_name)
    donor_geometry = UnstructuredFiniteVolumePlan.from_owner_local_geometry(
        donor_view,
        component_names=system.component_names,
    ).prepare()
    receptor_geometry = UnstructuredFiniteVolumePlan.from_owner_local_geometry(
        receptor_view,
        component_names=system.component_names,
    ).prepare()
    donor_blocks = tuple(
        block for block in donor_view.blocks if np.asarray(block.cell_ids).size
    )
    if len(donor_blocks) > 1:
        raise ValueError("FV archive source requires one donor block per owner.")
    local_reconstructions: dict[str, Any] = {}
    coefficient_ids: dict[str, np.ndarray] = {}
    if donor_blocks:
        donor_block = donor_blocks[0]
        local_reconstructions[donor_name] = (
            phx.discretization.prepare_owner_local_finite_volume_field_reconstruction(
                donor_view,
                donor_block.name,
                context.field_name,
            )
        )
        coefficient_ids[donor_name] = np.asarray(donor_block.cell_ids, dtype=np.int64)
    source_transfer = prepare_overset_field_transfer(
        source_candidate, local_reconstructions, context.field_name
    )
    source_connectivity = source_transfer.prepare_owner_local_exchange(
        coefficient_ids,
        execution_group,
        source_views,
    ).connectivity
    if not source_connectivity.owner_local or not source_connectivity.complete:
        raise ValueError("FV archive source connectivity did not replay completely.")
    source_overset = _owner_local_fv_overset_map(
        donor_view,
        receptor_view,
        donor_geometry,
        receptor_geometry,
        source_connectivity,
        donor_name,
        receptor_name,
    )

    precision = _rebuild_finite_volume_precision_policy(context.precision_policy)
    if validate_expected and (
        precision.policy_id != context.expected_precision_policy_id
        or system.system_id != context.expected_system_id
    ):
        raise ValueError("FV archive system or precision identity changed.")

    def build_dynamics(
        geometry: Any,
        overset: Any,
        expected_patches: tuple[str, ...],
        expected_boundary_set_id: str,
        expected_dynamics_id: str,
        /,
    ) -> tuple[Any, Any, Any]:
        patch_names = tuple(geometry.boundary_patch_names)
        if validate_expected and patch_names != expected_patches:
            raise ValueError("FV geometry boundary patch list changed.")
        boundaries = UnstructuredFiniteVolumeBoundarySet(
            patch_names,
            {name: phx.discretization.ExtrapolationBoundary() for name in patch_names},
        )
        if validate_expected and boundaries.boundary_set_id != expected_boundary_set_id:
            raise ValueError("FV boundary policy identity changed.")
        method = UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(
                smooth_epsilon=context.flux_smooth_epsilon
            ),
        )
        if validate_expected and method.method_id != context.expected_method_id:
            raise ValueError("FV numerical method identity changed.")
        problem = ConservationProblemIR(
            context.problem_name,
            context.field_name,
            system,
            boundaries,
        )
        coupling = UnstructuredFiniteVolumeCouplingPlan(
            overset=overset,
            topology_event_capacity=context.topology_event_capacity,
            topology_event_policy=context.topology_event_policy,
        )
        dynamics = compile_conservation_problem(
            problem,
            geometry,
            method,
            precision=precision,
            coupling=coupling,
        ).dynamics
        if validate_expected and dynamics.dynamics_id != expected_dynamics_id:
            raise ValueError("FV compiled dynamics identity changed.")
        return dynamics, boundaries, method

    source_dynamics, source_boundaries, method = build_dynamics(
        receptor_geometry,
        source_overset,
        context.source_boundary_patch_names,
        context.expected_source_boundary_set_id,
        context.expected_source_dynamics_id,
    )
    positivity = phx.discretization.FluxPositivityPlan(
        context.positivity_iterations,
        fallback_flux=phx.discretization.EinfeldtHLLFluxPlan(),
    )
    step_policy = FiniteVolumeStepPolicy(
        cfl=context.step_cfl,
        maximum_retries=context.maximum_retries,
        reduction_factor=context.step_reduction_factor,
        minimum_step_size=context.step_minimum_step_size,
    )
    if validate_expected and (
        positivity.plan_id != context.expected_positivity_id
        or step_policy.policy_id != context.expected_step_policy_id
    ):
        raise ValueError("FV time-step or positivity policy identity changed.")
    source_runtime = PreparedFiniteVolumeRuntime(
        source_dynamics,
        positivity,
        step_policy,
        execution_group=execution_group,
        owner_local_geometry=receptor_view,
        owner_local_axis_name=context.owner_local_axis_name,
    )
    if validate_expected:
        archived_source_epoch = TopologyEpoch.from_archive_record(
            context.source_topology_epoch_record
        )
        archived_source_artifacts = FiniteVolumeTopologyArtifacts.from_archive_record(
            context.source_topology_artifacts_record,
            archived_source_epoch,
        )
        if (
            source_runtime.runtime_id != context.expected_source_runtime_id
            or source_runtime.topology_epoch_id != context.expected_source_epoch_id
            or source_runtime.initial_topology_epoch.epoch_id
            != archived_source_epoch.epoch_id
            or source_runtime.initial_topology_artifacts.artifacts_id
            != context.expected_source_artifacts_id
            or source_runtime.initial_topology_artifacts.artifacts_id
            != archived_source_artifacts.artifacts_id
            or source_connectivity.connectivity_id
            != context.expected_source_connectivity_id
        ):
            raise ValueError("FV source runtime/topology authority changed.")

    target_plan = GraphPartitionPlan.from_stored_fields(
        context.target_graph_partition_plan
    )
    target_specification = phx.meshing.SurfaceMeshingSpec.from_stored_fields(
        context.target_specification
    )
    receptor_part = source_assembly.part(receptor_name)
    receptor_manifest = next(
        item for item in source_assembly.part_manifest if item.name == receptor_name
    )
    receptor_assembly = MeshAssembly(
        (receptor_part,),
        global_parts=(receptor_manifest,),
        local_rank=local_rank,
    )
    migration = partition_owner_local_graph(
        receptor_assembly,
        target_plan,
        execution_group,
        axis_name=f"{context.owner_local_axis_name}_target_partition",
    )
    target_assembly = _repartition_owner_local_periodic_assembly(
        receptor_assembly,
        migration,
        execution_group,
        current_specification=target_specification,
        graph_partition_plan=target_plan,
        axis_name=f"{context.owner_local_axis_name}_target_view",
    )
    target_part = target_assembly.part(receptor_name)
    replayed_target_view = target_part.carrier
    if not isinstance(replayed_target_view, OwnerLocalCellGeometry):
        raise TypeError("Graph repartition did not rebuild its owner-local target view.")
    if validate_expected and len(context.target_overset_specs) != 1:
        raise ValueError("FV archive must retain its exact target overset declaration.")
    if context.target_overset_specs:
        stored_target_spec = _finite_volume_archive_overset_part_spec(
            context.target_overset_specs[0]
        )
        derived_target_spec = source_receptor_spec.rebound(target_part)
        target_overset_spec = _finite_volume_archive_overset_part_spec(
            derived_target_spec
        )
        if stored_target_spec.spec_id != target_overset_spec.spec_id:
            raise ValueError("Stored target overset spec differs from owner replay.")
    else:
        target_overset_spec = source_overset_specs[
            tuple(spec.part_name for spec in source_overset_specs).index(receptor_name)
        ].rebound(target_part)
    target_candidate = source_connectivity.reregister(
        {receptor_name: target_part},
        successor_specs={receptor_name: target_overset_spec},
    )
    target_transfer = prepare_overset_field_transfer(
        target_candidate, local_reconstructions, context.field_name
    )
    target_connectivity = target_transfer.prepare_owner_local_exchange(
        coefficient_ids,
        execution_group,
        source_views,
    ).connectivity
    target_view = target_connectivity.assembly.part(receptor_name).carrier
    if not isinstance(target_view, OwnerLocalCellGeometry):
        raise TypeError("Candidate connectivity lost its owner-local target view.")
    if validate_expected and (
        target_view.distribution_id != context.expected_target_distribution_id
        or context.target_view.distribution_id != replayed_target_view.distribution_id
        or not np.array_equal(
            np.asarray(context.target_view.cell_ids, dtype=np.int64),
            np.asarray(replayed_target_view.cell_ids, dtype=np.int64),
        )
        or not np.array_equal(
            np.asarray(context.target_view.owned, dtype=np.bool_),
            np.asarray(replayed_target_view.owned, dtype=np.bool_),
        )
        or not np.array_equal(
            np.asarray(context.target_view.vertex_ids, dtype=np.int64),
            np.asarray(replayed_target_view.vertex_ids, dtype=np.int64),
        )
        or target_plan.plan_id != context.expected_target_plan_id
        or target_connectivity.connectivity_id != context.expected_target_connectivity_id
    ):
        raise ValueError("FV target view, plan, or connectivity identity changed.")
    target_geometry = UnstructuredFiniteVolumePlan.from_owner_local_geometry(
        target_view, component_names=system.component_names
    ).prepare()
    target_overset = _owner_local_fv_overset_map(
        donor_view,
        target_view,
        donor_geometry,
        target_geometry,
        target_connectivity,
        donor_name,
        receptor_name,
    )
    target_dynamics, target_boundaries, _ = build_dynamics(
        target_geometry,
        target_overset,
        context.target_boundary_patch_names,
        context.expected_target_boundary_set_id,
        context.expected_target_dynamics_id,
    )
    source_epoch = source_runtime.initial_topology_epoch
    graph_migration = target_view.graph_migration_evidence
    if graph_migration is None:
        raise ValueError("Replayed target view lost its graph migration evidence.")
    target_epoch = TopologyEpoch(
        source_epoch.index + 1,
        source_epoch.geometry_id,
        source_epoch.topology_id,
        target_view.distribution_id,
    )
    target_artifacts = FiniteVolumeTopologyArtifacts(
        target_epoch,
        target_geometry.prepared_id,
        topology_artifact_id=target_connectivity.connectivity_id,
        metrics_artifact_id=graph_migration.evidence_id,
        operators_artifact_id=target_dynamics.dynamics_id,
    )
    if validate_expected:
        archived_target_epoch = TopologyEpoch.from_archive_record(
            context.target_topology_epoch_record
        )
        archived_target_artifacts = FiniteVolumeTopologyArtifacts.from_archive_record(
            context.target_topology_artifacts_record,
            archived_target_epoch,
        )
        if (
            target_epoch.epoch_id != context.expected_target_epoch_id
            or target_artifacts.artifacts_id != context.expected_target_artifacts_id
            or target_epoch.epoch_id != archived_target_epoch.epoch_id
            or target_artifacts.artifacts_id != archived_target_artifacts.artifacts_id
        ):
            raise ValueError("Replayed FV target epoch/artifact authority changed.")
    rebuilt_context = context._replace(
        source_views=tuple(source_views[name] for name in context.part_names),
        source_overset_specs=source_overset_specs,
        source_overset_policy=source_policy,
        target_overset_specs=(target_overset_spec,),
        target_graph_partition_plan=target_plan,
        target_specification=target_specification,
        target_view=target_view,
        source_boundary_patch_names=tuple(source_boundaries.patch_names),
        target_boundary_patch_names=tuple(target_boundaries.patch_names),
        expected_target_plan_id=target_plan.plan_id,
        expected_source_dynamics_id=source_dynamics.dynamics_id,
        expected_source_boundary_set_id=source_boundaries.boundary_set_id,
        expected_target_boundary_set_id=target_boundaries.boundary_set_id,
        expected_source_runtime_id=source_runtime.runtime_id,
        expected_source_epoch_id=source_runtime.topology_epoch_id,
        expected_source_artifacts_id=source_runtime.initial_topology_artifacts.artifacts_id,
        source_topology_epoch_record=source_runtime.initial_topology_epoch.to_archive_record(),
        source_topology_artifacts_record=source_runtime.initial_topology_artifacts.to_archive_record(),
        target_topology_epoch_record=target_epoch.to_archive_record(),
        target_topology_artifacts_record=target_artifacts.to_archive_record(),
        expected_source_connectivity_id=source_connectivity.connectivity_id,
        expected_target_distribution_id=target_view.distribution_id,
        expected_target_connectivity_id=target_connectivity.connectivity_id,
        expected_target_dynamics_id=target_dynamics.dynamics_id,
        expected_system_id=system.system_id,
        expected_method_id=method.method_id,
        expected_positivity_id=positivity.plan_id,
        expected_precision_policy_id=precision.policy_id,
        expected_step_policy_id=step_policy.policy_id,
        expected_target_epoch_id=target_epoch.epoch_id,
        expected_target_artifacts_id=target_artifacts.artifacts_id,
    )
    return {
        "context": rebuilt_context,
        "group": execution_group,
        "source_assembly": source_assembly,
        "source_views": source_views,
        "source_connectivity": source_connectivity,
        "source_runtime": source_runtime,
        "source_geometry": receptor_geometry,
        "source_dynamics": source_dynamics,
        "target_view": target_view,
        "target_connectivity": target_connectivity,
        "target_dynamics": target_dynamics,
        "system": system,
        "source_cell_ids": np.asarray(
            source_runtime.owner_local_cell_ids(), dtype=np.int64
        ),
        "receptor_name": receptor_name,
        "source_epoch": source_runtime.initial_topology_epoch,
        "source_artifacts": source_runtime.initial_topology_artifacts,
        "target_epoch": target_epoch,
        "target_artifacts": target_artifacts,
    }


def _write_native_owner_local_fv_repartition_archive(
    artifact_root: str,
    /,
) -> None:
    import json
    from pathlib import Path

    from jax.experimental import multihost_utils

    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax._fingerprint import canonical_fingerprint
    from phydrax._model._structure import (
        FiniteVolumeOwnerLocalRepartitionContextArchiveRecipe,
    )
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.graph._partition import GraphPartitionPlan
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._assembly import MeshAssembly
    from phydrax.meshing._model_codec import write_model_artifact
    from phydrax.solver._finite_volume_runtime import (
        FiniteVolumeStepPolicy,
    )
    from tests.unit.meshing.test_native_overset import (
        _owner_local_periodic_assembly,
    )

    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    process_indices = tuple(device.process_index for device in devices)
    if (
        jax.process_count() != 3
        or process_indices != (0, 1, 2)
        or len(jax.local_devices()) != 1
    ):
        raise RuntimeError("FV archive writer requires three fresh process owners.")
    rank = jax.process_index()
    group_spec = ExecutionGroupSpec(
        "native-owner-local-fv-cold-context",
        process_indices,
        tuple((device.process_index, device.id) for device in devices),
        mesh_axes=(("owner_local_fv_archive", 3),),
    )
    group = ExecutionGroup(group_spec, devices)
    donor_name, receptor_name = "fv-archive-donor", "fv-archive-receptor"
    donor_points = np.asarray(
        ((0.11, 0.13), (0.58, 0.17), (0.31, 0.62), (0.84, 0.77)),
        dtype=np.float64,
    )
    receptor_points = np.asarray(
        ((0.19, 0.21), (0.63, 0.11), (0.23, 0.70), (0.77, 0.83)),
        dtype=np.float64,
    )
    point_owners = np.asarray((1, 1, 0, 0), dtype=np.int32)
    donor_assembly = _owner_local_periodic_assembly(
        donor_name, donor_points, point_owners, group
    )
    receptor_assembly = _owner_local_periodic_assembly(
        receptor_name, receptor_points, point_owners, group
    )
    source_assembly = MeshAssembly(
        (*donor_assembly.parts, *receptor_assembly.parts),
        global_parts=(
            *donor_assembly.part_manifest,
            *receptor_assembly.part_manifest,
        ),
        local_rank=rank,
    )
    source_parts = {part.name: part for part in source_assembly.parts}
    source_views: dict[str, OwnerLocalCellGeometry] = {}
    for name, part in source_parts.items():
        view = part.carrier
        if type(view) is not OwnerLocalCellGeometry:
            raise TypeError(
                "FV archive writer requires canonical owner-local source views."
            )
        source_views[name] = view
    part_names = tuple(source_assembly.global_part_names)
    source_specifications = tuple(
        _finite_volume_archive_surface_spec(source_views[name].source_value)
        for name in part_names
    )
    receptor_boundary = _finite_volume_archive_boundary_scope(
        source_views[receptor_name],
        rank,
    )
    source_overset_specs = tuple(
        OversetPartSpec(
            donor_name,
            priority=2,
            owner_rank=0,
        )
        if name == donor_name
        else OversetPartSpec(
            receptor_name,
            boundary=receptor_boundary,
            priority=1,
            owner_rank=1,
        )
        for name in part_names
    )
    source_overset_policy = OversetPolicy(
        fringe_layers=1,
        donor_priority="part-priority",
    )
    receptor_specification = source_specifications[part_names.index(receptor_name)]
    target_graph_plan = GraphPartitionPlan(
        len(process_indices),
        target_shares=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        maximum_imbalance=1.0,
        require_nonempty=False,
    )
    system = phx.equations.EulerSystem(2)
    flux_plan = phx.discretization.RusanovFluxPlan()
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        flux_plan,
    )
    precision = phx.discretization.FiniteVolumePrecisionPolicy()
    positivity = phx.discretization.FluxPositivityPlan()
    step_policy = FiniteVolumeStepPolicy(maximum_retries=2)
    context = FiniteVolumeOwnerLocalRepartitionContextArchiveRecipe(
        group_id=group_spec.group_id,
        process_indices=group_spec.process_indices,
        device_keys=group_spec.device_keys,
        mesh_axes=group_spec.mesh_axes,
        parent_group_id=group_spec.parent_group_id,
        process_host_ids=group_spec.process_host_ids,
        part_names=part_names,
        donor_part_name=donor_name,
        receptor_part_name=receptor_name,
        source_views=tuple(source_views[name] for name in part_names),
        source_specifications=source_specifications,
        source_overset_specs=source_overset_specs,
        source_overset_policy=source_overset_policy,
        target_overset_specs=(),
        target_graph_partition_plan=target_graph_plan,
        target_specification=receptor_specification,
        target_view=None,
        source_topology_epoch_record=None,
        source_topology_artifacts_record=None,
        target_topology_epoch_record=None,
        target_topology_artifacts_record=None,
        source_boundary_patch_names=(),
        target_boundary_patch_names=(),
        system_kind="euler",
        spatial_dimension=system.dimension,
        expected_system_id=system.system_id,
        problem_name="owner-local-fv-cold-repartition",
        field_name="state",
        source_id=None,
        reconstruction_kind="piecewise-constant",
        flux_kind="rusanov",
        expected_method_id=method.method_id,
        flux_smooth_epsilon=flux_plan.smooth_epsilon,
        material_gamma=system.material.gamma,
        material_gas_constant=system.material.gas_constant,
        material_density_floor=system.material.density_floor,
        material_pressure_floor=system.material.pressure_floor,
        precision_policy=precision,
        positivity_iterations=positivity.iterations,
        positivity_fallback_kind="einfeldt-hll",
        step_cfl=step_policy.cfl,
        step_reduction_factor=step_policy.reduction_factor,
        step_minimum_step_size=step_policy.minimum_step_size,
        boundary_condition_kind="extrapolation",
        maximum_retries=step_policy.maximum_retries,
        topology_event_capacity=8,
        topology_event_policy="accepted_step",
        owner_local_axis_name="owner_local_fv_archive_runtime",
        expected_target_plan_id=target_graph_plan.plan_id,
        expected_source_dynamics_id="",
        expected_source_boundary_set_id="",
        expected_target_boundary_set_id="",
        expected_positivity_id=positivity.plan_id,
        expected_source_runtime_id="",
        expected_source_epoch_id="",
        expected_precision_policy_id=precision.policy_id,
        expected_step_policy_id=step_policy.policy_id,
        expected_source_artifacts_id="",
        expected_source_connectivity_id="",
        expected_target_distribution_id="",
        expected_target_connectivity_id="",
        expected_target_dynamics_id="",
        expected_target_epoch_id="",
        expected_target_artifacts_id="",
    )
    bundle = _finite_volume_archive_build_context(context, group, validate_expected=False)
    context = bundle["context"]
    source_runtime = bundle["source_runtime"]
    source_views = bundle["source_views"]
    source_geometry = bundle["source_geometry"]
    source_cell_ids = bundle["source_cell_ids"]
    receptor_view = source_views[receptor_name]
    cell_ids = np.asarray(source_geometry.cell_global_ids, dtype=np.int64)
    primitive = np.zeros((cell_ids.size, 4), dtype=np.float64)
    primitive[:, 0] = 1.0 + (cell_ids % 19) * 0.01
    primitive[:, 3] = 1.0
    primitive[:, 1] = 0.2 + (cell_ids % 5) * 0.01
    primitive[:, 2] = -0.1 + (cell_ids % 3) * 0.01
    input_primitive = primitive.copy()
    owned = np.asarray(receptor_view.owned, dtype=np.bool_)
    input_primitive[~owned, 0] = 3.0
    input_primitive[~owned, 3] = 3.0
    initial = source_runtime.initialize_state(
        system.primitive_to_conserved(jnp.asarray(input_primitive)),
        0.0,
        1.0e-4,
    )
    accepted = source_runtime.advance(initial)
    accepted_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(bool(np.asarray(accepted.accepted))),), dtype=np.int32)
        )
    ).reshape((-1,))
    if not np.all(accepted_flags == 1):
        raise ValueError("Archive source FV solve did not produce an accepted boundary.")
    state_id = source_runtime.accepted_state_id(accepted.runtime_state, source_cell_ids)
    token = source_runtime.validate_restored_accepted_state(
        accepted.runtime_state,
        source_cell_ids,
        expected_runtime_id=source_runtime.runtime_id,
        expected_epoch_id=source_runtime.topology_epoch_id,
        expected_state_id=state_id,
    )
    history_cell_ids = np.sort(np.asarray(receptor_view.owned_cell_ids, dtype=np.int64))
    history_values = (history_cell_ids * 3 + 7).reshape((-1, 1))
    bank_spec = phx.solver.FiniteVolumeOwnerLocalCellStateBankSpec(
        "test/owner-local-cell-history",
        history_cell_ids,
        0,
    )
    history_revision = _finite_volume_cell_state_bank_revision(bank_spec, history_values)
    context_entry = CompositionEntry(
        context,
        entry_id="finite-volume/context",
        role="discretization",
        owner_id="phydrax.solver.PreparedFiniteVolumeRuntime",
        structure_id=context.expected_source_epoch_id,
        revision_id=context.expected_source_runtime_id,
        semantics_id=context.expected_source_connectivity_id,
    )
    state_entry = CompositionEntry(
        token,
        entry_id="finite-volume/state",
        role="physical-state",
        owner_id="phydrax.solver.PreparedFiniteVolumeRuntime",
        structure_id=token.runtime_state.content_state.geometry_layout_id,
        revision_id=token.state_id,
        semantics_id=system.system_id,
        dependencies=(context_entry.binding("revision"),),
    )
    history_entry = CompositionEntry(
        history_values,
        entry_id=bank_spec.entry_id,
        role="history",
        owner_id="test.native-owner-local-fv",
        structure_id=token.runtime_state.content_state.geometry_layout_id,
        revision_id=history_revision,
        semantics_id="stable-cell-id-history",
        dependencies=(
            context_entry.binding("structure"),
            state_entry.binding("revision"),
        ),
    )
    schema_revision = canonical_fingerprint(
        {
            "kind": "finite-volume-cell-state-bank-layouts",
            "schemas": (
                (
                    bank_spec.entry_id,
                    tuple(
                        int(cell_id)
                        for cell_id in np.asarray(bank_spec.cell_ids, dtype=np.int64)
                    ),
                    bank_spec.cell_axis,
                ),
            ),
        }
    )
    schema_entry = CompositionEntry(
        (bank_spec,),
        entry_id="finite-volume/cell-state-bank-layouts",
        role="discretization",
        owner_id="phydrax.solver",
        structure_id=token.runtime_state.content_state.geometry_layout_id,
        revision_id=schema_revision,
        semantics_id="finite-volume-cell-state-bank-layouts",
        dependencies=(state_entry.binding("revision"),),
    )
    artifact = Composition(
        (context_entry, state_entry, history_entry, schema_entry),
        boundary_id=canonical_fingerprint(
            {
                "kind": "finite-volume-owner-local-repartition-archive",
                "runtime": token.runtime_id,
                "state": token.state_id,
                "target": context.expected_target_connectivity_id,
            }
        ),
    )
    write_model_artifact(
        Path(artifact_root) / f"owner-local-fv-state-{rank}.phx",
        artifact,
        limits=None,
    )
    multihost_utils.sync_global_devices("finite-volume-context-archive-written")
    print(
        json.dumps(
            {
                "rank": rank,
                "phase": "write",
                "source_runtime_id": context.expected_source_runtime_id,
                "state_id": token.state_id,
                "target_connectivity_id": context.expected_target_connectivity_id,
            },
            sort_keys=True,
        )
    )


def _read_native_owner_local_fv_repartition_archive(
    artifact_root: str,
    /,
) -> None:
    import json
    from pathlib import Path

    from jax.experimental import multihost_utils

    from phydrax._fingerprint import canonical_fingerprint
    from phydrax._model._structure import (
        FiniteVolumeOwnerLocalCellStateBankArchiveRecipe,
        FiniteVolumeOwnerLocalRepartitionContextArchiveRecipe,
        FiniteVolumeRuntimeStateArchiveRecipe,
    )
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import read_model_artifact
    from phydrax.solver._finite_volume_runtime import (
        FiniteVolumeOwnerLocalCellStateBankSpec,
        FiniteVolumeValidatedRuntimeState,
    )
    from tests.unit.interchange._native_cad_archive_worker import (
        _rebuild_finite_volume_runtime_state_archive_recipe,
    )

    rank = jax.process_index()
    archive_path = Path(artifact_root) / f"owner-local-fv-state-{rank}.phx"
    restored_context: dict[str, Any] = {}

    def restore_archive(value: Any, /) -> Any:
        expected_entries = {
            "finite-volume/context",
            "finite-volume/state",
            "test/owner-local-cell-history",
            "finite-volume/cell-state-bank-layouts",
        }
        if not isinstance(value, Composition) or set(value.entry_ids) != expected_entries:
            raise TypeError("Cold FV archive lost its exact typed context/state entries.")
        entries = {entry.entry_id: entry for entry in value.entries}
        stored_context_entry = entries["finite-volume/context"]
        stored_context = stored_context_entry.value
        if (
            type(stored_context)
            is not FiniteVolumeOwnerLocalRepartitionContextArchiveRecipe
        ):
            raise TypeError(
                "Cold FV archive lost its callable-free runtime context recipe."
            )
        group = _finite_volume_archive_execution_group(stored_context)
        validated_context = _finite_volume_archive_rebuild_context_views(
            stored_context, group
        )
        bundle = _finite_volume_archive_build_context(
            validated_context, group, validate_expected=True
        )
        if (
            stored_context_entry.revision_id
            != bundle["context"].expected_source_runtime_id
            or stored_context_entry.structure_id
            != bundle["context"].expected_source_epoch_id
            or stored_context_entry.semantics_id
            != bundle["context"].expected_source_connectivity_id
        ):
            raise ValueError("Cold FV context Composition identity changed.")

        stored_state_entry = entries["finite-volume/state"]
        state_recipe = stored_state_entry.value
        if type(state_recipe) is not FiniteVolumeRuntimeStateArchiveRecipe:
            raise TypeError("Cold FV archive lost its accepted-state recipe.")
        token = _rebuild_finite_volume_runtime_state_archive_recipe(
            state_recipe, bundle["source_runtime"]
        )
        if not isinstance(token, FiniteVolumeValidatedRuntimeState):
            raise TypeError("FV archive restore did not produce its validated token.")
        stored_history_entry = entries["test/owner-local-cell-history"]
        raw_schemas = entries["finite-volume/cell-state-bank-layouts"].value
        if not isinstance(raw_schemas, tuple) or len(raw_schemas) != 1:
            raise ValueError("Cold FV archive needs one explicit cell-bank schema.")
        schema = raw_schemas[0]
        if type(schema) is not FiniteVolumeOwnerLocalCellStateBankArchiveRecipe:
            raise TypeError(
                "Cold FV cell-bank layout is not its canonical archive recipe."
            )
        bank_spec = FiniteVolumeOwnerLocalCellStateBankSpec(
            schema.entry_id, schema.cell_ids, schema.cell_axis
        )
        history_values = np.asarray(stored_history_entry.value)
        axis = bank_spec.cell_axis
        if axis < 0:
            axis += history_values.ndim
        schema_revision = canonical_fingerprint(
            {
                "kind": "finite-volume-cell-state-bank-layouts",
                "schemas": (
                    (
                        bank_spec.entry_id,
                        tuple(
                            int(cell_id)
                            for cell_id in np.asarray(bank_spec.cell_ids, dtype=np.int64)
                        ),
                        bank_spec.cell_axis,
                    ),
                ),
            }
        )
        if (
            history_values.ndim < 1
            or not 0 <= axis < history_values.ndim
            or history_values.shape[axis] != np.asarray(bank_spec.cell_ids).size
            or stored_history_entry.entry_id != bank_spec.entry_id
            or stored_history_entry.revision_id
            != _finite_volume_cell_state_bank_revision(bank_spec, history_values)
            or stored_history_entry.structure_id
            != bundle["source_runtime"].owner_local_geometry.geometry_layout_id
            or stored_history_entry.semantics_id != "stable-cell-id-history"
            or entries["finite-volume/cell-state-bank-layouts"].revision_id
            != schema_revision
            or stored_state_entry.revision_id != token.state_id
            or stored_state_entry.structure_id
            != token.runtime_state.content_state.geometry_layout_id
            or stored_state_entry.semantics_id != bundle["system"].system_id
        ):
            raise ValueError("Cold FV state/history/context identity changed.")

        replacements = {
            "finite-volume/context": bundle["context"],
            "finite-volume/state": token,
            "finite-volume/cell-state-bank-layouts": (bank_spec,),
        }
        rebuilt_entries = tuple(
            CompositionEntry(
                replacements.get(stored.entry_id, stored.value),
                entry_id=stored.entry_id,
                role=stored.role,
                owner_id=stored.owner_id,
                structure_id=stored.structure_id,
                revision_id=stored.revision_id,
                semantics_id=stored.semantics_id,
                dependencies=stored.dependencies,
            )
            for stored in value.entries
        )
        rebuilt = Composition(rebuilt_entries, boundary_id=value.boundary_id)
        if (
            rebuilt.composition_id != value.composition_id
            or rebuilt.structure_id != value.structure_id
        ):
            raise ValueError("Cold FV context Composition identity changed.")
        restored_context.update(
            {
                **bundle,
                "token": token,
                "history_entry": stored_history_entry,
                "bank_spec": bank_spec,
            }
        )
        return rebuilt

    composition = read_model_artifact(
        archive_path,
        expected_type=Composition,
        restore=restore_archive,
        limits=None,
    )
    context = restored_context["context"]
    group = restored_context["group"]
    source_runtime = restored_context["source_runtime"]
    source_connectivity = restored_context["source_connectivity"]
    target_connectivity = restored_context["target_connectivity"]
    target_dynamics = restored_context["target_dynamics"]
    token = composition.value("finite-volume/state")
    history_record = composition.entry("test/owner-local-cell-history")
    bank_spec = composition.value("finite-volume/cell-state-bank-layouts")[0]
    route_entries = source_connectivity.composition_entries()
    route = next(entry for entry in route_entries if entry.role == "interface-route")
    state = token.runtime_state
    state_entry = CompositionEntry(
        state,
        entry_id="finite-volume/state",
        role="physical-state",
        owner_id="phydrax.solver.PreparedFiniteVolumeRuntime",
        structure_id=state.content_state.geometry_layout_id,
        revision_id=source_runtime.accepted_state_id(state, token.source_cell_ids),
        semantics_id=restored_context["system"].system_id,
        dependencies=(route.binding("revision"),),
    )
    history_entry = CompositionEntry(
        history_record.value,
        entry_id=history_record.entry_id,
        role=history_record.role,
        owner_id=history_record.owner_id,
        structure_id=history_record.structure_id,
        revision_id=history_record.revision_id,
        semantics_id=history_record.semantics_id,
        dependencies=(route.binding("revision"),),
    )
    accepted_composition = Composition(
        (*route_entries, state_entry, history_entry),
        boundary_id=canonical_fingerprint(
            {
                "kind": "finite-volume-cold-accepted-boundary",
                "epoch": token.epoch_id,
                "state": token.state_id,
            }
        ),
    )
    repartitioned = source_runtime.commit_owner_local_repartition(
        token,
        source_connectivity,
        target_connectivity,
        target_dynamics,
        accepted_composition=accepted_composition,
        execution_group=group,
        owner_local_part_name=context.receptor_part_name,
        cell_state_banks=(bank_spec,),
    )
    accepted_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(bool(np.asarray(repartitioned.accepted))),), dtype=np.int32)
        )
    ).reshape((-1,))
    if not np.all(accepted_flags == 1):
        raise ValueError("Cold accepted FV graph repartition was refused.")
    if (
        repartitioned.successor_runtime is None
        or repartitioned.repartition_receipt is None
    ):
        raise ValueError("Cold FV repartition did not publish its successor runtime.")
    target_ids = np.asarray(
        restored_context["target_view"].owned_cell_ids, dtype=np.int64
    )
    migrated_history = np.asarray(
        repartitioned.repartition_receipt.composition.value(bank_spec.entry_id)
    )
    np.testing.assert_array_equal(migrated_history, (target_ids * 3 + 7).reshape((-1, 1)))
    source_epoch = token.runtime_state.topology_journal.epoch_table[-1]
    target_epoch = repartitioned.runtime_state.topology_journal.epoch_table[-1]
    if target_epoch.partition_id == source_epoch.partition_id:
        raise ValueError("Cold FV graph migration did not change partition identity.")
    target_artifacts = repartitioned.runtime_state.topology_journal.artifact_table[-1]
    if (
        target_epoch.epoch_id != context.expected_target_epoch_id
        or target_artifacts.artifacts_id != context.expected_target_artifacts_id
    ):
        raise ValueError("Cold FV repartition did not reproduce target epoch artifacts.")
    target_owner_mask = repartitioned.runtime_state.content_state.owned_cell_mask
    if target_owner_mask is None:
        raise ValueError("Cold FV repartition lost explicit target ownership.")
    target_counts = np.asarray(
        multihost_utils.process_allgather(np.asarray((target_ids.size,), dtype=np.int32))
    ).reshape((-1,))
    if (
        target_counts.shape != (3,)
        or target_counts[0] != 0
        or target_counts[1] != 0
        or target_counts[2] <= 0
    ):
        raise ValueError("Cold FV graph plan did not replay its declared target owners.")
    if target_ids.size == 0 and np.any(np.asarray(target_owner_mask, dtype=np.bool_)):
        raise ValueError("Cold FV empty target owner retained owned state rows.")
    next_step = repartitioned.successor_runtime.advance(repartitioned.runtime_state)
    next_step_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(bool(np.asarray(next_step.accepted))),), dtype=np.int32)
        )
    ).reshape((-1,))
    if not np.all(next_step_flags == 1):
        raise ValueError("Cold FV repartitioned runtime did not accept its next solve.")
    next_state = next_step.runtime_state
    prior_content = np.asarray(
        repartitioned.runtime_state.content_state.conservative_content
    )
    next_content = np.asarray(next_state.content_state.conservative_content)
    owner_mask = np.asarray(target_owner_mask, dtype=np.bool_)
    if (
        prior_content.shape != next_content.shape
        or owner_mask.shape != prior_content.shape[:1]
        or target_ids.size != np.count_nonzero(owner_mask)
        or not np.all(np.isfinite(next_content[owner_mask]))
    ):
        raise ValueError("Cold FV continuation lost finite owned physical content.")
    owned_state_changed = bool(
        target_ids.size and np.any(prior_content[owner_mask] != next_content[owner_mask])
    )
    changed_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(owned_state_changed),), dtype=np.int32)
        )
    ).reshape((-1,))
    if changed_flags.shape != (3,) or changed_flags[2] != 1:
        raise ValueError("Cold FV next solve did not change target-owned physical state.")
    positivity = next_step.positivity
    local_admissible = bool(np.asarray(positivity.fallback_valid)) and bool(
        np.asarray(positivity.limited_state_valid)
    )
    if target_ids.size:
        minimum_density = float(np.asarray(positivity.minimum_density))
        local_admissible = (
            local_admissible and np.isfinite(minimum_density) and minimum_density > 0.0
        )
    admissible_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(local_admissible),), dtype=np.int32)
        )
    ).reshape((-1,))
    if admissible_flags.shape != (3,) or admissible_flags[2] != 1:
        raise ValueError("Cold FV next solve lacks positive admissibility evidence.")
    from phydrax.discretization._conservation_ledger import (
        AcceptedConservationIntegralLedger,
    )

    ledger = next_step.accepted_flux_integrals
    if not isinstance(ledger, AcceptedConservationIntegralLedger):
        raise ValueError("Cold FV next solve lacks its accepted flux-balance ledger.")
    source_sum, boundary_sum, net_sum = ledger.conservation_sums()
    if (
        ledger.start_topology_epoch_id != target_epoch.epoch_id
        or ledger.end_topology_epoch_id != target_epoch.epoch_id
        or float(np.asarray(ledger.end_time)) <= float(np.asarray(ledger.start_time))
        or int(np.asarray(ledger.accepted_step))
        != int(np.asarray(next_state.accepted_step))
        or any(
            not np.all(np.isfinite(np.asarray(values)))
            for values in (source_sum, boundary_sum, net_sum)
        )
    ):
        raise ValueError("Cold FV next solve has invalid native balance evidence.")
    if (
        int(np.asarray(next_state.accepted_step))
        != int(np.asarray(repartitioned.runtime_state.accepted_step)) + 1
    ):
        raise ValueError("Cold FV continuation did not advance one accepted step.")
    print(
        json.dumps(
            {
                "phase": "read",
                "rank": rank,
                "state_id": token.state_id,
                "target_epoch": target_epoch.epoch_id,
                "next_step": int(np.asarray(next_step.runtime_state.accepted_step)),
            },
            sort_keys=True,
        )
    )


def _native_owner_local_fv_archive_rank_worker(
    artifact_root: str,
    phase: str,
    /,
) -> None:
    import sys
    import traceback

    from phydrax._execution_launch import require_local_process_supervision

    supervisor = require_local_process_supervision()
    try:
        if phase == "write":
            _write_native_owner_local_fv_repartition_archive(artifact_root)
        elif phase == "read":
            _read_native_owner_local_fv_repartition_archive(artifact_root)
        else:
            raise ValueError("FV archive worker phase must be write or read.")
    except BaseException as error:
        traceback.print_exc()
        sys.stderr.flush()
        supervisor.fail(
            f"FV archive {phase!r} failed on rank {supervisor.process_id}: "
            f"{type(error).__name__}: {error}"
        )
        raise


def test_native_owner_local_fv_archive_cold_repartitions_and_continues(
    tmp_path: Any,
) -> None:
    """Cold-read typed state in a new three-owner group before graph migration."""
    import json
    import os
    import shutil
    import socket
    import sys
    from pathlib import Path

    from phydrax import config as pxconfig
    from phydrax._array_archive import read_array_archive, write_array_archive
    from phydrax._execution_launch import (
        launch_local_processes,
        LocalProcessLaunchError,
        LocalProcessLaunchPlan,
    )
    from phydrax._model._structure import FiniteVolumeRuntimeStateArchiveRecipe

    if jax.process_count() != 1:
        pytest.skip("The cold FV coordinator must run outside a live process group.")
    if os.name != "posix":
        pytest.skip("The cold FV process group requires POSIX supervision.")
    if os.environ.get("PYTEST_XDIST_WORKER") is not None:
        pytest.skip("The three-owner cold FV qualification runs outside xdist.")
    repository = Path(__file__).resolve().parents[3]
    artifact_root = Path(tmp_path) / "owner-local-fv-cold"
    artifact_root.mkdir()
    worker_script = (
        "import runpy, sys; "
        "runpy.run_path(sys.argv[1])['_native_owner_local_fv_archive_rank_worker']"
        "(sys.argv[2], sys.argv[3])"
    )
    base_environment = dict(os.environ)
    for name in (
        "PHYDRAX_COORDINATOR_ADDRESS",
        "PHYDRAX_NUM_PROCESSES",
        "PHYDRAX_PROCESS_ID",
        "PHYDRAX_LOCAL_DEVICE_IDS",
        "PHYDRAX_COORDINATOR_BIND_ADDRESS",
    ):
        base_environment.pop(name, None)
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

    def launch_phase(root: Path, phase: str) -> list[dict[str, Any]]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        try:
            results = launch_local_processes(
                (
                    sys.executable,
                    "-c",
                    worker_script,
                    str(Path(__file__).resolve()),
                    str(root),
                    phase,
                ),
                LocalProcessLaunchPlan(3, f"127.0.0.1:{port}", devices_per_process=1),
                environment=environment,
                cwd=str(repository),
                timeout_seconds=900.0,
            )
        except LocalProcessLaunchError as error:
            output_directory = root / "rank-failure-logs" / phase
            output_directory.mkdir(parents=True, exist_ok=True)
            for result in error.results:
                rank_directory = output_directory / f"rank-{result.process_id}"
                rank_directory.mkdir(exist_ok=True)
                (rank_directory / "stdout.txt").write_bytes(result.stdout)
                (rank_directory / "stderr.txt").write_bytes(result.stderr)
                (rank_directory / "status.txt").write_text(
                    f"returncode={result.returncode}\n"
                    f"output_truncated={result.output_truncated}\n",
                    encoding="utf-8",
                )
            error.add_note(f"Full per-rank output saved under {output_directory}.")
            raise
        reports = [
            json.loads(result.stdout.decode().strip().splitlines()[-1])
            for result in results
        ]
        assert [item["rank"] for item in reports] == [0, 1, 2]
        assert all(item["phase"] == phase for item in reports)
        return reports

    launch_phase(artifact_root, "write")
    for rank in range(3):
        assert (artifact_root / f"owner-local-fv-state-{rank}.phx").is_file()
    launch_phase(artifact_root, "read")

    tampered_root = Path(tmp_path) / "owner-local-fv-tampered"
    state_identity_index = FiniteVolumeRuntimeStateArchiveRecipe._fields.index(
        "expected_state_id"
    )
    tampered_root.mkdir()
    for rank in range(3):
        source = artifact_root / f"owner-local-fv-state-{rank}.phx"
        destination = tampered_root / source.name
        if rank != 0:
            shutil.copyfile(source, destination)
            continue
        manifest, arrays = read_array_archive(source, limits=None)
        manifest.pop("arrays")

        def tamper_state_identity(node: Any) -> bool:
            if isinstance(node, dict):
                if str(node.get("type", "")).endswith(
                    "FiniteVolumeRuntimeStateArchiveRecipe"
                ):
                    fields = node.get("fields")
                    items = node.get("items")
                    identity = (
                        fields.get("expected_state_id")
                        if isinstance(fields, dict)
                        else items[state_identity_index]
                        if isinstance(items, list) and len(items) > state_identity_index
                        else None
                    )
                    if isinstance(identity, dict) and isinstance(
                        identity.get("value"), str
                    ):
                        current = identity["value"]
                        identity["value"] = "0" * 64 if current != "0" * 64 else "1" * 64
                        return True
                return any(tamper_state_identity(child) for child in node.values())
            if isinstance(node, list):
                return any(tamper_state_identity(child) for child in node)
            return False

        assert tamper_state_identity(manifest["model_recipe"])
        write_array_archive(destination, manifest=manifest, arrays=arrays, limits=None)

    with pytest.raises(LocalProcessLaunchError) as refused:
        launch_phase(tampered_root, "read")
    assert any(
        "Restored source state identity changed" in result.stderr.decode(errors="replace")
        for result in refused.value.results
    )
