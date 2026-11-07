#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

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
from phydrax.meshing._overset_finite_volume import (
    _unique_receptor_faces,
    prepare_owner_local_finite_volume_overset,
)
from phydrax.solver._finite_volume_archive_context import (
    _finite_volume_archive_target_receptor_spec,
    _resolve_finite_volume_archive_source,
)
from tools.meshing_qualification.scenarios.cold_owner_local_fv import (
    _finite_volume_cell_state_bank_revision,
)


def test_fv_archive_resolves_registered_source_by_exact_identity() -> None:
    def source(time: Any, state: Any, coordinates: Any, args: Any) -> Any:
        return state * args

    assert _resolve_finite_volume_archive_source(None, None) is None
    assert _resolve_finite_volume_archive_source("driven", {"driven": source}) is source
    with pytest.raises(ValueError, match="Unregistered FV source"):
        _resolve_finite_volume_archive_source("other", {"driven": source})
    invalid_source: Any = None
    with pytest.raises(TypeError, match="must be callable"):
        _resolve_finite_volume_archive_source("driven", {"driven": invalid_source})


def test_fv_cell_key_address_uses_both_global_id_words() -> None:
    from phydrax._sampling._addressing import derive_key, SampleAddress

    root = jax.random.key(731)
    address = SampleAddress("finite-volume", "owner-local-repartition", role="cell")
    identifiers = (7, (1 << 32) + 7, (2 << 32) + 7)
    keys = [
        derive_key(root, address, identifier >> 32, identifier & 0xFFFFFFFF)
        for identifier in identifiers
    ]
    words = [np.asarray(jax.random.key_data(key)) for key in keys]
    assert all(not np.array_equal(words[0], value) for value in words[1:])
    replay = derive_key(root, address, identifiers[1] >> 32, identifiers[1] & 0xFFFFFFFF)
    np.testing.assert_array_equal(jax.random.key_data(replay), words[1])


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


def test_owner_local_receptor_faces_are_independent_of_routes_and_storage() -> None:
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    )
    cells = np.asarray(((0, 1, 4, 3), (1, 2, 5, 4)), dtype=np.int32)
    vertex_ids = np.arange(10, 16)
    cell_ids = np.asarray((101, 103))

    def prepare(vertex_order: np.ndarray, cell_order: np.ndarray) -> Any:
        inverse = np.argsort(vertex_order)
        return phx.discretization.UnstructuredFiniteVolumePlan(
            vertices[vertex_order],
            quadrilaterals=inverse[cells[cell_order]],
            vertex_global_ids=vertex_ids[vertex_order],
            cell_global_ids=cell_ids[cell_order],
        ).prepare()

    def artifacts(geometry: Any, ids: tuple[int, ...]) -> dict[int, Any]:
        positions = {
            int(identifier): row
            for row, identifier in enumerate(np.asarray(geometry.cell_global_ids))
        }
        rows = np.asarray([positions[identifier] for identifier in ids], dtype=np.int32)
        faces, points, normals, measures, orientations = _unique_receptor_faces(
            geometry, rows
        )
        overset = phx.discretization.UnstructuredOversetPlan(
            geometry,
            geometry,
            rows,
            np.arange(len(ids) + 1, dtype=np.int32),
            rows,
            np.ones(len(ids)),
            receptor_face_ids=faces,
            receptor_face_points=points,
            receptor_face_normals=normals,
            receptor_face_measures=measures,
            receptor_face_cells=rows,
            receptor_face_orientations=orientations,
            receptor_route_faces=np.arange(len(ids), dtype=np.int32),
        )
        result = {}
        for index, identifier in enumerate(ids):
            # Quadrature is a weighted set; native edge ordering may reverse it.
            order = np.lexsort(np.asarray(overset.receptor_face_points)[index].T[::-1])
            result[identifier] = (
                np.asarray(overset.receptor_face_points)[index, order],
                np.asarray(overset.receptor_face_normals)[index, order],
                np.asarray(overset.receptor_face_measures)[index, order],
            )
        return result

    first = prepare(np.arange(6), np.asarray((0, 1)))
    permuted = prepare(np.asarray((5, 2, 4, 1, 3, 0)), np.asarray((1, 0)))
    reference = artifacts(first, (101, 103))
    for candidate in (
        artifacts(first, (103, 101)),
        artifacts(first, (101,)),
        artifacts(first, (103,)),
        artifacts(permuted, (101, 103)),
        artifacts(permuted, (103,)),
    ):
        for identifier, values in candidate.items():
            for actual, expected in zip(values, reference[identifier], strict=True):
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(reference[101][0][:, 0], 1.0)
    np.testing.assert_allclose(reference[103][0][:, 0], 2.0)
    for _, normals, _ in reference.values():
        np.testing.assert_allclose(normals, np.asarray(((1.0, 0.0), (1.0, 0.0))))


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
        receptor_face_orientations=np.asarray(
            (
                1
                if int(np.asarray(discretization.owner_cells)[face_id]) == receptor_cell
                else -1,
            ),
            dtype=np.int8,
        ),
        receptor_route_faces=np.asarray((0,), dtype=np.int32),
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
            receptor_face_orientations=np.where(
                np.asarray(geometry.owner_cells)[route_face_ids] == route_receptor_cells,
                1,
                -1,
            ).astype(np.int8),
            receptor_route_faces=np.arange(route_receptor_cells.size, dtype=np.int32),
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


def test_repeated_overset_advances_reuse_one_compiled_step() -> None:
    """The stationary overset step compiles once per runtime and is reused."""

    system, discretization, _, _, runtime = _compiled()
    state = runtime.initialize_state(_nonuniform_state(system, discretization), 0.0, 1e-3)
    state = runtime.advance(state).runtime_state
    compiles: list[str] = []

    def listener(event: str, duration_secs: float, **kwargs: str | int) -> None:
        del duration_secs, kwargs
        if event == "/jax/core/compile/backend_compile_duration":
            compiles.append(event)

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        for _ in range(3):
            result = runtime.advance(state)
            assert bool(np.asarray(result.accepted))
            state = result.runtime_state
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
    assert compiles == []
    assert int(np.asarray(state.accepted_step)) == 4


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


def _three_cell_overset_dynamics(
    donor_cells: tuple[int, ...], overlap_measures: tuple[float, ...], /
) -> tuple[Any, Any, Any, Any]:
    """Receptor cell 1 routed to ``donor_cells`` through its one face with cell 0."""
    system = phx.equations.EulerSystem(2)
    vertices = np.asarray([(float(i), float(j)) for j in range(2) for i in range(4)])
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(
            ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6)), dtype=np.int32
        ),
        vertex_global_ids=np.arange(20, 28),
        cell_global_ids=np.asarray((201, 202, 203)),
        component_names=system.component_names,
    ).prepare()
    owners = np.asarray(discretization.owner_cells)
    neighbors = np.asarray(discretization.neighbor_cells)
    face_id = int(
        np.flatnonzero(
            ((owners == 0) & (neighbors == 1)) | ((owners == 1) & (neighbors == 0))
        )[0]
    )
    unit_normal = np.asarray(discretization.area_vectors)[face_id] / float(
        np.asarray(discretization.face_measures)[face_id]
    )
    if int(owners[face_id]) != 1:
        unit_normal = -unit_normal
    face_points = np.asarray(discretization.face_quadrature_points)[face_id : face_id + 1]
    overset = phx.discretization.UnstructuredOversetPlan(
        discretization,
        discretization,
        np.asarray((1,), dtype=np.int32),
        np.asarray((0, len(donor_cells)), dtype=np.int32),
        np.asarray(donor_cells, dtype=np.int32),
        np.asarray(overlap_measures),
        epoch_id="shared-face-epoch",
        receptor_face_ids=np.asarray((face_id,), dtype=np.int32),
        receptor_face_points=face_points,
        receptor_face_normals=np.broadcast_to(unit_normal, face_points.shape),
        receptor_face_measures=np.asarray(discretization.face_quadrature_weights)[
            face_id : face_id + 1
        ],
        receptor_face_cells=np.asarray((1,), dtype=np.int32),
        receptor_face_orientations=np.asarray(
            (1 if owners[face_id] == 1 else -1,), dtype=np.int8
        ),
        receptor_route_faces=np.zeros((len(donor_cells),), dtype=np.int32),
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
    dynamics = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR(
            "shared-face-overset", "state", system, boundaries
        ),
        discretization,
        method,
        coupling=phx.discretization.UnstructuredFiniteVolumeCouplingPlan(overset=overset),
    ).dynamics
    return system, discretization, overset, dynamics


def test_overset_face_shared_by_several_donor_pieces_balances_its_face_flux() -> None:
    system, discretization, overset, dynamics = _three_cell_overset_dynamics(
        (0, 2), (0.25, 0.75)
    )
    np.testing.assert_array_equal(np.asarray(overset.receptor_route_faces), (0, 0))
    np.testing.assert_allclose(
        np.asarray(overset.receptor_route_face_fractions), (0.25, 0.75), rtol=0.0
    )
    primitive = np.asarray(
        ((1.0, 0.3, 0.0, 1.0), (2.0, 0.0, 0.1, 1.5), (0.5, -0.2, 0.0, 0.8))
    )
    state = system.primitive_to_conserved(jnp.asarray(primitive))
    metrics = phx.discretization.lower_static_unstructured_stage_metrics(discretization)
    block, _, measures = dynamics._overset_correction(state, metrics, None)
    assert block is not None
    np.testing.assert_array_equal(np.asarray(block.owner_cells), (0, 2))
    np.testing.assert_array_equal(np.asarray(block.neighbor_cells), (1, 1))
    route_flux = np.asarray(block.flux_rate)

    # Oracle: one route whose donor carries the overlap-weighted conserved mean.
    _, _, _, single = _three_cell_overset_dynamics((0,), (1.0,))
    mean_state = state.at[0].set(0.25 * state[0] + 0.75 * state[2])
    single_block, _, single_measures = single._overset_correction(
        mean_state, metrics, None
    )
    assert single_block is not None
    face_flux = np.asarray(single_block.flux_rate)[0]
    np.testing.assert_allclose(route_flux.sum(axis=0), face_flux, rtol=1e-13)
    np.testing.assert_allclose(route_flux[0], 0.25 * face_flux, rtol=1e-13)
    np.testing.assert_allclose(route_flux[1], 0.75 * face_flux, rtol=1e-13)
    np.testing.assert_allclose(
        np.sum(np.asarray(measures)), np.asarray(single_measures)[0], rtol=1e-14
    )
    scattered = np.zeros(discretization.state_shape, dtype=route_flux.dtype)
    np.add.at(scattered, np.asarray(block.owner_cells), -route_flux)
    np.add.at(scattered, np.asarray(block.neighbor_cells), route_flux)
    np.testing.assert_allclose(scattered.sum(axis=0), 0.0, atol=1e-13)

    with pytest.raises(ValueError, match="index certified receptor faces"):
        phx.discretization.UnstructuredOversetPlan(
            discretization,
            discretization,
            np.asarray((1,), dtype=np.int32),
            np.asarray((0, 2), dtype=np.int32),
            np.asarray((0, 2), dtype=np.int32),
            np.asarray((0.25, 0.75)),
            receptor_face_ids=overset.receptor_face_ids,
            receptor_face_points=overset.receptor_face_points,
            receptor_face_normals=overset.receptor_face_normals,
            receptor_face_measures=overset.receptor_face_measures,
            receptor_face_cells=np.asarray((1,), dtype=np.int32),
            receptor_face_orientations=overset.receptor_face_orientations,
            receptor_route_faces=np.asarray((0, 1), dtype=np.int32),
        )


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
        receptor_face_orientations=np.asarray(
            (
                1
                if int(np.asarray(discretization.owner_cells)[internal_face])
                == receptor_cell
                else -1,
            ),
            dtype=np.int8,
        ),
        receptor_route_faces=np.asarray((0,), dtype=np.int32),
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
    assert isinstance(
        rolled_back.topology_event_refusal, phx.solver.FiniteVolumeTopologyEventRefusal
    )
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
    assert isinstance(
        boundary_rejected.topology_event_refusal,
        phx.solver.FiniteVolumeTopologyEventRefusal,
    )
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


_SWEPT_CELL_FACES = {
    "prism": ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
    "hexahedron": (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ),
}


def _square_quads(lower: float, upper: float, count: int, /) -> tuple[Any, Any]:
    stations = np.linspace(lower, upper, count + 1)
    points = np.asarray([(x, y) for y in stations for x in stations])
    quads = [
        (
            row * (count + 1) + column,
            row * (count + 1) + column + 1,
            (row + 1) * (count + 1) + column + 1,
            (row + 1) * (count + 1) + column,
        )
        for row in range(count)
        for column in range(count)
    ]
    return points, np.asarray(quads, dtype=np.int64)


def _swept_polyhedra(mesh: Any, coordinates: Any, vertex_shift: int, /) -> list[Any]:
    """Swept cells as outward planar-faced polyhedra (straight extrusion faces)."""
    cells = []
    for block in mesh.blocks:
        for cell in np.asarray(block.vertices, dtype=np.int64):
            centroid = coordinates[cell].mean(axis=0)
            faces = []
            for template in _SWEPT_CELL_FACES[block.cell_kind]:
                loop = cell[list(template)]
                points = coordinates[loop]
                normal = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
                if float(np.dot(normal, points.mean(axis=0) - centroid)) < 0.0:
                    loop = loop[::-1]
                faces.append(tuple(int(vertex) + vertex_shift for vertex in loop))
            cells.append(tuple(faces))
    return cells


def test_native_whole_slab_layer_body_overset_on_background_conserves() -> None:
    """A natively swept graded prism slab is overset on a swept hex background.

    The whole slab is the layer domain (no core fill). Its fringe cells take
    donor fluxes from the background cells they overlap (exact common
    refinement volumes), and every accepted step closes the conservation
    ledger: content change equals the ledger net, the overset routes only
    move content between cells, and the physical boundary is the only
    exchange with the outside.
    """
    from phydrax.meshing._controls import LayerSchedule
    from phydrax.meshing._sweep import generate_sweep, SweepControl, SweepMapKind

    points, quads = _square_quads(0.0, 1.0, 4)
    background = generate_sweep(
        CellMesh(points, (CellBlock("quads", "quadrilateral", quads),)),
        SweepControl(SweepMapKind.EXTRUSION, LayerSchedule((0.25,) * 4), "background"),
    )
    body_points, body_quads = _square_quads(0.25, 0.75, 2)
    thicknesses = (0.08, 0.10, 0.12)
    body = generate_sweep(
        CellMesh.from_triangles(
            body_points,
            np.concatenate((body_quads[:, (0, 1, 2)], body_quads[:, (0, 2, 3)])),
        ),
        SweepControl(SweepMapKind.EXTRUSION, LayerSchedule(thicknesses), "body-layers"),
    )
    assert {block.cell_kind for block in background.mesh.blocks} == {"hexahedron"}
    assert {block.cell_kind for block in body.mesh.blocks} == {"prism"}
    np.testing.assert_allclose(
        body.stations, np.concatenate(((0.0,), np.cumsum(thicknesses))), atol=1e-15
    )
    background_points = np.asarray(background.mesh.coordinates, dtype=np.float64)
    # Lift the slab off the background's wall so its whole boundary is fringe.
    slab_points = np.asarray(body.mesh.coordinates, dtype=np.float64) + (0.0, 0.0, 0.25)
    background_cells = _swept_polyhedra(background.mesh, background_points, 0)
    slab_cells = _swept_polyhedra(body.mesh, slab_points, 0)
    background_ids = 100 + np.arange(len(background_cells))
    slab_ids = 500 + np.arange(len(slab_cells))
    background_vertex_ids = 1000 + np.arange(background_points.shape[0])
    slab_vertex_ids = 5000 + np.arange(slab_points.shape[0])
    background_mesh = CellMesh.from_polyhedra(
        background_points,
        background_cells,
        vertex_global_ids=background_vertex_ids,
        cell_global_ids=background_ids,
    )
    slab_mesh = CellMesh.from_polyhedra(
        slab_points,
        slab_cells,
        vertex_global_ids=slab_vertex_ids,
        cell_global_ids=slab_ids,
    )
    contract = phx.SpatialCoordinateContract.si()
    donor = MeshPart(
        "background", phx.meshing.certify_cell_mesh(background_mesh, contract)
    )
    receptor = MeshPart("slab", phx.meshing.certify_cell_mesh(slab_mesh, contract))
    slab_faces = np.asarray(slab_mesh.entity_set(2).entity_ids, dtype=np.int64)
    assert isinstance(slab_mesh.connectivity, phx.discretization.PolyhedralConnectivity)
    slab_boundary = np.asarray(slab_mesh.connectivity.boundary_faces, dtype=np.bool_)
    connectivity = prepare_overset_connectivity(
        MeshAssembly((donor, receptor)),
        (
            OversetPartSpec("background"),
            OversetPartSpec(
                "slab", boundary=receptor.scope(2, slab_faces[slab_boundary])
            ),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    assert connectivity.complete
    refinement = prepare_common_refinement(
        background_mesh,
        slab_mesh,
        policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.TARGET),
    )
    assert refinement.status is CommonRefinementStatus.SUCCESS

    # One single-device FV layout holds both parts as disjoint components.
    system = phx.equations.EulerSystem(3)
    composite = CellMesh.from_polyhedra(
        np.concatenate((background_points, slab_points)),
        background_cells
        + _swept_polyhedra(body.mesh, slab_points, background_points.shape[0]),
        vertex_global_ids=np.concatenate((background_vertex_ids, slab_vertex_ids)),
        cell_global_ids=np.concatenate((background_ids, slab_ids)),
    )
    geometry = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        composite, component_names=system.component_names
    ).prepare()
    row_of = {int(i): row for row, i in enumerate(np.asarray(geometry.cell_global_ids))}
    slab_rows = np.asarray([row_of[int(i)] for i in slab_ids], dtype=np.int32)
    # Each slab prism is a right prism over its profile triangle (area 1/32).
    slab_volumes = np.asarray(geometry.cell_volumes)[slab_rows]
    np.testing.assert_allclose(
        np.sort(slab_volumes),
        np.sort(np.repeat(np.asarray(thicknesses) / 32.0, 8)),
        rtol=0.0,
        atol=1e-16,
    )
    csr = np.asarray(refinement.target_offsets, dtype=np.int64)
    np.testing.assert_allclose(
        np.add.reduceat(np.asarray(refinement.volumes), csr[:-1]),
        slab_volumes,
        rtol=1e-12,
    )
    owners = np.asarray(geometry.owner_cells)
    neighbors = np.asarray(geometry.neighbor_cells)
    boundary_face_of = {
        int(owners[face]): int(face) for face in np.flatnonzero(neighbors < 0)[::-1]
    }
    fringe = np.asarray(
        [int(row) in boundary_face_of for row in slab_rows], dtype=np.bool_
    )
    receptors = slab_rows[fringe]
    routes = [
        np.arange(csr[index], csr[index + 1], dtype=np.int64)
        for index in np.flatnonzero(fringe)
    ]
    counts = np.asarray([route.size for route in routes], dtype=np.int64)
    selected = np.concatenate(routes)
    donor_rows = np.asarray(
        [
            row_of[int(background_ids[cell])]
            for cell in np.asarray(refinement.source_cells)[selected]
        ],
        dtype=np.int32,
    )
    face_ids = np.asarray([boundary_face_of[int(row)] for row in receptors], np.int32)
    face_points = np.asarray(geometry.face_quadrature_points)[face_ids]
    normals = (
        np.asarray(geometry.area_vectors)[face_ids]
        / np.asarray(geometry.face_measures)[face_ids, None]
    )
    overset = phx.discretization.UnstructuredOversetPlan(
        geometry,
        geometry,
        receptors,
        np.concatenate(((0,), np.cumsum(counts))),
        donor_rows,
        np.asarray(refinement.volumes)[selected],
        native_connectivity_id=connectivity.connectivity_id,
        receptor_face_ids=face_ids,
        receptor_face_points=face_points,
        receptor_face_normals=np.broadcast_to(normals[:, None, :], face_points.shape),
        # Triangle faces pad their quadrature with a zero-weight point.
        receptor_face_measures=np.asarray(geometry.face_quadrature_weights)[face_ids],
        receptor_face_cells=receptors,
        receptor_face_orientations=np.ones(receptors.size, dtype=np.int8),
        receptor_route_faces=np.repeat(np.arange(receptors.size, dtype=np.int32), counts),
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        geometry.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in geometry.boundary_patch_names
        },
    )
    dynamics = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR("slab-overset", "state", system, boundaries),
        geometry,
        phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
        coupling=phx.discretization.UnstructuredFiniteVolumeCouplingPlan(overset=overset),
    ).dynamics
    assert isinstance(
        dynamics, phx.discretization.PreparedUnstructuredFiniteVolumeDynamics
    )
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(maximum_retries=2),
    )
    centers = np.asarray(geometry.cell_centers)
    primitive = np.zeros(geometry.state_shape)
    primitive[:, 0] = 1.0 + 0.2 * centers[:, 2]
    primitive[:, 2] = 0.1
    primitive[:, 4] = 1.0
    state = runtime.initialize_state(
        system.primitive_to_conserved(jnp.asarray(primitive)), 0.0, 1.0e-3
    )
    for _ in range(3):
        before = np.asarray(state.content_state.volume_integral())
        result = runtime.advance(state)
        assert bool(np.asarray(result.accepted))
        state = result.runtime_state
        change = np.asarray(state.content_state.volume_integral()) - before
        source, outward, net = (
            np.asarray(value)
            for value in result.accepted_flux_integrals.conservation_sums()
        )
        np.testing.assert_array_equal(source, 0.0)
        np.testing.assert_allclose(net, -outward, rtol=0.0, atol=1e-18)
        np.testing.assert_allclose(change, net, rtol=0.0, atol=1e-14)
        overset_blocks = [
            block
            for block in result.accepted_flux_integrals.blocks
            if block.block_kind == "overset-correction"
        ]
        assert len(overset_blocks) == 1
        block = overset_blocks[0]
        route_flux = np.asarray(block.flux_integral)
        assert route_flux.shape == (selected.size, system.component_count)
        assert float(np.max(np.abs(route_flux[:, 0]))) > 0.0
        np.testing.assert_array_equal(np.asarray(block.owner_cells), donor_rows)
        scattered = np.zeros(geometry.state_shape)
        np.add.at(scattered, np.asarray(block.owner_cells), -route_flux)
        np.add.at(scattered, np.asarray(block.neighbor_cells), route_flux)
        np.testing.assert_allclose(scattered.sum(axis=0), 0.0, atol=1e-16)


def test_owner_local_composite_fv_layout_lowers_and_conserves_each_part() -> None:
    """Two owner-local periodic parts compile as one disjoint FV layout and advance."""
    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax.discretization._view_support import (
        OwnerLocalCellGeometry,
        OwnerLocalCompositeCellGeometry,
    )
    from phydrax.solver._finite_volume_runtime import FiniteVolumeStepPolicy
    from tools.meshing_qualification.scenarios.cold_owner_local_fv import (
        _owner_local_periodic_assembly,
    )

    if jax.process_count() != 1:
        pytest.skip("This composite layout contract uses one local process group.")
    device = jax.local_devices()[0]
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "owner-local-composite-fv",
            (jax.process_index(),),
            ((device.process_index, device.id),),
            mesh_axes=(("owner_local_composite", 1),),
        ),
        (device,),
    )
    owners = np.zeros((4,), dtype=np.int32)
    views: dict[str, OwnerLocalCellGeometry] = {}
    for name, points in (
        ("composite-receptor", ((0.19, 0.21), (0.63, 0.11), (0.23, 0.70), (0.77, 0.83))),
        ("composite-donor", ((0.11, 0.13), (0.58, 0.17), (0.31, 0.62), (0.84, 0.77))),
    ):
        carrier = (
            _owner_local_periodic_assembly(
                name, np.asarray(points, dtype=np.float64), owners, group
            )
            .parts[0]
            .carrier
        )
        assert isinstance(carrier, OwnerLocalCellGeometry)
        views[name] = carrier
    donor, receptor = views["composite-donor"], views["composite-receptor"]
    composite = OwnerLocalCompositeCellGeometry((receptor, donor))
    assert composite.part_names == ("composite-donor", "composite-receptor")
    assert composite.cell_offsets == (
        0,
        donor.cell_ids.size,
        donor.cell_ids.size + receptor.cell_ids.size,
    )
    np.testing.assert_array_equal(
        composite.cell_ids, np.concatenate((donor.cell_ids, receptor.cell_ids))
    )
    with pytest.raises(ValueError, match="unique names"):
        OwnerLocalCompositeCellGeometry((donor, donor))

    system = phx.equations.EulerSystem(2)
    geometry = phx.discretization.UnstructuredFiniteVolumePlan.from_owner_local_geometry(
        composite, component_names=system.component_names
    ).prepare()
    assert geometry.owner_local_view is composite
    np.testing.assert_array_equal(
        np.sort(np.asarray(geometry.cell_global_ids)), np.sort(composite.cell_ids)
    )
    # Each periodic part tiles the quotient cell once; the union holds both tilings.
    np.testing.assert_allclose(
        float(np.sum(np.asarray(geometry.cell_volumes))),
        float(np.sum(np.asarray(donor.cell_measures)))
        + float(np.sum(np.asarray(receptor.cell_measures))),
        rtol=1.0e-12,
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        geometry.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in geometry.boundary_patch_names
        },
    )
    dynamics = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR(
            "owner-local-composite", "state", system, boundaries
        ),
        geometry,
        phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
    ).dynamics
    assert isinstance(
        dynamics, phx.discretization.PreparedUnstructuredFiniteVolumeDynamics
    )
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        FiniteVolumeStepPolicy(maximum_retries=2),
        execution_group=group,
        owner_local_geometry=composite,
        owner_local_axis_name="owner_local_composite_fv",
    )
    cell_ids = np.asarray(geometry.cell_global_ids, dtype=np.int64)
    primitive = np.zeros((cell_ids.size, 4), dtype=np.float64)
    primitive[:, 0] = 1.0 + (cell_ids % 7) * 0.05
    primitive[:, 1] = 0.1
    primitive[:, 3] = 1.0 + (cell_ids % 5) * 0.05
    initial = runtime.initialize_state(
        system.primitive_to_conserved(jnp.asarray(primitive)), 0.0, 1.0e-3
    )
    advanced = runtime.advance(initial)
    assert bool(np.asarray(advanced.accepted))
    before = np.asarray(initial.content_state.conservative_content)
    after = np.asarray(advanced.runtime_state.content_state.conservative_content)
    assert not np.array_equal(before, after)
    # Parts share no faces: each part's periodic content is conserved on its own.
    donor_rows = np.isin(cell_ids, np.asarray(donor.cell_ids, dtype=np.int64))
    for rows in (donor_rows, ~donor_rows):
        np.testing.assert_allclose(
            np.sum(after[rows], axis=0),
            np.sum(before[rows], axis=0),
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def _serial_periodic_residual(
    view: Any, state: np.ndarray, dynamics: Any, /
) -> tuple[np.ndarray, int]:
    """Independent serial periodic FV residual of the owned rows of ``view``.

    Faces are paired through their midpoints reduced to the fundamental
    lattice cell, so a seam face meets its partner at whatever image the
    partner is resident; owned rows need every partner resident. Returns the
    residual and the number of owned faces whose partner sits at another image.
    """
    mesh, _ = view.reconstruct_local_geometry()
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    lift = np.asarray(view.periodic_lift_matrix, dtype=np.float64)
    inverse = np.linalg.inv(lift)
    row_by_id = {int(value): row for row, value in enumerate(np.asarray(view.cell_ids))}
    owned = np.asarray(view.owned, dtype=np.bool_)
    cell_faces: dict[int, list[tuple[tuple[int, ...], np.ndarray, np.ndarray]]] = {}
    faces: dict[tuple[int, ...], list[tuple[int, np.ndarray]]] = {}
    volumes = np.zeros((owned.size,), dtype=np.float64)
    for block in mesh.blocks:
        for vertices, identifier in zip(
            np.asarray(block.vertices), np.asarray(block.global_ids), strict=True
        ):
            row = row_by_id[int(identifier)]
            points = coordinates[np.asarray(vertices)]
            first, second = points[1] - points[0], points[2] - points[0]
            signed = 0.5 * (first[0] * second[1] - first[1] * second[0])
            if signed < 0.0:
                points = points[::-1]
            volumes[row] = abs(signed)
            for start, stop in ((0, 1), (1, 2), (2, 0)):
                tangent = points[stop] - points[start]
                area = np.asarray((tangent[1], -tangent[0]))
                midpoint = 0.5 * (points[start] + points[stop])
                fractional = (midpoint @ inverse) % 1.0
                key = tuple(
                    int(np.rint(value * 1.0e8)) % 100000000 for value in fractional
                )
                faces.setdefault(key, []).append((row, midpoint))
                cell_faces.setdefault(row, []).append((key, area, midpoint))
    solver = dynamics.method.interface_solver
    residual = np.zeros_like(state)
    seam_faces = 0
    for row in np.flatnonzero(owned):
        for key, area, midpoint in cell_faces[int(row)]:
            partners = [
                (other, other_midpoint)
                for other, other_midpoint in faces[key]
                if not (other == row and np.allclose(other_midpoint, midpoint))
            ]
            assert len(partners) == 1, "every owned face needs one resident partner"
            partner, partner_midpoint = partners[0]
            if not np.allclose(partner_midpoint, midpoint):
                seam_faces += 1
            measure = float(np.linalg.norm(area))
            flux = solver.normal_face_flux(
                dynamics.system,
                jnp.asarray(state[row])[None],
                jnp.asarray(state[partner])[None],
                jnp.asarray(area / measure)[None],
                None,
            ).normal_flux
            residual[row] -= np.asarray(flux)[0] * measure / volumes[row]
    return residual, seam_faces


def test_owner_local_periodic_fv_joins_seams_and_matches_serial_residual() -> None:
    """Periodic seam faces of owner-local patches are interior faces.

    One owner (every seam owned on both sides) or three process owners (seams
    against halo cells at other lattice images): owned rows reproduce an
    independent serial periodic FV residual, flux crosses the seams, and no
    periodic face is a physical boundary.
    """
    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from tools.meshing_qualification.scenarios.cold_owner_local_fv import (
        _owner_local_periodic_assembly,
    )

    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    ranks = tuple(device.process_index for device in devices)
    if jax.process_count() == 1:
        devices = (jax.local_devices()[0],)
        ranks = (jax.process_index(),)
        owners = np.zeros((4,), dtype=np.int32)
    elif jax.process_count() == 3 and ranks == (0, 1, 2) and len(devices) == 3:
        owners = np.asarray((1, 1, 0, 2), dtype=np.int32)
    else:
        pytest.skip("This contract runs on one owner or three process owners.")
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "owner-local-periodic-seams",
            ranks,
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("owner_local_periodic_seams", len(devices)),),
        ),
        devices,
    )
    view = (
        _owner_local_periodic_assembly(
            "periodic-seams",
            np.asarray(((0.11, 0.13), (0.58, 0.17), (0.31, 0.62), (0.84, 0.77))),
            owners,
            group,
        )
        .parts[0]
        .carrier
    )
    assert isinstance(view, OwnerLocalCellGeometry)
    if not np.any(np.asarray(view.owned)):
        return
    system = phx.equations.EulerSystem(2)
    geometry = phx.discretization.UnstructuredFiniteVolumePlan.from_owner_local_geometry(
        view, component_names=system.component_names
    ).prepare()
    owned = np.asarray(view.owned, dtype=np.bool_)
    neighbor = np.asarray(geometry.neighbor_cells)
    owner = np.asarray(geometry.owner_cells)
    # A periodic part has no physical boundary; only halo cuts stay open.
    assert not np.any(owned[owner[neighbor < 0]])
    physical = geometry.boundary_patch_names.index("__physical_boundary__")
    assert not np.any(np.asarray(geometry.boundary_patch_ids) == physical)
    assert geometry.periodic_face_translations is not None
    seams = np.any(np.asarray(geometry.periodic_face_translations) != 0.0, axis=1)
    assert np.any(seams)
    owned_ids = set(np.asarray(view.cell_ids)[owned].tolist())
    receptor_rows = np.asarray(
        [
            row
            for row, identifier in enumerate(np.asarray(geometry.cell_global_ids))
            if int(identifier) in owned_ids
        ],
        dtype=np.int32,
    )
    faces, points, normals, weights, orientations = _unique_receptor_faces(
        geometry, receptor_rows
    )
    overset = phx.discretization.UnstructuredOversetPlan(
        geometry,
        geometry,
        receptor_rows,
        np.arange(receptor_rows.size + 1, dtype=np.int32),
        receptor_rows,
        np.asarray(geometry.cell_volumes)[receptor_rows],
        receptor_face_ids=faces,
        receptor_face_points=points,
        receptor_face_normals=normals,
        receptor_face_measures=weights,
        receptor_face_cells=receptor_rows,
        receptor_face_orientations=orientations,
        receptor_route_faces=np.arange(receptor_rows.size, dtype=np.int32),
    )
    for index, row in enumerate(receptor_rows):
        single = _unique_receptor_faces(geometry, np.asarray((row,), dtype=np.int32))
        for actual, expected in zip(
            single,
            (
                overset.receptor_face_ids,
                overset.receptor_face_points,
                overset.receptor_face_normals,
                overset.receptor_face_measures,
                overset.receptor_face_orientations,
            ),
            strict=True,
        ):
            np.testing.assert_allclose(actual, np.asarray(expected)[index : index + 1])
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        geometry.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in geometry.boundary_patch_names
        },
    )
    dynamics: Any = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR(
            "owner-local-periodic-seams", "state", system, boundaries
        ),
        geometry,
        phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
    ).dynamics
    cell_ids = np.asarray(geometry.cell_global_ids, dtype=np.int64)
    primitive = np.zeros((cell_ids.size, 4), dtype=np.float64)
    primitive[:, 0] = 1.0 + (cell_ids % 7) * 0.05
    primitive[:, 1] = 0.1 + (cell_ids % 3) * 0.02
    primitive[:, 2] = -0.05 + (cell_ids % 4) * 0.01
    primitive[:, 3] = 1.0 + (cell_ids % 5) * 0.05
    state = np.asarray(system.primitive_to_conserved(jnp.asarray(primitive)))
    residual = np.asarray(dynamics(jnp.asarray(0.0), jnp.asarray(state)))
    view_rows = {int(value): row for row, value in enumerate(np.asarray(view.cell_ids))}
    plan_rows = np.asarray([view_rows[int(value)] for value in cell_ids], dtype=np.int64)
    view_state = np.empty_like(state)
    view_state[plan_rows] = state
    expected, seam_count = _serial_periodic_residual(view, view_state, dynamics)
    assert seam_count > 0
    owned_plan = owned[plan_rows]
    np.testing.assert_allclose(
        residual[owned_plan],
        expected[plan_rows][owned_plan],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    flux, _ = dynamics.face_fluxes(jnp.asarray(0.0), jnp.asarray(state))
    assert float(np.max(np.abs(np.asarray(flux)[seams]))) > 1.0e-3
    # A step with the joined operator conserves the periodic content of a
    # single owner exactly: no flux leaves through a seam.
    if len(devices) == 1:
        volumes = np.asarray(geometry.cell_volumes)[:, None]
        np.testing.assert_allclose(np.sum(volumes * residual, axis=0), 0.0, atol=1.0e-13)


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
    from tools.meshing_qualification.scenarios.cold_owner_local_fv import (
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

    def fv_overset(connectivity: Any, receptor: Any, donor: Any, label: str, /) -> Any:
        """The production owner-local FV overlap with donor halo expansion."""
        return prepare_owner_local_finite_volume_overset(
            connectivity,
            {donor_name: donor, receptor_name: receptor},
            donor_name,
            receptor_name,
            group,
            component_names=system.component_names,
            field_name="state",
            axis_name=f"owner_local_fv_event_{label}_overset",
        )

    source_coupling = fv_overset(connectivity, receptor_view, donor_view, "source")
    receptor_geometry = source_coupling.geometry
    receptor_overset = source_coupling.overset
    mapped_route_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((receptor_overset.receptor_cells.size,), dtype=np.int32)
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
        owner_local_geometry=source_coupling.composite,
        owner_local_axis_name="owner_local_fv_event_runtime",
    )
    cell_ids = np.asarray(receptor_geometry.cell_global_ids, dtype=np.int64)
    expected_primitive = np.zeros((cell_ids.size, 4), dtype=np.float64)
    expected_primitive[:, 0] = 1.0 + (cell_ids % 19) * 0.01
    expected_primitive[:, 3] = 1.0
    input_primitive = expected_primitive.copy()
    owned = np.asarray(runtime.owner_local_owned_cell_mask, dtype=np.bool_)
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
                # The root means its authored source, quotient topology, and
                # coordinate contract; motion changes only its geometry and
                # revision, so the meaning stays bound while distinct roots
                # keep distinct meanings.
                semantics_id=canonical_fingerprint(
                    {
                        "kind": "owner-local-source-root",
                        "source": view.source_id,
                        "topology": view.source_topology_id,
                        "coordinates": view.coordinate_contract_id,
                    }
                ),
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
    assert np.all(motion_flags == 1), motion.assessments[-1].failure_detail
    assert motion.local_result is not None
    if np.asarray(receptor_view.cell_ids).size:
        # Each owner certifies the complete stars of its own vertices; the
        # pinched halo vertices of the resident patch are their owners' scope.
        star_scope = motion.local_manifold_scope
        assert star_scope is not None and star_scope.kind == "owned-star"
        receptor_vertex_ids = np.asarray(receptor_view.vertex_ids, dtype=np.int64)
        np.testing.assert_array_equal(
            np.asarray(star_scope.vertex_ids),
            np.sort(
                receptor_vertex_ids[
                    np.asarray(receptor_view.vertex_owner_ranks, dtype=np.int32) == rank
                ]
            ),
        )
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
    successor_coupling = fv_overset(
        successor_connectivity, successor_view, donor_view, "successor"
    )
    successor_geometry = successor_coupling.geometry
    successor_overset = successor_coupling.overset
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
        # Both parts' derived geometry roots are reprepared against the
        # successor registration.
        reprepare_entries=tuple(
            entry
            for entry in with_geometry_roots(
                successor_assembly, successor_connectivity.composition_entries()
            )
            if entry.entry_id.startswith(("mesh/source-root:", "mesh/geometry:"))
        ),
    )
    assert committed.accepted
    assert committed.topology_event_refusal is None, committed.topology_event_refusal
    assert committed.successor_runtime is not None
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
    checkpoint_plan = phx.solver.FiniteVolumeCheckpointPlan(
        case, runtime=committed.successor_runtime
    )
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
    assert isinstance(
        rolled_back.topology_event_refusal, phx.solver.FiniteVolumeTopologyEventRefusal
    )
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
    repartitioned_receptor_part = repartitioned_owner_assembly.part(receptor_name)
    # The moved receptor's boundary scope names rank-local facets of the motion
    # target; graph migration moves them, so rebind it on the repartitioned part.
    repartitioned_specs = {
        receptor_name: _finite_volume_archive_target_receptor_spec(
            successor_specs[receptor_name],
            current_receptor_part,
            repartitioned_receptor_part,
            group,
        )
    }
    target_connectivity_candidate = successor_connectivity.reregister(
        {receptor_name: repartitioned_receptor_part},
        successor_specs=repartitioned_specs,
        execution_group=group,
        axis_name="owner_local_fv_repartition_reregister",
    )
    target_field_transfer = prepare_overset_field_transfer(
        target_connectivity_candidate,
        local_reconstructions,
        "state",
    )
    # The repartitioned receptor replaces the moved view in the search.
    target_search_views = {
        part.name: part.carrier
        for part in target_connectivity_candidate.assembly.parts
        if isinstance(part.carrier, OwnerLocalCellGeometry)
    }
    target_connectivity = target_field_transfer.prepare_owner_local_exchange(
        coefficient_ids,
        group,
        target_search_views,
    ).connectivity
    assert target_connectivity.owner_local and target_connectivity.complete
    target_receptor_carrier = target_connectivity.assembly.part(receptor_name).carrier
    if not isinstance(target_receptor_carrier, OwnerLocalCellGeometry):
        raise TypeError("Graph-migrated connectivity lacks its owner-local cell view.")
    target_receptor_view = target_receptor_carrier
    target_coupling = fv_overset(
        target_connectivity,
        target_receptor_view,
        current_source_views[donor_name],
        "target",
    )
    target_geometry = target_coupling.geometry
    target_overset = target_coupling.overset
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

    def accepted_repartition_composition_for(
        state: Any, history: Any, rng: Any, /
    ) -> Any:
        epoch = state.topology_journal.epoch_table[-1]
        return Composition(
            (*repartition_entries, repartition_state_entry_for(state), history, rng),
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

    def cell_keys(ids: np.ndarray) -> jax.Array:
        from phydrax._sampling._addressing import derive_key, SampleAddress

        unsigned = ids.astype(np.uint64)
        low = jnp.asarray(unsigned.astype(np.uint32))
        high = jnp.asarray((unsigned >> np.uint64(32)).astype(np.uint32))
        root = jax.random.key(731)
        address = SampleAddress("finite-volume", "owner-local-repartition", role="cell")
        return jax.vmap(lambda hi, lo: derive_key(root, address, hi, lo))(high, low)

    rng_spec = phx.solver.FiniteVolumeOwnerLocalCellStateBankSpec(
        "test/owner-local-cell-rng", history_cell_ids, 0
    )
    rng_entry = CompositionEntry(
        cell_keys(history_cell_ids),
        entry_id=rng_spec.entry_id,
        role="rng",
        owner_id="test.native-owner-local-fv",
        structure_id=current_receptor_view.geometry_layout_id,
        revision_id=canonical_fingerprint(
            {"kind": "test-cell-rng", "cell_ids": history_cell_ids}
        ),
        semantics_id="typed-stable-cell-rng",
        dependencies=(repartition_route.binding("revision"),),
    )
    repartition_composition = accepted_repartition_composition_for(
        repartition_state, history_entry, rng_entry
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
        cell_state_banks=(invalid_history_spec, rng_spec),
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
        repartition_state, history_entry, rng_entry
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
        cell_state_banks=(cell_state_bank_spec, rng_spec),
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
    migrated_rng = repartitioned.repartition_receipt.composition.entry(
        rng_entry.entry_id
    ).value
    assert jax.random.key_impl(migrated_rng) == jax.random.key_impl(rng_entry.value)
    np.testing.assert_array_equal(
        jax.random.key_data(migrated_rng),
        jax.random.key_data(cell_keys(target_history_ids)),
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
    # The composite layout keeps the unmoved donor on its owners; only the
    # repartitioned receptor migrates wholly to the last rank.
    receptor_owned_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                (int(np.count_nonzero(np.asarray(target_receptor_view.owned))),),
                dtype=np.int32,
            )
        )
    ).reshape((-1,))
    assert receptor_owned_counts[2] > 0
    assert np.all(receptor_owned_counts[:2] == 0)

    repartition_case = phx.solver.FiniteVolumeCaseSpec(
        "native-owner-local-fv-repartition",
        repartitioned.successor_runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 1000),
    )
    repartition_checkpoint_plan = phx.solver.FiniteVolumeCheckpointPlan(
        repartition_case, runtime=repartitioned.successor_runtime
    )
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


