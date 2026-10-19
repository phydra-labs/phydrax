# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Periodic polyhedral VEM construction and continuation records."""

from collections.abc import Callable
from dataclasses import asdict
from typing import Any, TypedDict

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax._meshcore import NativeExecutionScope
from phydrax.discretization import TransferGeometryBinding
from phydrax.discretization._cell_geometry import cell_geometry_preparation_query_upper
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization.fem import vertex_interpolation_transfer
from phydrax.discretization.vem import (
    prepare_polyhedral_h1_virtual_element_3d,
    PreparedPolyhedralH1VirtualElement3D,
)
from phydrax.lifecycle import read_meshing_source_closure
from phydrax.meshing import (
    NativePolyhedralSchedule,
    PeriodicConstraint,
    PiecewiseLinearComplex,
)
from phydrax.meshing._polyhedral_generation import (
    generate_polyhedral_volume,
    PolyhedralConstruction,
)
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.distributed_meshing.cold_lifecycle_identities import (
    array_record,
    coordinate_record,
)


class PeriodicRecipe(TypedDict):
    domain: PiecewiseLinearComplex
    sites: np.ndarray
    weights: np.ndarray
    constraints: tuple[PeriodicConstraint, ...]
    controls: dict[str, Any]


class PeriodicArchiveRoot(TypedDict):
    recipe: PeriodicRecipe
    mesh: phx.discretization.CellMesh
    geometry: phx.discretization.CellGeometrySpec
    state: ArrayLike
    coefficients: ArrayLike
    pinned_dof: int
    history: tuple[ArrayLike, ArrayLike]


def original_periodic_source() -> tuple[PolyhedralConstruction, PeriodicRecipe]:
    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)],
        dtype=np.float64,
    )
    loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    domain = PiecewiseLinearComplex(
        points, loops, np.arange(6, dtype=np.int64), [(-1, 0)] * 6, ("material",)
    )
    constraints: list[PeriodicConstraint] = []
    for axis, (left, right) in enumerate(((4, 5), (2, 3), (0, 1))):
        source = phx.meshing.MeshingScope(
            "vem-periodic-cube",
            "r1",
            phx.meshing.MeshingEntityKind.GEOMETRY,
            2,
            "source",
            np.asarray((left,), dtype=np.int64),
        )
        target = phx.meshing.MeshingScope(
            "vem-periodic-cube",
            "r1",
            phx.meshing.MeshingEntityKind.GEOMETRY,
            2,
            "target",
            np.asarray((right,), dtype=np.int64),
        )
        action = np.eye(4, dtype=np.float64)
        action[axis, 3] = 1
        constraints.append(
            PeriodicConstraint(
                source,
                target,
                action,
                tolerance=0,
                source_entity_ids=np.asarray((left,), dtype=np.int64),
                orientations=np.asarray((-1,), dtype=np.int32),
            )
        )
    sites = np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64)
    weights = np.asarray([0.0, 0.0], dtype=np.float64)
    schedule = NativePolyhedralSchedule()
    with NativeExecutionScope(schedule.native_limits()) as scope:
        construction = generate_polyhedral_volume(
            domain,
            sites=sites,
            weights=weights,
            schedule=schedule,
            periodic_constraints=constraints,
            scope=scope,
        )
    return construction, PeriodicRecipe(
        domain=domain,
        sites=sites,
        weights=weights,
        constraints=tuple(constraints),
        controls=asdict(schedule),
    )


def _value_error_message(operation: Callable[[], object]) -> str | None:
    try:
        operation()
    except ValueError as error:
        return str(error)
    return None


def _assemble_cellwise_constant_load(
    owner: PreparedPolyhedralH1VirtualElement3D, force: Array
) -> Array:
    return owner.assemble_cellwise_constant_load(force)


def _quotient_transfer_record(
    construction: PolyhedralConstruction,
    prepared: PreparedPolyhedralH1VirtualElement3D,
) -> dict[str, Any]:
    periodic = construction.mesh.periodic_topology
    if periodic is None:
        raise AssertionError("periodic_vem.topology must retain its quotient; got None")
    vertex_count = construction.mesh.coordinates.shape[0]
    identity = vertex_interpolation_transfer(
        np.arange(vertex_count, dtype=np.int64)[:, None],
        np.ones((vertex_count, 1), dtype=np.float64),
        np.ones((vertex_count, 1), dtype=np.bool_),
        source_size=vertex_count,
        source_topology_id=construction.mesh.topology_id,
        target_topology_id=construction.mesh.topology_id,
    )
    binding = TransferGeometryBinding(
        cell_geometry_id(construction.geometry),
        cell_geometry_id(construction.geometry),
        "identity",
        source_topology_id=construction.mesh.topology_id,
        target_topology_id=construction.mesh.topology_id,
    )
    transfer = prepared.vertex_field_transfer(
        prepared, identity, binding, field_name="temperature"
    )
    nodal_probe = jnp.linspace(-0.7, 0.8, prepared.dof_count, dtype=jnp.float64)
    dual = transfer.dual_pullback_operator
    if dual is None:
        raise AssertionError(
            "periodic_vem.transfer.dual must retain its pullback; got None"
        )
    layout = transfer.source.layout
    if not isinstance(layout, phx.discretization.EntityDofLayout):
        raise AssertionError(
            "periodic_vem.transfer.layout must be a quotient entity layout; got an incompatible layout"
        )
    return {
        "periodic_present": construction.mesh.periodic_topology is not None,
        "dof_count": prepared.dof_count,
        "quotient_vertex_count": periodic.quotient.entities(0).count,
        "physical_vertex_count": construction.mesh.connectivity.vertex_count,
        "nodal_probe": array_record(nodal_probe),
        "primal_probe": array_record(transfer.primal_operator.mv(nodal_probe)),
        "dual_probe": array_record(dual.mv(nodal_probe)),
        "layout_entity_set_id": layout.entity_set_id,
        "quotient_entity_set_id": periodic.quotient.entities(0).entity_set_id,
    }


def _physical_load_record(
    construction: PolyhedralConstruction,
    prepared: PreparedPolyhedralH1VirtualElement3D,
    pinned_dof: int,
) -> dict[str, Any]:
    physical_force = (
        jnp.arange(construction.mesh.connectivity.cell_count, dtype=jnp.float64) + 1
    )
    physical_load = prepared.assemble_cellwise_constant_load(physical_force)
    incompatible_physical_load = _value_error_message(
        lambda: prepared.compatible_pinned_load(
            physical_load, pinned_dof, compatibility_tolerance=1e-12
        )
    )
    return {
        "physical_load": array_record(physical_load),
        "jitted_physical_load": array_record(
            eqx.filter_jit(_assemble_cellwise_constant_load)(prepared, physical_force)
        ),
        "physical_load_sum": float(jnp.sum(physical_load)),
        "physical_force_integral": float(
            jnp.vdot(physical_force, prepared.evidence.cell_volumes)
        ),
        "incompatible_physical_load": incompatible_physical_load,
    }


def _affine_energy_record(
    construction: PolyhedralConstruction,
    prepared: PreparedPolyhedralH1VirtualElement3D,
) -> dict[str, Any]:
    connectivity = construction.mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolyhedralConnectivity):
        raise AssertionError(
            "periodic_vem.connectivity must be polyhedral; got an incompatible connectivity owner"
        )
    offsets = np.asarray(connectivity.cell_vertex_offsets)
    vertices = np.asarray(connectivity.cell_vertex_values)
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(construction.geometry),
    ) as budget:
        source_points = np.asarray(
            construction.geometry.source_coordinates(budget=budget), dtype=np.float64
        )
    affine_energy = np.zeros(3, dtype=np.float64)
    for arity, tensors in zip(
        sorted(set(np.diff(offsets))), prepared.operator.local_tensors(), strict=True
    ):
        cells = np.flatnonzero(np.diff(offsets) == arity)
        for cell, tensor in zip(cells, np.asarray(tensors), strict=True):
            local_points = source_points[vertices[offsets[cell] : offsets[cell + 1]]]
            affine_energy += np.diag(local_points.T @ tensor @ local_points)
    return {
        "maximum_reproduction_defect": prepared.evidence.maximum_reproduction_defect,
        "volume": float(jnp.sum(prepared.evidence.cell_volumes)),
        "affine_energy": array_record(affine_energy),
        "constant_image": array_record(
            prepared.mv(jnp.ones(prepared.dof_count, dtype=jnp.float64))
        ),
    }


def _diffusion_continuation(
    construction: PolyhedralConstruction,
    prepared: PreparedPolyhedralH1VirtualElement3D,
    state: ArrayLike | None,
    cell_coefficients: ArrayLike | None,
    pinned_dof: int,
) -> tuple[Array, np.ndarray, dict[str, Any]]:
    target = (
        jnp.linspace(0.0, 1.0, prepared.dof_count, dtype=jnp.float64)
        if state is None
        else jnp.asarray(state) * 1.25
    )
    target = target - target[0]
    local_energy = sum(
        float(jnp.einsum("ci,cij,cj->", target[gather], tensor, target[gather]))
        for gather, tensor in zip(
            prepared.operator.gathers, prepared.operator.local_tensors(), strict=True
        )
    )
    coefficients = (
        np.linspace(
            1.0,
            1.5,
            construction.mesh.connectivity.cell_count,
            dtype=np.float64,
        )
        if cell_coefficients is None
        else np.asarray(cell_coefficients)
    )
    diffusion = prepared.bind_scalar_diffusion(coefficients)
    load = diffusion.mv(target)
    rhs = prepared.compatible_pinned_load(load, 0, compatibility_tolerance=1e-11)
    operator = prepared.pinned_diffusion_operator(0, cell_coefficients=coefficients)
    result = phx.linalg.solve(
        phx.linalg.LinearSystem(operator),
        rhs,
        policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.ConjugateGradient(),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1e-10, absolute=1e-12, max_steps=100
            ),
        ),
    )
    full = jnp.concatenate((jnp.zeros(1, dtype=jnp.float64), result.value))
    probe = jnp.linspace(-0.3, 0.9, prepared.dof_count, dtype=jnp.float64)
    incompatible_unit_load = _value_error_message(
        lambda: prepared.compatible_pinned_load(
            jnp.ones(prepared.dof_count, dtype=jnp.float64),
            0,
            compatibility_tolerance=0,
        )
    )
    nonpositive_diffusion = _value_error_message(
        lambda: prepared.bind_scalar_diffusion(
            np.zeros(construction.mesh.connectivity.cell_count, dtype=np.float64)
        )
    )
    record = {
        "pinned_dof": pinned_dof,
        "global_energy": float(jnp.vdot(target, prepared.mv(target))),
        "local_energy": local_energy,
        "gather_aliases": [
            original is rebound
            for original, rebound in zip(
                prepared.operator.gathers, diffusion.gathers, strict=True
            )
        ],
        "successful": bool(result.successful),
        "full": array_record(full),
        "target": array_record(target),
        "diffusion_image": array_record(diffusion.mv(full)),
        "load": array_record(load),
        "primal_pairing": float(jnp.vdot(probe, diffusion.mv(full))),
        "dual_pairing": float(jnp.vdot(diffusion.transpose_mv(probe), full)),
        "incompatible_unit_load": incompatible_unit_load,
        "nonpositive_diffusion": nonpositive_diffusion,
    }
    return full, coefficients, record


def solve_and_continue(
    construction: PolyhedralConstruction,
    state: ArrayLike | None = None,
    cell_coefficients: ArrayLike | None = None,
    pinned_dof: int = 0,
) -> tuple[PreparedPolyhedralH1VirtualElement3D, Array, np.ndarray, dict[str, Any]]:
    prepared = prepare_polyhedral_h1_virtual_element_3d(construction.mesh)
    record = _quotient_transfer_record(construction, prepared)
    record.update(_physical_load_record(construction, prepared, pinned_dof))
    record.update(_affine_energy_record(construction, prepared))
    full, coefficients, diffusion = _diffusion_continuation(
        construction, prepared, state, cell_coefficients, pinned_dof
    )
    record.update(diffusion)
    return prepared, full, coefficients, record


def restore_and_continue(
    root: PeriodicArchiveRoot,
) -> tuple[PreparedPolyhedralH1VirtualElement3D, Array, np.ndarray, dict[str, Any]]:
    recipe = root["recipe"]
    mesh, geometry = root["mesh"], root["geometry"]
    state, coefficients = root["state"], root["coefficients"]
    gauge, history = root["pinned_dof"], root["history"]
    generation_schedule = NativePolyhedralSchedule(**recipe["controls"])
    with NativeExecutionScope(generation_schedule.native_limits()) as scope:
        construction = generate_polyhedral_volume(
            recipe["domain"],
            sites=recipe["sites"],
            weights=recipe["weights"],
            schedule=generation_schedule,
            periodic_constraints=recipe["constraints"],
            scope=scope,
        )
    prepared, full, continued_coefficients, record = solve_and_continue(
        construction, history[-1], np.asarray(coefficients) * 1.25, gauge
    )
    with coordinate_reference_budget(
        queries_upper=(
            cell_geometry_preparation_query_upper(construction.geometry)
            + cell_geometry_preparation_query_upper(geometry)
        ),
    ) as budget:
        record["restore"] = {
            "history_last": array_record(history[-1]),
            "state": array_record(state),
            "regenerated_mesh_id": construction.mesh.mesh_id,
            "archived_mesh_id": mesh.mesh_id,
            "regenerated_source_coordinates": coordinate_record(
                construction.geometry.source_coordinates(budget=budget)
            ),
            "archived_source_coordinates": coordinate_record(
                geometry.source_coordinates(budget=budget)
            ),
            "regenerated_coordinates": array_record(construction.mesh.coordinates),
            "archived_coordinates": array_record(mesh.coordinates),
        }
    return prepared, full, continued_coefficients, record


def periodic_vem(path: str, content_id: str) -> dict[str, Any]:
    root = read_meshing_source_closure(path, expected_content_id=content_id)
    _, _, _, record = restore_and_continue(root[0])
    return record
