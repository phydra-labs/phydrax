from __future__ import annotations

import os
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._execution_runtime import ExecutionGroup
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.meshing._assembly import MeshAssembly
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._overset import OversetPartSpec, OversetPolicy
from phydrax.meshing._periodic import PeriodicPointOrbits
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeCompliancePolicy, UniformSizeControl
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_periodic import (
    NativePeriodicSource,
    prepare_native_periodic,
)
from phydrax.solver._finite_volume_archive_context import (
    _finite_volume_archive_build_context,
    _finite_volume_archive_execution_group,
    _finite_volume_archive_rebuild_context_views,
    _finite_volume_archive_target_receptor_spec,
)


def _owner_local_periodic_assembly(
    name: str,
    fractional_points: np.ndarray,
    point_owners: np.ndarray,
    execution_group: ExecutionGroup,
) -> MeshAssembly:
    revision = f"{name}-revision-1"
    cell = PeriodicCell(np.asarray(((1.0, 0.2), (0.3, 1.1)), dtype=np.float64))
    point_ids = np.asarray((91, 4, 53, 17), dtype=np.int64)
    local_rows = point_owners == jax.process_index()
    source = NativePeriodicSource(
        PeriodicPointOrbits(
            cell,
            np.asarray(cell.cartesian(fractional_points[local_rows]), dtype=np.float64),
            point_ids[local_rows],
        ),
        name,
        revision,
        point_owner_ranks=point_owners[local_rows],
    )
    domain_scope = MeshingScope(
        name,
        revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            2,
            CellFamilyPolicy(required=("triangle",)),
        ),
        domain_scope,
        size_controls=(UniformSizeControl(domain_scope, 0.35),),
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
    )
    return prepare_native_periodic(
        source,
        specification,
        NativeMeshingOptions("periodic_delaunay"),
        phx.SpatialCoordinateContract.si(),
    ).execute(execution_group=execution_group)


def _finite_volume_cell_state_bank_revision(bank_spec: Any, values: Any, /) -> str:
    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint

    typed_key = hasattr(values, "dtype") and jax.dtypes.issubdtype(
        values.dtype, jax.dtypes.prng_key
    )
    fingerprint_values = jax.random.key_data(values) if typed_key else values

    return canonical_fingerprint(
        {
            "kind": "finite-volume-owner-local-cell-state-bank",
            "entry_id": bank_spec.entry_id,
            "cell_ids": tuple(bank_spec.cell_ids),
            "cell_axis": bank_spec.cell_axis,
            "values": array_tree_fingerprint(fingerprint_values),
            "key_impl": str(jax.random.key_impl(values)) if typed_key else None,
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


def _finite_volume_archive_driven_source(
    time: Any, state: Any, coordinates: Any, args: Any, /
) -> Any:
    from phydrax.discretization.finite_volume._driven_euler_source import (
        DrivenEulerSource,
    )

    return DrivenEulerSource(1.0)(time, state, coordinates, args)


_FV_ARCHIVE_SOURCE_ID = "test.owner-local-euler-force-and-heating"

_FV_COLD_BANK_NAMES = ("material", "history", "keys", "clock")
_FV_COLD_BANK_IDS = tuple(f"test/owner-local-cell-{name}" for name in _FV_COLD_BANK_NAMES)


def _fv_cold_initial_banks(ids: np.ndarray, /) -> tuple[Any, ...]:
    from phydrax.discretization.finite_volume._driven_euler_source import (
        driven_euler_cell_keys,
    )

    unsigned = ids.astype(np.uint64)
    high = jnp.asarray((unsigned >> np.uint64(32)).astype(np.uint32))
    low = jnp.asarray(unsigned.astype(np.uint32))
    keys = driven_euler_cell_keys(jax.random.key(731), high, low)
    material = np.stack(
        (0.2 + (ids % 7) * 0.01, 0.4 + (ids % 5) * 0.01, np.full(ids.shape, 0.1)),
        axis=1,
    )
    return (
        jnp.asarray(material),
        jnp.asarray(0.1 + (ids % 11) * 0.01),
        keys,
        jnp.asarray(0.01 + (ids % 3) * 0.001),
    )


def _fv_cold_rows(runtime: Any, ids: np.ndarray, /) -> np.ndarray:
    geometry_ids = np.asarray(
        runtime.dynamics.discretization.cell_global_ids, dtype=np.int64
    )
    row_by_id = {int(identifier): row for row, identifier in enumerate(geometry_ids)}
    return np.asarray([row_by_id[int(identifier)] for identifier in ids], dtype=np.int32)


def _fv_cold_full_banks(runtime: Any, ids: np.ndarray, banks: tuple[Any, ...], /) -> Any:
    from phydrax.discretization.finite_volume._driven_euler_source import (
        DrivenEulerSourceArgs,
    )

    rows = _fv_cold_rows(runtime, ids)
    count = int(runtime.dynamics.discretization.cell_global_ids.shape[0])
    material, history, keys, clock = banks
    return DrivenEulerSourceArgs(
        jnp.zeros((count, 3), dtype=material.dtype).at[rows].set(material),
        jnp.zeros((count,), dtype=history.dtype).at[rows].set(history),
        jnp.broadcast_to(jax.random.key(0), (count,)).at[rows].set(keys),
        jnp.zeros((count,), dtype=clock.dtype).at[rows].set(clock),
        jnp.zeros((count,), dtype=jnp.bool_).at[rows].set(True),
    )


def _fv_cold_compact_banks(banks: Any, rows: np.ndarray, /) -> tuple[Any, ...]:
    return banks.material[rows], banks.history[rows], banks.keys[rows], banks.clock[rows]


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
        source_id=_FV_ARCHIVE_SOURCE_ID,
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
    bundle = _finite_volume_archive_build_context(
        context,
        group,
        validate_expected=False,
        sources={_FV_ARCHIVE_SOURCE_ID: _finite_volume_archive_driven_source},
    )
    context = bundle["context"]
    source_runtime = bundle["source_runtime"]
    source_views = bundle["source_views"]
    source_geometry = bundle["source_geometry"]
    source_cell_ids = bundle["source_cell_ids"]
    receptor_view = source_views[receptor_name]
    from phydrax.meshing._overset_exchange import _owner_local_all_gather_rows

    def owned_blanking_rows(connectivity: Any, view: Any, axis: str) -> np.ndarray:
        blanking = connectivity.blanking_of(receptor_name)
        status_by_id = dict(
            zip(
                np.asarray(blanking.cell_ids, dtype=np.int64),
                np.asarray(blanking.cell_status, dtype=np.int64),
                strict=True,
            )
        )
        rows = np.asarray(
            [
                (int(identifier), int(status_by_id[int(identifier)]))
                for identifier in np.asarray(view.owned_cell_ids, dtype=np.int64)
            ],
            dtype=np.int64,
        ).reshape((-1, 2))
        gathered = _owner_local_all_gather_rows(rows, group.devices, axis)
        combined = np.concatenate(gathered, axis=0)
        return combined[np.argsort(combined[:, 0], kind="stable")]

    def scoped_edge_keys(connectivity: Any, view: Any, axis: str) -> np.ndarray:
        spec = next(
            item for item in connectivity.specs if item.part_name == receptor_name
        )
        vertex_ids = np.asarray(view.vertex_ids, dtype=np.int64)
        loops = (
            ()
            if spec.boundary is None
            else view.entity_vertex_loops(1, spec.boundary.entity_ids)
        )
        rows = np.asarray(
            [
                sorted(map(int, vertex_ids[np.asarray(loop, dtype=np.int64)]))
                for loop in loops
            ],
            dtype=np.int64,
        ).reshape((-1, 2))
        gathered = _owner_local_all_gather_rows(rows, group.devices, axis)
        return np.unique(np.concatenate(gathered, axis=0), axis=0)

    # Exercise independent scoped declarations against the actual graph
    # migration, without changing the default archived writer/read case.
    source_part = bundle["source_assembly"].part(receptor_name)
    target_part = bundle["target_connectivity"].assembly.part(receptor_name)
    owned_ids = np.asarray(receptor_view.owned_cell_ids, dtype=np.int64)
    excluded = (
        source_part.scope(receptor_view.topological_dimension, owned_ids[:1])
        if rank == 1 and owned_ids.size
        else None
    )
    nondefault = OversetPartSpec(
        receptor_name,
        boundary=receptor_boundary,
        wall=receptor_boundary,
        excluded=excluded,
        solid_index=3,
        priority=7,
        owner_rank=1,
        image_rotation=np.asarray(((0.0, -1.0), (1.0, 0.0))),
        image_translation=np.asarray((0.25, -0.5)),
    )
    migrated = _finite_volume_archive_target_receptor_spec(
        nondefault, source_part, target_part, group
    )
    assert migrated.solid_index == 3
    assert migrated.priority == 7 and migrated.owner_rank == 1
    np.testing.assert_array_equal(migrated.image_rotation, nondefault.image_rotation)
    np.testing.assert_array_equal(
        migrated.image_translation, nondefault.image_translation
    )
    assert migrated.solid_query is nondefault.solid_query

    def scope_keys(spec: Any, view: Any, name: str, axis: str) -> np.ndarray:
        scope = getattr(spec, name)
        if name == "excluded":
            rows = np.asarray(
                () if scope is None else scope.entity_ids, dtype=np.int64
            ).reshape((-1, 1))
        else:
            vertices = np.asarray(view.vertex_ids, dtype=np.int64)
            loops = () if scope is None else view.entity_vertex_loops(1, scope.entity_ids)
            rows = np.asarray(
                [
                    sorted(map(int, vertices[np.asarray(loop, dtype=np.int64)]))
                    for loop in loops
                ],
                dtype=np.int64,
            ).reshape((-1, 2))
        return np.unique(
            np.concatenate(
                _owner_local_all_gather_rows(rows, group.devices, axis), axis=0
            ),
            axis=0,
        )

    for name in ("boundary", "wall", "excluded"):
        before = scope_keys(nondefault, receptor_view, name, f"fv_scope_{name}_before")
        after = scope_keys(
            migrated, bundle["target_view"], name, f"fv_scope_{name}_after"
        )
        assert before.shape[0] > 0
        np.testing.assert_array_equal(before, after)
    invalid = OversetPartSpec(receptor_name, excluded=receptor_boundary)
    stale = OversetPartSpec(receptor_name, boundary=receptor_boundary)
    for spec, source in ((invalid, source_part), (stale, target_part)):
        try:
            _finite_volume_archive_target_receptor_spec(spec, source, target_part, group)
        except ValueError:
            pass  # The invalid/stale scope must be refused.
        else:
            raise RuntimeError("Cold FV invalid or stale scope was accepted.")

    np.testing.assert_array_equal(
        scoped_edge_keys(
            bundle["source_connectivity"], receptor_view, "fv_archive_source_scope"
        ),
        scoped_edge_keys(
            bundle["target_connectivity"],
            bundle["target_view"],
            "fv_archive_target_scope",
        ),
    )

    np.testing.assert_array_equal(
        owned_blanking_rows(
            bundle["source_connectivity"], receptor_view, "fv_archive_source_blanking"
        ),
        owned_blanking_rows(
            bundle["target_connectivity"],
            bundle["target_view"],
            "fv_archive_target_blanking",
        ),
    )
    # The writer only needs the source runtime and the serialized target recipe;
    # release the rebuilt target dynamics before compiling either PDE step.
    del bundle
    cell_ids = np.asarray(source_geometry.cell_global_ids, dtype=np.int64)
    primitive = np.zeros((cell_ids.size, 4), dtype=np.float64)
    primitive[:, 0] = 1.0 + (cell_ids % 19) * 0.01
    primitive[:, 3] = 1.0
    primitive[:, 1] = 0.2 + (cell_ids % 5) * 0.01
    primitive[:, 2] = -0.1 + (cell_ids % 3) * 0.01
    input_primitive = primitive.copy()
    owned = np.asarray(source_runtime.owner_local_owned_cell_mask, dtype=np.bool_)
    input_primitive[~owned, 0] = 3.0
    input_primitive[~owned, 3] = 3.0
    initial = source_runtime.initialize_state(
        system.primitive_to_conserved(jnp.asarray(input_primitive)),
        0.0,
        1.0e-4,
    )
    from phydrax.discretization.finite_volume._driven_euler_source import (
        advance_driven_euler_banks,
    )

    history_cell_ids = np.sort(np.asarray(receptor_view.owned_cell_ids, dtype=np.int64))
    initial_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray([history_cell_ids.size], dtype=np.int32)
        )
    ).reshape((-1,))
    if (
        initial_counts.shape != (3,)
        or np.any(initial_counts[:2] <= 0)
        or initial_counts[2] != 0
    ):
        raise ValueError(
            "Cold FV source partition must have two active owners and one empty owner."
        )
    rows = _fv_cold_rows(source_runtime, history_cell_ids)
    initial_banks = _fv_cold_full_banks(
        source_runtime, history_cell_ids, _fv_cold_initial_banks(history_cell_ids)
    )
    accepted, accepted_banks = source_runtime.advance_with_auxiliary(
        initial,
        initial_banks,
        advance_driven_euler_banks,
        args=initial_banks.select_rows(source_runtime.dynamics.source_cell_indices),
    )
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
    bank_values = _fv_cold_compact_banks(accepted_banks, rows)
    bank_specs = tuple(
        phx.solver.FiniteVolumeOwnerLocalCellStateBankSpec(entry_id, history_cell_ids, 0)
        for entry_id in _FV_COLD_BANK_IDS
    )
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
    bank_entries = tuple(
        CompositionEntry(
            value,
            entry_id=spec.entry_id,
            role=role,
            owner_id="test.native-owner-local-fv",
            structure_id=token.runtime_state.content_state.geometry_layout_id,
            revision_id=_finite_volume_cell_state_bank_revision(spec, value),
            semantics_id=f"driven-euler-{name}",
            dependencies=(
                context_entry.binding("structure"),
                state_entry.binding("revision"),
            ),
        )
        for name, role, spec, value in zip(
            _FV_COLD_BANK_NAMES,
            ("model-parameter", "history", "rng", "model-state"),
            bank_specs,
            bank_values,
            strict=True,
        )
    )
    schema_revision = canonical_fingerprint(
        {
            "kind": "finite-volume-cell-state-bank-layouts",
            "schemas": tuple(
                (
                    spec.entry_id,
                    tuple(map(int, np.asarray(spec.cell_ids).reshape(-1))),
                    spec.cell_axis,
                )
                for spec in bank_specs
            ),
        }
    )
    schema_entry = CompositionEntry(
        bank_specs,
        entry_id="finite-volume/cell-state-bank-layouts",
        role="discretization",
        owner_id="phydrax.solver",
        structure_id=token.runtime_state.content_state.geometry_layout_id,
        revision_id=schema_revision,
        semantics_id="finite-volume-cell-state-bank-layouts",
        dependencies=(state_entry.binding("revision"),),
    )
    artifact = Composition(
        (context_entry, state_entry, *bank_entries, schema_entry),
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
    reference, reference_banks = source_runtime.advance_with_auxiliary(
        accepted.runtime_state,
        accepted_banks,
        advance_driven_euler_banks,
        args=accepted_banks.select_rows(source_runtime.dynamics.source_cell_indices),
    )
    if not bool(np.asarray(reference.accepted)):
        raise ValueError("Uninterrupted FV reference step was rejected.")
    reference_values = _fv_cold_compact_banks(reference_banks, rows)
    np.savez(
        Path(artifact_root) / f"owner-local-fv-reference-{rank}.npz",
        ids=history_cell_ids,
        content=np.asarray(reference.runtime_state.content_state.conservative_content)[
            rows
        ],
        material=np.asarray(reference_values[0]),
        history=np.asarray(reference_values[1]),
        keys=np.asarray(jax.random.key_data(reference_values[2])),
        clock=np.asarray(reference_values[3]),
    )
    multihost_utils.sync_global_devices("finite-volume-context-archive-written")
    print(
        json.dumps(
            {
                "rank": rank,
                "pid": os.getpid(),
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
    from phydrax.solver._finite_volume_archive_context import (
        _rebuild_finite_volume_runtime_state_archive_recipe,
    )
    from phydrax.solver._finite_volume_runtime import (
        FiniteVolumeOwnerLocalCellStateBankSpec,
        FiniteVolumeValidatedRuntimeState,
    )

    rank = jax.process_index()
    archive_path = Path(artifact_root) / f"owner-local-fv-state-{rank}.phx"
    restored_context: dict[str, Any] = {}

    def restore_archive(value: Any, /) -> Any:
        expected_entries = {
            "finite-volume/context",
            "finite-volume/state",
            *_FV_COLD_BANK_IDS,
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
            validated_context,
            group,
            validate_expected=True,
            sources={_FV_ARCHIVE_SOURCE_ID: _finite_volume_archive_driven_source},
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
        raw_schemas = entries["finite-volume/cell-state-bank-layouts"].value
        if (
            not isinstance(raw_schemas, tuple)
            or len(raw_schemas) != len(_FV_COLD_BANK_IDS)
            or any(
                type(schema) is not FiniteVolumeOwnerLocalCellStateBankArchiveRecipe
                for schema in raw_schemas
            )
        ):
            raise TypeError("Cold FV archive lost its explicit cell-bank schemas.")
        bank_specs = tuple(
            FiniteVolumeOwnerLocalCellStateBankSpec(
                schema.entry_id, schema.cell_ids, schema.cell_axis
            )
            for schema in raw_schemas
        )
        if tuple(spec.entry_id for spec in bank_specs) != _FV_COLD_BANK_IDS:
            raise ValueError("Cold FV bank schema identities changed.")
        schema_revision = canonical_fingerprint(
            {
                "kind": "finite-volume-cell-state-bank-layouts",
                "schemas": tuple(
                    (
                        spec.entry_id,
                        tuple(map(int, np.asarray(spec.cell_ids).reshape(-1))),
                        spec.cell_axis,
                    )
                    for spec in bank_specs
                ),
            }
        )
        for name, spec in zip(_FV_COLD_BANK_NAMES, bank_specs, strict=True):
            entry = entries[spec.entry_id]
            if (
                spec.cell_axis != 0
                or entry.value.shape[0] != np.asarray(spec.cell_ids).size
                or entry.revision_id
                != _finite_volume_cell_state_bank_revision(spec, entry.value)
                or entry.structure_id != bundle["source_runtime"].geometry_layout_id
                or entry.semantics_id != f"driven-euler-{name}"
            ):
                raise ValueError("Cold FV bank identity or rows changed.")
        if (
            entries["finite-volume/cell-state-bank-layouts"].revision_id
            != schema_revision
            or stored_state_entry.revision_id != token.state_id
            or stored_state_entry.structure_id
            != token.runtime_state.content_state.geometry_layout_id
            or stored_state_entry.semantics_id != bundle["system"].system_id
        ):
            raise ValueError("Cold FV state/context identity changed.")

        replacements = {
            "finite-volume/context": bundle["context"],
            "finite-volume/state": token,
            "finite-volume/cell-state-bank-layouts": bank_specs,
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
                "bank_specs": bank_specs,
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
    bank_records = tuple(composition.entry(entry_id) for entry_id in _FV_COLD_BANK_IDS)
    bank_specs = composition.value("finite-volume/cell-state-bank-layouts")
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
    bank_entries = tuple(
        CompositionEntry(
            record.value,
            entry_id=record.entry_id,
            role=record.role,
            owner_id=record.owner_id,
            structure_id=record.structure_id,
            revision_id=record.revision_id,
            semantics_id=record.semantics_id,
            dependencies=(route.binding("revision"),),
        )
        for record in bank_records
    )
    accepted_composition = Composition(
        (*route_entries, state_entry, *bank_entries),
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
        cell_state_banks=bank_specs,
    )
    accepted_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(bool(np.asarray(repartitioned.accepted))),), dtype=np.int32)
        )
    ).reshape((-1,))
    if not np.all(accepted_flags == 1):
        raise ValueError(
            f"Cold accepted FV graph repartition was refused: {repartitioned.refusal}"
        )
    if (
        repartitioned.successor_runtime is None
        or repartitioned.repartition_receipt is None
    ):
        raise ValueError("Cold FV repartition did not publish its successor runtime.")
    target_ids = np.sort(
        np.asarray(restored_context["target_view"].owned_cell_ids, dtype=np.int64)
    )
    migrated_banks = tuple(
        repartitioned.repartition_receipt.composition.value(entry_id)
        for entry_id in _FV_COLD_BANK_IDS
    )
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
    # The composite layout keeps the donor part; ownership checks below are on
    # the migrated receptor rows.
    receptor_rows = np.isin(
        np.asarray(
            repartitioned.successor_runtime.dynamics.discretization.cell_global_ids,
            dtype=np.int64,
        ),
        np.asarray(restored_context["target_view"].cell_ids, dtype=np.int64),
    )
    receptor_owner_mask = np.asarray(target_owner_mask, dtype=np.bool_) & receptor_rows
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
    if target_ids.size == 0 and np.any(receptor_owner_mask):
        raise ValueError("Cold FV empty target owner retained owned state rows.")
    from phydrax.discretization.finite_volume._driven_euler_source import (
        advance_driven_euler_banks,
    )

    runtime = repartitioned.successor_runtime
    target_rows = _fv_cold_rows(runtime, target_ids)
    full_banks = _fv_cold_full_banks(runtime, target_ids, migrated_banks)
    next_step, next_banks = runtime.advance_with_auxiliary(
        repartitioned.runtime_state,
        full_banks,
        advance_driven_euler_banks,
        args=full_banks.select_rows(runtime.dynamics.source_cell_indices),
    )
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
    compact_next = _fv_cold_compact_banks(next_banks, target_rows)
    if target_ids.size:
        for name, before, after in zip(
            _FV_COLD_BANK_NAMES, migrated_banks, compact_next, strict=True
        ):
            if name == "keys":
                before, after = jax.random.key_data(before), jax.random.key_data(after)
            if np.array_equal(np.asarray(before), np.asarray(after)):
                raise ValueError(f"Accepted FV step failed to advance the {name} bank.")
        references = [
            np.load(Path(artifact_root) / f"owner-local-fv-reference-{owner}.npz")
            for owner in range(3)
        ]
        reference_ids = np.concatenate([item["ids"] for item in references])
        reference_rows = {
            int(identifier): row for row, identifier in enumerate(reference_ids)
        }
        order = np.asarray(
            [reference_rows[int(identifier)] for identifier in target_ids], dtype=np.int64
        )
        np.testing.assert_allclose(
            next_content[target_rows],
            np.concatenate([item["content"] for item in references])[order],
            rtol=2e-10,
            atol=2e-11,
        )
        for name, values in zip(_FV_COLD_BANK_NAMES, compact_next, strict=True):
            actual = jax.random.key_data(values) if name == "keys" else values
            expected = np.concatenate([item[name] for item in references])[order]
            np.testing.assert_allclose(
                np.asarray(actual), expected, rtol=2e-10, atol=2e-11
            )
        for item in references:
            item.close()

    # Perturb each restored input independently and run the real native solve.
    variants = (
        full_banks._replace(material=full_banks.material.at[:, 0].add(0.2)),
        full_banks._replace(history=full_banks.history + 0.5),
        full_banks._replace(
            keys=jax.vmap(lambda key: jax.random.fold_in(key, 93))(full_banks.keys)
        ),
        full_banks._replace(clock=full_banks.clock + 0.5),
    )
    for name, variant in zip(_FV_COLD_BANK_NAMES, variants, strict=True):
        perturbed = runtime.advance(
            repartitioned.runtime_state,
            variant.select_rows(runtime.dynamics.source_cell_indices),
        )
        changed = target_ids.size > 0 and np.any(
            np.abs(
                np.asarray(perturbed.runtime_state.content_state.conservative_content)[
                    target_rows
                ]
                - next_content[target_rows]
            )
            > 1e-13
        )
        flags = np.asarray(
            multihost_utils.process_allgather(np.asarray([changed], dtype=np.int32))
        )
        if not np.all(np.asarray(perturbed.accepted)) or not np.any(flags):
            raise ValueError(
                f"Perturbing the restored {name} bank did not affect physical evolution."
            )
    invalid_args = full_banks._replace(material=full_banks.material.at[:, 1].set(-1.0e12))
    rejected, retained = runtime.advance_with_auxiliary(
        repartitioned.runtime_state,
        full_banks,
        advance_driven_euler_banks,
        args=invalid_args.select_rows(runtime.dynamics.source_cell_indices),
    )
    if (
        bool(np.asarray(rejected.accepted))
        or float(np.asarray(rejected.accepted_step_size)) != 0.0
        or int(np.asarray(rejected.retries)) < 1
    ):
        raise ValueError("FV rejection did not retain its native retry evidence.")
    np.testing.assert_array_equal(
        np.asarray(rejected.runtime_state.content_state.conservative_content),
        prior_content,
    )
    np.testing.assert_array_equal(
        np.asarray(rejected.runtime_state.time),
        np.asarray(repartitioned.runtime_state.time),
    )
    for attribute in (
        "accepted_step",
        "step_size",
        "output_cursor",
        "controller_state",
        "integrator_state",
    ):
        previous = getattr(repartitioned.runtime_state, attribute)
        current = getattr(rejected.runtime_state, attribute)
        jax.tree.map(
            lambda left, right: np.testing.assert_array_equal(
                np.asarray(left), np.asarray(right)
            ),
            previous,
            current,
        )
    for name, before, after in zip(
        _FV_COLD_BANK_NAMES,
        _fv_cold_compact_banks(full_banks, target_rows),
        _fv_cold_compact_banks(retained, target_rows),
        strict=True,
    ):
        if name == "keys":
            before, after = jax.random.key_data(before), jax.random.key_data(after)
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    owner_mask = receptor_owner_mask
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
    global_source = np.sum(
        np.asarray(multihost_utils.process_allgather(np.asarray(source_sum))).reshape(
            (-1, 4)
        ),
        axis=0,
    )
    if global_source[1] <= 0.0 or global_source[3] <= 0.0:
        raise ValueError("Cold FV accepted source ledger lost momentum or heating.")
    global_boundary = np.sum(
        np.asarray(multihost_utils.process_allgather(np.asarray(boundary_sum))).reshape(
            (-1, 4)
        ),
        axis=0,
    )
    global_net = np.sum(
        np.asarray(multihost_utils.process_allgather(np.asarray(net_sum))).reshape(
            (-1, 4)
        ),
        axis=0,
    )
    local_delta = np.sum(
        (next_content - prior_content)[np.asarray(target_owner_mask)], axis=0
    )
    global_delta = np.sum(
        np.asarray(multihost_utils.process_allgather(local_delta)).reshape((-1, 4)),
        axis=0,
    )
    np.testing.assert_allclose(
        global_net, global_source - global_boundary, atol=2e-11, rtol=2e-10
    )
    np.testing.assert_allclose(global_delta, global_net, atol=2e-11, rtol=2e-10)
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
                "pid": os.getpid(),
                "source_runtime_id": context.expected_source_runtime_id,
                "target_connectivity_id": context.expected_target_connectivity_id,
                "source_partition": source_epoch.partition_id,
                "target_partition": target_epoch.partition_id,
                "target_owned_counts": target_counts.tolist(),
                "source_integral": global_source.tolist(),
                "boundary_integral": global_boundary.tolist(),
                "net_integral": global_net.tolist(),
                "content_change": global_delta.tolist(),
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Expected FV archive directory and write/read phase.")
    _native_owner_local_fv_archive_rank_worker(sys.argv[1], sys.argv[2])
