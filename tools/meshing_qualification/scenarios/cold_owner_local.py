from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_AXIS_NAME = "cold_owner_local_fe"
_PROFILE_PARAMETERS = {
    "routine": (
        "native-distributed-routine",
        0.43,
        "routine-periodic-generation-adaptation-repartition",
        (0.5, 0.5, 0.0),
    ),
    "larger": (
        "native-distributed-larger",
        0.28,
        "larger-periodic-generation-adaptation-repartition",
        (0.75, 0.25, 0.0),
    ),
}
_INPUT_FRACTIONAL_POINTS = np.asarray(
    (
        (0.11, 0.13),
        (0.58, 0.17),
        (0.31, 0.62),
        (0.84, 0.77),
        (0.21, 0.88),
        (0.73, 0.39),
        (0.44, 0.42),
        (0.92, 0.05),
    ),
    dtype=np.float64,
)
_INPUT_POINT_IDS = np.asarray((91, 4, 53, 17, 28, 61, 6, 43), dtype=np.int64)
_INPUT_POINT_OWNERS = np.asarray((0, 1, 0, 1, 0, 1, 0, 1), dtype=np.int32)
_CELL_BASIS = np.asarray(((1.2, 0.1), (0.35, 1.05)), dtype=np.float64)


def _source_graph_partition_plan(profile: str, /) -> Any:
    from phydrax.graph._partition import GraphPartitionPlan

    try:
        shares = _PROFILE_PARAMETERS[profile][3]
    except KeyError as error:
        raise ValueError(f"Unknown owner-local FE source profile {profile!r}.") from error
    return GraphPartitionPlan(
        3,
        target_shares=np.asarray(shares, dtype=np.float64),
        maximum_imbalance=1.5,
        require_nonempty=False,
        refinement_passes=4,
    )


def _current_group() -> tuple[Any, int]:
    import jax

    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup

    ranks = (0, 1, 2)
    device_by_process: dict[int, Any] = {}
    for device in sorted(
        jax.devices(), key=lambda value: (value.process_index, value.id)
    ):
        device_by_process.setdefault(device.process_index, device)
    if jax.process_count() != 3 or tuple(device_by_process) != ranks:
        raise RuntimeError(
            "Owner-local FE qualification requires three real CPU process owners."
        )
    rank = jax.process_index()
    devices = tuple(device_by_process[process] for process in ranks)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "native-cold-owner-local-fe",
            ranks,
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=((_AXIS_NAME, 3),),
        ),
        devices,
    )
    return group, rank


def _target_graph_partition_plan() -> Any:
    from phydrax.graph._partition import GraphPartitionPlan

    return GraphPartitionPlan(
        3,
        target_shares=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        maximum_imbalance=1.0,
        require_nonempty=False,
        refinement_passes=0,
    )


def _source_and_specification(
    group: Any,
    profile: str,
    authored_source: Any | None = None,
    graph_partition_plan: Any | None = None,
    authored_specification: Any | None = None,
    *,
    include_size_controls: bool = True,
) -> tuple[Any, Any]:
    import jax

    import phydrax as phx
    from phydrax.discretization._periodic_cell import PeriodicCell
    from phydrax.graph._partition import GraphPartitionPlan
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        SizeCompliancePolicy,
        SurfaceMeshingSpec,
    )
    from phydrax.meshing._periodic import PeriodicPointOrbits
    from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
    from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
    from phydrax.meshing.providers._native_options import NativeMeshingOptions
    from phydrax.meshing.providers._native_periodic import (
        NativePeriodicSource,
        prepare_native_periodic,
    )

    try:
        expected_source_id, size, _, _ = _PROFILE_PARAMETERS[profile]
    except KeyError as error:
        raise ValueError(f"Unknown owner-local FE source profile {profile!r}.") from error
    expected_source_revision = f"{expected_source_id}-revision-1"
    point_owners = _INPUT_POINT_OWNERS
    local_rows = point_owners == jax.process_index()
    cell = PeriodicCell(_CELL_BASIS)
    if authored_source is None:
        source = NativePeriodicSource(
            PeriodicPointOrbits(
                cell,
                np.asarray(
                    cell.cartesian(_INPUT_FRACTIONAL_POINTS[local_rows]),
                    dtype=np.float64,
                ),
                _INPUT_POINT_IDS[local_rows],
            ),
            expected_source_id,
            expected_source_revision,
            point_owner_ranks=point_owners[local_rows],
        )
    else:
        if not isinstance(authored_source, NativePeriodicSource):
            raise TypeError(
                "Cold FE archive did not restore a NativePeriodicSource root."
            )
        source = authored_source
    if (
        source.source_id != expected_source_id
        or source.source_revision != expected_source_revision
        or not isinstance(source.domain, PeriodicPointOrbits)
        or source.domain.cell.cell_id != cell.cell_id
        or source.point_owner_ranks is None
    ):
        raise ValueError("Cold FE replay changed its exact authored periodic source.")
    expected_ids = _INPUT_POINT_IDS[local_rows]
    expected_order = np.argsort(expected_ids, kind="stable")
    source_ids = np.asarray(source.domain.point_ids, dtype=np.int64)
    source_order = np.argsort(source_ids, kind="stable")
    expected_points = np.asarray(
        cell.cartesian(_INPUT_FRACTIONAL_POINTS[local_rows]), dtype=np.float64
    )[expected_order]
    source_points = np.asarray(source.domain.points, dtype=np.float64)[source_order]
    expected_owners = point_owners[local_rows][expected_order]
    source_owners = np.asarray(source.point_owner_ranks, dtype=np.int32)[source_order]
    if (
        not np.array_equal(source_ids[source_order], expected_ids[expected_order])
        or not np.array_equal(source_points, expected_points)
        or not np.array_equal(source_owners, expected_owners)
    ):
        raise ValueError(
            "Cold FE artifact differs from its authored source IDs, points, or owners."
        )
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    if authored_specification is None:
        specification = SurfaceMeshingSpec(
            CellMeshingTarget(2, 2, CellFamilyPolicy(required=("triangle",))),
            scope,
            size_controls=(
                (UniformSizeControl(scope, size, strength=SizeControlStrength.SOFT),)
                if include_size_controls
                else ()
            ),
            size_compliance=SizeCompliancePolicy(
                relative_tolerance=0.8,
                target_statistics=("p50",),
            ),
        )
    else:
        if not isinstance(authored_specification, SurfaceMeshingSpec):
            raise TypeError(
                "Cold FE archive did not restore its typed surface specification."
            )
        specification = authored_specification
        if (
            specification.scope.source_id != source.source_id
            or specification.scope.source_revision != source.source_revision
            or specification.scope.entity_set_id != source.scope_entity_set_id(2)
            or not np.array_equal(
                np.sort(np.asarray(specification.scope.entity_ids, dtype=np.int64)),
                np.sort(np.asarray(source.scope_entity_ids(2), dtype=np.int64)),
            )
        ):
            raise ValueError(
                "Archived surface specification is stale for the exact periodic source."
            )
    if bool(specification.size_controls) != include_size_controls:
        raise ValueError(
            "Archived periodic specification changed its base/adaptation role."
        )
    if graph_partition_plan is not None and not isinstance(
        graph_partition_plan, GraphPartitionPlan
    ):
        raise TypeError(
            "graph_partition_plan must be the exact archived GraphPartitionPlan."
        )
    plan = prepare_native_periodic(
        source,
        specification,
        NativeMeshingOptions("periodic_delaunay"),
        phx.SpatialCoordinateContract.si(),
    )
    assembly = (
        plan.execute(execution_group=group)
        if graph_partition_plan is None
        else plan.execute(
            execution_group=group,
            graph_partition_plan=graph_partition_plan,
        )
    )
    return assembly, specification


def _owner_local_views(
    group: Any,
    profile: str,
    authored_source: Any | None = None,
    authored_specification: Any | None = None,
    authored_adaptation_specification: Any | None = None,
    source_graph_plan: Any | None = None,
    target_graph_plan: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    import jax

    import phydrax as phx
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.graph._partition import GraphPartitionPlan
    from phydrax.meshing._device_adaptation import OwnerLocalPeriodicAdaptiveResult
    from phydrax.meshing._distribution import OwnerLocalGraphPartitionEvidence
    from phydrax.meshing.providers._native_periodic import (
        _repartition_owner_local_periodic_assembly,
    )

    source_graph_plan = (
        _source_graph_partition_plan(profile)
        if source_graph_plan is None
        else source_graph_plan
    )
    if not isinstance(source_graph_plan, GraphPartitionPlan):
        raise TypeError("The archived source graph request is not GraphPartitionPlan.")
    source_assembly, specification = _source_and_specification(
        group,
        profile,
        authored_source,
        source_graph_plan,
        authored_specification,
        include_size_controls=False,
    )
    source_view = source_assembly.parts[0].carrier
    if not isinstance(source_view, OwnerLocalCellGeometry):
        raise TypeError(
            "Native periodic production did not return a compact source view."
        )
    if authored_source is None:
        authored_source = source_view.source_value
    if authored_source is None or source_view.source_value is not authored_source:
        raise ValueError(
            "The base FE view did not retain the exact authored source object."
        )
    adaptation_assembly, adaptation_specification = _source_and_specification(
        group,
        profile,
        authored_source,
        source_graph_plan,
        authored_adaptation_specification,
        include_size_controls=True,
    )
    target_graph_plan = (
        _target_graph_partition_plan() if target_graph_plan is None else target_graph_plan
    )
    if not isinstance(target_graph_plan, GraphPartitionPlan):
        raise TypeError("The archived repartition request is not GraphPartitionPlan.")
    adaptation_view = adaptation_assembly.parts[0].carrier
    if (
        not isinstance(adaptation_view, OwnerLocalCellGeometry)
        or adaptation_view.source_value is not source_view.source_value
    ):
        raise TypeError(
            "Periodic adaptation did not retain the exact authored source object."
        )
    pre_graph_adaptive_result = adaptation_view.adaptive_result
    if (
        not isinstance(pre_graph_adaptive_result, OwnerLocalPeriodicAdaptiveResult)
        or pre_graph_adaptive_result.source_topology_id != source_view.source_topology_id
        or pre_graph_adaptive_result.target_topology_id
        != adaptation_view.source_topology_id
        or pre_graph_adaptive_result.specification_id
        != adaptation_specification.specification_id
    ):
        raise RuntimeError(
            "The periodic adaptation does not connect the archived base mesh to its exact target."
        )
    local_rank = group.spec.process_indices.index(jax.process_index())
    _, adaptation_compiled, _, _ = _compile_owner_local_problem(
        adaptation_view, group, local_rank
    )
    graph_evidence = phx.solver.partition_owner_local_finite_element_graph(
        adaptation_view,
        adaptation_compiled,
        target_graph_plan,
        group,
        axis_name=f"{_AXIS_NAME}_{profile}_fe_graph",
    )
    if not isinstance(graph_evidence, OwnerLocalGraphPartitionEvidence):
        raise TypeError(
            "FE-weighted periodic graph partition returned no typed evidence."
        )
    target_assembly = _repartition_owner_local_periodic_assembly(
        adaptation_assembly,
        graph_evidence,
        group,
        current_specification=adaptation_specification,
        graph_partition_plan=target_graph_plan,
        axis_name=f"{_AXIS_NAME}_{profile}_geometry",
    )
    target_view = target_assembly.parts[0].carrier
    if not isinstance(target_view, OwnerLocalCellGeometry):
        raise TypeError(
            "Native FE-weighted graph repartition did not return a compact target view."
        )
    if target_view.source_value is not source_view.source_value:
        raise TypeError(
            "FE-weighted graph target changed the retained periodic source root."
        )
    target_adaptive_result = target_view.adaptive_result
    if (
        not isinstance(target_adaptive_result, OwnerLocalPeriodicAdaptiveResult)
        or target_adaptive_result.source_topology_id
        != pre_graph_adaptive_result.source_topology_id
        or target_adaptive_result.target_topology_id
        != pre_graph_adaptive_result.target_topology_id
        or target_adaptive_result.specification_id
        != pre_graph_adaptive_result.specification_id
        or target_adaptive_result.transition_id != pre_graph_adaptive_result.transition_id
        or target_adaptive_result.source_distribution_id != source_view.distribution_id
        or target_adaptive_result.target_distribution_id != target_view.distribution_id
        or target_adaptive_result.target_owner_directory_id != target_view.distribution_id
        or target_adaptive_result.target_part_id != target_view.part_id
        or target_adaptive_result.target_owner_rank != target_view.owner_rank
        or target_adaptive_result.target_local_view_id != target_view.local_view_id
        or (
            None
            if target_adaptive_result.vertex_stencil is None
            else target_adaptive_result.vertex_stencil.stencil_id
        )
        != (
            None
            if pre_graph_adaptive_result.vertex_stencil is None
            else pre_graph_adaptive_result.vertex_stencil.stencil_id
        )
        or (
            None
            if target_adaptive_result.lineage is None
            else target_adaptive_result.lineage.lineage_id
        )
        != (
            None
            if pre_graph_adaptive_result.lineage is None
            else pre_graph_adaptive_result.lineage.lineage_id
        )
        or (
            None
            if target_adaptive_result.vertex_transfer is None
            else target_adaptive_result.vertex_transfer.transfer_id
        )
        != (
            None
            if pre_graph_adaptive_result.vertex_transfer is None
            else pre_graph_adaptive_result.vertex_transfer.transfer_id
        )
    ):
        raise RuntimeError(
            "Graph repartition did not rebind the exact periodic adaptation ancestry to its target view."
        )
    if target_view.graph_migration_evidence is None:
        raise RuntimeError("Current target view omitted graph migration evidence.")
    source_view.reconstruct_local_geometry()
    adaptation_view.reconstruct_local_geometry()
    target_view.reconstruct_local_geometry()
    return (
        source_assembly,
        adaptation_assembly,
        target_assembly,
        source_view,
        adaptation_view,
        target_view,
        specification,
        adaptation_specification,
        source_graph_plan,
        target_graph_plan,
        graph_evidence,
        target_adaptive_result,
        pre_graph_adaptive_result,
    )


def _compile_owner_local_problem(
    view: Any, group: Any, local_rank: int, /
) -> tuple[Any, Any, tuple[Any, ...], tuple[Any, ...]]:
    import jax.numpy as jnp

    import phydrax as phx
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.discretization.fem._distributed import finite_element_dof_identity_plan
    from phydrax.discretization.fem._generic import FiniteElementDiscretization

    if not isinstance(view, OwnerLocalCellGeometry):
        raise TypeError("Owner-local FE compilation requires OwnerLocalCellGeometry.")
    source_value = view.source_value
    source_authority = view.source_geometry_authority
    if (
        source_value is None
        or source_authority is None
        or getattr(source_value, "source_id", None) != view.source_id
        or getattr(source_value, "source_revision", None) != view.source_revision
        or getattr(source_authority, "source_id", None) != view.source_id
        or getattr(source_authority, "source_revision", None) != view.source_revision
    ):
        raise ValueError("Compact FE view lost its source value or geometry authority.")
    cell_scope = view.scope(view.topological_dimension, view.cell_ids)
    if not np.array_equal(
        np.sort(np.asarray(cell_scope.entity_ids, dtype=np.int64)),
        np.sort(np.asarray(view.cell_ids, dtype=np.int64)),
    ):
        raise ValueError("Compact FE view scope changed its locally resident cell IDs.")
    mesh, coordinate_spec = view.reconstruct_local_geometry()
    if mesh is None:
        authority = phx.discretization.EmptyLocalCellMeshAuthority(
            view.source_topology_id,
            view.source_geometry_id,
            view.distribution_id,
            group.spec.group_id,
            local_rank,
            view.owner_directory_data.part_count,
            view.blocks[0].name,
            view.blocks[0].cell_kind,
            view.topological_dimension,
            view.coordinates.shape[1],
        )
        mesh = phx.discretization.CellMesh.empty_local(authority)
    local_cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    if not np.array_equal(local_cell_ids, np.asarray(view.cell_ids, dtype=np.int64)):
        raise ValueError("FE carrier rows differ from the accepted owner-local view.")
    if not np.array_equal(
        np.sort(np.asarray(cell_scope.entity_ids, dtype=np.int64)),
        np.sort(local_cell_ids),
    ):
        raise ValueError(
            "Source-bound FE scope differs from the reconstructed local mesh."
        )
    field_specs = (
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
        phx.discretization.FiniteElementFieldSpec(
            "content", phx.discretization.discontinuous_element("triangle", 0)
        ),
    )
    plan = (
        phx.discretization.FiniteElementPlan(
            mesh, field_specs, coordinate_spec=coordinate_spec
        )
        if coordinate_spec is not None
        else phx.discretization.FiniteElementPlan(mesh, field_specs)
    )
    discretization = plan.prepare()
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError(
            "Owner-local FE plan did not prepare a finite-element discretization."
        )
    periodic_cell = getattr(getattr(source_value, "domain", None), "cell", None)
    if periodic_cell is None:
        raise TypeError("Owner-local P0 source requires its exact periodic lattice.")
    lattice_origin = np.asarray(periodic_cell.origin, dtype=np.float64)
    inverse_vectors = np.asarray(periodic_cell.inverse_vectors, dtype=np.float64)
    if (
        lattice_origin.shape != (2,)
        or inverse_vectors.shape != (2, 2)
        or not np.all(np.isfinite(lattice_origin))
        or not np.all(np.isfinite(inverse_vectors))
    ):
        raise ValueError("The periodic P0 source lattice has invalid coordinates.")

    def periodic_content_source(points: Any, args: object) -> Any:
        del args
        fractional = (jnp.asarray(points) - jnp.asarray(lattice_origin)) @ jnp.asarray(
            inverse_vectors
        )
        return jnp.sin(2.0 * jnp.pi * (fractional[..., 0] + 2.0 * fractional[..., 1]))

    content_rule = phx.integration.ReferenceTriangleRule(
        phx.integration.GaussLegendreRule(20)
    )
    content_coefficient = phx.equations.coefficient(
        periodic_content_source,
        coefficient_id=(
            f"periodic-p0-sine:{view.source_id}:{view.source_revision}:"
            f"{periodic_cell.cell_id}"
        ),
    )
    form = phx.equations.FiniteElementForm(
        "cold-owner-local-positive-mass-source",
        ("u", "content"),
        (
            phx.equations.MassAction("u"),
            phx.equations.SourceAction("u", 1.0),
            phx.equations.MassAction("content", rules={"triangles": content_rule}),
            phx.equations.SourceAction(
                "content",
                content_coefficient,
                action_id="periodic-p0-sine-source",
                rules={"triangles": content_rule},
            ),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    layouts = (
        phx.solver.FiniteElementAcceptedFieldLayout(
            field_specs[0],
            transfer_semantics=phx.solver.FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION,
        ),
        phx.solver.FiniteElementAcceptedFieldLayout(
            field_specs[1],
            transfer_semantics=phx.solver.FiniteElementFieldTransferSemantics.CONSERVATIVE_CELL_AVERAGE,
        ),
    )
    identities = tuple(
        finite_element_dof_identity_plan(
            discretization,
            index,
            field_layout_id=layout.layout_id,
            distribution_id=view.distribution_id,
            topology_id=view.source_topology_id,
            geometry_id=view.source_geometry_id,
            local_rank=local_rank,
            part_count=view.owner_directory_data.part_count,
            cell_ids=view.cell_ids,
            cell_owner_ranks=view.cell_owner_ranks,
            vertex_ids=view.vertex_ids,
            vertex_owner_ranks=view.vertex_owner_ranks,
        )
        for index, layout in enumerate(layouts)
    )
    return discretization, compiled, identities, layouts


def _periodic_p0_content_values(view: Any, discretization: Any, /) -> np.ndarray:
    from phydrax.discretization._view_support import OwnerLocalCellGeometry

    if (
        not isinstance(view, OwnerLocalCellGeometry)
        or len(view.blocks) != 1
        or view.blocks[0].cell_kind != "triangle"
        or view.source_value is None
        or not hasattr(view.source_value.domain, "cell")
    ):
        raise TypeError(
            "Periodic P0 content requires one native periodic triangle block."
        )
    block = view.blocks[0]
    cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
    block_cell_ids = np.asarray(block.cell_ids, dtype=np.int64)
    vertices = np.asarray(block.vertices, dtype=np.int64)
    valid = np.asarray(block.vertex_valid, dtype=np.bool_)
    coordinates = np.asarray(view.vertex_coordinates, dtype=np.float64)
    shifts = (
        None
        if block.cell_vertex_shifts is None
        else np.asarray(block.cell_vertex_shifts, dtype=np.int64)
    )
    basis = np.asarray(view.source_value.domain.cell.vectors, dtype=np.float64)
    if (
        not np.array_equal(block_cell_ids, cell_ids)
        or vertices.shape != (cell_ids.size, 3)
        or valid.shape != vertices.shape
        or not np.all(valid)
        or np.any(vertices < 0)
        or np.any(vertices >= coordinates.shape[0])
        or coordinates.shape[1] != 2
        or shifts is None
        or shifts.shape != (cell_ids.size, 3, 2)
        or basis.shape != (2, 2)
        or not np.all(np.isfinite(basis))
    ):
        raise ValueError(
            "Periodic P0 cell rows differ from the compact triangle geometry."
        )
    fractional_vertices = (coordinates[vertices] + shifts @ basis) @ np.linalg.inv(basis)
    nodes, weights = np.polynomial.legendre.leggauss(20)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    cell_values = np.zeros((cell_ids.size,), dtype=np.float64)
    for s, weight_s in zip(unit_nodes, unit_weights, strict=True):
        for t, weight_t in zip(unit_nodes, unit_weights, strict=True):
            points = (
                fractional_vertices[:, 0]
                + s * (fractional_vertices[:, 1] - fractional_vertices[:, 0])
                + (1.0 - s) * t * (fractional_vertices[:, 2] - fractional_vertices[:, 0])
            )
            cell_values += (
                2.0
                * weight_s
                * weight_t
                * (1.0 - s)
                * np.sin(2.0 * np.pi * (points[:, 0] + 2.0 * points[:, 1]))
            )
    cell_dofs = np.asarray(discretization.dof_maps[1].cell_dofs[0], dtype=np.int64)
    dof_count = int(discretization.dof_maps[1].global_dof_count)
    if (
        cell_dofs.shape != (cell_ids.size, 1)
        or np.unique(cell_dofs).size != cell_dofs.size
        or (cell_dofs.size and (cell_dofs.min() < 0 or cell_dofs.max() >= dof_count))
        or dof_count != cell_ids.size
    ):
        raise ValueError(
            "Periodic conservative P0 layout is not one owned average per cell."
        )
    values = np.empty((dof_count,), dtype=np.float64)
    values[cell_dofs[:, 0]] = cell_values
    if not np.all(np.isfinite(values)):
        raise ValueError("Periodic P0 field contains nonfinite cell averages.")
    return values


def _periodic_p0_owner_integral(
    view: Any,
    discretization: Any,
    identity: Any,
    owned_values: Any,
    /,
) -> float:
    values = np.full(
        (int(discretization.dof_maps[1].global_dof_count),),
        np.nan,
        dtype=np.float64,
    )
    local_indices = np.asarray(identity.owned_local_indices, dtype=np.int64)
    owned_field = np.asarray(owned_values, dtype=np.float64)
    if owned_field.shape != local_indices.shape:
        raise ValueError(
            "Conservative P0 state rows differ from their owner identity plan."
        )
    values[local_indices] = owned_field
    cell_dofs = np.asarray(discretization.dof_maps[1].cell_dofs[0], dtype=np.int64)
    cell_values = values[cell_dofs[:, 0]]
    owned_cells = np.asarray(view.owned, dtype=np.bool_)
    measures = np.asarray(view.cell_measures, dtype=np.float64)
    if (
        owned_cells.shape != cell_values.shape
        or measures.shape != cell_values.shape
        or np.any(~np.isfinite(cell_values[owned_cells]))
        or np.any(~np.isfinite(measures[owned_cells]))
    ):
        raise ValueError(
            "Conservative P0 content cannot be paired with owned cell measures."
        )
    return float(np.dot(measures[owned_cells], cell_values[owned_cells]))


def _solve_physical_problem(
    compiled: Any, local_dof_counts: tuple[int, ...], /
) -> dict[str, Any]:
    import jax.numpy as jnp

    import phydrax as phx

    field_names = tuple(compiled.form.field_names)
    if field_names != ("u", "content") or len(local_dof_counts) != len(field_names):
        raise ValueError(
            "The physical FE solve requires its accepted P1/P0 field blocks."
        )
    if all(count == 0 for count in local_dof_counts):
        empty = tuple(
            jnp.empty((count,), dtype=jnp.float64) for count in local_dof_counts
        )
        return {
            "values": empty[0],
            "content_values": empty[1],
            "residual_norm": 0.0,
            "rhs_norm": 0.0,
            "minimum_value": None,
            "content_minimum_value": None,
            "physical_integral": 0.0,
            "successful": True,
        }
    if any(count == 0 for count in local_dof_counts):
        raise ValueError("The physical FE solve has only a subset of P1/P0 rows.")
    operator, rhs = compiled.linear_system()
    solved = phx.linalg.solve(operator, rhs)
    expanded = compiled.expand(solved.value)
    residual = compiled.residual(solved.value)
    if (
        not isinstance(expanded, tuple)
        or not isinstance(rhs, tuple)
        or not isinstance(residual, tuple)
        or any(len(values) != len(field_names) for values in (expanded, rhs, residual))
    ):
        raise RuntimeError("The mixed P1/P0 solve lost its field-block structure.")
    values = tuple(jnp.asarray(value) for value in expanded)
    field_values = tuple(np.asarray(value, dtype=np.float64) for value in values)
    rhs_values = tuple(np.asarray(value, dtype=np.float64) for value in rhs)
    residual_values = tuple(np.asarray(value, dtype=np.float64) for value in residual)
    if any(
        field_values[index].shape != (local_dof_counts[index],)
        or rhs_values[index].shape != (local_dof_counts[index],)
        or residual_values[index].shape != (local_dof_counts[index],)
        for index in range(len(field_names))
    ):
        raise RuntimeError("The mixed P1/P0 solve rows differ from their FE maps.")
    residual_norm = float(
        np.sqrt(sum(float(np.vdot(value, value).real) for value in residual_values))
    )
    rhs_norm = float(
        np.sqrt(sum(float(np.vdot(value, value).real) for value in rhs_values))
    )
    physical_integral = float(np.dot(rhs_values[0], field_values[0]))
    minimum_value = float(np.min(field_values[0]))
    content_minimum_value = float(np.min(field_values[1]))
    tolerance = 2.0e-9
    successful = bool(np.asarray(solved.successful))
    if (
        not successful
        or any(not np.all(np.isfinite(value)) for value in field_values)
        or minimum_value <= 0.0
        or not np.isfinite(residual_norm)
        or residual_norm > tolerance * max(1.0, rhs_norm)
    ):
        raise RuntimeError(
            "The mixed P1/P0 physical solve failed its positivity or residual oracle."
        )
    return {
        "values": values[0],
        "content_values": values[1],
        "residual_norm": residual_norm,
        "rhs_norm": rhs_norm,
        "minimum_value": minimum_value,
        "content_minimum_value": content_minimum_value,
        "physical_integral": physical_integral,
        "successful": True,
    }


def _periodic_adaptation_transfer_oracle(view: Any, /) -> dict[str, Any]:
    adaptation = view.adaptive_result
    if adaptation is None:
        if np.asarray(view.cell_ids).size or np.asarray(view.vertex_ids).size:
            raise RuntimeError(
                "A nonempty periodic owner view omitted its adaptation result."
            )
        return {
            "accepted": True,
            "empty_owner": True,
            "source_cell_count": 0,
            "target_cell_count": 0,
            "topology_changed": False,
            "shared_representative_winding_count": 0,
            "constant_field_transfer_max_error": 0.0,
            "transition_id": None,
        }
    if adaptation.refusal is not None or not bool(
        np.asarray(adaptation.size_evidence.compliant)
    ):
        raise RuntimeError(
            f"Orbit-synchronized native periodic adaptation was refused: {adaptation.refusal!r}."
        )
    source_cell_ids = np.asarray(adaptation.source_cell_ids, dtype=np.int64)
    target_cell_ids = np.asarray(adaptation.target_cell_ids, dtype=np.int64)
    target_vertex_ids = np.asarray(adaptation.target_vertex_ids, dtype=np.int64)
    if (
        not np.asarray(view.cell_ids).size
        and not np.asarray(view.vertex_ids).size
        and adaptation.vertex_stencil is None
    ):
        return {
            "accepted": True,
            "empty_owner": True,
            "source_cell_count": int(source_cell_ids.size),
            "target_cell_count": int(target_cell_ids.size),
            "topology_changed": False,
            "shared_representative_winding_count": 0,
            "constant_field_transfer_max_error": 0.0,
            "transition_id": adaptation.transition_id,
        }
    stencil = adaptation.vertex_stencil
    if (
        adaptation.transition_id is None
        or stencil is None
        or adaptation.target_local_view_id != view.local_view_id
        or adaptation.target_topology_id != view.source_topology_id
        or adaptation.target_geometry_id != view.source_geometry_id
        or adaptation.target_periodic_topology_id != view.periodic_topology_id
        or not np.array_equal(
            np.sort(target_cell_ids), np.sort(np.asarray(view.cell_ids, dtype=np.int64))
        )
        or not np.array_equal(
            np.sort(target_vertex_ids),
            np.sort(np.asarray(view.vertex_ids, dtype=np.int64)),
        )
    ):
        raise RuntimeError(
            "Periodic adaptation target differs from its compact owner-local view."
        )
    if not stencil.preserves_constants or not np.all(
        np.asarray(stencil.valid, dtype=np.bool_)
    ):
        raise RuntimeError(
            "Periodic FE transfer stencil is incomplete or not constant preserving."
        )

    def keyed_rows(ids: Any, values: Any, /) -> dict[int, tuple[int, ...]]:
        identifiers = np.asarray(ids, dtype=np.int64)
        rows = np.asarray(values, dtype=np.int64)
        if rows.shape[0] != identifiers.size:
            raise RuntimeError(
                "Periodic orbit identity rows do not align with their global IDs."
            )
        return {
            int(identifier): tuple(int(component) for component in row)
            for identifier, row in zip(identifiers, rows, strict=True)
        }

    if (
        keyed_rows(adaptation.target_vertex_ids, adaptation.vertex_source_ids)
        != keyed_rows(view.vertex_ids, view.vertex_source_ids)
        or keyed_rows(adaptation.target_vertex_ids, adaptation.vertex_absolute_windings)
        != keyed_rows(view.vertex_ids, view.vertex_absolute_windings)
        or keyed_rows(adaptation.target_cell_ids, adaptation.cell_metadata.cell_windings)
        != keyed_rows(view.cell_ids, view.cell_windings)
        or keyed_rows(
            adaptation.target_cell_ids,
            np.asarray(adaptation.target_cell_owner_ranks, dtype=np.int64).reshape(
                (-1, 1)
            ),
        )
        != keyed_rows(
            view.cell_ids,
            np.asarray(view.cell_owner_ranks, dtype=np.int64).reshape((-1, 1)),
        )
    ):
        raise RuntimeError(
            "Periodic adaptation changed shared representative or winding identities."
        )
    source_ids = np.asarray(_INPUT_POINT_IDS, dtype=np.int64)
    source_fractional = np.asarray(_INPUT_FRACTIONAL_POINTS, dtype=np.float64)
    stencil_source_ids = np.asarray(stencil.source_global_ids, dtype=np.int64)
    active_source_ids = stencil_source_ids[np.asarray(stencil.valid, dtype=np.bool_)]
    if not np.all(np.isin(active_source_ids, source_ids)):
        raise RuntimeError(
            "Periodic transfer stencil references a source orbit outside the authored point registry."
        )
    wave = np.asarray((1.0, 2.0), dtype=np.float64)
    source_field = np.sin(2.0 * np.pi * (source_fractional @ wave))
    transferred_constant = np.asarray(
        stencil.apply(source_ids, np.ones(source_ids.size, dtype=np.float64)),
        dtype=np.float64,
    )
    transfer_error = float(np.max(np.abs(transferred_constant - 1.0), initial=0.0))
    if transfer_error > 1.0e-12:
        raise RuntimeError("Periodic adaptation failed constant FE transfer.")
    transferred_periodic = np.asarray(
        stencil.apply(source_ids, source_field), dtype=np.float64
    )
    target_vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(view.vertex_ids, dtype=np.int64).tolist()
        )
    }
    target_coordinates = np.asarray(view.vertex_coordinates, dtype=np.float64)[
        np.asarray(
            [
                target_vertex_rows[int(identifier)]
                for identifier in stencil.target_global_ids
            ],
            dtype=np.int64,
        )
    ]
    basis = np.asarray(view.source_value.domain.cell.vectors, dtype=np.float64)
    inverse_basis = np.linalg.inv(basis)
    target_fractional = target_coordinates @ inverse_basis
    target_periodic_exact = np.sin(2.0 * np.pi * (target_fractional @ wave))
    nonconstant_error = float(
        np.max(np.abs(transferred_periodic - target_periodic_exact), initial=0.0)
    )
    hessian_vector = 2.0 * np.pi * np.linalg.solve(basis, wave)
    hessian_bound = float(np.dot(hessian_vector, hessian_vector))
    source_fractional_by_id = {
        int(identifier): source_fractional[row]
        for row, identifier in enumerate(source_ids.tolist())
    }
    interpolation_bound = 0.0
    weights = np.asarray(stencil.weights, dtype=np.float64)
    validity = np.asarray(stencil.valid, dtype=np.bool_)
    source_rows = np.asarray(stencil.source_global_ids, dtype=np.int64)
    for target_row, target_point in enumerate(target_fractional):
        row_mask = validity[target_row]
        source_points = np.asarray(
            [
                source_fractional_by_id[int(value)]
                for value in source_rows[target_row, row_mask]
            ],
            dtype=np.float64,
        )
        fractional_delta = source_points - target_point
        fractional_delta -= np.round(fractional_delta)
        physical_delta = fractional_delta @ basis
        row_bound = (
            0.5
            * hessian_bound
            * float(
                np.sum(
                    np.abs(weights[target_row, row_mask])
                    * np.sum(physical_delta * physical_delta, axis=1)
                )
            )
        )
        interpolation_bound = max(interpolation_bound, row_bound)
    if nonconstant_error > interpolation_bound + 1.0e-12:
        raise RuntimeError(
            "Periodic sinusoidal FE transfer exceeded its independent Hessian interpolation bound."
        )
    key_source_ids = np.asarray(view.cell_key_source_ids, dtype=np.int64)
    key_windings = np.asarray(view.cell_key_relative_windings, dtype=np.int64)
    if key_windings.ndim != 2 or key_windings.shape[0] != key_source_ids.size:
        raise RuntimeError(
            "Periodic winding-key rows differ from their stable source representatives."
        )
    windings_by_source: dict[int, set[tuple[int, ...]]] = {}
    lifted_x_by_source: dict[int, list[float]] = {}
    for source_id, winding in zip(key_source_ids, key_windings, strict=True):
        source_id = int(source_id)
        if source_id not in source_fractional_by_id:
            raise RuntimeError(
                "Periodic winding row references an unknown authored source representative."
            )
        winding_tuple = tuple(int(value) for value in winding.tolist())
        windings_by_source.setdefault(source_id, set()).add(winding_tuple)
        lifted_x_by_source.setdefault(source_id, []).append(
            float(source_fractional_by_id[source_id][0] + winding[0])
        )
    shared_winding_ids = [
        source_id
        for source_id, windings in windings_by_source.items()
        if len(windings) > 1
    ]
    shared_winding_count = len(shared_winding_ids)
    winding_sensitive_separation = max(
        (
            max(lifted_x_by_source[source_id]) - min(lifted_x_by_source[source_id])
            for source_id in shared_winding_ids
        ),
        default=0.0,
    )
    return {
        "accepted": True,
        "empty_owner": False,
        "source_cell_count": int(source_cell_ids.size),
        "target_cell_count": int(target_cell_ids.size),
        "topology_changed": not np.array_equal(
            np.sort(source_cell_ids), np.sort(target_cell_ids)
        ),
        "source_vertex_count": int(source_ids.size),
        "target_vertex_count": int(target_vertex_ids.size),
        "transition_id": adaptation.transition_id,
        "vertex_stencil_id": stencil.stencil_id,
        "periodic_topology_id": adaptation.target_periodic_topology_id,
        "owner_directory_id": adaptation.target_owner_directory_id,
        "constant_field_transfer_max_error": transfer_error,
        "nonconstant_periodic_field": "sin(2*pi*(xi_0+2*xi_1))",
        "nonconstant_interpolation_error": nonconstant_error,
        "nonconstant_interpolation_bound": interpolation_bound,
        "orbit_and_winding_identity_preserved": True,
        "shared_representative_winding_count": shared_winding_count,
        "winding_sensitive_lifted_field_image_separation": winding_sensitive_separation,
    }


def _mesh_manifest_recipe(manifest: Any, /) -> tuple[Any, ...]:
    return (
        "mesh-part-manifest-v1",
        manifest.name,
        manifest.part_id,
        manifest.coordinate_contract.to_dict(),
        manifest.intrinsic_dimension,
        manifest.ambient_dimension,
        manifest.source_topology_id,
        manifest.source_geometry_id,
        manifest.distribution_id,
        manifest.manifest_id,
    )


def _source_composition(
    source_view: Any,
    geometry_manifest: Any,
    restart_manifest: Any,
    field_layouts: tuple[Any, ...],
    specification: Any,
    adaptation_specification: Any,
    source_graph_plan: Any,
    target_graph_plan: Any,
    /,
    *,
    archive_safe: bool = False,
) -> Any:
    from phydrax.graph._partition import GraphPartitionPlan
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._contracts import SurfaceMeshingSpec

    state = restart_manifest.state

    source_value = source_view.source_value
    source_authority = source_view.source_geometry_authority
    if (
        source_value is None
        or source_authority is None
        or getattr(source_value, "source_id", None) != source_view.source_id
        or getattr(source_value, "source_revision", None) != source_view.source_revision
        or getattr(source_authority, "source_id", None) != source_view.source_id
        or getattr(source_authority, "source_revision", None)
        != source_view.source_revision
        or source_view.owner_directory_data.source_manifest_id == ""
    ):
        raise ValueError(
            "Compact owner-local source, geometry authority, or directory data is stale."
        )
    source_entry = CompositionEntry(
        source_value,
        entry_id="mesh/source",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=source_view.source_topology_id,
        revision_id=source_view.source_revision,
        semantics_id=source_value.binding_id,
    )
    if (
        not isinstance(specification, SurfaceMeshingSpec)
        or not isinstance(adaptation_specification, SurfaceMeshingSpec)
        or not isinstance(source_graph_plan, GraphPartitionPlan)
        or not isinstance(target_graph_plan, GraphPartitionPlan)
        or specification.scope.source_id != source_view.source_id
        or specification.scope.source_revision != source_view.source_revision
        or adaptation_specification.scope.source_id != source_view.source_id
        or adaptation_specification.scope.source_revision != source_view.source_revision
    ):
        raise TypeError(
            "Cold FE archive request roots are not the exact typed source plans."
        )
    specification_entry = CompositionEntry(
        specification,
        entry_id="mesh/specification",
        role="discretization",
        owner_id="phydrax.meshing",
        structure_id=specification.specification_id,
        revision_id=specification.specification_id,
        semantics_id="native-periodic-surface-specification",
        dependencies=(
            source_entry.binding("structure"),
            source_entry.binding("revision"),
        ),
    )
    adaptation_specification_entry = CompositionEntry(
        adaptation_specification,
        entry_id="mesh/adaptation-specification",
        role="discretization",
        owner_id="phydrax.meshing",
        structure_id=adaptation_specification.specification_id,
        revision_id=adaptation_specification.specification_id,
        semantics_id="native-periodic-adaptation-surface-specification",
        dependencies=(
            source_entry.binding("structure"),
            source_entry.binding("revision"),
        ),
    )
    source_graph_entry = CompositionEntry(
        source_graph_plan,
        entry_id="mesh/source-graph-plan",
        role="prepared-graph",
        owner_id="phydrax.graph",
        structure_id=source_graph_plan.plan_id,
        revision_id=source_graph_plan.plan_id,
        semantics_id="native-periodic-source-graph-plan",
        dependencies=(specification_entry.binding("revision"),),
    )
    target_graph_entry = CompositionEntry(
        target_graph_plan,
        entry_id="mesh/target-graph-plan",
        role="prepared-graph",
        owner_id="phydrax.graph",
        structure_id=target_graph_plan.plan_id,
        revision_id=target_graph_plan.plan_id,
        semantics_id="native-periodic-repartition-graph-plan",
        dependencies=(adaptation_specification_entry.binding("revision"),),
    )
    manifest_value = (
        _mesh_manifest_recipe(geometry_manifest) if archive_safe else geometry_manifest
    )
    topology_entry = CompositionEntry(
        manifest_value,
        entry_id="mesh/topology",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=geometry_manifest.source_topology_id,
        revision_id=geometry_manifest.manifest_id,
        semantics_id="cell-mesh",
        dependencies=(
            source_entry.binding("structure"),
            source_entry.binding("revision"),
            source_entry.binding("semantics"),
            specification_entry.binding("revision"),
            source_graph_entry.binding("revision"),
        ),
    )
    geometry_entry = CompositionEntry(
        manifest_value,
        entry_id="mesh/geometry",
        role="discretization",
        owner_id="phydrax.meshing",
        structure_id=geometry_manifest.source_geometry_id,
        revision_id=geometry_manifest.manifest_id,
        semantics_id="compact-owner-local-geometry",
        dependencies=(
            topology_entry.binding("structure"),
            source_entry.binding("structure"),
            source_entry.binding("revision"),
            source_entry.binding("semantics"),
        ),
    )
    state_entry = CompositionEntry(
        state,
        entry_id="solver/state",
        role="physical-state",
        owner_id="phydrax.solver",
        structure_id=state.topology_id,
        revision_id=state.accepted_id,
        semantics_id="accepted-owner-local-fe-state",
        dependencies=(
            topology_entry.binding("structure"),
            geometry_entry.binding("structure"),
        ),
    )
    if (
        state.field_layouts is None
        or len(state.field_layouts) != len(state.fields)
        or tuple(layout.layout_id for layout in field_layouts)
        != tuple(layout.layout_id for layout in state.field_layouts)
        or len(field_layouts) != len(state.fields)
    ):
        raise ValueError("Cold FE state lost its exact typed field-layout set.")
    field_entries = [
        CompositionEntry(
            field,
            entry_id=f"solver/state/field/{layout.field_spec.name}",
            role="physical-state",
            owner_id="phydrax.solver",
            structure_id=state.topology_id,
            revision_id=state.accepted_id,
            semantics_id=layout.layout_id,
            dependencies=(state_entry.binding("revision"),),
        )
        for field, layout in zip(state.fields, field_layouts, strict=True)
    ]
    restart_entry = CompositionEntry(
        restart_manifest,
        entry_id="solver/restart-manifest",
        role="history",
        owner_id="phydrax.solver",
        structure_id=state.topology_id,
        revision_id=restart_manifest.manifest_id,
        semantics_id="typed-fe-restart-banks",
        dependencies=(state_entry.binding("revision"),),
    )
    entries = [
        source_entry,
        specification_entry,
        adaptation_specification_entry,
        source_graph_entry,
        target_graph_entry,
        topology_entry,
        geometry_entry,
        state_entry,
        *field_entries,
    ]
    if archive_safe:
        entries.append(restart_entry)
    if restart_manifest.auxiliary_layouts is None:
        raise ValueError(
            "FE restart manifest omitted declared auxiliary layout contracts."
        )
    for (name, bank), layout in zip(
        restart_manifest.auxiliary_state,
        restart_manifest.auxiliary_layouts,
        strict=True,
    ):
        entries.append(
            CompositionEntry(
                bank,
                entry_id=f"solver/restart/auxiliary/{name}",
                role=layout.role.value,
                owner_id="phydrax.solver",
                structure_id=layout.layout_id,
                revision_id=restart_manifest.manifest_id,
                semantics_id=layout.layout_id,
                dependencies=(state_entry.binding("structure"),),
            )
        )
    if restart_manifest.integrator_layouts is None:
        raise ValueError(
            "FE restart manifest omitted declared integrator layout contracts."
        )
    for (name, bank), layout in zip(
        restart_manifest.integrator_state,
        restart_manifest.integrator_layouts,
        strict=True,
    ):
        entries.append(
            CompositionEntry(
                bank,
                entry_id=f"solver/restart/integrator/{name}",
                role=layout.role.value,
                owner_id="phydrax.solver",
                structure_id=layout.layout_id,
                revision_id=restart_manifest.manifest_id,
                semantics_id=layout.layout_id,
                dependencies=(state_entry.binding("structure"),),
            )
        )
    if restart_manifest.material_layouts is None:
        raise ValueError(
            "FE restart manifest omitted its declared material layout contract."
        )
    if state.materials is None:
        raise ValueError(
            "The owner-local FE profile requires its typed material transaction."
        )
    entries.append(
        CompositionEntry(
            state.materials,
            entry_id="solver/state/materials",
            role="model-state",
            owner_id="phydrax.solver",
            structure_id=state.materials.layout_id,
            revision_id=state.materials.transaction_id,
            semantics_id="replicated-material-transaction",
            dependencies=(state_entry.binding("structure"),),
        )
    )
    return Composition(entries, boundary_id="native-cold-owner-local-fe-restart")


def _restore_composition(value: Any, /) -> Any:
    from phydrax._model._structure import FiniteElementAcceptedStateArchiveRecipe
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing.providers._native_periodic import NativePeriodicSource
    from phydrax.solver._finite_element_schedule import (
        FiniteElementAcceptedState,
        FiniteElementRestartManifest,
    )

    if not isinstance(value, Composition):
        raise TypeError("Owner-local FE artifact root is not a Composition.")
    rebuilt_entries = []
    rebuilt_state = None
    rebuilt_restart = None
    for entry in value.entries:
        item = entry.value
        if isinstance(item, NativePeriodicSource):
            item = NativePeriodicSource.from_stored_fields(item)
        if entry.entry_id == "solver/state":
            if type(item) is FiniteElementAcceptedStateArchiveRecipe:
                item = FiniteElementAcceptedState.from_archive_recipe(item)
            elif type(item) is FiniteElementAcceptedState:
                item = FiniteElementAcceptedState.from_stored_fields(item)
            else:
                raise TypeError(
                    "Owner-local FE archive omitted its canonical accepted-state recipe."
                )
            if item.accepted_id != entry.revision_id:
                raise ValueError(
                    "Owner-local FE archive changed accepted-state identity."
                )
            rebuilt_state = item
        elif entry.entry_id == "solver/restart-manifest":
            if not isinstance(item, FiniteElementRestartManifest):
                raise TypeError(
                    "Owner-local FE archive omitted its typed restart manifest."
                )
            item = FiniteElementRestartManifest.from_stored_fields(item)
            rebuilt_restart = item
        rebuilt_entries.append(
            CompositionEntry(
                item,
                entry_id=entry.entry_id,
                role=entry.role,
                owner_id=entry.owner_id,
                structure_id=entry.structure_id,
                revision_id=entry.revision_id,
                semantics_id=entry.semantics_id,
                dependencies=entry.dependencies,
            )
        )
    if (
        rebuilt_state is None
        or rebuilt_restart is None
        or rebuilt_restart.state.accepted_id != rebuilt_state.accepted_id
    ):
        raise ValueError("Owner-local FE state and restart-bank recipes disagree.")
    rebuilt = Composition(rebuilt_entries, boundary_id=value.boundary_id)
    if (
        rebuilt.composition_id != value.composition_id
        or rebuilt.structure_id != value.structure_id
    ):
        raise ValueError(
            "Owner-local FE archive reconstruction changed composition identity."
        )
    return rebuilt


def _worker(phase: str, archive_root: Path, profile: str, /) -> dict[str, Any]:
    import equinox as eqx
    import jax
    from jax.experimental import multihost_utils

    import phydrax as phx
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.graph._partition import GraphPartitionPlan
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._assembly import MeshPartManifest
    from phydrax.meshing._contracts import SurfaceMeshingSpec
    from phydrax.meshing._distribution import (
        OwnerLocalGraphMigrationEvidence,
        OwnerLocalGraphPartitionEvidence,
    )
    from phydrax.meshing._model_codec import read_model_artifact, write_model_artifact
    from phydrax.meshing.providers._native_periodic import NativePeriodicSource
    from phydrax.solver._finite_element_adaptivity import (
        FiniteElementOwnerLocalFieldTransferEvidence,
        FiniteElementOwnerLocalTopologyResult,
        FiniteElementTopologyTransaction,
    )
    from phydrax.solver._finite_element_schedule import (
        FiniteElementAcceptedState,
        FiniteElementAcceptedStepSchedule,
        FiniteElementAttemptResult,
        FiniteElementRestartManifest,
    )

    if phase not in ("write", "read"):
        raise ValueError(f"Unknown owner-local FE worker phase {phase!r}.")
    group, rank = _current_group()
    archive = archive_root / f"{profile}-rank-{rank}.phx"

    def gather_scalar(value: Any, dtype: Any = np.float64) -> np.ndarray:
        local = np.asarray(value, dtype=dtype)
        return np.asarray(multihost_utils.process_allgather(local), dtype=dtype).reshape(
            (3,)
        )

    def gather_rows(values: Any, dtype: Any = np.int64) -> np.ndarray:
        local = np.asarray(values, dtype=dtype)
        return np.asarray(multihost_utils.process_allgather(local), dtype=dtype).reshape(
            (3, local.size)
        )

    def require_global_flag(local_success: bool, /, contract: str) -> None:
        flags = gather_scalar(int(local_success), dtype=np.int32)
        if not np.all(flags == 1):
            raise RuntimeError(
                f"Owner-local FE ranks disagree on {contract}: {flags.tolist()}."
            )

    def solve_record(compiled: Any, discretization: Any, /) -> dict[str, Any]:
        local_dof_counts = tuple(
            int(dof_map.global_dof_count) for dof_map in discretization.dof_maps
        )
        return _solve_physical_problem(compiled, local_dof_counts)

    def verify_periodic_adaptation(view: Any, stage: str, /) -> dict[str, Any]:
        local_error = None
        record: dict[str, Any]
        try:
            record = _periodic_adaptation_transfer_oracle(view)
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
            record = {
                "accepted": False,
                "source_cell_count": 0,
                "target_cell_count": 0,
                "topology_changed": False,
                "shared_representative_winding_count": 0,
                "constant_field_transfer_max_error": float("inf"),
                "nonconstant_interpolation_error": float("inf"),
                "nonconstant_interpolation_bound": 0.0,
                "winding_sensitive_lifted_field_image_separation": 0.0,
            }
        require_global_flag(
            local_error is None and bool(record["accepted"]),
            f"{stage} orbit-synchronized FE adaptation and transfer",
        )
        counts = gather_rows(
            (
                int(record["source_cell_count"]),
                int(record["target_cell_count"]),
                int(bool(record.get("topology_changed", False))),
                int(record["shared_representative_winding_count"]),
            )
        )
        if not np.any(counts[:, 3] > 0):
            raise RuntimeError(
                f"{stage} has no distinct winding entities sharing a source representative."
            )
        if not np.any(counts[:, 2] > 0):
            raise RuntimeError(
                f"{stage} did not publish a nontrivial periodic topology adaptation."
            )
        constant_errors = gather_scalar(
            float(record["constant_field_transfer_max_error"])
        )
        if float(np.max(constant_errors, initial=0.0)) > 1.0e-12:
            raise RuntimeError(f"{stage} constant FE transfer exceeded 1e-12.")
        nonconstant_errors = gather_scalar(
            float(record["nonconstant_interpolation_error"])
        )
        nonconstant_bounds = gather_scalar(
            float(record["nonconstant_interpolation_bound"])
        )
        if np.any(nonconstant_errors > nonconstant_bounds + 1.0e-12):
            raise RuntimeError(
                f"{stage} periodic interpolation exceeded its independent analytic bound."
            )
        winding_separations = gather_scalar(
            float(record["winding_sensitive_lifted_field_image_separation"])
        )
        if not np.any(winding_separations > 0.0):
            raise RuntimeError(
                f"{stage} winding-sensitive field did not distinguish periodic images."
            )
        record["global_source_cell_count"] = int(np.sum(counts[:, 0]))
        record["global_target_cell_count"] = int(np.sum(counts[:, 1]))
        record["global_topology_changed_owner_count"] = int(np.sum(counts[:, 2]))
        record["global_shared_representative_winding_count"] = int(np.sum(counts[:, 3]))
        record["global_constant_field_transfer_max_error"] = float(
            np.max(constant_errors, initial=0.0)
        )
        record["global_nonconstant_interpolation_max_error"] = float(
            np.max(nonconstant_errors, initial=0.0)
        )
        record["global_nonconstant_interpolation_analytic_bound"] = float(
            np.max(nonconstant_bounds, initial=0.0)
        )
        record["global_winding_sensitive_field_image_separation"] = float(
            np.max(winding_separations, initial=0.0)
        )
        record["stage"] = stage
        return record

    if phase == "write":
        source_graph_plan = _source_graph_partition_plan(profile)
        target_graph_plan = _target_graph_partition_plan()
        (
            source_assembly,
            adaptation_assembly,
            target_assembly,
            source_view,
            adaptation_view,
            target_view,
            specification,
            adaptation_specification,
            source_graph_plan,
            target_graph_plan,
            finite_element_graph_evidence,
            adaptive_result,
            pre_graph_adaptive_result,
        ) = _owner_local_views(
            group,
            profile,
            source_graph_plan=source_graph_plan,
            target_graph_plan=target_graph_plan,
        )
        if not (
            source_view.source_value
            is adaptation_view.source_value
            is target_view.source_value
        ):
            raise RuntimeError(
                "Base, adapted, and target views changed the retained source root."
            )
        if any(
            not isinstance(view, OwnerLocalCellGeometry)
            for view in (source_view, adaptation_view, target_view)
        ):
            raise TypeError(
                "Native periodic FE production did not return compact owner-local views."
            )
        source_part = source_assembly.parts[0]
        source_manifest = MeshPartManifest.from_part(source_part)
        source_view.require_owner_directory_current(
            group, axis_name=f"{_AXIS_NAME}_{profile}_writer_source_directory"
        )
        adaptation_view.require_owner_directory_current(
            group, axis_name=f"{_AXIS_NAME}_{profile}_writer_adaptation_directory"
        )
        target_view.require_owner_directory_current(
            group, axis_name=f"{_AXIS_NAME}_{profile}_writer_target_directory"
        )
        if (
            adaptive_result is not target_view.adaptive_result
            or pre_graph_adaptive_result is not adaptation_view.adaptive_result
            or adaptive_result.source_topology_id
            != pre_graph_adaptive_result.source_topology_id
            or adaptive_result.target_topology_id
            != pre_graph_adaptive_result.target_topology_id
            or adaptive_result.specification_id
            != pre_graph_adaptive_result.specification_id
            or adaptive_result.transition_id != pre_graph_adaptive_result.transition_id
            or (
                None
                if adaptive_result.lineage is None
                else adaptive_result.lineage.lineage_id
            )
            != (
                None
                if pre_graph_adaptive_result.lineage is None
                else pre_graph_adaptive_result.lineage.lineage_id
            )
            or (
                None
                if adaptive_result.vertex_transfer is None
                else adaptive_result.vertex_transfer.transfer_id
            )
            != (
                None
                if pre_graph_adaptive_result.vertex_transfer is None
                else pre_graph_adaptive_result.vertex_transfer.transfer_id
            )
            or (
                None
                if adaptive_result.vertex_stencil is None
                else adaptive_result.vertex_stencil.stencil_id
            )
            != (
                None
                if pre_graph_adaptive_result.vertex_stencil is None
                else pre_graph_adaptive_result.vertex_stencil.stencil_id
            )
            or adaptive_result.target_topology_id != target_view.source_topology_id
            or adaptive_result.target_local_view_id != target_view.local_view_id
            or adaptive_result.target_distribution_id != target_view.distribution_id
            or adaptive_result.target_owner_directory_id != target_view.distribution_id
            or not np.array_equal(
                np.sort(np.asarray(adaptive_result.target_cell_ids, dtype=np.int64)),
                np.sort(np.asarray(target_view.cell_ids, dtype=np.int64)),
            )
            or not np.array_equal(
                np.sort(
                    np.asarray(adaptive_result.target_cell_owner_ranks, dtype=np.int64)
                ),
                np.sort(np.asarray(target_view.cell_owner_ranks, dtype=np.int64)),
            )
        ):
            raise RuntimeError(
                "Rebound writer adaptation lost its pre-graph ancestry or target-view authority."
            )
        writer_pre_graph_adaptation = verify_periodic_adaptation(
            adaptation_view, "fresh-writer-pre-graph-orbit-adaptation"
        )
        writer_adaptation = verify_periodic_adaptation(
            target_view, "fresh-writer-rebound-orbit-adaptation"
        )
        if source_view.source_value is None:
            raise ValueError("Owner-local FE source lacks its exact authored point root.")
        source_discretization, source_compiled, source_identities, field_layouts = (
            _compile_owner_local_problem(source_view, group, rank)
        )
        if len(source_identities) != 2 or len(field_layouts) != 2:
            raise RuntimeError(
                "Cold FE source plan omitted its P1/P0 identity or layout."
            )
        source_identity, content_identity = source_identities
        u_layout, content_layout = field_layouts
        domain_measure = abs(
            float(
                np.linalg.det(
                    np.asarray(
                        source_view.source_value.domain.cell.basis, dtype=np.float64
                    )
                )
            )
        )
        physical = solve_record(source_compiled, source_discretization)
        source_content_reference = _periodic_p0_content_values(
            source_view, source_discretization
        )
        source_content_full = np.asarray(physical["content_values"], dtype=np.float64)
        if (
            source_content_full.shape != source_content_reference.shape
            or np.max(np.abs(source_content_full - source_content_reference), initial=0.0)
            > 2.0e-10
        ):
            raise RuntimeError(
                "P0 FE mass/source solution differs from independent periodic cell averages."
            )
        source_content_values, source_content_ownership = content_identity.shard_values(
            source_content_full
        )
        source_content_integral_local = _periodic_p0_owner_integral(
            source_view,
            source_discretization,
            content_identity,
            source_content_values,
        )
        source_content_minimum_local = float(
            np.min(np.asarray(source_content_values, dtype=np.float64), initial=np.inf)
        )
        source_values, source_ownership = source_identity.shard_values(physical["values"])
        material_site = phx.equations.MaterialSiteId("steel:integration-site:0")
        materials = phx.equations.MaterialTransaction(
            (
                phx.equations.MaterialState(
                    material_site,
                    "linear-elastic:steel",
                    np.asarray((2.5, 3.5), dtype=np.float64),
                    trial=np.asarray((2.5, 3.5), dtype=np.float64),
                    state_version=3,
                ),
            )
        )
        state = FiniteElementAcceptedState(
            (source_values, source_content_values),
            np.asarray(0.125, dtype=np.float64),
            7,
            source_view.source_topology_id,
            source_discretization.prepared_id,
            source_compiled.compilation_id,
            field_layouts=field_layouts,
            materials=materials,
            schedule_cursor=3,
            state_version=5,
            field_ownership=(source_ownership, source_content_ownership),
        )
        u_history_name = "accepted-u-history"
        content_history_name = "accepted-content-history"
        clock_history_name = "time-history"
        previous_name = "previous-step"
        rng_name = "rng"
        u_history = source_values
        content_history = source_content_values
        clock_history = np.asarray((0.0, float(np.asarray(state.time))), dtype=np.float64)
        previous_step = np.asarray((0.125,), dtype=np.float64)
        rng = jax.random.key(2026)
        u_history_layout = phx.solver.FiniteElementRestartStateLayout(
            u_history_name,
            phx.solver.FiniteElementRestartStateRole.HISTORY,
            phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS,
            ownership=source_ownership,
        )
        content_history_layout = phx.solver.FiniteElementRestartStateLayout(
            content_history_name,
            phx.solver.FiniteElementRestartStateRole.HISTORY,
            phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS,
            ownership=source_content_ownership,
        )
        clock_history_layout = phx.solver.FiniteElementRestartStateLayout(
            clock_history_name,
            phx.solver.FiniteElementRestartStateRole.HISTORY,
            phx.solver.FiniteElementRestartStateAssociation.REPLICATED,
        )
        previous_layout = phx.solver.FiniteElementRestartStateLayout(
            previous_name,
            phx.solver.FiniteElementRestartStateRole.HISTORY,
            phx.solver.FiniteElementRestartStateAssociation.REPLICATED,
        )
        rng_layout = phx.solver.FiniteElementRestartStateLayout(
            rng_name,
            phx.solver.FiniteElementRestartStateRole.RNG,
            phx.solver.FiniteElementRestartStateAssociation.REPLICATED,
            prng_impl=str(jax.random.key_impl(rng)),
        )
        material_layout = phx.solver.FiniteElementRestartStateLayout(
            material_site.key,
            phx.solver.FiniteElementRestartStateRole.MODEL_STATE,
            phx.solver.FiniteElementRestartStateAssociation.REPLICATED,
        )
        restart_manifest = FiniteElementRestartManifest(
            state,
            auxiliary_state=(
                (u_history_name, u_history),
                (content_history_name, content_history),
                (clock_history_name, clock_history),
            ),
            integrator_state=((previous_name, previous_step), (rng_name, rng)),
            auxiliary_layouts=(
                u_history_layout,
                content_history_layout,
                clock_history_layout,
            ),
            integrator_layouts=(previous_layout, rng_layout),
            material_layouts=(material_layout,),
        )
        composition = _source_composition(
            source_view,
            source_manifest,
            restart_manifest,
            field_layouts,
            specification,
            adaptation_specification,
            source_graph_plan,
            target_graph_plan,
            archive_safe=True,
        )
        write_model_artifact(archive, composition, limits=None)
        source_placement = gather_rows(
            (
                int(np.asarray(source_view.owned_cell_ids).size),
                int(np.asarray(source_view.cell_ids).size),
                int(source_discretization.dof_maps[0].global_dof_count),
            )
        )
        source_integral = float(np.sum(gather_scalar(physical["physical_integral"])))
        source_content_integral = float(
            np.sum(gather_scalar(source_content_integral_local))
        )
        source_content_absolute_integral = float(
            np.sum(
                gather_scalar(
                    _periodic_p0_owner_integral(
                        source_view,
                        source_discretization,
                        content_identity,
                        np.abs(np.asarray(source_content_values, dtype=np.float64)),
                    )
                )
            )
        )
        source_content_rounding_bound = (
            16.0
            * np.finfo(np.float64).eps
            * max(1.0, source_content_absolute_integral)
            * max(1, int(np.sum(source_placement[:, 0])) + 8)
        )
        source_content_minimum_by_rank = gather_scalar(source_content_minimum_local)
        source_content_maximum_by_rank = gather_scalar(
            float(
                np.max(
                    np.asarray(source_content_values, dtype=np.float64), initial=-np.inf
                )
            )
        )
        source_content_minimum = float(
            np.min(source_content_minimum_by_rank, initial=np.inf)
        )
        source_content_maximum = float(
            np.max(source_content_maximum_by_rank, initial=-np.inf)
        )
        source_minimum = float(
            np.min(
                gather_scalar(
                    np.inf
                    if physical["minimum_value"] is None
                    else physical["minimum_value"]
                )
            )
        )
        if (
            np.any(source_placement[:2, 0] <= 0)
            or source_placement[2, 0] != 0
            or abs(source_integral - domain_measure)
            > 2.0e-9 * max(1.0, abs(domain_measure))
            or abs(source_content_integral) > source_content_rounding_bound
            or source_minimum <= 0.0
            or not np.isfinite(source_content_minimum)
            or not np.isfinite(source_content_maximum)
            or source_content_maximum - source_content_minimum <= 1.0e-8
        ):
            raise RuntimeError(
                "Native source FE state failed its placement or P1/P0 physical oracle."
            )
        return {
            "phase": phase,
            "profile_key": profile,
            "rank": rank,
            "pid": os.getpid(),
            "source_id": source_view.source_value.source_id,
            "source_revision": source_view.source_value.source_revision,
            "source_view_id": source_view.local_view_id,
            "source_manifest_id": source_manifest.manifest_id,
            "specification_id": specification.specification_id,
            "adaptation_specification_id": adaptation_specification.specification_id,
            "adaptation_view_id": adaptation_view.local_view_id,
            "adapted_topology_id": adaptive_result.target_topology_id,
            "adapted_geometry_id": adaptive_result.target_geometry_id,
            "adaptive_result_id": adaptive_result.result_id,
            "pre_graph_adaptive_result_id": pre_graph_adaptive_result.result_id,
            "adaptive_lineage_id": (
                None
                if adaptive_result.lineage is None
                else adaptive_result.lineage.lineage_id
            ),
            "pre_graph_lineage_id": (
                None
                if pre_graph_adaptive_result.lineage is None
                else pre_graph_adaptive_result.lineage.lineage_id
            ),
            "adaptive_vertex_transfer_id": (
                None
                if adaptive_result.vertex_transfer is None
                else adaptive_result.vertex_transfer.transfer_id
            ),
            "pre_graph_vertex_transfer_id": (
                None
                if pre_graph_adaptive_result.vertex_transfer is None
                else pre_graph_adaptive_result.vertex_transfer.transfer_id
            ),
            "finite_element_graph_evidence_id": finite_element_graph_evidence.evidence_id,
            "finite_element_graph_result_id": finite_element_graph_evidence.result.result_id,
            "finite_element_graph_plan_id": finite_element_graph_evidence.result.plan_id,
            "finite_element_graph_owner_directory_id": (
                finite_element_graph_evidence.owner_directory_id
            ),
            "source_graph_plan_id": source_graph_plan.plan_id,
            "target_graph_plan_id": target_graph_plan.plan_id,
            "source_topology_id": source_manifest.source_topology_id,
            "source_geometry_id": source_manifest.source_geometry_id,
            "source_state_id": state.accepted_id,
            "restart_manifest_id": restart_manifest.manifest_id,
            "accepted_time": float(np.asarray(state.time)),
            "accepted_step": int(state.step),
            "source_owned_cell_counts": source_placement[:, 0].tolist(),
            "source_resident_cell_counts": source_placement[:, 1].tolist(),
            "source_dof_counts": source_placement[:, 2].tolist(),
            "source_content_integral": source_content_integral,
            "source_content_absolute_integral": source_content_absolute_integral,
            "source_content_rounding_bound": source_content_rounding_bound,
            "source_content_minimum": source_content_minimum,
            "source_content_maximum": source_content_maximum,
            "source_content_minimum_by_rank": [
                float(value) if np.isfinite(value) else None
                for value in source_content_minimum_by_rank
            ],
            "source_content_maximum_by_rank": [
                float(value) if np.isfinite(value) else None
                for value in source_content_maximum_by_rank
            ],
            "source_field_layout_ids": [layout.layout_id for layout in field_layouts],
            "source_identity_plan_ids": [
                identity.plan_id for identity in source_identities
            ],
            "source_integral": source_integral,
            "domain_measure": domain_measure,
            "positive_state_minimum": source_minimum,
            "source_residual_norm": float(
                np.max(gather_scalar(physical["residual_norm"]))
            ),
            "periodic_adaptation_transfer": writer_adaptation,
            "periodic_pre_graph_adaptation_transfer": writer_pre_graph_adaptation,
            "archive_bytes": archive.stat().st_size,
        }

    composition = read_model_artifact(
        archive,
        expected_type=Composition,
        restore=_restore_composition,
        limits=None,
    )
    specification_entry = composition.entry("mesh/specification")
    adaptation_specification_entry = composition.entry("mesh/adaptation-specification")
    source_graph_plan_entry = composition.entry("mesh/source-graph-plan")
    target_graph_plan_entry = composition.entry("mesh/target-graph-plan")
    specification = specification_entry.value
    adaptation_specification = adaptation_specification_entry.value
    source_graph_plan = source_graph_plan_entry.value
    target_graph_plan = target_graph_plan_entry.value
    if (
        not isinstance(specification, SurfaceMeshingSpec)
        or specification.size_controls
        or specification_entry.entry_id != "mesh/specification"
        or specification_entry.structure_id != specification.specification_id
        or specification_entry.revision_id != specification.specification_id
        or specification_entry.semantics_id != "native-periodic-surface-specification"
        or not isinstance(adaptation_specification, SurfaceMeshingSpec)
        or not adaptation_specification.size_controls
        or adaptation_specification_entry.entry_id != "mesh/adaptation-specification"
        or adaptation_specification_entry.structure_id
        != adaptation_specification.specification_id
        or adaptation_specification_entry.revision_id
        != adaptation_specification.specification_id
        or adaptation_specification_entry.semantics_id
        != "native-periodic-adaptation-surface-specification"
        or not isinstance(source_graph_plan, GraphPartitionPlan)
        or source_graph_plan_entry.entry_id != "mesh/source-graph-plan"
        or source_graph_plan_entry.structure_id != source_graph_plan.plan_id
        or source_graph_plan_entry.revision_id != source_graph_plan.plan_id
        or source_graph_plan_entry.semantics_id != "native-periodic-source-graph-plan"
        or not isinstance(target_graph_plan, GraphPartitionPlan)
        or target_graph_plan_entry.entry_id != "mesh/target-graph-plan"
        or target_graph_plan_entry.structure_id != target_graph_plan.plan_id
        or target_graph_plan_entry.revision_id != target_graph_plan.plan_id
        or target_graph_plan_entry.semantics_id
        != "native-periodic-repartition-graph-plan"
    ):
        raise TypeError(
            "Cold FE archive lost typed base/adaptation specification or graph-plan roots."
        )
    source_roots = tuple(
        entry
        for entry in composition.entries
        if isinstance(entry.value, NativePeriodicSource)
    )
    if len(source_roots) != 1:
        raise ValueError(
            "Cold FE archive must contain one exact NativePeriodicSource root."
        )
    archived_source_entry = source_roots[0]
    stored_state_entry = composition.entry("solver/state")
    stored_restart_entry = composition.entry("solver/restart-manifest")
    stored_state = stored_state_entry.value
    restart_manifest = stored_restart_entry.value
    if (
        not isinstance(stored_state, FiniteElementAcceptedState)
        or not isinstance(restart_manifest, FiniteElementRestartManifest)
        or stored_state.accepted_id != restart_manifest.state.accepted_id
        or stored_state.accepted_id != stored_state_entry.revision_id
    ):
        raise TypeError(
            "Cold FE archive lost its exact accepted state or typed restart manifest."
        )
    (
        source_assembly,
        adaptation_assembly,
        target_assembly,
        source_view,
        adaptation_view,
        target_view,
        current_specification,
        current_adaptation_specification,
        current_source_graph_plan,
        current_target_graph_plan,
        finite_element_graph_evidence,
        adaptive_result,
        pre_graph_adaptive_result,
    ) = _owner_local_views(
        group,
        profile,
        authored_source=archived_source_entry.value,
        authored_specification=specification,
        authored_adaptation_specification=adaptation_specification,
        source_graph_plan=source_graph_plan,
        target_graph_plan=target_graph_plan,
    )
    source_part = source_assembly.parts[0]
    adaptation_part = adaptation_assembly.parts[0]
    target_part = target_assembly.parts[0]
    if any(
        not isinstance(view, OwnerLocalCellGeometry)
        for view in (source_view, adaptation_view, target_view)
    ):
        raise TypeError(
            "Cold FE replay did not rebuild all compact owner-local geometry views."
        )
    if not (
        source_view.source_value
        is adaptation_view.source_value
        is target_view.source_value
    ):
        raise RuntimeError(
            "Fresh base, adapted, and target views do not share the archived source root."
        )
    if (
        current_specification.specification_id != specification.specification_id
        or current_adaptation_specification.specification_id
        != adaptation_specification.specification_id
        or current_specification.scope.source_id != archived_source_entry.value.source_id
        or current_adaptation_specification.scope.source_id
        != archived_source_entry.value.source_id
        or current_specification.scope.source_revision
        != archived_source_entry.value.source_revision
        or current_adaptation_specification.scope.source_revision
        != archived_source_entry.value.source_revision
        or current_source_graph_plan.plan_id != source_graph_plan.plan_id
        or current_target_graph_plan.plan_id != target_graph_plan.plan_id
        or not isinstance(finite_element_graph_evidence, OwnerLocalGraphPartitionEvidence)
        or finite_element_graph_evidence.source_view_id != adaptation_view.local_view_id
        or finite_element_graph_evidence.owner_directory_id
        != adaptation_view.distribution_id
        or finite_element_graph_evidence.result.plan_id != target_graph_plan.plan_id
        or finite_element_graph_evidence.result.result_id is None
        or adaptive_result is not target_view.adaptive_result
        or pre_graph_adaptive_result is not adaptation_view.adaptive_result
        or adaptive_result.source_topology_id
        != pre_graph_adaptive_result.source_topology_id
        or adaptive_result.target_topology_id
        != pre_graph_adaptive_result.target_topology_id
        or adaptive_result.specification_id != pre_graph_adaptive_result.specification_id
        or adaptive_result.transition_id != pre_graph_adaptive_result.transition_id
        or (
            None
            if adaptive_result.lineage is None
            else adaptive_result.lineage.lineage_id
        )
        != (
            None
            if pre_graph_adaptive_result.lineage is None
            else pre_graph_adaptive_result.lineage.lineage_id
        )
        or (
            None
            if adaptive_result.vertex_transfer is None
            else adaptive_result.vertex_transfer.transfer_id
        )
        != (
            None
            if pre_graph_adaptive_result.vertex_transfer is None
            else pre_graph_adaptive_result.vertex_transfer.transfer_id
        )
        or (
            None
            if adaptive_result.vertex_stencil is None
            else adaptive_result.vertex_stencil.stencil_id
        )
        != (
            None
            if pre_graph_adaptive_result.vertex_stencil is None
            else pre_graph_adaptive_result.vertex_stencil.stencil_id
        )
        or adaptive_result.source_topology_id != source_view.source_topology_id
        or adaptive_result.target_topology_id != target_view.source_topology_id
        or adaptive_result.specification_id != adaptation_specification.specification_id
        or adaptive_result.target_local_view_id != target_view.local_view_id
        or adaptive_result.target_distribution_id != target_view.distribution_id
        or adaptive_result.target_owner_directory_id != target_view.distribution_id
        or not np.array_equal(
            np.sort(np.asarray(adaptive_result.target_cell_ids, dtype=np.int64)),
            np.sort(np.asarray(target_view.cell_ids, dtype=np.int64)),
        )
        or not np.array_equal(
            np.sort(np.asarray(adaptive_result.target_cell_owner_ranks, dtype=np.int64)),
            np.sort(np.asarray(target_view.cell_owner_ranks, dtype=np.int64)),
        )
    ):
        raise RuntimeError(
            "Fresh owner-local replay changed a typed request, adaptation, or FE graph identity."
        )
    source_view.require_owner_directory_current(
        group, axis_name=f"{_AXIS_NAME}_{profile}_reader_source_directory"
    )
    adaptation_view.require_owner_directory_current(
        group, axis_name=f"{_AXIS_NAME}_{profile}_reader_adaptation_directory"
    )
    target_view.require_owner_directory_current(
        group, axis_name=f"{_AXIS_NAME}_{profile}_reader_target_directory"
    )
    reader_pre_graph_adaptation = verify_periodic_adaptation(
        adaptation_view, "fresh-reader-pre-graph-orbit-adaptation-after-restore"
    )
    reader_adaptation = verify_periodic_adaptation(
        target_view, "fresh-reader-rebound-orbit-adaptation-after-restore"
    )
    source_manifest = MeshPartManifest.from_part(source_part)
    adaptation_manifest = MeshPartManifest.from_part(adaptation_part)
    target_manifest = MeshPartManifest.from_part(target_part)
    if (
        archived_source_entry.value is not source_view.source_value
        or archived_source_entry.entry_id != "mesh/source"
        or archived_source_entry.structure_id != source_view.source_topology_id
        or archived_source_entry.revision_id != source_view.source_revision
        or archived_source_entry.semantics_id != source_view.source_value.binding_id
        or specification.scope.source_id != source_view.source_id
        or specification.scope.source_revision != source_view.source_revision
        or adaptation_specification.scope.source_id != source_view.source_id
        or adaptation_specification.scope.source_revision != source_view.source_revision
        or composition.value("mesh/topology") != _mesh_manifest_recipe(source_manifest)
        or composition.value("mesh/geometry") != _mesh_manifest_recipe(source_manifest)
        or stored_state.topology_id != source_view.source_topology_id
        or source_manifest.distribution_id != source_view.distribution_id
        or adaptation_manifest.source_topology_id != adaptive_result.target_topology_id
        or adaptation_manifest.source_geometry_id != adaptive_result.target_geometry_id
        or target_manifest.source_topology_id != adaptation_manifest.source_topology_id
        or target_manifest.source_geometry_id != adaptation_manifest.source_geometry_id
    ):
        raise RuntimeError(
            "Fresh source or adapted geometry differs from the archived periodic evidence."
        )
    source_discretization, source_compiled, source_identities, source_layouts = (
        _compile_owner_local_problem(source_view, group, rank)
    )
    (
        adaptation_discretization,
        adaptation_compiled,
        adaptation_identities,
        adaptation_layouts,
    ) = _compile_owner_local_problem(adaptation_view, group, rank)
    target_discretization, target_compiled, target_identities, target_layouts = (
        _compile_owner_local_problem(target_view, group, rank)
    )
    if (
        len(source_identities) != 2
        or len(adaptation_identities) != 2
        or len(target_identities) != 2
        or len(source_layouts) != 2
        or len(adaptation_layouts) != 2
        or len(target_layouts) != 2
        or tuple(layout.layout_id for layout in source_layouts)
        != tuple(layout.layout_id for layout in stored_state.field_layouts)
        or tuple(layout.layout_id for layout in source_layouts)
        != tuple(layout.layout_id for layout in target_layouts)
        or stored_state.prepared_id != source_discretization.prepared_id
        or stored_state.compilation_id != source_compiled.compilation_id
        or restart_manifest.state.prepared_id != source_discretization.prepared_id
    ):
        raise RuntimeError(
            "Fresh owner-local FE plans changed archived P1/P0 field identities."
        )
    source_u_identity, source_content_identity = source_identities
    adaptation_u_identity, adaptation_content_identity = adaptation_identities
    target_u_identity, target_content_identity = target_identities
    domain_measure = abs(
        float(
            np.linalg.det(
                np.asarray(source_view.source_value.domain.cell.basis, dtype=np.float64)
            )
        )
    )
    source_physical = solve_record(source_compiled, source_discretization)
    expected_source_values, _ = source_u_identity.shard_values(source_physical["values"])
    expected_source_content_reference = _periodic_p0_content_values(
        source_view, source_discretization
    )
    expected_source_content_full = np.asarray(
        source_physical["content_values"], dtype=np.float64
    )
    if (
        expected_source_content_full.shape != expected_source_content_reference.shape
        or np.max(
            np.abs(expected_source_content_full - expected_source_content_reference),
            initial=0.0,
        )
        > 2.0e-10
    ):
        raise RuntimeError(
            "Fresh P0 mass/source solve differs from independent periodic cell averages."
        )
    expected_source_content, _ = source_content_identity.shard_values(
        expected_source_content_full
    )
    stored_u_values = np.asarray(stored_state.fields[0], dtype=np.float64)
    stored_content_values = np.asarray(stored_state.fields[1], dtype=np.float64)
    expected_source_values_np = np.asarray(expected_source_values, dtype=np.float64)
    expected_source_content_np = np.asarray(expected_source_content, dtype=np.float64)
    if (
        stored_u_values.shape != expected_source_values_np.shape
        or stored_content_values.shape != expected_source_content_np.shape
    ):
        raise RuntimeError(
            "Restored FE state rows differ from rebuilt P1/P0 identity plans."
        )
    source_u_restore_error = float(
        np.max(np.abs(stored_u_values - expected_source_values_np), initial=0.0)
    )
    source_content_restore_error = float(
        np.max(np.abs(stored_content_values - expected_source_content_np), initial=0.0)
    )
    source_restore_error = max(source_u_restore_error, source_content_restore_error)
    source_integral = float(np.sum(gather_scalar(source_physical["physical_integral"])))
    source_content_integral = float(
        np.sum(
            gather_scalar(
                _periodic_p0_owner_integral(
                    source_view,
                    source_discretization,
                    source_content_identity,
                    stored_content_values,
                )
            )
        )
    )
    source_content_absolute_integral = float(
        np.sum(
            gather_scalar(
                _periodic_p0_owner_integral(
                    source_view,
                    source_discretization,
                    source_content_identity,
                    np.abs(stored_content_values),
                )
            )
        )
    )
    source_content_minimum_by_rank = gather_scalar(
        float(np.min(stored_content_values, initial=np.inf))
    )
    source_content_maximum_by_rank = gather_scalar(
        float(np.max(stored_content_values, initial=-np.inf))
    )
    source_content_minimum = float(np.min(source_content_minimum_by_rank, initial=np.inf))
    source_content_maximum = float(
        np.max(source_content_maximum_by_rank, initial=-np.inf)
    )
    source_minimum = float(
        np.min(
            gather_scalar(
                np.inf
                if source_physical["minimum_value"] is None
                else source_physical["minimum_value"]
            )
        )
    )
    source_placement = gather_rows(
        (
            int(np.asarray(source_view.owned_cell_ids).size),
            int(np.asarray(source_view.cell_ids).size),
            int(source_discretization.dof_maps[0].global_dof_count),
        )
    )
    target_placement = gather_rows(
        (
            int(np.asarray(target_view.owned_cell_ids).size),
            int(np.asarray(target_view.cell_ids).size),
            int(target_discretization.dof_maps[0].global_dof_count),
        )
    )
    source_cell_count = int(np.sum(source_placement[:, 0]))
    source_content_rounding_bound = (
        16.0
        * np.finfo(np.float64).eps
        * max(1.0, source_content_absolute_integral)
        * max(1, source_cell_count + 8)
    )
    if (
        source_restore_error > 2.0e-9
        or abs(source_integral - domain_measure) > 2.0e-9 * max(1.0, abs(domain_measure))
        or abs(source_content_integral) > source_content_rounding_bound
        or source_minimum <= 0.0
        or not np.isfinite(source_content_minimum)
        or not np.isfinite(source_content_maximum)
        or source_content_maximum - source_content_minimum <= 1.0e-8
        or np.any(source_placement[:2, 0] <= 0)
        or source_placement[2, 0] != 0
        or np.any(target_placement[:2, 0] != 0)
        or target_placement[2, 0] <= 0
        or source_view.distribution_id == target_view.distribution_id
        or source_manifest.source_topology_id == target_manifest.source_topology_id
        or source_manifest.source_geometry_id == target_manifest.source_geometry_id
    ):
        raise RuntimeError(
            "Fresh owner-local FE placement or source P1/P0 oracle changed."
        )
    source_composition = _source_composition(
        source_view,
        source_manifest,
        restart_manifest,
        source_layouts,
        specification,
        adaptation_specification,
        source_graph_plan,
        target_graph_plan,
    )
    source_root_entry = source_composition.entry("mesh/source")
    target_root_entry = source_root_entry
    target_topology_id = target_manifest.source_topology_id
    target_geometry_id = target_manifest.source_geometry_id
    if target_topology_id is None or target_geometry_id is None:
        raise RuntimeError(
            "Current target manifest lacks exact topology or geometry identity."
        )
    target_topology_entry = CompositionEntry(
        target_manifest,
        entry_id="mesh/topology",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=target_topology_id,
        revision_id=target_manifest.manifest_id,
        semantics_id="cell-mesh",
        dependencies=(
            target_root_entry.binding("structure"),
            target_root_entry.binding("revision"),
            target_root_entry.binding("semantics"),
            source_composition.entry("mesh/specification").binding("revision"),
            source_composition.entry("mesh/adaptation-specification").binding("revision"),
            source_composition.entry("mesh/source-graph-plan").binding("revision"),
            source_composition.entry("mesh/target-graph-plan").binding("revision"),
        ),
    )
    target_geometry_entry = CompositionEntry(
        target_manifest,
        entry_id="mesh/geometry",
        role="discretization",
        owner_id="phydrax.meshing",
        structure_id=target_geometry_id,
        revision_id=target_manifest.manifest_id,
        semantics_id="compact-owner-local-geometry",
        dependencies=(
            target_topology_entry.binding("structure"),
            target_root_entry.binding("structure"),
            target_root_entry.binding("revision"),
            target_root_entry.binding("semantics"),
        ),
    )
    target_reprepared_entries = (target_topology_entry, target_geometry_entry)
    migration = target_view.graph_migration_evidence
    if migration is None:
        raise RuntimeError(
            "Current target view omitted its identity-bound graph migration evidence."
        )
    if (
        migration.result.plan_id != target_graph_plan.plan_id
        or migration.result.result_id is None
        or migration.source_directory_id != adaptation_view.distribution_id
        or migration.target_directory_id != target_view.distribution_id
    ):
        raise RuntimeError(
            "Current target migration evidence differs from its archived graph plan."
        )
    auxiliary_layouts = restart_manifest.auxiliary_layouts
    integrator_layouts = restart_manifest.integrator_layouts
    material_layouts = restart_manifest.material_layouts
    if (
        auxiliary_layouts is None
        or integrator_layouts is None
        or material_layouts is None
        or len(auxiliary_layouts) != 3
        or len(integrator_layouts) != 2
        or len(material_layouts) != 1
    ):
        raise ValueError(
            "Cold FE manifest omitted a declared history, clock, RNG, or material layout."
        )
    u_history_name = auxiliary_layouts[0].name
    content_history_name = auxiliary_layouts[1].name
    clock_history_name = auxiliary_layouts[2].name
    previous_name = integrator_layouts[0].name
    rng_name = integrator_layouts[1].name
    target_u_history_layout = phx.solver.FiniteElementRestartStateLayout(
        u_history_name,
        phx.solver.FiniteElementRestartStateRole.HISTORY,
        phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS,
        ownership=target_u_identity.ownership(),
    )
    target_content_history_layout = phx.solver.FiniteElementRestartStateLayout(
        content_history_name,
        phx.solver.FiniteElementRestartStateRole.HISTORY,
        phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS,
        ownership=target_content_identity.ownership(),
    )
    target_auxiliary_layouts = (
        target_u_history_layout,
        target_content_history_layout,
        phx.solver.FiniteElementRestartStateLayout.from_stored_fields(
            auxiliary_layouts[2]
        ),
    )
    transaction = FiniteElementTopologyTransaction(
        lambda mesh, fields, materials, lineage, args: (
            len(fields) == 2
            and np.all(np.isfinite(np.asarray(fields[0])))
            and np.all(np.asarray(fields[0]) > 0.0)
            and np.all(np.isfinite(np.asarray(fields[1])))
        ),
        transaction_id="native-cold-owner-local-fe-topology-adaptation",
    )

    def adapt(
        candidate_target_view: OwnerLocalCellGeometry,
        declared_auxiliary_layouts: tuple[Any, ...],
        /,
    ) -> FiniteElementOwnerLocalTopologyResult:
        return transaction.adapt_owner_local(
            restart_manifest,
            source_compiled,
            target_compiled,
            source_identities,
            target_identities,
            adaptive_result,
            source_composition,
            group,
            source_view,
            candidate_target_view,
            state_entry_id="solver/state",
            geometry_entry_id="mesh/geometry",
            field_entry_ids=(
                ("u", "solver/state/field/u"),
                ("content", "solver/state/field/content"),
            ),
            material_entry_id="solver/state/materials",
            auxiliary_entry_ids=(
                (u_history_name, f"solver/restart/auxiliary/{u_history_name}"),
                (
                    content_history_name,
                    f"solver/restart/auxiliary/{content_history_name}",
                ),
                (clock_history_name, f"solver/restart/auxiliary/{clock_history_name}"),
            ),
            integrator_entry_ids=(
                (previous_name, f"solver/restart/integrator/{previous_name}"),
                (rng_name, f"solver/restart/integrator/{rng_name}"),
            ),
            target_auxiliary_layouts=declared_auxiliary_layouts,
            target_integrator_layouts=integrator_layouts,
            target_material_layouts=material_layouts,
            retain_entry_ids=(
                "mesh/source",
                "mesh/specification",
                "mesh/adaptation-specification",
                "mesh/source-graph-plan",
                "mesh/target-graph-plan",
            ),
            reprepare_entries=target_reprepared_entries,
            axis_name=f"{_AXIS_NAME}_topology_adaptation",
        )

    source_composition_id = source_composition.composition_id

    def refused_without_partial_commit(
        refusal: FiniteElementOwnerLocalTopologyResult, /
    ) -> bool:
        def same_bytes(left: Any, right: Any, /) -> bool:
            left_array = np.asarray(left)
            right_array = np.asarray(right)
            return (
                left_array.dtype == right_array.dtype
                and left_array.shape == right_array.shape
                and left_array.tobytes() == right_array.tobytes()
            )

        def layout_ids(layouts: Any, /) -> tuple[str, ...] | None:
            return (
                None if layouts is None else tuple(layout.layout_id for layout in layouts)
            )

        if (
            not isinstance(refusal, FiniteElementOwnerLocalTopologyResult)
            or bool(np.asarray(refusal.committed))
            or refusal.state.accepted_id != stored_state.accepted_id
            or refusal.manifest.manifest_id != restart_manifest.manifest_id
            or (refusal.rebind_receipt is not None and refusal.rebind_receipt.published)
            or refusal.state.time != stored_state.time
            or refusal.state.step != stored_state.step
            or refusal.state.state_version != stored_state.state_version
            or refusal.state.schedule_cursor != stored_state.schedule_cursor
            or refusal.state.topology_id != stored_state.topology_id
            or refusal.state.prepared_id != stored_state.prepared_id
            or refusal.state.compilation_id != stored_state.compilation_id
            or layout_ids(refusal.state.field_layouts)
            != layout_ids(stored_state.field_layouts)
            or len(refusal.state.fields) != 2
            or any(
                not same_bytes(actual, expected)
                for actual, expected in zip(
                    refusal.state.fields, stored_state.fields, strict=True
                )
            )
            or layout_ids(refusal.manifest.auxiliary_layouts)
            != layout_ids(restart_manifest.auxiliary_layouts)
            or layout_ids(refusal.manifest.integrator_layouts)
            != layout_ids(restart_manifest.integrator_layouts)
            or layout_ids(refusal.manifest.material_layouts)
            != layout_ids(restart_manifest.material_layouts)
        ):
            return False
        source_auxiliary = dict(restart_manifest.auxiliary_state)
        refused_auxiliary = dict(refusal.manifest.auxiliary_state)
        source_integrators = dict(restart_manifest.integrator_state)
        refused_integrators = dict(refusal.manifest.integrator_state)
        if (
            source_auxiliary.keys() != refused_auxiliary.keys()
            or source_integrators.keys() != refused_integrators.keys()
            or any(
                not same_bytes(refused_auxiliary[name], source_auxiliary[name])
                for name in source_auxiliary
            )
            or any(
                same_bytes(
                    jax.random.key_data(refused_integrators[name]),
                    jax.random.key_data(source_integrators[name]),
                )
                if name == rng_name
                else same_bytes(refused_integrators[name], source_integrators[name])
                for name in source_integrators
            )
        ):
            return False
        source_materials = stored_state.materials
        refused_materials = refusal.state.materials
        if source_materials is None or refused_materials is None:
            return source_materials is refused_materials
        site = phx.equations.MaterialSiteId("steel:integration-site:0")
        try:
            source_material = source_materials.state(site)
            refused_material = refused_materials.state(site)
        except KeyError:
            return False
        return (
            source_material.model_id == refused_material.model_id
            and source_material.state_version == refused_material.state_version
            and same_bytes(source_material.committed, refused_material.committed)
            and same_bytes(source_material.trial, refused_material.trial)
        )

    stale_migration = OwnerLocalGraphMigrationEvidence(
        migration.result,
        migration.source_candidate_id,
        migration.source_part_id,
        migration.periodic_face_coverage_id,
        migration.source_directory_id,
        migration.target_part_id,
        f"{migration.target_directory_id}:stale",
        migration.group_id,
        migration.local_rank,
        migration.part_count,
        migration.target_cell_ids,
    )
    stale_target_view = eqx.tree_at(
        lambda current: current.graph_migration_evidence,
        target_view,
        stale_migration,
    )
    refused_migration = adapt(stale_target_view, target_auxiliary_layouts)
    stale_refused = (
        refused_without_partial_commit(refused_migration)
        and bool(refused_migration.diagnostics)
        and source_composition.composition_id == source_composition_id
    )
    require_global_flag(stale_refused, "collective stale graph-migration refusal")
    invalid_content_history_layout = phx.solver.FiniteElementRestartStateLayout(
        content_history_name,
        phx.solver.FiniteElementRestartStateRole.HISTORY,
        phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS,
        ownership=target_u_identity.ownership(),
    )
    rank_layouts = (
        (
            target_u_history_layout,
            invalid_content_history_layout,
            target_auxiliary_layouts[2],
        )
        if rank == 1
        else target_auxiliary_layouts
    )
    refused_bank = adapt(target_view, rank_layouts)
    bank_refused = (
        refused_without_partial_commit(refused_bank)
        and bool(refused_bank.diagnostics)
        and source_composition.composition_id == source_composition_id
    )
    require_global_flag(bank_refused, "collective P1/P0 restart-bank refusal")
    result = adapt(target_view, target_auxiliary_layouts)
    if (
        not isinstance(result, FiniteElementOwnerLocalTopologyResult)
        or not bool(np.asarray(result.committed))
        or result.rebind_receipt is None
        or not result.rebind_receipt.published
    ):
        diagnostics = "invalid owner-local FE topology result"
        if isinstance(result, FiniteElementOwnerLocalTopologyResult):
            diagnostics = result.diagnostics
        raise RuntimeError(f"Owner-local FE topology adaptation refused: {diagnostics!r}")
    continued_state = result.state
    if (
        continued_state.time <= 0.0
        or continued_state.time != stored_state.time
        or continued_state.step != stored_state.step
        or continued_state.topology_id != target_view.source_topology_id
        or continued_state.prepared_id != target_discretization.prepared_id
        or continued_state.compilation_id != target_compiled.compilation_id
        or result.manifest.state.accepted_id != continued_state.accepted_id
        or continued_state.materials is None
    ):
        raise RuntimeError(
            "FE commit lost accepted clock, target solver authority, or material state."
        )
    target_physical = solve_record(target_compiled, target_discretization)
    target_integral = float(np.sum(gather_scalar(target_physical["physical_integral"])))
    target_minimum = float(
        np.min(
            gather_scalar(
                np.inf
                if target_physical["minimum_value"] is None
                else target_physical["minimum_value"]
            )
        )
    )
    target_residual = float(np.max(gather_scalar(target_physical["residual_norm"])))
    target_rhs_norm = float(np.max(gather_scalar(target_physical["rhs_norm"])))
    target_normalized_residual = target_residual / max(1.0, target_rhs_norm)
    expected_target_values, _ = target_u_identity.shard_values(target_physical["values"])
    migrated_values = np.asarray(continued_state.fields[0], dtype=np.float64)
    target_content_values = np.asarray(continued_state.fields[1], dtype=np.float64)
    expected_values = np.asarray(expected_target_values, dtype=np.float64)
    if migrated_values.shape != expected_values.shape:
        raise RuntimeError(
            "Committed P1 FE rows differ from the current target identity plan."
        )
    if (
        result.adaptive_result_id != adaptive_result.result_id
        or adaptive_result.vertex_stencil is None
        or result.vertex_stencil_id != adaptive_result.vertex_stencil.stencil_id
        or tuple(plan.plan_id for plan in result.target_identity_plans)
        != tuple(plan.plan_id for plan in target_identities)
        or len(result.field_transfers) != 2
        or any(
            not isinstance(transfer, FiniteElementOwnerLocalFieldTransferEvidence)
            for transfer in result.field_transfers
        )
    ):
        raise RuntimeError(
            "Atomic FE topology result lost exact P1/P0 transfer identities."
        )
    transfer_by_layout = {
        transfer.field_layout_id: transfer for transfer in result.field_transfers
    }
    if set(transfer_by_layout) != {layout.layout_id for layout in target_layouts}:
        raise RuntimeError(
            "Atomic FE topology result omitted a registered P1 or P0 field transfer."
        )
    u_transfer = transfer_by_layout[target_layouts[0].layout_id]
    content_transfer = transfer_by_layout[target_layouts[1].layout_id]
    expected_content_values = np.asarray(content_transfer.target_values, dtype=np.float64)
    if tuple(plan.plan_id for plan in source_identities) != tuple(
        transfer.source_plan_id for transfer in result.field_transfers
    ) and set(transfer.source_plan_id for transfer in result.field_transfers) != {
        plan.plan_id for plan in source_identities
    }:
        raise RuntimeError(
            "Atomic FE field transfers do not bind both source identity plans."
        )
    if (
        u_transfer.conservative
        or u_transfer.semantics
        is not phx.solver.FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION
        or u_transfer.adaptive_result_id != adaptive_result.result_id
        or u_transfer.vertex_stencil_id != result.vertex_stencil_id
        or u_transfer.source_plan_id != source_u_identity.plan_id
        or u_transfer.target_plan_id != target_u_identity.plan_id
        or u_transfer.source_structure_id != source_view.source_topology_id
        or u_transfer.target_structure_id != target_view.source_topology_id
    ):
        raise RuntimeError(
            "P1 owner-local transfer lost its interpolation/stencil identity."
        )
    if (
        not content_transfer.conservative
        or content_transfer.semantics
        is not phx.solver.FiniteElementFieldTransferSemantics.CONSERVATIVE_CELL_AVERAGE
        or content_transfer.adaptive_result_id != adaptive_result.result_id
        or content_transfer.source_plan_id != source_content_identity.plan_id
        or content_transfer.target_plan_id != target_content_identity.plan_id
        or content_transfer.source_structure_id != source_view.source_topology_id
        or content_transfer.target_structure_id != target_view.source_topology_id
        or not content_transfer.conservation_evidence_id
        or content_transfer.source_integral is None
        or content_transfer.target_integral is None
        or content_transfer.conservation_defect is None
        or content_transfer.conservation_bound is None
        or expected_content_values.shape != target_content_values.shape
    ):
        raise RuntimeError(
            "Conservative P0 content transfer omitted its exact identity or integral evidence."
        )
    u_transfer_values = np.asarray(u_transfer.target_values, dtype=np.float64)
    if u_transfer_values.shape != migrated_values.shape:
        raise RuntimeError(
            "P1 field-transfer evidence rows differ from the committed accepted field."
        )
    target_content_transfer_error = float(
        np.max(np.abs(target_content_values - expected_content_values), initial=0.0)
    )
    target_u_transfer_error = float(
        np.max(np.abs(migrated_values - u_transfer_values), initial=0.0)
    )
    transfer_error = max(
        float(np.max(np.abs(migrated_values - expected_values), initial=0.0)),
        target_content_transfer_error,
        target_u_transfer_error,
    )
    target_content_integral = float(
        np.sum(
            gather_scalar(
                _periodic_p0_owner_integral(
                    target_view,
                    target_discretization,
                    target_content_identity,
                    target_content_values,
                )
            )
        )
    )
    target_content_absolute_integral = float(
        np.sum(
            gather_scalar(
                _periodic_p0_owner_integral(
                    target_view,
                    target_discretization,
                    target_content_identity,
                    np.abs(target_content_values),
                )
            )
        )
    )
    target_content_minimum_by_rank = gather_scalar(
        float(np.min(target_content_values, initial=np.inf))
    )
    target_content_maximum_by_rank = gather_scalar(
        float(np.max(target_content_values, initial=-np.inf))
    )
    target_content_minimum = float(np.min(target_content_minimum_by_rank, initial=np.inf))
    target_content_maximum = float(
        np.max(target_content_maximum_by_rank, initial=-np.inf)
    )
    target_content_rounding_bound = (
        16.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            source_content_absolute_integral + target_content_absolute_integral,
        )
        * max(
            1,
            source_cell_count + int(np.sum(target_placement[:, 0])) + 8,
        )
    )
    content_transfer_source_integral = float(np.asarray(content_transfer.source_integral))
    content_transfer_target_integral = float(np.asarray(content_transfer.target_integral))
    content_transfer_defect = float(np.asarray(content_transfer.conservation_defect))
    content_transfer_bound = float(content_transfer.conservation_bound)
    if (
        abs(source_content_integral - content_transfer_source_integral)
        > source_content_rounding_bound
        or abs(target_content_integral - content_transfer_target_integral)
        > target_content_rounding_bound
        or abs(
            content_transfer_defect
            - (content_transfer_target_integral - content_transfer_source_integral)
        )
        > target_content_rounding_bound
        or abs(content_transfer_defect) > content_transfer_bound
        or abs(target_content_integral - source_content_integral)
        > target_content_rounding_bound
        or abs(target_content_integral) > target_content_rounding_bound
        or not np.isfinite(target_content_minimum)
        or not np.isfinite(target_content_maximum)
        or target_content_maximum - target_content_minimum <= 1.0e-8
    ):
        raise RuntimeError(
            "Conservative periodic P0 content failed its owner-reduced integral or variation oracle."
        )
    committed_banks = dict(result.manifest.auxiliary_state)
    source_banks = dict(restart_manifest.auxiliary_state)
    if (
        not np.array_equal(
            np.asarray(committed_banks[u_history_name]),
            migrated_values,
        )
        or not np.array_equal(
            np.asarray(committed_banks[content_history_name]),
            np.asarray(continued_state.fields[1]),
        )
        or not np.array_equal(
            np.asarray(committed_banks[clock_history_name]),
            np.asarray(source_banks[clock_history_name]),
        )
    ):
        raise RuntimeError(
            "Owner-local FE P1/P0 history or clock banks changed during commit."
        )
    committed_integrators = dict(result.manifest.integrator_state)
    source_integrators = dict(restart_manifest.integrator_state)
    if (
        not np.array_equal(
            np.asarray(committed_integrators[previous_name]),
            np.asarray(source_integrators[previous_name]),
        )
        or not jax.dtypes.issubdtype(
            committed_integrators[rng_name].dtype, jax.dtypes.prng_key
        )
        or not np.array_equal(
            jax.random.key_data(committed_integrators[rng_name]),
            jax.random.key_data(source_integrators[rng_name]),
        )
    ):
        raise RuntimeError(
            "Owner-local FE integrator or typed RNG banks changed during commit."
        )
    source_materials = stored_state.materials
    target_materials = continued_state.materials
    if source_materials is None:
        raise RuntimeError("Cold FE source state lost its material transaction.")
    material_site = phx.equations.MaterialSiteId("steel:integration-site:0")
    source_material = source_materials.state(material_site)
    target_material = target_materials.state(material_site)
    if (
        source_material.model_id != target_material.model_id
        or source_material.state_version != target_material.state_version
        or not np.array_equal(
            np.asarray(source_material.committed),
            np.asarray(target_material.committed),
        )
        or not np.array_equal(
            np.asarray(source_material.trial),
            np.asarray(target_material.trial),
        )
    ):
        raise RuntimeError(
            "Replicated FE material state changed during owner-local commit."
        )
    receipt_composition = result.rebind_receipt.composition
    if (
        receipt_composition.entry("mesh/geometry").revision_id
        != target_manifest.manifest_id
        or receipt_composition.entry("mesh/geometry").structure_id
        != target_manifest.source_geometry_id
        or receipt_composition.entry("solver/state").revision_id
        != continued_state.accepted_id
    ):
        raise RuntimeError(
            "Published FE state is not bound to the current compact target manifest."
        )
    if (
        transfer_error > 2.0e-9
        or abs(target_integral - source_integral) > 2.0e-9 * max(1.0, abs(domain_measure))
        or abs(target_integral - domain_measure) > 2.0e-9 * max(1.0, abs(domain_measure))
        or target_minimum <= 0.0
        or not np.isfinite(target_residual)
        or target_normalized_residual > 2.0e-9
        or source_composition.composition_id != source_composition_id
        or archive.stat().st_size <= 0
    ):
        raise RuntimeError(
            "Owner-local FE next solve failed conservation, positivity, or rebind oracles."
        )
    next_step_evidence: dict[str, float] = {}

    target_content_reference_full = _periodic_p0_content_values(
        target_view, target_discretization
    )
    target_content_reference_values, _ = target_content_identity.shard_values(
        target_content_reference_full
    )

    def solve_next_step(
        accepted: FiniteElementAcceptedState,
        start_time: Any,
        end_time: Any,
        time_law: Any,
        args: Any,
    ) -> FiniteElementAttemptResult:
        local_ok = len(accepted.fields) == 2
        candidate_values = accepted.fields[0]
        candidate_content_values = accepted.fields[1] if len(accepted.fields) == 2 else ()
        physical_next: dict[str, Any] | None = None
        content_reanalysis_error = float("inf")
        try:
            law_value = float(np.asarray(time_law.value(end_time, args)))
            physical_next = solve_record(target_compiled, target_discretization)
            candidate_values, _ = target_u_identity.shard_values(physical_next["values"])
            candidate_content_values, _ = target_content_identity.shard_values(
                physical_next["content_values"]
            )
            content_reanalysis_error = float(
                np.max(
                    np.abs(
                        np.asarray(candidate_content_values, dtype=np.float64)
                        - np.asarray(target_content_reference_values, dtype=np.float64)
                    ),
                    initial=0.0,
                )
            )
            local_ok = (
                local_ok
                and np.isfinite(law_value)
                and law_value == 1.0
                and np.shape(candidate_values) == np.shape(accepted.fields[0])
                and np.shape(candidate_content_values) == np.shape(accepted.fields[1])
                and np.all(np.isfinite(np.asarray(candidate_content_values)))
                and content_reanalysis_error <= 2.0e-10
            )
        except Exception:
            local_ok = False
        all_local_solves_ok = bool(
            np.all(gather_scalar(int(local_ok), dtype=np.int32) == 1)
        )
        if not all_local_solves_ok or physical_next is None:
            return FiniteElementAttemptResult(
                accepted.fields,
                False,
                materials=accepted.materials,
                retry_requested=False,
                diagnostics={"accepted_physical_solve": False},
            )
        step_integral = float(np.sum(gather_scalar(physical_next["physical_integral"])))
        step_minimum = float(
            np.min(
                gather_scalar(
                    np.inf
                    if physical_next["minimum_value"] is None
                    else physical_next["minimum_value"]
                )
            )
        )
        step_residual = float(np.max(gather_scalar(physical_next["residual_norm"])))
        step_rhs_norm = float(np.max(gather_scalar(physical_next["rhs_norm"])))
        normalized_residual = step_residual / max(1.0, step_rhs_norm)
        step_content_integral_local = _periodic_p0_owner_integral(
            target_view,
            target_discretization,
            target_content_identity,
            candidate_content_values,
        )
        step_content_integral = float(np.sum(gather_scalar(step_content_integral_local)))
        step_content_absolute_integral = float(
            np.sum(
                gather_scalar(
                    _periodic_p0_owner_integral(
                        target_view,
                        target_discretization,
                        target_content_identity,
                        np.abs(np.asarray(candidate_content_values, dtype=np.float64)),
                    )
                )
            )
        )
        step_content_minimum = float(
            np.min(
                gather_scalar(
                    float(
                        np.min(
                            np.asarray(candidate_content_values, dtype=np.float64),
                            initial=np.inf,
                        )
                    )
                )
            )
        )
        step_content_maximum = float(
            np.max(
                gather_scalar(
                    float(
                        np.max(
                            np.asarray(candidate_content_values, dtype=np.float64),
                            initial=-np.inf,
                        )
                    )
                )
            )
        )
        step_content_rounding_bound = (
            16.0
            * np.finfo(np.float64).eps
            * max(1.0, target_content_absolute_integral + step_content_absolute_integral)
            * max(1, int(np.sum(target_placement[:, 0])) + 8)
        )
        step_accepted = (
            abs(step_integral - domain_measure) <= 2.0e-9 * max(1.0, abs(domain_measure))
            and step_minimum > 0.0
            and normalized_residual <= 2.0e-9
            and abs(step_content_integral - target_content_integral)
            <= step_content_rounding_bound
            and np.isfinite(step_content_minimum)
            and np.isfinite(step_content_maximum)
            and step_content_maximum - step_content_minimum > 1.0e-8
        )
        if not step_accepted:
            return FiniteElementAttemptResult(
                accepted.fields,
                False,
                materials=accepted.materials,
                retry_requested=False,
                diagnostics={"accepted_physical_solve": False},
            )
        next_step_evidence.update(
            {
                "integral": step_integral,
                "minimum": step_minimum,
                "residual": step_residual,
                "normalized_residual": normalized_residual,
                "content_integral": step_content_integral,
                "content_absolute_integral": step_content_absolute_integral,
                "content_minimum": step_content_minimum,
                "content_maximum": step_content_maximum,
                "content_reanalysis_error": content_reanalysis_error,
                "content_rounding_bound": step_content_rounding_bound,
            }
        )
        return FiniteElementAttemptResult(
            (candidate_values, candidate_content_values),
            True,
            materials=accepted.materials,
            diagnostics={"accepted_physical_solve": True},
        )

    step_size = float(
        np.asarray(source_integrators[previous_name], dtype=np.float64).reshape(-1)[0]
    )
    next_time = float(np.asarray(continued_state.time)) + step_size
    schedule = FiniteElementAcceptedStepSchedule(
        solve_next_step,
        schedule_id=f"{profile}-owner-local-fe-next-solve",
    )
    next_state, next_diagnostics = schedule.advance(
        continued_state,
        np.asarray(next_time, dtype=np.float64),
        phx.solver.TimeLaw.constant(1.0, law_id=f"{profile}-steady-unit-source"),
    )
    next_step_accepted = bool(np.asarray(next_diagnostics.accepted))
    require_global_flag(next_step_accepted, "accepted next FE solve step")
    if (
        next_state.time <= continued_state.time
        or next_state.step != continued_state.step + 1
        or next_state.state_version != continued_state.state_version + 1
        or next_state.accepted_id == continued_state.accepted_id
        or next_state.materials is None
    ):
        raise RuntimeError(
            "Accepted FE next solve did not advance time, step, and state identity."
        )
    next_material = next_state.materials.state(material_site)
    if (
        next_material.model_id != source_material.model_id
        or not np.array_equal(
            np.asarray(next_material.committed),
            np.asarray(source_material.committed),
        )
        or not np.array_equal(
            np.asarray(next_material.trial),
            np.asarray(source_material.trial),
        )
    ):
        raise RuntimeError(
            "The stationary FE next step changed its registered material values."
        )
    next_restart_manifest = FiniteElementRestartManifest(
        next_state,
        auxiliary_state=(
            (u_history_name, next_state.fields[0]),
            (content_history_name, next_state.fields[1]),
            (
                clock_history_name,
                np.asarray(
                    (
                        float(np.asarray(continued_state.time)),
                        float(np.asarray(next_state.time)),
                    ),
                    dtype=np.float64,
                ),
            ),
        ),
        integrator_state=(
            (previous_name, np.asarray((step_size,), dtype=np.float64)),
            (rng_name, source_integrators[rng_name]),
        ),
        auxiliary_layouts=target_auxiliary_layouts,
        integrator_layouts=integrator_layouts,
        material_layouts=material_layouts,
    )
    next_auxiliary = dict(next_restart_manifest.auxiliary_state)
    next_integrators = dict(next_restart_manifest.integrator_state)
    if (
        next_restart_manifest.state.accepted_id != next_state.accepted_id
        or not np.array_equal(
            np.asarray(next_auxiliary[u_history_name]),
            np.asarray(next_state.fields[0]),
        )
        or not np.array_equal(
            np.asarray(next_auxiliary[content_history_name]),
            np.asarray(next_state.fields[1]),
        )
        or not np.array_equal(
            np.asarray(next_auxiliary[clock_history_name]),
            np.asarray(
                (
                    float(np.asarray(continued_state.time)),
                    float(np.asarray(next_state.time)),
                ),
                dtype=np.float64,
            ),
        )
        or not np.array_equal(
            np.asarray(next_integrators[previous_name]),
            np.asarray((step_size,), dtype=np.float64),
        )
        or not jax.dtypes.issubdtype(
            next_integrators[rng_name].dtype, jax.dtypes.prng_key
        )
        or not np.array_equal(
            jax.random.key_data(next_integrators[rng_name]),
            jax.random.key_data(source_integrators[rng_name]),
        )
    ):
        raise RuntimeError("Accepted FE next solve lost its typed restart-bank state.")
    if (
        abs(next_step_evidence["integral"] - target_integral)
        > 2.0e-9 * max(1.0, abs(domain_measure))
        or next_step_evidence["minimum"] <= 0.0
        or next_step_evidence["normalized_residual"] > 2.0e-9
        or abs(next_step_evidence["content_integral"] - target_content_integral)
        > next_step_evidence["content_rounding_bound"]
        or not np.isfinite(next_step_evidence["content_minimum"])
        or not np.isfinite(next_step_evidence["content_maximum"])
        or next_step_evidence["content_maximum"] - next_step_evidence["content_minimum"]
        <= 1.0e-8
        or not np.array_equal(
            np.asarray(next_state.fields[1]),
            np.asarray(continued_state.fields[1]),
        )
    ):
        raise RuntimeError(
            "Accepted FE next solve changed positive P1 or nonconstant P0 state."
        )
    return {
        "phase": phase,
        "profile_key": profile,
        "rank": rank,
        "pid": os.getpid(),
        "source_id": source_view.source_value.source_id,
        "source_revision": source_view.source_value.source_revision,
        "source_view_id": source_view.local_view_id,
        "target_view_id": target_view.local_view_id,
        "source_manifest_id": source_manifest.manifest_id,
        "target_manifest_id": target_manifest.manifest_id,
        "source_topology_id": source_manifest.source_topology_id,
        "target_topology_id": target_manifest.source_topology_id,
        "source_geometry_id": source_manifest.source_geometry_id,
        "target_geometry_id": target_manifest.source_geometry_id,
        "source_distribution_id": source_view.distribution_id,
        "specification_id": specification.specification_id,
        "source_graph_plan_id": source_graph_plan.plan_id,
        "target_graph_plan_id": target_graph_plan.plan_id,
        "finite_element_graph_evidence_id": finite_element_graph_evidence.evidence_id,
        "finite_element_graph_result_id": finite_element_graph_evidence.result.result_id,
        "finite_element_graph_plan_id": finite_element_graph_evidence.result.plan_id,
        "finite_element_graph_owner_directory_id": (
            finite_element_graph_evidence.owner_directory_id
        ),
        "target_distribution_id": target_view.distribution_id,
        "adaptation_specification_id": adaptation_specification.specification_id,
        "adaptation_view_id": adaptation_view.local_view_id,
        "adaptation_distribution_id": adaptation_view.distribution_id,
        "adapted_topology_id": adaptive_result.target_topology_id,
        "adapted_geometry_id": adaptive_result.target_geometry_id,
        "adaptive_result_id": adaptive_result.result_id,
        "pre_graph_adaptive_result_id": pre_graph_adaptive_result.result_id,
        "adaptive_lineage_id": (
            None
            if adaptive_result.lineage is None
            else adaptive_result.lineage.lineage_id
        ),
        "pre_graph_lineage_id": (
            None
            if pre_graph_adaptive_result.lineage is None
            else pre_graph_adaptive_result.lineage.lineage_id
        ),
        "adaptive_vertex_transfer_id": (
            None
            if adaptive_result.vertex_transfer is None
            else adaptive_result.vertex_transfer.transfer_id
        ),
        "pre_graph_vertex_transfer_id": (
            None
            if pre_graph_adaptive_result.vertex_transfer is None
            else pre_graph_adaptive_result.vertex_transfer.transfer_id
        ),
        "source_field_layout_ids": [layout.layout_id for layout in source_layouts],
        "target_field_layout_ids": [layout.layout_id for layout in target_layouts],
        "source_identity_plan_ids": [plan.plan_id for plan in source_identities],
        "target_identity_plan_ids": [plan.plan_id for plan in target_identities],
        "finite_element_topology_adaptive_result_id": result.adaptive_result_id,
        "finite_element_topology_vertex_stencil_id": result.vertex_stencil_id,
        "graph_migration_evidence_id": migration.evidence_id,
        "graph_partition_result_id": migration.result.result_id,
        "graph_migration_plan_id": migration.result.plan_id,
        "graph_migration_source_directory_id": migration.source_directory_id,
        "graph_migration_target_directory_id": migration.target_directory_id,
        "p1_field_transfer_id": u_transfer.transfer_id,
        "p1_field_transfer_stencil_id": u_transfer.vertex_stencil_id,
        "p0_field_transfer_id": content_transfer.transfer_id,
        "p0_conservation_evidence_id": content_transfer.conservation_evidence_id,
        "restored_state_id": stored_state.accepted_id,
        "restored_restart_manifest_id": restart_manifest.manifest_id,
        "next_restart_manifest_id": next_restart_manifest.manifest_id,
        "continued_state_id": continued_state.accepted_id,
        "next_solve_state_id": next_state.accepted_id,
        "restored_time": float(np.asarray(stored_state.time)),
        "restored_step": int(stored_state.step),
        "topology_adaptation_time": float(np.asarray(continued_state.time)),
        "topology_adaptation_step": int(continued_state.step),
        "accepted_time": float(np.asarray(next_state.time)),
        "accepted_step": int(next_state.step),
        "source_owned_cell_counts": source_placement[:, 0].tolist(),
        "target_owned_cell_counts": target_placement[:, 0].tolist(),
        "source_integral": source_integral,
        "continued_integral": next_step_evidence["integral"],
        "domain_measure": domain_measure,
        "source_content_integral": source_content_integral,
        "source_content_absolute_integral": source_content_absolute_integral,
        "source_content_rounding_bound": source_content_rounding_bound,
        "source_content_minimum": source_content_minimum,
        "source_content_maximum": source_content_maximum,
        "source_content_minimum_by_rank": [
            float(value) if np.isfinite(value) else None
            for value in source_content_minimum_by_rank
        ],
        "source_content_maximum_by_rank": [
            float(value) if np.isfinite(value) else None
            for value in source_content_maximum_by_rank
        ],
        "target_content_integral": target_content_integral,
        "target_content_absolute_integral": target_content_absolute_integral,
        "target_content_rounding_bound": target_content_rounding_bound,
        "target_content_minimum": target_content_minimum,
        "target_content_maximum": target_content_maximum,
        "target_content_minimum_by_rank": [
            float(value) if np.isfinite(value) else None
            for value in target_content_minimum_by_rank
        ],
        "target_content_maximum_by_rank": [
            float(value) if np.isfinite(value) else None
            for value in target_content_maximum_by_rank
        ],
        "p0_transfer_source_integral": content_transfer_source_integral,
        "p0_transfer_target_integral": content_transfer_target_integral,
        "p0_transfer_conservation_defect": content_transfer_defect,
        "p0_transfer_conservation_bound": content_transfer_bound,
        "p0_transfer_conservative": content_transfer.conservative,
        "p0_transfer_max_error": target_content_transfer_error,
        "next_content_integral": next_step_evidence["content_integral"],
        "next_content_minimum": next_step_evidence["content_minimum"],
        "next_content_maximum": next_step_evidence["content_maximum"],
        "next_content_rounding_bound": next_step_evidence["content_rounding_bound"],
        "field_minimum": next_step_evidence["minimum"],
        "fe_residual_norm": next_step_evidence["residual"],
        "fe_normalized_residual": next_step_evidence["normalized_residual"],
        "source_restore_max_error": source_restore_error,
        "state_transfer_max_error": transfer_error,
        "periodic_adaptation_transfer": reader_adaptation,
        "periodic_pre_graph_adaptation_transfer": reader_pre_graph_adaptation,
        "stale_migration_collectively_refused": stale_refused,
        "stale_migration_refusal_diagnostics": refused_migration.diagnostics,
        "invalid_bank_refusal_diagnostics": refused_bank.diagnostics,
        "invalid_bank_collectively_refused": bank_refused,
        "source_reused_after_refusal": source_composition.composition_id
        == source_composition_id,
        "history_rebound": True,
        "clock_history_rebound": True,
        "integrator_rebound": True,
        "rng_rebound": True,
        "material_rebound": True,
        "rebind_published": True,
        "next_fe_solve_accepted": bool(np.asarray(next_diagnostics.accepted)),
    }


def _launch_ranks(
    phase: str,
    archive_root: Path,
    environment: dict[str, str],
    profile: str,
    /,
) -> list[dict[str, Any]]:
    from phydrax._execution_launch import (
        launch_local_processes,
        LocalProcessLaunchPlan,
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    results = launch_local_processes(
        (
            sys.executable,
            "-m",
            "tools.meshing_qualification.scenarios.cold_owner_local",
            phase,
            str(archive_root),
            profile,
        ),
        LocalProcessLaunchPlan(
            3,
            f"127.0.0.1:{port}",
        ),
        environment=environment,
        cwd=str(_REPOSITORY_ROOT),
        timeout_seconds=900.0,
    )
    records: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda value: value.process_id):
        if result.returncode != 0:
            raise RuntimeError(
                f"Owner-local FE {phase} rank {result.process_id} exited with {result.returncode}."
            )
        lines = [
            line
            for line in result.stdout.decode("utf-8", errors="strict").splitlines()
            if line.strip().startswith("{")
        ]
        if not lines:
            raise RuntimeError(
                f"Owner-local FE {phase} rank {result.process_id} returned no JSON evidence."
            )
        record = json.loads(lines[-1])
        if not isinstance(record, dict) or record.get("rank") != result.process_id:
            raise RuntimeError(
                f"Owner-local FE {phase} rank {result.process_id} returned malformed evidence."
            )
        records.append(record)
    return records


def _qualify_profile(
    profile_key: str,
    temporary_root: Path,
    environment: dict[str, str],
    /,
) -> dict[str, Any]:
    source_id, requested_size, profile_name, source_graph_shares = _PROFILE_PARAMETERS[
        profile_key
    ]
    archive_root = temporary_root / profile_key
    archive_root.mkdir()
    writer = _launch_ranks("write", archive_root, environment, profile_key)
    archive_bytes = tuple(
        (archive_root / f"{profile_key}-rank-{rank}.phx").stat().st_size
        for rank in range(3)
    )
    if any(size <= 0 for size in archive_bytes):
        raise RuntimeError(f"{profile_key} producer omitted a rank-local model artifact.")
    reader = _launch_ranks("read", archive_root, environment, profile_key)
    writer_by_rank = {int(record["rank"]): record for record in writer}
    reader_by_rank = {int(record["rank"]): record for record in reader}
    if tuple(sorted(writer_by_rank)) != (0, 1, 2) or tuple(sorted(reader_by_rank)) != (
        0,
        1,
        2,
    ):
        raise RuntimeError(f"{profile_key} did not report all three owner processes.")
    if set(record["pid"] for record in writer).intersection(
        record["pid"] for record in reader
    ):
        raise RuntimeError(f"{profile_key} reader reused a producer process.")
    for stage, by_rank in (
        ("producer", writer_by_rank),
        ("fresh reader", reader_by_rank),
    ):
        for field in (
            "specification_id",
            "source_graph_plan_id",
            "adaptation_specification_id",
            "target_graph_plan_id",
        ):
            if len({by_rank[rank][field] for rank in range(3)}) != 1:
                raise RuntimeError(
                    f"{profile_key} {stage} ranks disagree on typed request root {field}."
                )

    def require_adaptation_records(
        records: list[dict[str, Any]],
        stage: str,
        /,
        record_key: str = "periodic_adaptation_transfer",
    ) -> list[dict[str, Any]]:
        adaptations = [record[record_key] for record in records]
        for rank, adaptation in enumerate(adaptations):
            if (
                not adaptation.get("accepted", False)
                or adaptation["global_source_cell_count"] <= 0
                or adaptation["global_target_cell_count"] <= 0
                or adaptation["global_topology_changed_owner_count"] <= 0
                or adaptation["global_shared_representative_winding_count"] <= 0
                or adaptation["global_constant_field_transfer_max_error"] > 1.0e-12
            ):
                raise RuntimeError(
                    f"{profile_key} {stage} rank {rank} lacks nontrivial, "
                    "constant-preserving periodic adaptation evidence."
                )
            if not adaptation.get("empty_owner", False) and (
                not adaptation.get("orbit_and_winding_identity_preserved", False)
                or not adaptation.get("transition_id")
                or not adaptation.get("vertex_stencil_id")
                or not adaptation.get("periodic_topology_id")
                or not adaptation.get("owner_directory_id")
            ):
                raise RuntimeError(
                    f"{profile_key} {stage} rank {rank} lacks exact orbit/winding transfer identity."
                )
        return adaptations

    writer_adaptation = require_adaptation_records(writer, "producer")
    reader_adaptation = require_adaptation_records(reader, "fresh reader")
    writer_pre_graph_adaptation = require_adaptation_records(
        writer, "producer pre-graph", "periodic_pre_graph_adaptation_transfer"
    )
    reader_pre_graph_adaptation = require_adaptation_records(
        reader, "fresh reader pre-graph", "periodic_pre_graph_adaptation_transfer"
    )
    for stage, written_records, restored_records in (
        ("rebound target", writer_adaptation, reader_adaptation),
        ("pre-graph target", writer_pre_graph_adaptation, reader_pre_graph_adaptation),
    ):
        for rank, (written_adaptation, restored_adaptation) in enumerate(
            zip(written_records, restored_records, strict=True)
        ):
            for field in (
                "empty_owner",
                "source_cell_count",
                "target_cell_count",
                "topology_changed",
                "transition_id",
                "vertex_stencil_id",
                "periodic_topology_id",
                "owner_directory_id",
                "orbit_and_winding_identity_preserved",
            ):
                if written_adaptation.get(field) != restored_adaptation.get(field):
                    raise RuntimeError(
                        f"{profile_key} fresh reader changed rank {rank} {stage} adaptation {field}."
                    )
            for field in (
                "global_source_cell_count",
                "global_target_cell_count",
                "global_topology_changed_owner_count",
                "global_shared_representative_winding_count",
            ):
                if written_adaptation[field] != restored_adaptation[field]:
                    raise RuntimeError(
                        f"{profile_key} fresh reader changed global {stage} adaptation {field}."
                    )
            if (
                abs(
                    written_adaptation["global_constant_field_transfer_max_error"]
                    - restored_adaptation["global_constant_field_transfer_max_error"]
                )
                > 1.0e-12
            ):
                raise RuntimeError(
                    f"{profile_key} fresh reader changed the {stage} constant-transfer oracle."
                )
    for rank in range(3):
        written = writer_by_rank[rank]
        restored = reader_by_rank[rank]
        if (
            written["profile_key"] != profile_key
            or restored["profile_key"] != profile_key
            or written["source_id"] != source_id
            or restored["source_id"] != source_id
            or written["source_state_id"] != restored["restored_state_id"]
            or written["restart_manifest_id"] != restored["restored_restart_manifest_id"]
            or written["source_manifest_id"] != restored["source_manifest_id"]
            or written["source_geometry_id"] != restored["source_geometry_id"]
            or written["source_topology_id"] != restored["source_topology_id"]
            or written["source_revision"] != restored["source_revision"]
            or written["specification_id"] != restored["specification_id"]
            or written["adaptation_specification_id"]
            != restored["adaptation_specification_id"]
            or written["adapted_topology_id"] != restored["adapted_topology_id"]
            or written["adapted_geometry_id"] != restored["adapted_geometry_id"]
            or written["adaptive_result_id"] != restored["adaptive_result_id"]
            or written["pre_graph_adaptive_result_id"]
            != restored["pre_graph_adaptive_result_id"]
            or written["adaptive_lineage_id"] != restored["adaptive_lineage_id"]
            or written["pre_graph_lineage_id"] != restored["pre_graph_lineage_id"]
            or written["adaptive_vertex_transfer_id"]
            != restored["adaptive_vertex_transfer_id"]
            or written["pre_graph_vertex_transfer_id"]
            != restored["pre_graph_vertex_transfer_id"]
            or written["adaptive_lineage_id"] != written["pre_graph_lineage_id"]
            or written["adaptive_vertex_transfer_id"]
            != written["pre_graph_vertex_transfer_id"]
            or written["source_field_layout_ids"] != restored["source_field_layout_ids"]
            or written["source_identity_plan_ids"] != restored["source_identity_plan_ids"]
            or written["finite_element_graph_evidence_id"]
            != restored["finite_element_graph_evidence_id"]
            or written["finite_element_graph_result_id"]
            != restored["finite_element_graph_result_id"]
            or written["finite_element_graph_plan_id"]
            != restored["finite_element_graph_plan_id"]
            or written["finite_element_graph_owner_directory_id"]
            != restored["finite_element_graph_owner_directory_id"]
            or written["source_graph_plan_id"] != restored["source_graph_plan_id"]
            or written["target_graph_plan_id"] != restored["target_graph_plan_id"]
            or restored["finite_element_graph_plan_id"]
            != restored["target_graph_plan_id"]
            or restored["finite_element_graph_owner_directory_id"]
            != restored["adaptation_distribution_id"]
            or restored["graph_migration_source_directory_id"]
            != restored["adaptation_distribution_id"]
            or restored["target_topology_id"] == restored["source_topology_id"]
            or restored["adapted_topology_id"] != restored["target_topology_id"]
            or not restored["finite_element_graph_evidence_id"]
            or not restored["finite_element_graph_result_id"]
            or written["accepted_time"] != restored["restored_time"]
            or written["accepted_step"] != restored["restored_step"]
            or restored["accepted_time"] <= written["accepted_time"]
            or restored["accepted_step"] != written["accepted_step"] + 1
            or restored["next_solve_state_id"] == restored["continued_state_id"]
            or restored["next_restart_manifest_id"]
            == restored["restored_restart_manifest_id"]
        ):
            raise RuntimeError(
                f"{profile_key} rank {rank} changed its exact source or accepted-state identity."
            )
        if (
            not restored["stale_migration_collectively_refused"]
            or not restored["invalid_bank_collectively_refused"]
            or not restored["source_reused_after_refusal"]
            or not restored["history_rebound"]
            or not restored["clock_history_rebound"]
            or not restored["integrator_rebound"]
            or not restored["rng_rebound"]
            or not restored["material_rebound"]
            or not restored["rebind_published"]
            or not restored["next_fe_solve_accepted"]
            or restored["accepted_time"] <= 0.0
            or restored["field_minimum"] <= 0.0
        ):
            raise RuntimeError(
                f"{profile_key} rank {rank} lacks complete migration or next-solve evidence."
            )
    source_owned = np.asarray(writer[0]["source_owned_cell_counts"], dtype=np.int64)
    target_owned = np.asarray(reader[0]["target_owned_cell_counts"], dtype=np.int64)
    if (
        source_owned.shape != (3,)
        or target_owned.shape != (3,)
        or np.any(source_owned[:2] <= 0)
        or source_owned[2] != 0
        or np.any(target_owned[:2] != 0)
        or target_owned[2] <= 0
    ):
        raise RuntimeError(
            f"{profile_key} graph repartition did not move source owners 0/1 to target owner 2."
        )
    source_content_minima = writer[0]["source_content_minimum_by_rank"]
    source_content_maxima = writer[0]["source_content_maximum_by_rank"]
    if (
        len(source_content_minima) != 3
        or len(source_content_maxima) != 3
        or source_content_minima[2] is not None
        or source_content_maxima[2] is not None
        or any(value is None for value in source_content_minima[:2])
        or any(value is None for value in source_content_maxima[:2])
        or any(not np.isfinite(value) for value in source_content_minima[:2])
        or any(not np.isfinite(value) for value in source_content_maxima[:2])
        or max(source_content_maxima[:2]) - min(source_content_minima[:2]) <= 1.0e-8
    ):
        raise RuntimeError(
            f"{profile_key} source P0 content lacks rank-local nonconstant variation."
        )
    source_content_integral = float(writer[0]["source_content_integral"])
    source_content_absolute_integral = float(
        writer[0]["source_content_absolute_integral"]
    )
    source_content_rounding_bound = float(writer[0]["source_content_rounding_bound"])
    target_content_integral = float(reader[0]["target_content_integral"])
    target_content_absolute_integral = float(
        reader[0]["target_content_absolute_integral"]
    )
    target_content_rounding_bound = float(reader[0]["target_content_rounding_bound"])
    p0_transfer_defect = float(reader[0]["p0_transfer_conservation_defect"])
    p0_transfer_bound = float(reader[0]["p0_transfer_conservation_bound"])
    p0_summed_absolute_rounding_bound = (
        16.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            source_content_absolute_integral + target_content_absolute_integral,
        )
        * max(1, int(source_owned.sum() + target_owned.sum()) + 8)
    )
    target_content_minima = reader[0]["target_content_minimum_by_rank"]
    target_content_maxima = reader[0]["target_content_maximum_by_rank"]
    if (
        abs(float(reader[0]["source_content_integral"]) - source_content_integral)
        > source_content_rounding_bound
        or abs(
            float(reader[0]["source_content_absolute_integral"])
            - source_content_absolute_integral
        )
        > source_content_rounding_bound
        or reader[0]["source_content_minimum_by_rank"]
        != writer[0]["source_content_minimum_by_rank"]
        or reader[0]["source_content_maximum_by_rank"]
        != writer[0]["source_content_maximum_by_rank"]
        or not reader[0]["p0_transfer_conservative"]
        or not reader[0]["p0_conservation_evidence_id"]
        or not reader[0]["p0_field_transfer_id"]
        or abs(source_content_integral) > source_content_rounding_bound
        or abs(target_content_integral) > target_content_rounding_bound
        or abs(target_content_integral - source_content_integral)
        > p0_summed_absolute_rounding_bound
        or abs(p0_transfer_defect) > p0_transfer_bound
        or abs(p0_transfer_defect) > p0_summed_absolute_rounding_bound
        or abs(p0_transfer_defect - (target_content_integral - source_content_integral))
        > p0_summed_absolute_rounding_bound
        or len(target_content_minima) != 3
        or len(target_content_maxima) != 3
        or target_content_minima[0] is not None
        or target_content_minima[1] is not None
        or target_content_maxima[0] is not None
        or target_content_maxima[1] is not None
        or target_content_minima[2] is None
        or target_content_maxima[2] is None
        or not np.isfinite(target_content_minima[2])
        or not np.isfinite(target_content_maxima[2])
        or target_content_maxima[2] - target_content_minima[2] <= 1.0e-8
    ):
        raise RuntimeError(
            f"{profile_key} conservative signed periodic P0 content failed its owner-reduced rounding oracle."
        )
    source_integral = float(writer[0]["source_integral"])
    continued_integral = float(reader[0]["continued_integral"])
    quotient_measure = float(reader[0]["domain_measure"])
    exact_lattice_measure = abs(float(np.linalg.det(_CELL_BASIS)))
    if (
        abs(quotient_measure - exact_lattice_measure) > 1.0e-10
        or abs(source_integral - quotient_measure)
        > 2.0e-9 * max(1.0, abs(quotient_measure))
        or abs(continued_integral - quotient_measure)
        > 2.0e-9 * max(1.0, abs(quotient_measure))
        or abs(source_integral - continued_integral)
        > 2.0e-9 * max(1.0, abs(quotient_measure))
    ):
        raise RuntimeError(
            f"{profile_key} failed its independent quotient-measure or solve-balance oracle."
        )
    return {
        "status": "passed",
        "profile": profile_name,
        "profile_key": profile_key,
        "source": {
            "source_id": source_id,
            "source_revision": writer[0]["source_revision"],
            "requested_size": requested_size,
            "periodic_cell_basis": _CELL_BASIS.tolist(),
            "fractional_point_ids": _INPUT_POINT_IDS.tolist(),
            "fractional_points": _INPUT_FRACTIONAL_POINTS.tolist(),
            "point_owner_ranks": _INPUT_POINT_OWNERS.tolist(),
            "source_graph_target_shares": list(source_graph_shares),
            "source_topology_id": writer[0]["source_topology_id"],
            "source_geometry_id": writer[0]["source_geometry_id"],
            "source_manifest_id": writer[0]["source_manifest_id"],
            "specification_id": writer[0]["specification_id"],
            "adaptation_specification_id": writer[0]["adaptation_specification_id"],
            "adapted_topology_id": writer[0]["adapted_topology_id"],
            "adapted_geometry_id": writer[0]["adapted_geometry_id"],
            "adaptive_result_id": writer[0]["adaptive_result_id"],
            "source_content_integral": writer[0]["source_content_integral"],
            "source_content_absolute_integral": writer[0][
                "source_content_absolute_integral"
            ],
            "source_content_rounding_bound": writer[0]["source_content_rounding_bound"],
            "source_content_minimum_by_rank": writer[0]["source_content_minimum_by_rank"],
            "source_content_maximum_by_rank": writer[0]["source_content_maximum_by_rank"],
            "source_graph_plan_id": writer[0]["source_graph_plan_id"],
            "target_graph_plan_id": writer[0]["target_graph_plan_id"],
        },
        "producer_processes_exited_before_reader": True,
        "fresh_processes": True,
        "archive_bytes_by_rank": list(archive_bytes),
        "typed_request_roots": {
            "specification_ids_by_writer_rank": [
                writer_by_rank[rank]["specification_id"] for rank in range(3)
            ],
            "specification_ids_by_reader_rank": [
                reader_by_rank[rank]["specification_id"] for rank in range(3)
            ],
            "adaptation_specification_ids_by_writer_rank": [
                writer_by_rank[rank]["adaptation_specification_id"] for rank in range(3)
            ],
            "adaptation_specification_ids_by_reader_rank": [
                reader_by_rank[rank]["adaptation_specification_id"] for rank in range(3)
            ],
            "source_graph_plan_ids_by_writer_rank": [
                writer_by_rank[rank]["source_graph_plan_id"] for rank in range(3)
            ],
            "source_graph_plan_ids_by_reader_rank": [
                reader_by_rank[rank]["source_graph_plan_id"] for rank in range(3)
            ],
            "target_graph_plan_ids_by_writer_rank": [
                writer_by_rank[rank]["target_graph_plan_id"] for rank in range(3)
            ],
            "target_graph_plan_ids_by_reader_rank": [
                reader_by_rank[rank]["target_graph_plan_id"] for rank in range(3)
            ],
        },
        "finite_element_graph_partition": {
            "evidence_ids_by_writer_rank": [
                writer_by_rank[rank]["finite_element_graph_evidence_id"]
                for rank in range(3)
            ],
            "evidence_ids_by_reader_rank": [
                reader_by_rank[rank]["finite_element_graph_evidence_id"]
                for rank in range(3)
            ],
            "result_ids_by_writer_rank": [
                writer_by_rank[rank]["finite_element_graph_result_id"]
                for rank in range(3)
            ],
            "result_ids_by_reader_rank": [
                reader_by_rank[rank]["finite_element_graph_result_id"]
                for rank in range(3)
            ],
            "plan_ids_by_writer_rank": [
                writer_by_rank[rank]["finite_element_graph_plan_id"] for rank in range(3)
            ],
            "plan_ids_by_reader_rank": [
                reader_by_rank[rank]["finite_element_graph_plan_id"] for rank in range(3)
            ],
            "owner_directory_ids_by_writer_rank": [
                writer_by_rank[rank]["finite_element_graph_owner_directory_id"]
                for rank in range(3)
            ],
            "owner_directory_ids_by_reader_rank": [
                reader_by_rank[rank]["finite_element_graph_owner_directory_id"]
                for rank in range(3)
            ],
        },
        "finite_element_dof_identity_plans": {
            "source_layout_ids_by_writer_rank": [
                writer_by_rank[rank]["source_field_layout_ids"] for rank in range(3)
            ],
            "source_layout_ids_by_reader_rank": [
                reader_by_rank[rank]["source_field_layout_ids"] for rank in range(3)
            ],
            "source_plan_ids_by_writer_rank": [
                writer_by_rank[rank]["source_identity_plan_ids"] for rank in range(3)
            ],
            "source_plan_ids_by_reader_rank": [
                reader_by_rank[rank]["source_identity_plan_ids"] for rank in range(3)
            ],
            "target_layout_ids_by_rank": [
                reader_by_rank[rank]["target_field_layout_ids"] for rank in range(3)
            ],
            "target_plan_ids_by_rank": [
                reader_by_rank[rank]["target_identity_plan_ids"] for rank in range(3)
            ],
        },
        "periodic_p0_content_oracle": {
            "field": "content",
            "field_layout_id": reader[0]["target_field_layout_ids"][1],
            "analytic_field": "sin(2*pi*(xi_0+2*xi_1))",
            "transfer_semantics": "conservative_cell_average",
            "source_owner_reduced_integral": source_content_integral,
            "target_owner_reduced_integral": target_content_integral,
            "source_absolute_cell_content_sum": source_content_absolute_integral,
            "target_absolute_cell_content_sum": target_content_absolute_integral,
            "source_integral_rounding_bound": source_content_rounding_bound,
            "target_integral_rounding_bound": target_content_rounding_bound,
            "summed_absolute_content_rounding_bound": p0_summed_absolute_rounding_bound,
            "source_minimum_by_rank": source_content_minima,
            "source_maximum_by_rank": source_content_maxima,
            "target_minimum_by_rank": target_content_minima,
            "target_maximum_by_rank": target_content_maxima,
            "transfer_id": reader[0]["p0_field_transfer_id"],
            "conservation_evidence_id": reader[0]["p0_conservation_evidence_id"],
            "transfer_source_integral": reader[0]["p0_transfer_source_integral"],
            "transfer_target_integral": reader[0]["p0_transfer_target_integral"],
            "conservation_defect": p0_transfer_defect,
            "conservation_bound": p0_transfer_bound,
        },
        "producer": writer,
        "reader": reader,
        "periodic_adaptation_transfer": {
            "producer": writer_adaptation,
            "fresh_reader": reader_adaptation,
            "global_source_cell_count": writer_adaptation[0]["global_source_cell_count"],
            "global_target_cell_count": writer_adaptation[0]["global_target_cell_count"],
            "global_topology_changed_owner_count": writer_adaptation[0][
                "global_topology_changed_owner_count"
            ],
            "global_shared_representative_winding_count": writer_adaptation[0][
                "global_shared_representative_winding_count"
            ],
            "global_constant_field_transfer_max_error": writer_adaptation[0][
                "global_constant_field_transfer_max_error"
            ],
            "transition_ids": [
                record.get("transition_id") for record in writer_adaptation
            ],
            "vertex_stencil_ids": [
                record.get("vertex_stencil_id") for record in writer_adaptation
            ],
            "periodic_topology_ids": [
                record.get("periodic_topology_id") for record in writer_adaptation
            ],
        },
        "periodic_pre_graph_adaptation_transfer": {
            "producer": writer_pre_graph_adaptation,
            "fresh_reader": reader_pre_graph_adaptation,
            "global_source_cell_count": writer_pre_graph_adaptation[0][
                "global_source_cell_count"
            ],
            "global_target_cell_count": writer_pre_graph_adaptation[0][
                "global_target_cell_count"
            ],
            "global_topology_changed_owner_count": writer_pre_graph_adaptation[0][
                "global_topology_changed_owner_count"
            ],
            "global_shared_representative_winding_count": writer_pre_graph_adaptation[0][
                "global_shared_representative_winding_count"
            ],
        },
        "adaptive_result_identity": {
            "pre_graph_result_ids_by_writer_rank": [
                writer_by_rank[rank]["pre_graph_adaptive_result_id"] for rank in range(3)
            ],
            "pre_graph_result_ids_by_reader_rank": [
                reader_by_rank[rank]["pre_graph_adaptive_result_id"] for rank in range(3)
            ],
            "rebound_result_ids_by_writer_rank": [
                writer_by_rank[rank]["adaptive_result_id"] for rank in range(3)
            ],
            "rebound_result_ids_by_reader_rank": [
                reader_by_rank[rank]["adaptive_result_id"] for rank in range(3)
            ],
            "pre_graph_lineage_ids_by_rank": [
                writer_by_rank[rank]["pre_graph_lineage_id"] for rank in range(3)
            ],
            "rebound_lineage_ids_by_rank": [
                writer_by_rank[rank]["adaptive_lineage_id"] for rank in range(3)
            ],
            "pre_graph_vertex_transfer_ids_by_rank": [
                writer_by_rank[rank]["pre_graph_vertex_transfer_id"] for rank in range(3)
            ],
            "rebound_vertex_transfer_ids_by_rank": [
                writer_by_rank[rank]["adaptive_vertex_transfer_id"] for rank in range(3)
            ],
            "target_view_ids_by_rank": [
                reader_by_rank[rank]["target_view_id"] for rank in range(3)
            ],
            "target_distribution_ids_by_rank": [
                reader_by_rank[rank]["target_distribution_id"] for rank in range(3)
            ],
        },
        "placement_oracles": {
            "source_owned_cell_counts_by_rank": source_owned.tolist(),
            "target_owned_cell_counts_by_rank": target_owned.tolist(),
            "source_distribution_id": reader[0]["source_distribution_id"],
            "target_distribution_id": reader[0]["target_distribution_id"],
            "migration_evidence_ids_by_rank": [
                reader_by_rank[rank]["graph_migration_evidence_id"] for rank in range(3)
            ],
        },
        "scientific_oracles": {
            "source_integral": source_integral,
            "continued_integral": continued_integral,
            "independent_periodic_quotient_measure": exact_lattice_measure,
            "positive_state_minimum": reader[0]["field_minimum"],
            "accepted_time": reader[0]["accepted_time"],
            "accepted_step": reader[0]["accepted_step"],
            "source_state_ids": [
                writer_by_rank[rank]["source_state_id"] for rank in range(3)
            ],
            "restored_state_ids": [
                reader_by_rank[rank]["restored_state_id"] for rank in range(3)
            ],
            "state_transfer_max_errors": [
                reader_by_rank[rank]["state_transfer_max_error"] for rank in range(3)
            ],
            "fe_normalized_residual": reader[0]["fe_normalized_residual"],
            "signed_p0_source_integral": source_content_integral,
            "signed_p0_target_integral": target_content_integral,
            "signed_p0_integral_defect": p0_transfer_defect,
            "signed_p0_integral_rounding_bound": p0_summed_absolute_rounding_bound,
        },
        "collective_oracles": {
            "stale_migration_refused_without_partial_commit": True,
            "one_rank_restart_layout_refusal_without_partial_commit": True,
            "saved_source_reused_after_both_refusals": True,
            "history_clock_integrator_rng_material_rebound": True,
            "target_rebind_published": True,
            "next_compiled_fe_solve_accepted": True,
            "stale_graph_migration_refusal_diagnostics_by_rank": [
                reader_by_rank[rank]["stale_migration_refusal_diagnostics"]
                for rank in range(3)
            ],
            "invalid_bank_refusal_diagnostics_by_rank": [
                reader_by_rank[rank]["invalid_bank_refusal_diagnostics"]
                for rank in range(3)
            ],
            "rollback_evidence_by_rank": [
                {
                    "stale_refusal": reader_by_rank[rank][
                        "stale_migration_collectively_refused"
                    ],
                    "invalid_bank_refusal": reader_by_rank[rank][
                        "invalid_bank_collectively_refused"
                    ],
                    "source_reused": reader_by_rank[rank]["source_reused_after_refusal"],
                }
                for rank in range(3)
            ],
        },
        "tolerances": {
            "quotient_measure_absolute": 1.0e-10,
            "fe_normalized_residual": 2.0e-9,
            "source_and_target_integral_relative": 2.0e-9,
            "owner_local_state_transfer_absolute": 2.0e-9,
            "p0_owner_reduced_integral_rounding": p0_summed_absolute_rounding_bound,
            "p0_transaction_conservation_bound": p0_transfer_bound,
        },
    }


def native_cold_owner_local() -> dict[str, Any]:
    import jax

    from phydrax import config as phx_config
    from phydrax._config_types import RuntimeConfig

    if jax.process_count() != 1:
        raise RuntimeError(
            "Owner-local FE qualification must launch its own fresh three-process groups."
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
    environment = phx_config.environment(
        RuntimeConfig(
            enable_x64=True,
            platforms=("cpu",),
            cpu_collectives_implementation="gloo",
        ),
        base=base_environment,
    )
    meshcore_library = os.environ.get("PHYDRAX_MESHCORE_LIBRARY")
    if meshcore_library is None:
        environment.pop("PHYDRAX_MESHCORE_LIBRARY", None)
    else:
        environment["PHYDRAX_MESHCORE_LIBRARY"] = meshcore_library
    environment["PYTHONPATH"] = str(_REPOSITORY_ROOT)
    with TemporaryDirectory(prefix="phydrax-native-cold-owner-local-fe-") as temporary:
        root = Path(temporary)
        profiles = tuple(
            _qualify_profile(profile_key, root, environment)
            for profile_key in _PROFILE_PARAMETERS
        )
    if len(profiles) != len(_PROFILE_PARAMETERS) or any(
        profile["status"] != "passed" for profile in profiles
    ):
        raise RuntimeError(
            "The complete owner-local FE routine/larger profile matrix did not pass."
        )
    return {
        "status": "passed",
        "workflow": (
            "fresh-process three-owner FE source archive, owner-local topology adaptation "
            "with changed placement, restart-bank rebind, and next physical solve"
        ),
        "source_registry": (
            "tools/meshing_qualification/native_case_migration.json#distributed-lifecycle"
        ),
        "producer_processes_exited_before_each_reader": True,
        "profiles": list(profiles),
        "missing_production_contracts": [],
    }


if __name__ == "__main__" and len(sys.argv) == 4:
    print(
        json.dumps(
            _worker(sys.argv[1], Path(sys.argv[2]), sys.argv[3]),
            sort_keys=True,
        )
    )
