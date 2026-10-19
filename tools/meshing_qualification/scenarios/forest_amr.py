# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import Any, final

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import model_array_bytes
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import (
    _measured,
    _runtime_identity,
    qualification_failure,
)


def _forest_balance(topology: phx.discretization.ForestHierarchyTopology, /) -> Any:
    """Level histogram and the largest level jump across shared faces."""
    levels = topology.leaf_levels().astype(np.int64)
    workset = topology.workset
    valid = np.asarray(workset.face_valid)
    minus = levels[np.asarray(workset.face_minus)[valid]]
    plus = levels[np.asarray(workset.face_plus)[valid]]
    return {
        "topology_id": topology.topology_id,
        "epoch": topology.epoch.index,
        "leaves": topology.leaf_count,
        "faces": topology.face_count,
        "level_histogram": np.bincount(
            levels, minlength=topology.plan.maximum_level + 1
        ).tolist(),
        "maximum_face_level_jump": int(np.max(np.abs(minus - plus))),
        "hanging_faces": int(
            np.count_nonzero(np.asarray(workset.face_coarse_fine)[valid])
        ),
    }


@final
class _ForestAdaptationPhase(phx.StrictModule):
    plan: phx.discretization.ForestPlan
    source: phx.discretization.ForestHierarchyTopology
    adapted: phx.discretization.ForestAdaptResult
    before: dict[str, Any]
    after: dict[str, Any]
    closures: int
    initialize_stage: dict[str, float]
    refinement_stages: list[dict[str, float]]
    adapt_stage: dict[str, float]


@final
class _ForestTransferPhase(phx.StrictModule):
    transition: phx.discretization.ForestFieldTransition
    field: jax.Array
    transferred: phx.discretization.ForestTransferResult
    content_before: float
    content_after: float
    residual: float
    reported: float
    constant_error: float
    transition_stage: dict[str, float]
    apply_stage: dict[str, float]


def _forest_adaptation_phase() -> _ForestAdaptationPhase:
    """Refine with face-balance closure, then accept one mixed refine/coarsen epoch."""
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4),
            phx.discretization.UniformCellAxisSpec(4),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    plan = phx.discretization.ForestPlan(
        grid,
        maximum_level=4,
        balance=phx.discretization.AMRBalanceStencil.FACE,
        maximum_leaf_capacity=1 << 12,
    )
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    initial, initialize_stage = _measured(lambda: compiler.initialize(1))

    topology = initial.topology
    finest = np.asarray(plan.root_shape, dtype=np.int64) << plan.maximum_level
    # Drive one interior point to the maximum level; every step forces 2:1
    # closure refinements of coarser face neighbors.
    point = (finest * 3 // 8)[None, :]
    refinement_stages, closures = [], 0
    for _ in range(plan.maximum_level - 1):
        marks = np.zeros((topology.signature.leaf_capacity,), dtype=np.int8)
        marks[
            topology.locate_cells(np.asarray([plan.maximum_level], dtype=np.int64), point)
        ] = 1
        refined, stage = _measured(
            lambda topology=topology, marks=marks: compiler.adapt(topology, marks)
        )

        if not refined.status.successful:
            raise qualification_failure(
                "qualify_forest_amr",
                (f"Forest refinement failed: {refined.status.message}"),
                {"refined": refined},
            )

        closures += refined.evidence.balance_refinements
        refinement_stages.append(stage)

        topology = refined.topology
    source = topology
    # One mixed epoch: refine near the center, coarsen every family on the right.
    levels = source.leaf_levels().astype(np.int64)
    shift = plan.maximum_level - levels
    lower = source.leaf_coordinates().astype(np.int64) << shift[:, None]
    centers = (lower + (1 << shift)[:, None] / 2.0) / finest
    marks = np.zeros((source.signature.leaf_capacity,), dtype=np.int8)
    marks[: source.leaf_count] = np.where(centers[:, 0] > 0.75, -1, 0)
    marks[
        source.locate_cells(
            np.asarray([plan.maximum_level], dtype=np.int64), (finest * 5 // 8)[None, :]
        )
    ] = 1
    adapted, adapt_stage = _measured(lambda: compiler.adapt(source, marks))

    evidence = adapted.evidence

    target = adapted.topology
    before, after = _forest_balance(source), _forest_balance(target)
    if (
        not adapted.status.successful
        or evidence.coarsened_families == 0
        or evidence.requested_refinements == 0
        or closures == 0
        or max(before["maximum_face_level_jump"], after["maximum_face_level_jump"]) > 1
    ):
        raise qualification_failure(
            "qualify_forest_amr",
            ("Forest adaptation did not refine, coarsen, and stay 2:1."),
            {
                "adapted": adapted,
                "after": after,
                "before": before,
                "closures": closures,
                "evidence": evidence,
            },
        )
    return _ForestAdaptationPhase(
        plan,
        source,
        adapted,
        before,
        after,
        closures,
        initialize_stage,
        refinement_stages,
        adapt_stage,
    )


def _forest_transfer_phase(
    source: phx.discretization.ForestHierarchyTopology,
    target: phx.discretization.ForestHierarchyTopology,
) -> _ForestTransferPhase:
    """Apply the field transition and certify conservation and constants."""
    transition, transition_stage = _measured(
        lambda: phx.discretization.ForestFieldTransition(source, target)
    )

    routes = transition.routes
    capacity = source.signature.leaf_capacity
    rng = np.random.default_rng(11)
    field = jnp.asarray(
        np.where(
            np.asarray(source.workset.leaf_valid),
            3.0 + rng.normal(size=(capacity,)),
            0.0,
        )
    )
    transferred, apply_stage = _measured(
        lambda: jax.block_until_ready(routes.apply(field))
    )
    content_before = float(jnp.sum(routes.source_measures * field))

    content_after = float(jnp.sum(routes.target_measures * transferred.values))
    residual = abs(content_after - content_before) / abs(content_before)

    reported = float(jnp.max(jnp.abs(transferred.conservation_residual)))
    constant = np.asarray(routes.apply(jnp.full((capacity,), 2.5)).values)
    constant_error = float(
        np.max(np.abs(constant[np.asarray(target.workset.leaf_valid)] - 2.5))
    )
    if (
        not bool(transferred.successful)
        or not transition.properties.conservative
        or not transition.properties.constant_preserving
        or residual > 1e-13
        or reported > 1e-12 * abs(content_before)
        or constant_error > 1e-14
    ):
        raise qualification_failure(
            "qualify_forest_amr",
            ("Forest transfer failed conservation or constants."),
            {
                "constant_error": constant_error,
                "content_before": content_before,
                "reported": reported,
                "residual": residual,
                "transferred": transferred,
                "transition": transition,
            },
        )
    return _ForestTransferPhase(
        transition,
        field,
        transferred,
        content_before,
        content_after,
        residual,
        reported,
        constant_error,
        transition_stage,
        apply_stage,
    )


@final
class _ForestMigrationPhase(phx.StrictModule):
    partition_plan: phx.discretization.ForestPartitionPlan
    source_partition: phx.discretization.PreparedForestPartition
    target_partition: phx.discretization.PreparedForestPartition
    migration: phx.discretization.ForestMigrationPlan
    migration_error: float
    source_partition_stage: dict[str, float]
    target_partition_stage: dict[str, float]
    migration_stage: dict[str, float]


def _forest_migration_phase(
    source: phx.discretization.ForestHierarchyTopology,
    target: phx.discretization.ForestHierarchyTopology,
    transfer: _ForestTransferPhase,
) -> _ForestMigrationPhase:
    """Certify partition migration against the accepted conservative field transfer."""
    transition, field, transferred = (
        transfer.transition,
        transfer.field,
        transfer.transferred,
    )
    partition_plan = phx.discretization.ForestPartitionPlan(4)
    source_partition, source_partition_stage = _measured(
        lambda: partition_plan.prepare(source)
    )
    target_partition, target_partition_stage = _measured(
        lambda: partition_plan.prepare(target)
    )
    migration, migration_stage = _measured(
        lambda: source_partition.migration_to(target_partition, transition=transition)
    )

    migrated = target_partition.unpack(migration.migrate(source_partition.pack(field)))

    migration_error = float(jnp.max(jnp.abs(migrated - transferred.values)))
    round_trip = bool(
        jnp.array_equal(source_partition.unpack(source_partition.pack(field)), field)
    )
    partitions = {
        "source": source_partition.evidence,
        "target": target_partition.evidence,
    }
    if (
        migration_error > 1e-12
        or not round_trip
        or any(min(value.ghost_counts) == 0 for value in partitions.values())
    ):
        raise qualification_failure(
            "qualify_forest_amr",
            ("Distributed forest migration disagreed with the transfer."),
            {
                "migration_error": migration_error,
                "partitions": partitions,
                "round_trip": round_trip,
            },
        )
    return _ForestMigrationPhase(
        partition_plan,
        source_partition,
        target_partition,
        migration,
        migration_error,
        source_partition_stage,
        target_partition_stage,
        migration_stage,
    )


def qualify_forest_amr() -> dict[str, object]:
    """Qualify balanced adaptation, conservative transfer, and distributed migration."""
    adaptation = _forest_adaptation_phase()
    plan, source, adapted = adaptation.plan, adaptation.source, adaptation.adapted
    target, evidence = adapted.topology, adapted.evidence
    before, after, closures = adaptation.before, adaptation.after, adaptation.closures
    initialize_stage = adaptation.initialize_stage
    refinement_stages, adapt_stage = adaptation.refinement_stages, adaptation.adapt_stage
    transfer = _forest_transfer_phase(source, target)
    transition = transfer.transition
    content_before, content_after = transfer.content_before, transfer.content_after
    residual, reported, constant_error = (
        transfer.residual,
        transfer.reported,
        transfer.constant_error,
    )
    transition_stage, apply_stage = transfer.transition_stage, transfer.apply_stage
    distribution = _forest_migration_phase(source, target, transfer)
    partition_plan = distribution.partition_plan
    source_partition, target_partition = (
        distribution.source_partition,
        distribution.target_partition,
    )
    migration, migration_error = distribution.migration, distribution.migration_error
    source_partition_stage = distribution.source_partition_stage
    target_partition_stage, migration_stage = (
        distribution.target_partition_stage,
        distribution.migration_stage,
    )
    partitions = {
        "source": source_partition.evidence,
        "target": target_partition.evidence,
    }
    return {
        "status": "passed",
        "source": {
            "plan_id": plan.plan_id,
            "topology_id": source.topology_id,
            "epoch_id": source.epoch.epoch_id,
            "leaves": source.leaf_count,
            "partition_id": source_partition.partition_id,
        },
        "target": {
            "topology_id": target.topology_id,
            "epoch_id": target.epoch.epoch_id,
            "leaves": target.leaf_count,
            "adapt_result_id": adapted.result_id,
            "adapt_evidence_id": evidence.evidence_id,
            "partition_id": target_partition.partition_id,
        },
        "runtime": _runtime_identity(),
        "quality": {"before": before, "after": after},
        "balance": {
            "stencil": plan.balance.value,
            "closure_refinements_while_refining": closures,
            "closure_refinements_mixed_epoch": evidence.balance_refinements,
            "balance_vetoed_families": evidence.balance_vetoed_families,
            "depth_limited_refinements": evidence.depth_limited_refinements,
        },
        "adaptation": {
            "requested_refinements": evidence.requested_refinements,
            "requested_coarsenings": evidence.requested_coarsenings,
            "coarsened_families": evidence.coarsened_families,
            "retained_leaves": transition.retained_leaves,
            "prolonged_leaves": transition.prolonged_leaves,
            "restricted_leaves": transition.restricted_leaves,
        },
        "transfer": {
            "transition_id": transition.transition_id,
            "conservative": transition.properties.conservative,
            "constant_preserving": transition.properties.constant_preserving,
            "constant_error": constant_error,
            "distributed_migration_error": migration_error,
        },
        "conservation": {
            "content_before": content_before,
            "content_after": content_after,
            "relative_residual": residual,
            "reported_residual": reported,
        },
        "distribution": {
            "parts": partition_plan.part_count,
            "ghost_stencil": partition_plan.ghost_stencil.value,
            "migration_id": migration.migration_id,
            "moved_leaves": migration.moved_leaves,
            **{
                name: {
                    "evidence_id": value.evidence_id,
                    "part_leaf_counts": list(value.part_leaf_counts),
                    "part_costs": list(value.part_costs),
                    "ghost_counts": list(value.ghost_counts),
                    "maximum_imbalance": value.maximum_imbalance,
                    "cut_faces": value.cut_faces,
                }
                for name, value in partitions.items()
            },
        },
        "resources": {
            "initialize": initialize_stage,
            "refinements": refinement_stages,
            "mixed_adaptation": adapt_stage,
            "transition": transition_stage,
            "transfer_apply": apply_stage,
            "source_partition": source_partition_stage,
            "target_partition": target_partition_stage,
            "migration": migration_stage,
            "retained_bytes": {
                "source_topology": model_array_bytes(source),
                "target_topology": model_array_bytes(target),
                "transition": model_array_bytes(transition),
                "source_partition": model_array_bytes(source_partition),
                "target_partition": model_array_bytes(target_partition),
                "migration": model_array_bytes(migration),
            },
        },
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_forest_amr()
