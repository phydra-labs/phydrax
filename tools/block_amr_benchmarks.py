#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Synchronized fixed-block AMR benchmark and comparison harness."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._comparison import (
    compare_performance,
    PerformancePolicy,
)
from benchmarks._runtime import (
    capture_environment,
    CompilationTiming,
    compiler_evidence,
    CompilerEvidence,
    DurationDistribution,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    synchronize,
)
from phydrax import lifecycle, linalg
from phydrax.discretization import (
    AdaptiveDyadicGridPlan,
    FiniteVolumePlan,
    MortonAddressPlan,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.discretization.amr import (
    BlockAMRPartitionPlan,
    BlockHierarchyPlan,
    BlockHierarchyState,
    BlockLevelPlan,
    BlockLevelState,
    CompositeAMRCellLayout,
    FDAMRHierarchyPlan,
)
from phydrax.discretization.finite_volume import (
    BlockAMRFiniteVolumePlan,
    composite_amr_multigrid_builder,
    CompositeAMRDiffusionPlan,
    DyadicFiniteVolumePlan,
    ExtrapolationBoundary,
    FiniteVolumeBoundaryPair,
    FiniteVolumeBoundarySet,
    FiniteVolumeMethodPlan,
    PiecewiseConstantReconstruction,
    PreparedFiniteVolumeDynamics,
    RusanovFluxPlan,
    UnstructuredFiniteVolumeBoundarySet,
    UnstructuredFiniteVolumeMethodPlan,
)
from phydrax.equations import (
    compile_conservation_problem,
    ConservationProblemIR,
    ScalarConservationSystem,
)
from phydrax.execution import ExecutionGroup, ExecutionGroupSpec
from tools.block_amr_qualification import (
    _boundaries,
    _cell_centers,
    _configuration,
    _halo_evidence,
    _hierarchy,
    _method,
    _runtime,
    _state,
    _system,
    _topology_evidence,
    _tree_norm,
)


GateStatus = Literal["pass", "fail", "inconclusive"]


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    smoke: bool
    warmup: int
    repeats: int
    comparison_cells: int
    dyadic_depth: int


def _settings(
    *, smoke: bool, warmup: int | None, repeats: int | None
) -> BenchmarkSettings:
    default_warmup, default_repeats = (1, 2) if smoke else (3, 9)
    warmup_ = default_warmup if warmup is None else int(warmup)
    repeats_ = default_repeats if repeats is None else int(repeats)
    if warmup_ < 0 or repeats_ <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive.")
    if smoke:
        warmup_ = min(warmup_, 1)
        repeats_ = min(repeats_, 2)
    return BenchmarkSettings(
        smoke=smoke,
        warmup=warmup_,
        repeats=repeats_,
        comparison_cells=8 if smoke else 16,
        dyadic_depth=3 if smoke else 4,
    )


def _status(passed: bool) -> GateStatus:
    return "pass" if passed else "fail"


def _timing_dict(value: CompilationTiming) -> dict[str, float]:
    return {
        "lowering_seconds": value.lowering_seconds,
        "compilation_seconds": value.compilation_seconds,
    }


def _compiler_dict(value: CompilerEvidence) -> dict[str, Any]:
    return {
        "source": value.source,
        "flops": value.flops,
        "bytes_accessed": value.bytes_accessed,
        "argument_bytes": value.argument_bytes,
        "output_bytes": value.output_bytes,
        "temporary_bytes": value.temporary_bytes,
        "generated_code_bytes": value.generated_code_bytes,
        "estimated_device_memory_bytes": value.estimated_device_memory_bytes,
        "unavailable_reason": value.unavailable_reason,
    }


def _compiled_phase(
    name: str,
    function: Callable[..., Any],
    arguments: tuple[Any, ...],
    settings: BenchmarkSettings,
    /,
    *,
    evidence: Mapping[str, Any],
    passed: Callable[[Any], bool],
    required: bool = True,
) -> tuple[Any, dict[str, Any]]:
    jitted = eqx.filter_jit(function)
    executable, compilation = measure_lower_and_compile(
        lambda: jitted.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    result, samples = measure_repeated(
        lambda: executable(*arguments),
        warmup=settings.warmup,
        repeats=settings.repeats,
    )
    cost = executable.compiled.cost_analysis()
    memory = executable.compiled.memory_analysis()
    unavailable = (
        "Compiler cost and memory analysis were both unavailable on this backend."
        if not cost and memory is None
        else None
    )
    compiler = compiler_evidence(
        cost,
        memory,
        source="jax-lowered-compiled-executable",
        unavailable_reason=unavailable,
    )
    gate_passed = bool(passed(result))
    return result, {
        "name": name,
        "required": required,
        "status": _status(gate_passed),
        "reason": None,
        "compilation": _timing_dict(compilation),
        "steady": samples.to_seconds_dict(),
        "compiler": _compiler_dict(compiler),
        "logical_argument_bytes": logical_array_bytes(arguments),
        "logical_result_bytes": logical_array_bytes(result),
        "evidence": dict(evidence),
    }


def _host_phase(
    name: str,
    operation: Callable[[], Any],
    settings: BenchmarkSettings,
    /,
    *,
    evidence: Callable[[Any], Mapping[str, Any]],
    passed: Callable[[Any], bool],
    required: bool = True,
) -> tuple[Any, dict[str, Any]]:
    for _ in range(settings.warmup):
        operation()
    samples = []
    result = None
    for _ in range(settings.repeats):
        result, elapsed = measure_host(operation)
        samples.append(elapsed)
    distribution = DurationDistribution(tuple(samples))
    if result is None:
        raise RuntimeError("Host benchmark phase did not execute.")
    gate_passed = bool(passed(result))
    return result, {
        "name": name,
        "required": required,
        "status": _status(gate_passed),
        "reason": None,
        "steady": distribution.to_seconds_dict(),
        "evidence": dict(evidence(result)),
    }


def _inconclusive_phase(
    name: str,
    reason: str,
    evidence: Mapping[str, Any],
    /,
    *,
    required: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "required": required,
        "status": "inconclusive",
        "reason": reason,
        "steady": DurationDistribution(()).to_seconds_dict(),
        "evidence": dict(evidence),
    }


def _hierarchy_from_values(topology: Any, values: Sequence[Any]) -> BlockHierarchyState:
    return BlockHierarchyState(
        topology,
        tuple(
            BlockLevelState(plan, metadata, value)
            for plan, metadata, value in zip(
                topology.plan.levels,
                topology.levels,
                values,
                strict=True,
            )
        ),
    )


def _occupancy(topology: Any) -> dict[str, Any]:
    active = [
        int(np.count_nonzero(np.asarray(metadata.active))) for metadata in topology.levels
    ]
    capacities = [level.maximum_blocks for level in topology.plan.levels]
    return {
        "active_blocks": active,
        "block_capacities": capacities,
        "occupancy_fraction": [
            count / capacity for count, capacity in zip(active, capacities, strict=True)
        ],
        "active_cells": [
            count * int(np.prod(level.block_shape))
            for count, level in zip(active, topology.plan.levels, strict=True)
        ],
    }


def _compiler_phases(
    settings: BenchmarkSettings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    prepared, middle, refined = _hierarchy(periodic=True)
    topology = refined.topology
    state = _state(
        refined,
        lambda x: 1.0 + 0.1 * jnp.sin(2.0 * jnp.pi * x),
    )
    values = tuple(level.values for level in state.levels)

    coarse_tags = jnp.zeros((4, 4), dtype="bool").at[1:3].set(True)
    middle_tags = jnp.zeros((16, 2), dtype="bool").at[3, 1].set(True)
    _, topology_phase = _host_phase(
        "host-topology-compilation",
        lambda: prepared.compile_topology(
            middle.topology,
            (coarse_tags, middle_tags),
        ),
        settings,
        evidence=lambda result: {
            "configuration": _configuration(result.topology.plan),
            "topology": _topology_evidence(result),
            "compiler_id": prepared.topology_compiler.compiler_id,
            "result_id": result.result_id,
            "route_graph_id": result.routes.route_graph_id,
        },
        passed=lambda result: (
            bool(result.status.successful)
            and result.topology.topology_id == topology.topology_id
        ),
    )
    phases.append(topology_phase)

    finite_volume_plan = BlockAMRFiniteVolumePlan(
        prepared,
        _system(),
        _method(),
        _boundaries(periodic=True),
    )
    dynamics, route_phase = _host_phase(
        "host-route-preparation",
        lambda: finite_volume_plan.prepare(topology),
        settings,
        evidence=lambda result: {
            "dynamics_id": result.dynamics_id,
            "face_route_ids": [route.block_id for route in result.face_routes],
            "fill_patch_route_ids": [plan.plan_id for plan in result.fill_patch_plans],
            "coarse_fine_route_pairs": [
                list(pair) for pair in result.coarse_fine_route_pairs
            ],
            "face_routes": [
                {
                    "route_id": route.block_id,
                    "kind": route.block_kind,
                    "level": route.level,
                    "axis": route.axis,
                    "faces": route.owner_cells.size,
                }
                for route in result.face_routes
            ],
        },
        passed=lambda result: (
            len(result.face_routes) > 0 and len(result.coarse_fine_route_pairs) == 2
        ),
    )
    phases.append(route_phase)

    def fill_operation(level_values):
        current = _hierarchy_from_values(topology, level_values)
        fill = dynamics.fill_patch(0.0, current)
        return (
            tuple(workspace.values for workspace in fill.workspaces),
            tuple(workspace.valid for workspace in fill.workspaces),
            fill.complete,
        )

    fill, fill_phase = _compiled_phase(
        "fill-patch",
        fill_operation,
        (values,),
        settings,
        evidence={
            "fill_patch_route_ids": [plan.plan_id for plan in dynamics.fill_patch_plans],
            "halo_widths": [list(level.halo_width) for level in topology.plan.levels],
        },
        passed=lambda result: (
            bool(result[2])
            and all(bool(jnp.all(jnp.isfinite(value))) for value in result[0])
        ),
    )
    phases.append(fill_phase)
    concrete_fill = synchronize(dynamics.fill_patch(0.0, state))

    def finite_volume_operation(level_values, fill_patch):
        current = _hierarchy_from_values(topology, level_values)
        stage = dynamics.evaluate(0.0, current, fill_patch)
        return stage.residuals, stage.maximum_rate

    stage_arrays, finite_volume_phase = _compiled_phase(
        "block-finite-volume",
        finite_volume_operation,
        (values, concrete_fill),
        settings,
        evidence={
            "finite_volume_plan_id": finite_volume_plan.plan_id,
            "dynamics_id": dynamics.dynamics_id,
            "precision_policy_id": finite_volume_plan.precision.policy_id,
        },
        passed=lambda result: (
            all(bool(jnp.all(jnp.isfinite(value))) for value in result[0])
            and bool(jnp.all(jnp.isfinite(result[1])))
        ),
    )
    phases.append(finite_volume_phase)
    concrete_stage = synchronize(dynamics.evaluate(0.0, state, concrete_fill))

    ledger_result, ledger_phase = _compiled_phase(
        "stage-ledger-scatter",
        lambda ledger: ledger.scatter_content_rate(),
        (concrete_stage.ledger,),
        settings,
        evidence={
            "topology_epoch_id": concrete_stage.ledger.topology_epoch_id,
            "geometry_layout_id": concrete_stage.ledger.geometry_layout_id,
            "precision_policy_id": concrete_stage.ledger.evidence_policy_id,
            "ledger_block_count": len(concrete_stage.ledger.blocks),
        },
        passed=lambda result: bool(jnp.all(jnp.isfinite(result))),
    )
    phases.append(ledger_phase)

    runtime = _runtime(prepared, refined)
    runtime_state = runtime.initial_state(state)
    advanced = synchronize(runtime.advance(runtime_state, 0.005))

    def reflux_operation(level_values):
        updated = tuple(level_values)
        for register in advanced.flux_registers:
            updated = runtime.conservation.reflux(updated, register)
        return updated

    refluxed, reflux_phase = _compiled_phase(
        "accepted-ledger-reflux",
        reflux_operation,
        (values,),
        settings,
        evidence={
            "conservation_plan_id": runtime.conservation.plan_id,
            "flux_register_ids": [
                register.register_id for register in advanced.flux_registers
            ],
            "flux_register_count": len(advanced.flux_registers),
            "composite_conservation_defect": float(
                jnp.max(jnp.abs(advanced.composite_conservation_defect))
            ),
        },
        passed=lambda result: (
            bool(advanced.accepted)
            and float(jnp.max(jnp.abs(advanced.composite_conservation_defect))) <= 2.0e-12
            and all(bool(jnp.all(jnp.isfinite(value))) for value in result)
        ),
    )
    phases.append(reflux_phase)

    _, _, composite_result = _hierarchy(periodic=False, levels=2)
    layout = CompositeAMRCellLayout(composite_result.topology, dtype=jnp.float64)
    composite_plan = CompositeAMRDiffusionPlan(
        layout,
        boundaries={"x": ("dirichlet", "dirichlet")},
    )
    operator = composite_plan.prepare(1.0)
    rhs = operator.prepare_rhs(1.0, boundary_data={"x": (0.0, 0.0)})
    _, composite_route_phase = _host_phase(
        "host-composite-route-preparation",
        lambda: CompositeAMRDiffusionPlan(
            CompositeAMRCellLayout(composite_result.topology, dtype=jnp.float64),
            boundaries={"x": ("dirichlet", "dirichlet")},
        ).prepare(1.0),
        settings,
        evidence=lambda result: {
            "operator_id": result.operator_id,
            "route_id": result.plan.route_fingerprint,
            "layout_id": result.layout.layout_id,
            "physical_cell_count": result.layout.physical_cell_count,
            "allocated_cell_count": result.layout.cell_count,
            "coarse_fine_interface_count": int(
                jnp.count_nonzero(result.plan.routes.edge_level_jump)
            ),
        },
        passed=lambda result: (
            int(jnp.count_nonzero(result.plan.routes.edge_level_jump)) > 0
        ),
    )
    phases.append(composite_route_phase)

    image, operator_phase = _compiled_phase(
        "composite-operator",
        operator.mv,
        (rhs,),
        settings,
        evidence={
            "operator_id": operator.operator_id,
            "route_id": composite_plan.route_fingerprint,
            "precision_policy_id": composite_plan.precision.policy_id,
            "edge_count": operator.plan.routes.edge_left.size,
            "boundary_face_count": operator.plan.routes.boundary_cells.size,
        },
        passed=lambda result: all(
            bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(result)
        ),
    )
    phases.append(operator_phase)

    fine_size = operator.source.size
    centers = np.concatenate(
        tuple(np.asarray(value).reshape((-1,)) for value in _cell_centers(layout))
    )
    physical = np.flatnonzero(np.asarray(layout.flat_leaf_mask))
    physical = physical[np.argsort(centers[physical], kind="stable")]
    coarse_size = (physical.size + 1) // 2
    coarse_space = linalg.ArraySpace((coarse_size,), dtype=jnp.float64)
    prolongation_matrix = jnp.zeros((fine_size, coarse_size), dtype=jnp.float64)
    prolongation_matrix = prolongation_matrix.at[
        jnp.asarray(physical), jnp.arange(physical.size) // 2
    ].set(1.0)
    prolongation = linalg.DenseLinearOperator(
        prolongation_matrix,
        source=coarse_space,
        target=operator.source,
    )
    restriction = linalg.adjoint(prolongation)
    materialization = linalg.MaterializationPolicy(
        max_entries=1_000_000,
        max_bytes=32_000_000,
    )
    fine_matrix = linalg.materialize(operator, materialization)
    restriction_matrix = linalg.materialize(restriction, materialization)
    coarse_operator = linalg.DenseLinearOperator(
        restriction_matrix @ fine_matrix @ prolongation_matrix,
        source=coarse_space,
        target=coarse_space,
    )
    cycle_builder = composite_amr_multigrid_builder(
        (operator, coarse_operator),
        (
            linalg.JacobiPreconditionerBuilder(relaxation=2.0 / 3.0),
            linalg.DenseInversePreconditionerBuilder(),
        ),
        (restriction,),
        (prolongation,),
        coarse_operator_source="direct",
    )
    cycle = cycle_builder.prepare(operator, materialization=materialization)
    correction, cycle_phase = _compiled_phase(
        "composite-v-cycle",
        cycle.apply,
        (rhs,),
        settings,
        evidence={
            "fine_space_size": fine_size,
            "coarse_space_size": coarse_size,
            "cycle": "v",
        },
        passed=lambda result: all(
            bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(result)
        ),
    )
    cycle_residual = jax.tree.map(
        lambda target, applied: target - applied,
        rhs,
        operator.mv(correction),
    )
    cycle_residual_ratio = _tree_norm(operator.source, cycle_residual) / _tree_norm(
        operator.source, rhs
    )
    cycle_phase["evidence"]["residual_ratio"] = cycle_residual_ratio
    cycle_phase["evidence"]["residual_contracted"] = cycle_residual_ratio < 1.0
    if cycle_residual_ratio >= 1.0:
        cycle_phase["status"] = "inconclusive"
        cycle_phase["reason"] = (
            "The supplied aggregate transfer executed finitely but one standalone "
            "V-cycle did not contract the residual on this partial-patch topology."
        )
    phases.append(cycle_phase)

    solve_policy = linalg.LinearSolvePolicy(
        linalg.ConjugateGradient(),
        tolerance=linalg.TolerancePolicy(
            relative=1.0e-10,
            absolute=1.0e-12,
            max_steps=300,
        ),
    )

    def solve_operation(target):
        result = linalg.solve(operator.linear_system(), target, policy=solve_policy)
        return result.value, result.status, result.diagnostics.residual_norm

    solved, solve_phase = _compiled_phase(
        "composite-solve",
        solve_operation,
        (rhs,),
        settings,
        evidence={
            "operator_id": operator.operator_id,
            "solver": "conjugate-gradient",
            "relative_tolerance": 1.0e-10,
            "absolute_tolerance": 1.0e-12,
            "maximum_steps": 300,
        },
        passed=lambda result: (
            int(result[1]) == 0 and bool(jnp.all(jnp.isfinite(result[2])))
        ),
    )
    phases.append(solve_phase)

    initial = prepared.topology_compiler.initialize()
    transition = prepared.field_transition(
        initial.topology,
        middle.topology,
        "block-amr-benchmark-scalar",
        component_shape=(1,),
        dtype=jnp.float64,
    )
    transition_state = _state(initial, lambda x: 0.25 + x)
    transition_values = tuple(level.values for level in transition_state.levels)
    _, transition_prepare_phase = _host_phase(
        "host-transition-preparation",
        lambda: prepared.field_transition(
            initial.topology,
            middle.topology,
            "block-amr-benchmark-scalar",
            component_shape=(1,),
            dtype=jnp.float64,
        ),
        settings,
        evidence=lambda result: {
            "transition_id": result.transition_id,
            "transfer_id": result.transfer.transfer_id,
            "source_epoch_id": result.source_topology.epoch.epoch_id,
            "target_epoch_id": result.target_topology.epoch.epoch_id,
        },
        passed=lambda result: result.transition_id == transition.transition_id,
    )
    phases.append(transition_prepare_phase)

    def transition_operation(level_values):
        source = _hierarchy_from_values(initial.topology, level_values)
        result = transition.apply(source)
        return (
            tuple(level.values for level in result.state.levels),
            result.conservation_residual,
            result.successful,
        )

    transitioned, transition_phase = _compiled_phase(
        "topology-transition",
        transition_operation,
        (transition_values,),
        settings,
        evidence={
            "transition_id": transition.transition_id,
            "source_epoch_id": initial.topology.epoch.epoch_id,
            "target_epoch_id": middle.topology.epoch.epoch_id,
        },
        passed=lambda result: (
            bool(result[2]) and float(jnp.max(jnp.abs(result[1]))) <= 1.0e-13
        ),
    )
    phases.append(transition_phase)

    partition = BlockAMRPartitionPlan(topology.plan, 2)
    distributed, distributed_prepare_phase = _host_phase(
        "host-packed-route-preparation",
        lambda: partition.prepare(refined, prepared),
        settings,
        evidence=lambda result: {
            "partition_plan_id": result.partition.plan_id,
            "distributed_prepared_id": result.prepared_id,
            "resource_evidence_id": result.resource_evidence_id,
            "active_blocks": list(result.resources.active_blocks),
            "local_block_capacities": list(result.local_block_capacities),
            "allocated_block_slots": list(result.resources.allocated_block_slots),
            "same_level_routes": list(result.resources.same_level_routes),
            "coarse_fine_routes": list(result.resources.coarse_fine_routes),
            "interface_routes": list(result.resources.interface_routes),
            "route_array_bytes": result.resources.dynamic_route_array_bytes,
        },
        passed=lambda result: result.resources.dynamic_route_array_bytes >= 0,
    )
    phases.append(distributed_prepare_phase)
    weighted = partition.prepare(
        refined,
        prepared,
        costs=(
            jnp.asarray([100.0, 1.0, 1.0, 1.0]),
            None,
            None,
        ),
    )
    migration = distributed.migration_to(weighted)

    def packed_migration_operation(level_values):
        source = _hierarchy_from_values(topology, level_values)
        return migration.migrate(distributed.pack(source))

    packed, packed_phase = _compiled_phase(
        "packed-distribution-migration",
        packed_migration_operation,
        (values,),
        settings,
        evidence={
            "source_partition_id": distributed.prepared_id,
            "target_partition_id": weighted.prepared_id,
            "migration_id": migration.migration_id,
            "moved_blocks": list(migration.moved_block_counts),
            "source_loads": [
                np.bincount(
                    np.asarray(layout.block_owner)[: layout.active_count],
                    minlength=2,
                ).tolist()
                for layout in distributed.layouts
            ],
            "target_loads": [
                np.bincount(
                    np.asarray(layout.block_owner)[: layout.active_count],
                    minlength=2,
                ).tolist()
                for layout in weighted.layouts
            ],
        },
        passed=lambda result: (
            all(bool(jnp.all(jnp.isfinite(value))) for value in result)
            and sum(migration.moved_block_counts) > 0
        ),
    )
    phases.append(packed_phase)

    checkpoint_arrays = {
        **{f"level-{index}": value for index, value in enumerate(values)},
        "time": advanced.runtime_state.time,
        "accepted_step": advanced.runtime_state.accepted_step,
        "level_accepted_steps": advanced.runtime_state.level_accepted_steps,
    }
    encoded, encode_phase = _host_phase(
        "checkpoint-encode",
        lambda: lifecycle.encode_logical_arrays(
            checkpoint_arrays,
            logical_prefix="block-amr-benchmark",
        ),
        settings,
        evidence=lambda result: {
            "collection_id": result.collection_id,
            "manifest_bytes": len(result.manifest),
            "payload_bytes": sum(len(value) for value in result.payloads.values()),
            "logical_array_count": len(checkpoint_arrays),
            "execution_cache_arrays": 0,
        },
        passed=lambda result: (
            bool(result.collection_id) and len(result.payloads) == len(checkpoint_arrays)
        ),
    )
    phases.append(encode_phase)

    decoded, decode_phase = _host_phase(
        "checkpoint-decode",
        lambda: lifecycle.decode_logical_arrays(encoded.manifest, encoded.payloads),
        settings,
        evidence=lambda result: {
            "collection_id": encoded.collection_id,
            "decoded_array_count": len(result),
            "bitwise_roundtrip": all(
                np.array_equal(np.asarray(value), result[name])
                for name, value in checkpoint_arrays.items()
            ),
        },
        passed=lambda result: all(
            np.array_equal(np.asarray(value), result[name])
            for name, value in checkpoint_arrays.items()
        ),
    )
    phases.append(decode_phase)

    common_evidence = {
        "configuration": _configuration(topology.plan),
        "identifiers": {
            "hierarchy_plan_id": topology.plan.plan_id,
            "fd_hierarchy_plan_id": prepared.plan.plan_id,
            "fd_prepared_id": prepared.prepared_id,
            "topology_epoch_id": topology.epoch.epoch_id,
            "topology_id": topology.topology_id,
            "partition_id": topology.partition_id,
            "topology_route_graph_id": refined.routes.route_graph_id,
            "finite_volume_plan_id": finite_volume_plan.plan_id,
            "finite_volume_dynamics_id": dynamics.dynamics_id,
            "precision_policy_id": finite_volume_plan.precision.policy_id,
            "runtime_plan_id": runtime.plan.plan_id,
            "runtime_prepared_id": runtime.prepared_id,
        },
        "topology": _topology_evidence(refined),
        "occupancy": _occupancy(topology),
        "halo": _halo_evidence(concrete_fill),
        "routes": {
            "face_route_count": len(dynamics.face_routes),
            "coarse_fine_route_pairs": [
                list(pair) for pair in dynamics.coarse_fine_route_pairs
            ],
            "distributed_same_level": list(distributed.resources.same_level_routes),
            "distributed_coarse_fine": list(distributed.resources.coarse_fine_routes),
            "distributed_interface": list(distributed.resources.interface_routes),
        },
        "conservation": {
            "stage_global_content_rate": np.asarray(
                jnp.sum(ledger_result, axis=0)
            ).tolist(),
            "advance_composite_defect": float(
                jnp.max(jnp.abs(advanced.composite_conservation_defect))
            ),
            "transition_residual": np.asarray(transitioned[1]).tolist(),
        },
        "memory": {
            "hierarchy_state_logical_bytes": logical_array_bytes(state),
            "fill_patch_logical_bytes": logical_array_bytes(concrete_fill),
            "stage_logical_bytes": logical_array_bytes(concrete_stage),
            "packed_logical_bytes": logical_array_bytes(packed),
            "operator_logical_bytes": logical_array_bytes(operator),
            "v_cycle_logical_bytes": logical_array_bytes(cycle),
        },
        "composite": {
            "configuration": _configuration(composite_result.topology.plan),
            "topology_epoch_id": composite_result.topology.epoch.epoch_id,
            "layout_id": layout.layout_id,
            "operator_id": operator.operator_id,
            "route_id": composite_plan.route_fingerprint,
            "precision_policy_id": composite_plan.precision.policy_id,
            "physical_cells": layout.physical_cell_count,
            "allocated_cells": layout.cell_count,
            "solve_residual_norm": float(solved[2]),
            "v_cycle_correction_bytes": logical_array_bytes(correction),
            "operator_image_bytes": logical_array_bytes(image),
        },
        "checkpoint": {
            "collection_id": encoded.collection_id,
            "decoded_array_count": len(decoded),
        },
        "finite_volume_result_bytes": logical_array_bytes(stage_arrays),
        "reflux_result_bytes": logical_array_bytes(refluxed),
        "fill_result_bytes": logical_array_bytes(fill),
    }
    return phases, common_evidence


def _block_to_dense(topology: Any, values: Any) -> jax.Array:
    level = topology.plan.levels[0]
    metadata = topology.levels[0]
    output = jnp.zeros(topology.plan.global_cell_shapes[0] + values.shape[3:])
    logical = np.asarray(metadata.logical_indices, dtype=np.int32)
    for slot in np.flatnonzero(np.asarray(metadata.active, dtype="bool")):
        slices = tuple(
            slice(index * size, (index + 1) * size)
            for index, size in zip(logical[slot], level.block_shape, strict=True)
        )
        output = output.at[slices].set(values[int(slot)])
    return output


def _uniform_path_comparisons(
    settings: BenchmarkSettings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count = settings.comparison_cells
    grid = TensorGridPlan(
        (
            UniformCellAxisSpec(count),
            UniformCellAxisSpec(count),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    system = ScalarConservationSystem(
        2,
        lambda state, axis, args: (1.0 if axis == 0 else -0.25) * state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1], dtype=left.dtype),
        system_id="block-amr-benchmark-equivalent-paths",
    )
    method = FiniteVolumeMethodPlan(
        PiecewiseConstantReconstruction(),
        RusanovFluxPlan(),
    )
    boundary = ExtrapolationBoundary()
    boundaries = FiniteVolumeBoundarySet(
        ("x", "y"),
        (
            FiniteVolumeBoundaryPair(boundary, boundary),
            FiniteVolumeBoundaryPair(boundary, boundary),
        ),
    )
    row, column = jnp.meshgrid(
        (jnp.arange(count, dtype=jnp.float64) + 0.5) / count,
        (jnp.arange(count, dtype=jnp.float64) + 0.5) / count,
        indexing="ij",
    )
    dense_state = (
        1.0 + 0.1 * jnp.sin(2.0 * jnp.pi * row) * jnp.cos(2.0 * jnp.pi * column)
    )[..., None]
    dense_geometry = FiniteVolumePlan(grid).prepare()
    dense_dynamics = PreparedFiniteVolumeDynamics(
        system,
        dense_geometry,
        method,
        boundaries,
    )
    block_shape = (4, 4)
    block_capacity = (count // block_shape[0]) * (count // block_shape[1])
    block_hierarchy = BlockHierarchyPlan(
        grid,
        (
            BlockLevelPlan(
                0,
                block_shape,
                block_capacity,
                halo_width=1,
                refinement_ratio=2,
            ),
        ),
    )
    block_prepared = FDAMRHierarchyPlan(block_hierarchy).prepare()
    block_result = block_prepared.topology_compiler.initialize()
    block_topology = block_result.topology
    metadata = block_topology.levels[0]
    block_values = jnp.zeros((block_capacity, *block_shape, 1), dtype=dense_state.dtype)
    logical = np.asarray(metadata.logical_indices, dtype=np.int32)
    for slot in np.flatnonzero(np.asarray(metadata.active, dtype="bool")):
        slices = tuple(
            slice(index * size, (index + 1) * size)
            for index, size in zip(logical[slot], block_shape, strict=True)
        )
        block_values = block_values.at[int(slot)].set(dense_state[slices])
    block_state = BlockHierarchyState(
        block_topology,
        (BlockLevelState(block_hierarchy.levels[0], metadata, block_values),),
    )
    block_dynamics = BlockAMRFiniteVolumePlan(
        block_prepared,
        system,
        method,
        boundaries,
    ).prepare(block_topology)
    block_fill = synchronize(block_dynamics.fill_patch(0.0, block_state))

    dense_result, dense_phase = _compiled_phase(
        "equivalent-dense-finite-volume",
        lambda value: dense_dynamics(jnp.asarray(0.0), value),
        (dense_state,),
        settings,
        evidence={
            "logical_cells": count * count,
            "grid_shape": [count, count],
            "path": "dense",
        },
        passed=lambda result: bool(jnp.all(jnp.isfinite(result))),
    )
    block_residuals, block_phase = _compiled_phase(
        "equivalent-fixed-block-finite-volume",
        lambda value: block_dynamics.evaluate(
            0.0,
            _hierarchy_from_values(block_topology, (value,)),
            block_fill,
        ).residuals[0],
        (block_values,),
        settings,
        evidence={
            "logical_cells": count * count,
            "grid_shape": [count, count],
            "block_shape": list(block_shape),
            "block_capacity": block_capacity,
            "active_blocks": int(jnp.sum(metadata.active)),
            "path": "fixed-block",
        },
        passed=lambda result: bool(jnp.all(jnp.isfinite(result))),
    )
    block_dense = _block_to_dense(block_topology, block_residuals)
    block_dense_defect = float(jnp.max(jnp.abs(block_dense - dense_result)))
    if block_dense_defect > 2.0e-12:
        block_phase["status"] = "fail"
    block_phase["evidence"]["dense_residual_maximum_defect"] = block_dense_defect

    depth = settings.dyadic_depth
    capacity = (4 ** (depth + 1) - 1) // 3
    dyadic_grid = AdaptiveDyadicGridPlan(
        MortonAddressPlan((0.0, 0.0), (1.0, 1.0), depth),
        cell_capacity=capacity,
    )
    dyadic_topology = dyadic_grid.prepare()
    for _ in range(depth):
        refine = dyadic_topology.leaf_active & (dyadic_topology.levels < depth)
        dyadic_topology = dyadic_grid.adapt(
            dyadic_topology,
            refine_mask=refine,
        ).accepted
    dyadic = DyadicFiniteVolumePlan(dyadic_topology).prepare()
    dyadic_boundaries = UnstructuredFiniteVolumeBoundarySet(
        dyadic.boundary_patch_names,
        {name: ExtrapolationBoundary() for name in dyadic.boundary_patch_names},
    )
    dyadic_method = UnstructuredFiniteVolumeMethodPlan(
        PiecewiseConstantReconstruction(),
        RusanovFluxPlan(),
    )
    dyadic_problem = ConservationProblemIR(
        "block-amr-benchmark-dyadic-comparison",
        "state",
        system,
        dyadic_boundaries,
    )
    dyadic_dynamics = compile_conservation_problem(
        dyadic_problem,
        dyadic,
        dyadic_method,
    ).dynamics
    dyadic_state = jnp.ones((dyadic.cell_count, 1), dtype=jnp.float64)
    dyadic_residual, dyadic_phase = _compiled_phase(
        "equivalent-dyadic-finite-volume",
        lambda value: dyadic_dynamics(jnp.asarray(0.0), value, None),
        (dyadic_state,),
        settings,
        evidence={
            "logical_cells": dyadic.cell_count,
            "target_uniform_cells": count * count,
            "depth": depth,
            "cell_capacity": capacity,
            "face_count": dyadic.face_count,
            "path": "dyadic",
        },
        passed=lambda result: (
            bool(jnp.all(jnp.isfinite(result)))
            and float(jnp.max(jnp.abs(result))) <= 2.0e-12
            and dyadic.cell_count == count * count
        ),
    )

    policy = PerformancePolicy(
        "minimize",
        relative_tolerance=5.0,
        confidence=0.95,
        bootstrap_resamples=2_000,
        minimum_samples=5,
    )
    dense_distribution = DurationDistribution(
        tuple(dense_phase["steady"]["samples_seconds"])
    )
    block_distribution = DurationDistribution(
        tuple(block_phase["steady"]["samples_seconds"])
    )
    dyadic_distribution = DurationDistribution(
        tuple(dyadic_phase["steady"]["samples_seconds"])
    )
    block_comparison = compare_performance(
        dense_distribution,
        block_distribution,
        policy,
        comparison_id=f"fixed-block-vs-dense-{count}",
    )
    dyadic_comparison = compare_performance(
        dense_distribution,
        dyadic_distribution,
        policy,
        comparison_id=f"dyadic-vs-dense-{count}",
    )

    def comparison_gate(name: str, comparison: Any) -> dict[str, Any]:
        status: GateStatus = (
            "inconclusive"
            if comparison.regressed is None
            else "fail"
            if comparison.regressed
            else "pass"
        )
        return {
            "name": name,
            "required": False,
            "status": status,
            "reason": comparison.reason,
            "comparison": comparison.to_dict(),
            "policy": {
                "objective": policy.objective,
                "relative_tolerance": policy.relative_tolerance,
                "absolute_tolerance": policy.absolute_tolerance,
                "confidence": policy.confidence,
                "bootstrap_resamples": policy.bootstrap_resamples,
                "minimum_samples": policy.minimum_samples,
            },
        }

    phases = [
        dense_phase,
        block_phase,
        dyadic_phase,
        comparison_gate("fixed-block-vs-dense-performance", block_comparison),
        comparison_gate("dyadic-vs-dense-performance", dyadic_comparison),
    ]
    evidence = {
        "grid_shape": [count, count],
        "logical_cells": count * count,
        "fixed_block": {
            "configuration": _configuration(block_hierarchy),
            "topology_epoch_id": block_topology.epoch.epoch_id,
            "topology_id": block_topology.topology_id,
            "route_id": block_dynamics.dynamics_id,
            "precision_policy_id": block_dynamics.plan.precision.policy_id,
            "residual_maximum_defect_against_dense": block_dense_defect,
        },
        "dense": {
            "prepared_geometry_id": dense_geometry.prepared_id,
            "residual_bytes": logical_array_bytes(dense_result),
        },
        "dyadic": {
            "depth": depth,
            "cell_capacity": capacity,
            "active_leaves": dyadic.cell_count,
            "face_count": dyadic.face_count,
            "constant_state_defect": float(jnp.max(jnp.abs(dyadic_residual))),
            "residual_bytes": logical_array_bytes(dyadic_residual),
        },
    }
    return phases, evidence


def _real_device_phase(
    settings: BenchmarkSettings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    local_devices = tuple(jax.local_devices())
    accelerators = tuple(
        device for device in local_devices if device.platform in ("gpu", "tpu")
    )
    observed = {
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "requested_devices": 2,
        "observed_local_devices": [
            {
                "process_index": device.process_index,
                "device_id": device.id,
                "platform": device.platform,
                "kind": device.device_kind,
            }
            for device in local_devices
        ],
        "observed_accelerators": len(accelerators),
    }
    if len(accelerators) < 2:
        reason = (
            "Two locally addressable accelerator devices are unavailable; CPU "
            "logical devices are not reported as independent physical devices."
        )
        return (
            _inconclusive_phase(
                "real-device-distributed-fill-patch",
                reason,
                observed,
                required=False,
            ),
            {**observed, "status": "inconclusive", "reason": reason},
        )

    devices = accelerators[:2]
    group_id = "block-amr-benchmark:" + ":".join(
        f"{device.process_index}-{device.id}" for device in devices
    )
    specification = ExecutionGroupSpec(
        group_id,
        tuple(sorted({device.process_index for device in devices})),
        tuple((device.process_index, device.id) for device in devices),
        mesh_axes=(("block_parts", 2),),
    )
    group = ExecutionGroup(specification, devices)
    prepared, _, refined = _hierarchy(periodic=True)
    distributed = BlockAMRPartitionPlan(refined.topology.plan, 2).prepare(
        refined,
        prepared,
        execution_group=group,
    )
    state = _state(
        refined,
        lambda x: 1.0 + 0.1 * jnp.sin(2.0 * jnp.pi * x),
    )
    values = tuple(level.values for level in state.levels)
    reference = synchronize(distributed.serial_fill_patch(state))

    def operation(level_values):
        current = _hierarchy_from_values(refined.topology, level_values)
        fill = distributed.distributed_fill_patch(current)
        return (
            tuple(workspace.values for workspace in fill.workspaces),
            tuple(workspace.valid for workspace in fill.workspaces),
            fill.complete,
        )

    result, phase = _compiled_phase(
        "real-device-distributed-fill-patch",
        operation,
        (values,),
        settings,
        evidence={
            **observed,
            "execution_group_id": specification.group_id,
            "partition_plan_id": distributed.partition.plan_id,
            "distributed_prepared_id": distributed.prepared_id,
            "resource_evidence_id": distributed.resource_evidence_id,
            "route_array_bytes": distributed.resources.dynamic_route_array_bytes,
            "loads": [
                np.bincount(
                    np.asarray(layout.block_owner)[: layout.active_count],
                    minlength=2,
                ).tolist()
                for layout in distributed.layouts
            ],
        },
        passed=lambda output: (
            bool(output[2])
            and all(
                np.array_equal(np.asarray(expected.values), np.asarray(actual))
                and np.array_equal(np.asarray(expected.valid), np.asarray(valid))
                for expected, actual, valid in zip(
                    reference.workspaces,
                    output[0],
                    output[1],
                    strict=True,
                )
            )
        ),
        required=False,
    )
    evidence = {
        **observed,
        "status": phase["status"],
        "execution_group_id": specification.group_id,
        "partition_plan_id": distributed.partition.plan_id,
        "distributed_prepared_id": distributed.prepared_id,
        "logical_result_bytes": logical_array_bytes(result),
    }
    return phase, evidence


def benchmark(
    *,
    smoke: bool = False,
    warmup: int | None = None,
    repeats: int | None = None,
) -> dict[str, Any]:
    """Run synchronized phases and retain every raw steady timing sample."""
    settings = _settings(smoke=smoke, warmup=warmup, repeats=repeats)
    environment = capture_environment()
    phases, amr_evidence = _compiler_phases(settings)
    comparison_phases, comparison_evidence = _uniform_path_comparisons(settings)
    phases.extend(comparison_phases)
    device_phase, device_evidence = _real_device_phase(settings)
    phases.append(device_phase)
    failed = [
        phase["name"]
        for phase in phases
        if phase["required"] and phase["status"] == "fail"
    ]
    inconclusive = [
        phase["name"]
        for phase in phases
        if phase["required"] and phase["status"] == "inconclusive"
    ]
    status: GateStatus = "fail" if failed else "inconclusive" if inconclusive else "pass"
    report = {
        "kind": "block-amr-benchmark",
        "status": status,
        "failed_required_gates": failed,
        "inconclusive_required_gates": inconclusive,
        "settings": {
            "smoke": settings.smoke,
            "warmup": settings.warmup,
            "repeats": settings.repeats,
            "comparison_cells": settings.comparison_cells,
            "dyadic_depth": settings.dyadic_depth,
        },
        "environment": environment.to_dict(),
        "execution_id": environment.fingerprint,
        "evidence": {
            "block_amr": amr_evidence,
            "equivalent_paths": comparison_evidence,
            "real_device_distribution": device_evidence,
        },
        "phases": phases,
    }
    json.dumps(report, allow_nan=False)
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the public fixed-block AMR runtime."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/block_amr.json"),
        help="Machine-readable JSON destination.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use bounded shapes, at most one warmup, and at most two samples.",
    )
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    arguments = parser.parse_args(argv)
    report = benchmark(
        smoke=arguments.smoke,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    _write_report(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
