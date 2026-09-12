#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualify the public fixed-block AMR runtime with auditable numerical evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import capture_environment, synchronize
from phydrax import lifecycle, linalg
from phydrax.discretization import TensorGridPlan, UniformCellAxisSpec
from phydrax.discretization.amr import (
    BlockAMRPartitionPlan,
    BlockFieldTopologyTransition,
    BlockHierarchyPlan,
    BlockHierarchyState,
    BlockLevelPlan,
    BlockLevelState,
    BlockTopologyCompileResult,
    CompositeAMRCellLayout,
    FDAMRHierarchyPlan,
    FillPatchSource,
    PreparedFDAMRHierarchy,
)
from phydrax.discretization.finite_volume import (
    BlockAMRFiniteVolumePlan,
    CompositeAMRDiffusionPlan,
    ExtrapolationBoundary,
    FiniteVolumeBoundaryPair,
    FiniteVolumeBoundarySet,
    FiniteVolumeMethodPlan,
    PiecewiseConstantReconstruction,
    RusanovFluxPlan,
)
from phydrax.equations import ScalarConservationSystem
from phydrax.execution import ExecutionGroup, ExecutionGroupSpec
from phydrax.solver import BlockAMRRuntimePlan


GateStatus = Literal["pass", "fail", "inconclusive"]
_SOURCE_NAMES = {
    int(FillPatchSource.INACTIVE): "inactive",
    int(FillPatchSource.INTERIOR): "interior",
    int(FillPatchSource.SAME_LEVEL): "same_level",
    int(FillPatchSource.PERIODIC): "periodic",
    int(FillPatchSource.COARSE_TIME_INTERPOLATED): "coarse_time_interpolated",
    int(FillPatchSource.PHYSICAL_BOUNDARY): "physical_boundary",
    int(FillPatchSource.UNRESOLVED): "unresolved",
}


def _gate(
    name: str,
    status: GateStatus,
    evidence: Mapping[str, Any],
    /,
    *,
    reason: str | None = None,
    required: bool = True,
) -> dict[str, Any]:
    if status not in ("pass", "fail", "inconclusive"):
        raise ValueError("Unknown qualification gate status.")
    if status == "inconclusive" and not reason:
        raise ValueError("Inconclusive qualification gates require a reason.")
    return {
        "name": name,
        "required": required,
        "status": status,
        "reason": reason,
        "evidence": dict(evidence),
    }


def _configuration(plan: BlockHierarchyPlan) -> dict[str, Any]:
    return {
        "dimension": len(plan.grid.shape),
        "axis_names": list(plan.grid.axis_names),
        "bounds": [
            [float(axis.bounds[0]), float(axis.bounds[1])]
            for axis in plan.grid.structured_axes
        ],
        "periodic_axes": list(plan.periodic_axes),
        "global_cell_shapes": [list(shape) for shape in plan.global_cell_shapes],
        "block_lattice_shapes": [list(shape) for shape in plan.block_lattice_shapes],
        "levels": [
            {
                "level": level.level,
                "block_shape": list(level.block_shape),
                "halo_width": list(level.halo_width),
                "block_capacity": level.maximum_blocks,
                "refinement_ratio": level.refinement_ratio,
            }
            for level in plan.levels
        ],
    }


def _topology_evidence(result: BlockTopologyCompileResult) -> dict[str, Any]:
    topology = result.topology
    return {
        "status": result.status.code,
        "changed": result.status.changed,
        "message": result.status.message,
        "requested_blocks": list(result.evidence.requested_blocks),
        "realized_blocks": list(result.evidence.realized_blocks),
        "block_capacities": list(result.evidence.capacities),
        "buffered_tagged_cells": list(result.evidence.buffered_tagged_cells),
        "proper_nesting_rejections": list(result.evidence.proper_nesting_rejections),
        "overflow_level": result.evidence.overflow_level,
        "leaf_cells": [
            int(
                np.count_nonzero(
                    np.asarray(metadata.active, dtype=bool).reshape(
                        (metadata.active.shape[0],) + (1,) * len(level.block_shape)
                    )
                    & ~np.asarray(topology.covered_cells[index], dtype=bool)
                )
            )
            for index, (level, metadata) in enumerate(
                zip(topology.plan.levels, topology.levels, strict=True)
            )
        ],
        "interface_entries": [
            int(np.count_nonzero(np.asarray(level))) for level in topology.interfaces
        ],
    }


def _identifiers(
    prepared: PreparedFDAMRHierarchy,
    topology_result: BlockTopologyCompileResult,
    /,
    *,
    route_ids: Sequence[str] = (),
    execution_id: str,
    precision_id: str | None = None,
) -> dict[str, Any]:
    topology = topology_result.topology
    return {
        "hierarchy_plan_id": topology.plan.plan_id,
        "fd_hierarchy_plan_id": prepared.plan.plan_id,
        "fd_prepared_id": prepared.prepared_id,
        "topology_epoch_id": topology.epoch.epoch_id,
        "topology_id": topology.topology_id,
        "partition_id": topology.partition_id,
        "topology_result_id": topology_result.result_id,
        "topology_route_graph_id": topology_result.routes.route_graph_id,
        "route_ids": list(route_ids),
        "precision_policy_id": (
            prepared.plan.precision.policy_id if precision_id is None else precision_id
        ),
        "execution_id": execution_id,
    }


def _hierarchy(
    *,
    periodic: bool,
    levels: int = 3,
    fine_capacity: int = 16,
) -> tuple[
    PreparedFDAMRHierarchy, BlockTopologyCompileResult, BlockTopologyCompileResult
]:
    grid = TensorGridPlan(
        (UniformCellAxisSpec(16, periodic=periodic),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    level_plans = [BlockLevelPlan(0, (4,), 4, halo_width=1, refinement_ratio=2)]
    if levels >= 2:
        level_plans.append(
            BlockLevelPlan(
                1,
                (2,),
                fine_capacity,
                halo_width=1,
                refinement_ratio=2,
            )
        )
    if levels == 3:
        level_plans.append(BlockLevelPlan(2, (2,), 32, halo_width=1, refinement_ratio=2))
    prepared = FDAMRHierarchyPlan(BlockHierarchyPlan(grid, tuple(level_plans))).prepare()
    initial = prepared.topology_compiler.initialize()
    if levels == 1:
        return prepared, initial, initial
    coarse_tags = jnp.zeros((4, 4), dtype=bool).at[1:3].set(True)
    if levels == 2:
        refined = prepared.compile_topology(initial.topology, (coarse_tags,))
        return prepared, initial, refined
    empty_middle = jnp.zeros((fine_capacity, 2), dtype=bool)
    middle = prepared.compile_topology(
        initial.topology,
        (coarse_tags, empty_middle),
    )
    middle_tags = jnp.zeros((fine_capacity, 2), dtype=bool).at[3, 1].set(True)
    refined = prepared.compile_topology(
        middle.topology,
        (coarse_tags, middle_tags),
    )
    if not middle.status.successful or not refined.status.successful:
        raise RuntimeError("The declared qualification topology could not be realized.")
    return prepared, middle, refined


def _state(
    topology_result: BlockTopologyCompileResult,
    field: Callable[[jnp.ndarray], jnp.ndarray | float],
    /,
) -> BlockHierarchyState:
    topology = topology_result.topology
    levels = []
    lower = float(topology.plan.grid.structured_axes[0].bounds[0])
    for level_plan, metadata, spacing in zip(
        topology.plan.levels,
        topology.levels,
        topology.plan.level_spacings,
        strict=True,
    ):
        values = jnp.zeros(
            (level_plan.maximum_blocks, *level_plan.block_shape, 1),
            dtype=jnp.float64,
        )
        active = np.asarray(metadata.active, dtype=bool)
        logical = np.asarray(metadata.logical_indices, dtype=np.int32)
        local = jnp.arange(level_plan.block_shape[0], dtype=values.dtype)
        for slot in np.flatnonzero(active):
            x = (
                lower
                + (logical[slot, 0] * level_plan.block_shape[0] + local + 0.5)
                * spacing[0]
            )
            values = values.at[int(slot), :, 0].set(jnp.asarray(field(x)))
        levels.append(BlockLevelState(level_plan, metadata, values))
    return BlockHierarchyState(topology, tuple(levels))


def _system(*, source: bool = False) -> ScalarConservationSystem:
    return ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1], dtype=left.dtype),
        system_id=(
            "block-amr-qualification-source-advection"
            if source
            else "block-amr-qualification-advection"
        ),
    )


def _method() -> FiniteVolumeMethodPlan:
    return FiniteVolumeMethodPlan(
        PiecewiseConstantReconstruction(),
        RusanovFluxPlan(),
    )


def _boundaries(*, periodic: bool) -> FiniteVolumeBoundarySet:
    if periodic:
        return FiniteVolumeBoundarySet.periodic(("x",))
    boundary = ExtrapolationBoundary()
    return FiniteVolumeBoundarySet(
        ("x",),
        (FiniteVolumeBoundaryPair(boundary, boundary),),
    )


def _runtime(
    prepared: PreparedFDAMRHierarchy,
    topology_result: BlockTopologyCompileResult,
    /,
    *,
    source: Callable[..., Any] | None = None,
    source_id: str | None = None,
):
    finite_volume = BlockAMRFiniteVolumePlan(
        prepared,
        _system(source=source is not None),
        _method(),
        _boundaries(periodic=prepared.plan.hierarchy.periodic_axes[0]),
        source=source,
        source_id=source_id,
    )
    return BlockAMRRuntimePlan(finite_volume).prepare(topology_result.topology)


def _halo_evidence(fill: Any) -> dict[str, Any]:
    levels = []
    for workspace in fill.workspaces:
        source = np.asarray(workspace.source_class)
        levels.append(
            {
                "workspace_shape": list(workspace.values.shape),
                "valid_cells": int(np.count_nonzero(np.asarray(workspace.valid))),
                "total_cells": int(workspace.valid.size),
                "source_class_counts": {
                    name: int(np.count_nonzero(source == code))
                    for code, name in _SOURCE_NAMES.items()
                },
                "workspace_id": workspace.workspace_id,
            }
        )
    return {"complete": bool(fill.complete), "levels": levels}


def _constant_state_gate(execution_id: str) -> dict[str, Any]:
    prepared, _, refined = _hierarchy(periodic=True)
    topology = refined.topology
    state = _state(refined, lambda x: jnp.full_like(x, 2.5))
    plan = BlockAMRFiniteVolumePlan(
        prepared,
        _system(),
        _method(),
        _boundaries(periodic=True),
    )
    dynamics = plan.prepare(topology)
    fill = synchronize(dynamics.fill_patch(0.0, state))
    stage = synchronize(dynamics.evaluate(0.0, state, fill))
    residual = max(float(jnp.max(jnp.abs(value))) for value in stage.residuals)
    global_content_rate = float(
        jnp.max(jnp.abs(jnp.sum(stage.ledger.scatter_content_rate(), axis=0)))
    )
    finite = all(
        bool(jnp.all(jnp.isfinite(workspace.values))) for workspace in fill.workspaces
    ) and all(bool(jnp.all(jnp.isfinite(value))) for value in stage.residuals)
    passed = (
        bool(fill.complete)
        and finite
        and residual <= 1.0e-13
        and (global_content_rate <= 1.0e-13)
    )
    routes = tuple(route.block_id for route in dynamics.face_routes) + tuple(
        fill_plan.plan_id for fill_plan in dynamics.fill_patch_plans
    )
    evidence = {
        "configuration": _configuration(topology.plan),
        "identifiers": _identifiers(
            prepared,
            refined,
            route_ids=routes,
            precision_id=plan.precision.policy_id,
            execution_id=execution_id,
        ),
        "topology": _topology_evidence(refined),
        "halo": _halo_evidence(fill),
        "maximum_residual": residual,
        "global_content_rate_defect": global_content_rate,
        "finite": finite,
        "ledger_epoch_id": stage.ledger.topology_epoch_id,
        "ledger_precision_id": stage.ledger.evidence_policy_id,
        "face_route_count": len(dynamics.face_routes),
    }
    return _gate("constant-state", "pass" if passed else "fail", evidence)


def _topology_transition_gate(execution_id: str) -> dict[str, Any]:
    prepared, target, _ = _hierarchy(periodic=True)
    initial = prepared.topology_compiler.initialize()
    source_state = _state(initial, lambda x: 0.25 + x)
    transition = BlockFieldTopologyTransition(
        initial.topology,
        target.topology,
        "qualification-scalar",
        component_shape=(1,),
        dtype=source_state.levels[0].values.dtype,
    )
    transferred = synchronize(transition.apply(source_state))
    constant_transition = BlockFieldTopologyTransition(
        initial.topology,
        target.topology,
        "qualification-constant",
        component_shape=(1,),
        dtype=source_state.levels[0].values.dtype,
    )
    constant = synchronize(
        constant_transition.apply(_state(initial, lambda x: jnp.ones_like(x)))
    )
    constant_defect = max(
        float(
            jnp.max(
                jnp.where(
                    level.metadata.active.reshape((-1, 1, 1)),
                    jnp.abs(level.values - 1.0),
                    0.0,
                )
            )
        )
        for level in constant.state.levels
    )

    stale_refused = False
    stale_message = None
    try:
        transition.apply(transferred.state)
    except ValueError as error:
        stale_refused = True
        stale_message = str(error)

    overflow_grid = TensorGridPlan(
        (UniformCellAxisSpec(16, periodic=True),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    overflow_prepared = FDAMRHierarchyPlan(
        BlockHierarchyPlan(
            overflow_grid,
            (
                BlockLevelPlan(0, (4,), 4, halo_width=1),
                BlockLevelPlan(1, (2,), 1, halo_width=1),
            ),
        )
    ).prepare()
    overflow_initial = overflow_prepared.topology_compiler.initialize()
    overflow_tags = jnp.ones((4, 4), dtype=bool)
    overflow = overflow_prepared.compile_topology(
        overflow_initial.topology,
        (overflow_tags,),
    )
    unchanged_on_refusal = (
        overflow.topology.epoch.epoch_id == overflow_initial.topology.epoch.epoch_id
        and overflow.topology.topology_id == overflow_initial.topology.topology_id
    )
    conservation = float(jnp.max(jnp.abs(transferred.conservation_residual)))
    passed = (
        bool(transferred.successful)
        and conservation <= 1.0e-13
        and constant_defect <= 1.0e-13
        and stale_refused
        and not overflow.status.successful
        and overflow.status.code == "capacity_exceeded"
        and unchanged_on_refusal
    )
    evidence = {
        "configuration": _configuration(target.topology.plan),
        "identifiers": _identifiers(
            prepared,
            target,
            route_ids=(transition.transition_id, transition.transfer.transfer_id),
            execution_id=execution_id,
        ),
        "source_epoch_id": initial.topology.epoch.epoch_id,
        "target_epoch_id": target.topology.epoch.epoch_id,
        "transition_id": transition.transition_id,
        "source_content": np.asarray(transferred.source_content).tolist(),
        "target_content": np.asarray(transferred.target_content).tolist(),
        "conservation_residual": np.asarray(transferred.conservation_residual).tolist(),
        "maximum_conservation_residual": conservation,
        "constant_preservation_defect": constant_defect,
        "stale_epoch_refused": stale_refused,
        "stale_epoch_message": stale_message,
        "target_topology": _topology_evidence(target),
        "capacity_refusal": {
            "configuration": _configuration(overflow.topology.plan),
            "evidence": _topology_evidence(overflow),
            "unchanged_topology": unchanged_on_refusal,
        },
    }
    return _gate(
        "topology-transition-and-refusal", "pass" if passed else "fail", evidence
    )


def _advection_reflux_gate(execution_id: str) -> dict[str, Any]:
    prepared, _, refined = _hierarchy(periodic=True)
    runtime = _runtime(prepared, refined)
    initial = _state(
        refined,
        lambda x: 1.0 + 0.1 * jnp.sin(2.0 * jnp.pi * x),
    )
    runtime_state = runtime.initial_state(initial)
    result = synchronize(runtime.advance(runtime_state, 0.005))
    conservation = float(jnp.max(jnp.abs(result.composite_conservation_defect)))
    register_mismatches = [
        float(jnp.max(jnp.abs(register.mismatch()))) for register in result.flux_registers
    ]
    interface_finite = all(np.isfinite(value) for value in register_mismatches)
    ledgers_finite = all(
        bool(jnp.all(jnp.isfinite(ledger.source_integral)))
        and all(
            bool(jnp.all(jnp.isfinite(block.flux_integral))) for block in ledger.blocks
        )
        for ledger in result.accepted_ledgers
    )
    expected_steps = tuple(runtime.plan.schedule.level_substeps_per_root)
    observed_steps = tuple(
        int(value) for value in np.asarray(result.runtime_state.level_accepted_steps)
    )
    passed = (
        bool(result.accepted)
        and observed_steps == expected_steps
        and conservation <= 2.0e-12
        and interface_finite
        and ledgers_finite
        and len(result.flux_registers) > 0
    )
    route_ids = tuple(route.route_plan_id for route in runtime.edge_routes) + (
        runtime.dynamics.dynamics_id,
        runtime.conservation.plan_id,
        runtime.plan.schedule.schedule_id,
    )
    evidence = {
        "configuration": _configuration(refined.topology.plan),
        "identifiers": _identifiers(
            prepared,
            refined,
            route_ids=route_ids,
            precision_id=runtime.plan.finite_volume.precision.policy_id,
            execution_id=execution_id,
        ),
        "accepted": bool(result.accepted),
        "accepted_step_size": float(result.accepted_step_size),
        "edge_substeps": list(runtime.plan.schedule.edge_substeps),
        "level_substeps_per_root": list(expected_steps),
        "observed_level_accepted_steps": list(observed_steps),
        "level_attempt_order": list(result.level_attempt_order),
        "synchronization_order": list(result.synchronization_order),
        "accepted_ledger_count": len(result.accepted_ledgers),
        "edge_ledger_count": len(result.edge_accepted_ledgers),
        "flux_register_count": len(result.flux_registers),
        "flux_register_accumulated_times": [
            float(register.accumulated_time) for register in result.flux_registers
        ],
        "interface_flux_mismatch_maxima": register_mismatches,
        "interface_evidence_finite": interface_finite,
        "ledger_evidence_finite": ledgers_finite,
        "composite_conservation_defect": conservation,
        "precision_evidence": {
            "evidence_id": result.precision_evidence.evidence_id,
            "resolution_id": result.precision_evidence.resolution_id,
            "domain": result.precision_evidence.domain,
            "provider": result.precision_evidence.provider,
            "observed": [
                [
                    name,
                    value if value is None or isinstance(value, str) else value.to_dict(),
                ]
                for name, value in result.precision_evidence.observed
            ],
        },
    }
    return _gate("three-level-advection-reflux", "pass" if passed else "fail", evidence)


def _cell_centers(layout: CompositeAMRCellLayout) -> tuple[jax.Array, ...]:
    topology = layout.topology
    lower = float(topology.plan.grid.structured_axes[0].bounds[0])
    output = []
    for level_plan, metadata, spacing in zip(
        topology.plan.levels,
        topology.levels,
        topology.plan.level_spacings,
        strict=True,
    ):
        values = jnp.zeros(
            (level_plan.maximum_blocks, *level_plan.block_shape),
            dtype=layout.dtype,
        )
        active = np.asarray(metadata.active, dtype=bool)
        logical = np.asarray(metadata.logical_indices, dtype=np.int32)
        local = jnp.arange(level_plan.block_shape[0], dtype=layout.dtype)
        for slot in np.flatnonzero(active):
            x = (
                lower
                + (logical[slot, 0] * level_plan.block_shape[0] + local + 0.5)
                * spacing[0]
            )
            values = values.at[int(slot)].set(x)
        output.append(values)
    return tuple(output)


def _tree_norm(space: Any, value: Any) -> float:
    return float(jnp.sqrt(jnp.real(space.inner(value, value))))


def _poisson_gate(execution_id: str) -> dict[str, Any]:
    prepared, _, target = _hierarchy(periodic=False, levels=2)
    layout = CompositeAMRCellLayout(target.topology, dtype=jnp.float64)
    diffusion_plan = CompositeAMRDiffusionPlan(
        layout,
        boundaries={"x": ("dirichlet", "dirichlet")},
    )
    operator = diffusion_plan.prepare(1.0)
    rhs = operator.prepare_rhs(2.0, boundary_data={"x": (0.0, 0.0)})
    solve = synchronize(
        linalg.solve(
            operator.linear_system(),
            rhs,
            policy=linalg.LinearSolvePolicy(
                linalg.ConjugateGradient(),
                tolerance=linalg.TolerancePolicy(
                    relative=1.0e-11,
                    absolute=1.0e-13,
                    max_steps=300,
                ),
            ),
        )
    )
    residual = jax.tree.map(
        lambda target_value, image: target_value - image,
        rhs,
        operator.mv(solve.value),
    )
    residual_relative = _tree_norm(layout.space, residual) / _tree_norm(layout.space, rhs)
    centers = _cell_centers(layout)
    exact = layout.zero_masked(tuple(x * (1.0 - x) for x in centers))
    error = jax.tree.map(lambda actual, expected: actual - expected, solve.value, exact)
    solution_relative = _tree_norm(layout.space, error) / _tree_norm(layout.space, exact)
    left_flux, right_flux = operator.interface_flux_contributions(solve.value)
    interface_cancellation = float(jnp.max(jnp.abs(left_flux + right_flux)))
    interface_count = int(jnp.count_nonzero(operator.plan.routes.edge_level_jump))
    finite = all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(solve.value)
    )
    passed = (
        bool(solve.successful)
        and finite
        and residual_relative <= 1.0e-9
        and solution_relative <= 5.0e-2
        and interface_count > 0
        and interface_cancellation <= 1.0e-14
    )
    evidence = {
        "configuration": _configuration(target.topology.plan),
        "identifiers": _identifiers(
            prepared,
            target,
            route_ids=(
                layout.layout_id,
                diffusion_plan.route_fingerprint,
                operator.operator_id,
            ),
            precision_id=diffusion_plan.precision.policy_id,
            execution_id=execution_id,
        ),
        "manufactured_solution": "u=x*(1-x), -laplacian(u)=2, u(0)=u(1)=0",
        "physical_leaf_cells": layout.physical_cell_count,
        "allocated_cells": layout.cell_count,
        "leaf_measure_sum": float(
            jnp.sum(jnp.where(layout.flat_leaf_mask, layout.cell_measures, 0.0))
        ),
        "solve_successful": bool(solve.successful),
        "solve_steps": int(solve.diagnostics.iterations),
        "solve_residual_norm": _tree_norm(layout.space, residual),
        "solve_relative_residual": residual_relative,
        "manufactured_relative_l2_error": solution_relative,
        "coarse_fine_interface_count": interface_count,
        "interface_flux_cancellation_defect": interface_cancellation,
        "route_fingerprint": diffusion_plan.route_fingerprint,
        "coefficient_fingerprint": operator.coefficient_fingerprint,
        "numeric_fingerprint": operator.numeric_fingerprint,
        "finite": finite,
    }
    return _gate("manufactured-composite-poisson", "pass" if passed else "fail", evidence)


def _ad_topology_gate(execution_id: str) -> dict[str, Any]:
    def source(time, state, coordinates, rate):
        del time, coordinates
        return rate * state

    prepared, initial, _ = _hierarchy(periodic=False, levels=1)
    runtime = _runtime(
        prepared,
        initial,
        source=source,
        source_id="source:block-amr-qualification-linear",
    )
    state = runtime.initial_state(_state(initial, lambda x: jnp.ones_like(x)))
    step_size = 0.05

    def objective(rate):
        result = runtime.advance(state, step_size, rate)
        return jnp.sum(result.runtime_state.hierarchy_state.levels[0].values)

    rate = jnp.asarray(0.2, dtype=jnp.float64)
    eager = synchronize(objective(rate))
    compiled = synchronize(eqx.filter_jit(objective)(rate))
    rematerialized = synchronize(jax.checkpoint(objective)(rate))
    derivative = synchronize(jax.grad(objective)(rate))
    expected = 16.0 * (
        step_size + float(rate) * step_size**2 + 0.5 * float(rate) ** 2 * step_size**3
    )
    jit_defect = abs(float(compiled - eager))
    rematerialization_defect = abs(float(rematerialized - eager))
    derivative_defect = abs(float(derivative) - expected)

    stale_prepared, _, stale_refined = _hierarchy(periodic=False, levels=2)
    stale_runtime = _runtime(stale_prepared, stale_refined)
    stale_state = _state(
        stale_prepared.topology_compiler.initialize(),
        lambda x: jnp.ones_like(x),
    )
    stale_refused = False
    stale_message = None
    try:
        stale_runtime.initial_state(stale_state)
    except ValueError as error:
        stale_refused = True
        stale_message = str(error)

    passed = (
        bool(jnp.isfinite(derivative))
        and jit_defect <= 2.0e-13
        and rematerialization_defect <= 2.0e-13
        and derivative_defect <= 2.0e-11
        and stale_refused
    )
    evidence = {
        "configuration": _configuration(initial.topology.plan),
        "identifiers": _identifiers(
            prepared,
            initial,
            route_ids=(runtime.prepared_id, runtime.plan.schedule.schedule_id),
            precision_id=runtime.plan.finite_volume.precision.policy_id,
            execution_id=execution_id,
        ),
        "fixed_epoch_id": initial.topology.epoch.epoch_id,
        "objective": float(eager),
        "compiled_objective": float(compiled),
        "rematerialized_objective": float(rematerialized),
        "jit_defect": jit_defect,
        "rematerialization_defect": rematerialization_defect,
        "derivative": float(derivative),
        "analytic_derivative": expected,
        "derivative_defect": derivative_defect,
        "differentiated_inputs": ["source_rate"],
        "topology_is_static": True,
        "stale_epoch_refused": stale_refused,
        "stale_epoch_message": stale_message,
        "refused_epoch_id": stale_state.topology.epoch.epoch_id,
        "required_epoch_id": stale_refined.topology.epoch.epoch_id,
    }
    return _gate(
        "fixed-epoch-ad-and-topology-refusal", "pass" if passed else "fail", evidence
    )


def _restart_gate(execution_id: str) -> dict[str, Any]:
    prepared, initial, _ = _hierarchy(periodic=True, levels=1)
    runtime = _runtime(prepared, initial)
    hierarchy_state = _state(
        initial,
        lambda x: 1.0 + 0.1 * jnp.cos(2.0 * jnp.pi * x),
    )
    first = synchronize(runtime.advance(runtime.initial_state(hierarchy_state), 0.005))
    accepted = first.runtime_state
    arrays: dict[str, Any] = {
        f"level-{level}": state.values
        for level, state in enumerate(accepted.hierarchy_state.levels)
    }
    arrays.update(
        {
            "time": accepted.time,
            "accepted_step": accepted.accepted_step,
            "level_accepted_steps": accepted.level_accepted_steps,
            "last_status": accepted.last_status,
        }
    )
    encoded = lifecycle.encode_logical_arrays(
        arrays,
        logical_prefix="block-amr-restart",
    )
    decoded = lifecycle.decode_logical_arrays(encoded.manifest, encoded.payloads)
    restored_hierarchy = BlockHierarchyState(
        initial.topology,
        tuple(
            BlockLevelState(level_plan, metadata, decoded[f"level-{level}"])
            for level, (level_plan, metadata) in enumerate(
                zip(
                    initial.topology.plan.levels,
                    initial.topology.levels,
                    strict=True,
                )
            )
        ),
    )
    restored = runtime.initial_state(
        restored_hierarchy,
        time=decoded["time"],
        accepted_step=decoded["accepted_step"],
        level_accepted_steps=decoded["level_accepted_steps"],
    )
    uninterrupted = synchronize(runtime.advance(accepted, 0.005))
    resumed = synchronize(runtime.advance(restored, 0.005))
    state_defect = max(
        float(jnp.max(jnp.abs(left.values - right.values)))
        for left, right in zip(
            uninterrupted.runtime_state.hierarchy_state.levels,
            resumed.runtime_state.hierarchy_state.levels,
            strict=True,
        )
    )
    decoded_exact = all(
        np.array_equal(np.asarray(value), decoded[name]) for name, value in arrays.items()
    )
    counters_equal = (
        int(uninterrupted.runtime_state.accepted_step)
        == int(resumed.runtime_state.accepted_step)
        and np.array_equal(
            np.asarray(uninterrupted.runtime_state.level_accepted_steps),
            np.asarray(resumed.runtime_state.level_accepted_steps),
        )
        and float(uninterrupted.runtime_state.time) == float(resumed.runtime_state.time)
    )
    passed = (
        bool(first.accepted)
        and bool(uninterrupted.accepted)
        and bool(resumed.accepted)
        and decoded_exact
        and counters_equal
        and state_defect == 0.0
    )
    evidence = {
        "configuration": _configuration(initial.topology.plan),
        "identifiers": _identifiers(
            prepared,
            initial,
            route_ids=(runtime.prepared_id, runtime.topology_artifacts.artifacts_id),
            precision_id=runtime.plan.finite_volume.precision.policy_id,
            execution_id=execution_id,
        ),
        "collection_id": encoded.collection_id,
        "manifest_bytes": len(encoded.manifest),
        "payload_bytes": sum(len(value) for value in encoded.payloads.values()),
        "logical_array_count": len(arrays),
        "execution_cache_arrays_archived": 0,
        "decoded_arrays_exact": decoded_exact,
        "continued_state_maximum_defect": state_defect,
        "continued_counters_exact": counters_equal,
        "continued_time": float(resumed.runtime_state.time),
        "continued_accepted_step": int(resumed.runtime_state.accepted_step),
    }
    return _gate("restart", "pass" if passed else "fail", evidence)


def _distributed_gate(execution_id: str) -> dict[str, Any]:
    local_devices = tuple(jax.local_devices())
    accelerator_devices = tuple(
        device for device in local_devices if device.platform in ("gpu", "tpu")
    )
    observed = {
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "local_devices": [
            {
                "process_index": device.process_index,
                "device_id": device.id,
                "platform": device.platform,
                "kind": device.device_kind,
            }
            for device in local_devices
        ],
        "accelerator_device_count": len(accelerator_devices),
        "requested_device_count": 2,
        "environment_execution_id": execution_id,
    }
    if len(accelerator_devices) < 2:
        return _gate(
            "real-device-distributed-determinism",
            "inconclusive",
            observed,
            reason=(
                "At least two locally addressable accelerator devices are required; "
                "CPU logical devices are not claimed as independent physical devices."
            ),
        )

    devices = accelerator_devices[:2]
    group_id = "block-amr-qualification:" + ":".join(
        f"{device.process_index}-{device.id}" for device in devices
    )
    specification = ExecutionGroupSpec(
        group_id,
        tuple(sorted({device.process_index for device in devices})),
        tuple((device.process_index, device.id) for device in devices),
        mesh_axes=(("block_parts", len(devices)),),
    )
    group = ExecutionGroup(specification, devices)
    prepared, _, refined = _hierarchy(periodic=True)
    distributed = BlockAMRPartitionPlan(
        refined.topology.plan,
        len(devices),
    ).prepare(
        refined,
        prepared,
        execution_group=group,
    )
    state = _state(
        refined,
        lambda x: 1.0 + 0.1 * jnp.sin(2.0 * jnp.pi * x),
    )
    serial = synchronize(distributed.serial_fill_patch(state))
    first = synchronize(distributed.distributed_fill_patch(state))
    second = synchronize(distributed.distributed_fill_patch(state))
    parity_defect = max(
        float(jnp.max(jnp.abs(reference.values - actual.values)))
        for reference, actual in zip(serial.workspaces, first.workspaces, strict=True)
    )
    deterministic = all(
        np.array_equal(np.asarray(left.values), np.asarray(right.values))
        and np.array_equal(np.asarray(left.valid), np.asarray(right.valid))
        for left, right in zip(first.workspaces, second.workspaces, strict=True)
    )
    named_sharding = all(
        isinstance(array.sharding, jax.sharding.NamedSharding)
        and array.sharding.spec[0] == distributed.partition.axis_name
        for route in (
            *distributed.same_level_routes,
            *distributed.coarse_fine_routes,
            *distributed.interface_routes,
        )
        for array in route.dynamic_arrays
    )
    loads = [
        np.bincount(
            np.asarray(layout.block_owner)[: layout.active_count],
            minlength=len(devices),
        ).tolist()
        for layout in distributed.layouts
    ]
    passed = (
        bool(first.complete)
        and bool(second.complete)
        and parity_defect <= 1.0e-13
        and deterministic
        and named_sharding
    )
    evidence = {
        **observed,
        "execution_group_id": specification.group_id,
        "partition_plan_id": distributed.partition.plan_id,
        "distributed_prepared_id": distributed.prepared_id,
        "resource_evidence_id": distributed.resource_evidence_id,
        "topology_epoch_id": distributed.topology.epoch.epoch_id,
        "topology_id": distributed.topology.topology_id,
        "precision_policy_id": prepared.plan.precision.policy_id,
        "local_block_capacities": list(distributed.local_block_capacities),
        "active_blocks_by_level": list(distributed.resources.active_blocks),
        "allocated_block_slots_by_level": list(
            distributed.resources.allocated_block_slots
        ),
        "load_blocks_by_level_and_part": loads,
        "same_level_route_counts": list(distributed.resources.same_level_routes),
        "coarse_fine_route_counts": list(distributed.resources.coarse_fine_routes),
        "interface_route_counts": list(distributed.resources.interface_routes),
        "route_array_bytes": distributed.resources.dynamic_route_array_bytes,
        "distributed_serial_maximum_defect": parity_defect,
        "repeat_bitwise_deterministic": deterministic,
        "named_sharding_observed": named_sharding,
    }
    return _gate(
        "real-device-distributed-determinism",
        "pass" if passed else "fail",
        evidence,
    )


def qualify() -> dict[str, Any]:
    """Run all required gates and return JSON-safe machine-readable evidence."""
    environment = capture_environment()
    gates = [
        _constant_state_gate(environment.fingerprint),
        _topology_transition_gate(environment.fingerprint),
        _advection_reflux_gate(environment.fingerprint),
        _poisson_gate(environment.fingerprint),
        _ad_topology_gate(environment.fingerprint),
        _restart_gate(environment.fingerprint),
        _distributed_gate(environment.fingerprint),
    ]
    failed = [
        gate["name"] for gate in gates if gate["required"] and gate["status"] == "fail"
    ]
    inconclusive = [
        gate["name"]
        for gate in gates
        if gate["required"] and gate["status"] == "inconclusive"
    ]
    status: GateStatus = "fail" if failed else "inconclusive" if inconclusive else "pass"
    report = {
        "kind": "block-amr-qualification",
        "status": status,
        "failed_required_gates": failed,
        "inconclusive_required_gates": inconclusive,
        "environment": environment.to_dict(),
        "gates": gates,
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
        description="Qualify the public fixed-block AMR runtime."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON destination; stdout is always emitted.",
    )
    arguments = parser.parse_args(argv)
    report = qualify()
    if arguments.output is not None:
        _write_report(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 1 if report["failed_required_gates"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
