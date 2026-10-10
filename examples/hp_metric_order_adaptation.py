#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bounded, measured h/p and native-metric trials for the reaction PDE u = x**2.

The canonical FE compiler solves this coercive PDE. Existing tensor-hp routes
compete to attain an L2 target; --metric separately exercises the existing native
anisotropic triangle route and trusted physical acceptance/rollback. Coordinate
order remains one on the exact square. --with-history transports named physical
field histories; --archive and --resume use the canonical hp/restart owners.
Costs report actual compilation, solve, transfer and independent reanalysis.
--repeats reports ordered phase samples, not cost uncertainty or superiority.
Memory is shared CPU process-lifetime peak RSS, not a per-candidate allocation
delta. Run from the repository root with JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import NonTrainableState, StrictModule
from phydrax.discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementHPTransferPlan,
)
from phydrax.lifecycle import CompositionRebind
from phydrax.meshing import (
    AdaptationAction,
    DecisionBudget,
    MeasuredAdaptationCost,
    PhysicalErrorEvidence,
    RouteFeasibility,
    SolverAwareCandidate,
    SolverAwareDecision,
)


def exact(points: jax.Array) -> jax.Array:
    return points[..., 0] ** 2


def source(points: jax.Array, args: object) -> jax.Array:
    del args
    return exact(points)


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def retained_array_bytes(tree: object, /) -> int:
    """Count distinct retained numerical leaves without materializing them."""
    seen: set[int] = set()
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, (jax.Array, np.ndarray)) and id(leaf) not in seen:
            seen.add(id(leaf))
            total += int(leaf.size * leaf.dtype.itemsize)
    return total


@eqx.filter_jit
def solve_problem(compiled: Any) -> tuple[jax.Array, Any]:
    operator, rhs = compiled.linear_system()
    solved = phx.linalg.solve(
        operator, rhs, policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
    )
    return compiled.expand(solved.value), solved


def compile_and_solve(
    epoch: Any,
) -> tuple[Any, jax.Array, float, float, float, dict[str, Any]]:
    start = perf_counter()
    form = phx.equations.FiniteElementForm(
        "manufactured-reaction",
        "u",
        (
            phx.equations.MassAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(source, coefficient_id="reaction-x-squared"),
            ),
        ),
    )
    space = (
        epoch
        if isinstance(epoch, phx.discretization.FiniteElementDiscretization)
        else epoch.discretization
    )
    compiled = phx.equations.compile_finite_element_problem(form, space)
    frontend_seconds = perf_counter() - start
    lowering_started = perf_counter()
    lowered = solve_problem.lower(compiled)  # ty: ignore[unresolved-attribute]
    lowering_seconds = perf_counter() - lowering_started
    compilation_started = perf_counter()
    executable = lowered.compile()
    compilation_seconds = perf_counter() - compilation_started
    footprint_started = perf_counter()
    memory = executable.compiled.memory_analysis()
    retained_prepared_bytes = retained_array_bytes((space, compiled))
    footprint_seconds = perf_counter() - footprint_started
    compilation = perf_counter() - start
    start = perf_counter()
    values, solved = jax.block_until_ready(executable(compiled))
    first_execution_seconds = perf_counter() - start
    if not bool(jnp.all(solved.successful)):
        raise RuntimeError("Reaction solve failed.")
    operator, _ = compiled.linear_system()
    # The tiny actual compiled operator, not an element-count condition guess.
    matrix = np.asarray(
        jax.vmap(operator.operator.mv)(jnp.eye(values.size, dtype=jnp.float64))
    ).T
    condition = float(np.linalg.cond(matrix))
    solve_seconds = perf_counter() - start
    compiler = {
        "frontend_seconds": frontend_seconds,
        "lowering_seconds": lowering_seconds,
        "compile_seconds": compilation_seconds,
        "footprint_observation_seconds": footprint_seconds,
        "first_execution_seconds": first_execution_seconds,
        "condition_observation_seconds": solve_seconds - first_execution_seconds,
        "warm_seconds": None,
        "warm_scope": "No extra warm invocations; original solve and independent-reanalysis schedule retained.",
        "temporary_bytes": None if memory is None else memory.temp_size_in_bytes,
        "generated_code_bytes": None
        if memory is None
        else memory.generated_code_size_in_bytes,
        "argument_bytes": None if memory is None else memory.argument_size_in_bytes,
        "output_bytes": None if memory is None else memory.output_size_in_bytes,
        "retained_prepared_array_bytes": retained_prepared_bytes,
        "retained_solution_array_bytes": retained_array_bytes(values),
        "retained_scope": "Distinct declared scientific arrays of space/compiled owner and solved field; excludes native executable and process-global caches.",
        "memory_status": "unavailable" if memory is None else "compiler-reported",
        "compilation_count": 1,
        "compilation_count_scope": "explicit AOT compile invocations",
    }
    return compiled, values, condition, compilation, solve_seconds, compiler


def error_and_integral(epoch: Any, values: jax.Array, count: int) -> tuple[float, float]:
    # Independent Gauss/Duffy integration, not the compiler's quadrature ledger.
    axis, weights = np.polynomial.legendre.leggauss(count)
    axis, weights = (axis + 1.0) / 2.0, weights / 2.0
    tensor_points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    tensor_weights = np.outer(weights, weights).reshape(-1)
    squared, integral = jnp.asarray(0.0), jnp.asarray(0.0)
    space = (
        epoch
        if isinstance(epoch, phx.discretization.FiniteElementDiscretization)
        else epoch.discretization
    )
    for block, cell_block in enumerate(space.mesh.blocks):
        if cell_block.cell_kind == "triangle":
            points = jnp.asarray(
                np.column_stack(
                    (
                        tensor_points[:, 0],
                        (1.0 - tensor_points[:, 0]) * tensor_points[:, 1],
                    )
                )
            )
            quadrature = jnp.asarray(tensor_weights * (1.0 - tensor_points[:, 0]))
        elif cell_block.cell_kind == "quadrilateral":
            points, quadrature = jnp.asarray(tensor_points), jnp.asarray(tensor_weights)
        else:
            raise ValueError("This physical oracle requires triangles or quadrilaterals.")
        geometry = space.evaluate_block_geometry(
            "u", block, space.default_runtime.coordinates, points, quadrature
        )
        dofs = space.dof_maps[0].cell_dofs[block]
        discrete = phx.ein.contract("ql,cl->cq", geometry.basis_values, values[dofs])
        squared += jnp.sum(
            geometry.physical_weights * (exact(geometry.physical_points) - discrete) ** 2
        )
        integral += jnp.sum(geometry.physical_weights * discrete)
    return float(jnp.sqrt(squared)), float(integral)


def pack(epoch: Any, values: jax.Array, width: int) -> jax.Array:
    result = jnp.zeros((epoch.topology.capacity, width), dtype=jnp.float64)
    offset = 0
    for dofs in epoch.discretization.dof_maps[0].cell_dofs:
        for row in range(dofs.shape[0]):
            slot = int(epoch.active_cell_slots[offset + row])
            result = result.at[slot, : dofs.shape[1]].set(values[dofs[row]])
        offset += dofs.shape[0]
    return result


def unpack(epoch: Any, values: jax.Array) -> jax.Array:
    result = jnp.zeros(
        (epoch.discretization.field_spaces[0].layout.size,), dtype=jnp.float64
    )
    offset = 0
    for dofs in epoch.discretization.dof_maps[0].cell_dofs:
        for row in range(dofs.shape[0]):
            slot = int(epoch.active_cell_slots[offset + row])
            result = result.at[dofs[row]].set(values[slot, : dofs.shape[1]])
        offset += dofs.shape[0]
    return result


def physical(
    epoch: Any, values: jax.Array, estimator: str, count: int
) -> PhysicalErrorEvidence:
    error, _ = error_and_integral(epoch, values, count)
    # This measured total physical error already includes algebraic and refresh
    # defects; adding either again would double-count the same manufactured error.
    return PhysicalErrorEvidence(
        epoch.epoch_id,
        "reaction-L2-error",
        estimator,
        field_error=error,
        geometry_error=0.0,
        algebraic_error=0.0,
        transfer_error=0.0,
    )


class _SolvedFieldRefresh(StrictModule, NonTrainableState):
    values: jax.Array

    def __call__(
        self, fields: Any, candidate: Any, args: object
    ) -> tuple[jax.Array, ...]:
        del fields, candidate, args
        return (self.values,)


class _HPHistoryTransfer(StrictModule, NonTrainableState):
    transfer: FiniteElementHPTransferPlan

    def __call__(
        self, auxiliary: Any, integrator: Any, candidate: Any, args: object
    ) -> tuple[Any, Any]:
        del candidate, args
        carried_auxiliary = tuple(
            (name, self.transfer.apply_l2_projection(value)) for name, value in auxiliary
        )
        carried_integrator = tuple(
            (name, self.transfer.apply_l2_projection(value)) for name, value in integrator
        )
        return carried_auxiliary, carried_integrator


class _HPPhysicalCertifier(StrictModule, NonTrainableState):
    carried: jax.Array
    source_integral: jax.Array
    tolerance: float = eqx.field(static=True)

    def __call__(
        self, epoch: Any, fields: Any, materials: Any, candidate: Any, args: object
    ) -> bool:
        del materials, candidate, args
        carried_integral = error_and_integral(epoch, unpack(epoch, self.carried), 7)[1]
        error, integral = error_and_integral(epoch, unpack(epoch, fields[0]), 7)
        reference = float(self.source_integral)
        return (
            bool(jnp.all(jnp.isfinite(fields[0])))
            and max(
                abs(carried_integral - reference),
                abs(integral - reference),
            )
            < 1.0e-10
            and error < self.tolerance
        )


class _MetricPhysicalCertifier(StrictModule, NonTrainableState):
    space: FiniteElementDiscretization
    source_integral: jax.Array
    tolerance: float = eqx.field(static=True)

    def __call__(
        self, mesh: Any, fields: Any, materials: Any, lineage: Any, args: object
    ) -> bool:
        del mesh, materials, lineage, args
        error, integral = error_and_integral(self.space, fields[0], 7)
        return (
            error < self.tolerance
            and abs(integral - float(self.source_integral)) < 1.0e-10
        )


class _MetricPhysicalReanalysis(StrictModule, NonTrainableState):
    space: FiniteElementDiscretization
    revision_id: str = eqx.field(static=True)
    objective_id: str = eqx.field(static=True)

    def __call__(
        self, mesh: Any, fields: Any, materials: Any, args: object
    ) -> PhysicalErrorEvidence:
        del mesh, materials, args
        error, _ = error_and_integral(self.space, fields[0], 7)
        return PhysicalErrorEvidence(
            self.revision_id,
            self.objective_id,
            "independent-accepted-Duffy-7",
            field_error=error,
            geometry_error=0.0,
            algebraic_error=0.0,
            transfer_error=0.0,
        )


def hp_transfer_derivatives(
    transfer: FiniteElementHPTransferPlan, source_values: jax.Array
) -> dict[str, Any]:
    """Check the owner action against an independent NumPy stencil assembly."""
    if transfer.l2_projection is None:
        raise RuntimeError("Physical hp derivatives require the prepared L2 projection.")
    projections = np.asarray(transfer.l2_projection)
    target_width, source_width = projections.shape[1:]
    matrix = np.zeros(
        (transfer.target_capacity * target_width, transfer.source_capacity * source_width)
    )
    for route in np.flatnonzero(np.asarray(transfer.valid)):
        source_count = int(transfer.source_dof_count[route])
        target_count = int(transfer.target_dof_count[route])
        rows = int(transfer.target_slots[route]) * target_width + np.arange(target_count)
        columns = int(transfer.source_slots[route]) * source_width + np.arange(
            source_count
        )
        matrix[np.ix_(rows, columns)] += projections[route, :target_count, :source_count]
    direction = jnp.linspace(-0.2, 0.3, source_values.size).reshape(source_values.shape)
    dual = jnp.linspace(-0.4, 0.7, matrix.shape[0]).reshape(
        transfer.target_capacity, target_width
    )
    _, tangent = jax.jvp(transfer.apply_l2_projection, (source_values,), (direction,))
    _, pullback = jax.vjp(transfer.apply_l2_projection, source_values)
    adjoint = pullback(dual)[0]
    jvp_error = float(
        np.max(
            np.abs(
                np.asarray(tangent).reshape(-1)
                - matrix @ np.asarray(direction).reshape(-1)
            )
        )
    )
    vjp_error = float(
        np.max(
            np.abs(
                np.asarray(adjoint).reshape(-1) - matrix.T @ np.asarray(dual).reshape(-1)
            )
        )
    )
    duality_error = float(jnp.abs(jnp.vdot(dual, tangent) - jnp.vdot(adjoint, direction)))
    if max(jvp_error, vjp_error, duality_error) > 1.0e-10:
        raise RuntimeError(
            "Physical hp transfer derivatives disagree with the independent stencil."
        )
    return {
        "JVP_error": jvp_error,
        "VJP_error": vjp_error,
        "duality_error": duality_error,
        "scope": "fixed prepared L2 field-transfer action; p/h acceptance remains a stopped event",
    }


def run(
    *, p_degree: int = 2, with_history: bool = False, archive: Path | None = None
) -> dict[str, Any]:
    fem = phx.discretization.fem
    if p_degree < 2:
        raise ValueError("The p candidate must increase the field degree.")
    workflow_started = perf_counter()
    mesh = phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=jnp.float64),
        (
            phx.discretization.CellBlock(
                "square",
                "quadrilateral",
                jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
                global_ids=jnp.asarray((10,), dtype=jnp.int64),
            ),
        ),
    )
    topology, geometry = fem.initial_finite_element_hp_topology(mesh, 1, 8)
    initial = fem.prepare_finite_element_hp_epoch(
        topology, geometry, "u", conformity="L2"
    )
    baseline_preparation_seconds = perf_counter() - workflow_started
    (
        baseline_compiled,
        baseline_values,
        baseline_condition,
        baseline_compilation,
        baseline_solve,
        baseline_compiler,
    ) = compile_and_solve(initial)
    start = perf_counter()
    baseline = physical(initial, baseline_values, "independent-Gauss-5", 5)
    baseline_reanalysis_seconds = perf_counter() - start
    budget = DecisionBudget(
        tolerance=0.025,
        maximum_wall_seconds=180.0,
        maximum_memory_bytes=8 * 1024**3,
        maximum_dofs=64,
        maximum_condition=1000.0,
    )
    alternatives: list[SolverAwareCandidate] = []
    trials: dict[str, Any] = {}
    derivative_evidence: dict[str, dict[str, Any]] = {}
    compiler_evidence: dict[str, dict[str, Any]] = {}
    for action in (AdaptationAction.P, AdaptationAction.H):
        start = perf_counter()
        degrees = np.asarray(topology.cell_degrees).copy()
        marks = np.zeros((topology.capacity,), dtype=np.bool_)
        if action is AdaptationAction.P:
            degrees[0, 0] = p_degree
            target_topology = fem.FiniteElementHPTopology(
                topology.cell_kind,
                topology.topology_id,
                topology.cell_global_ids,
                topology.allocated,
                topology.active,
                degrees,
            )
            target_geometry = geometry
            lineage = fem.FiniteElementHPLineage(
                topology.topology_id,
                topology.topology_id,
                8,
                8,
                jnp.asarray((0,), dtype=jnp.int32),
                jnp.asarray((0,), dtype=jnp.int32),
                ("unchanged",),
            )
            transfer_kind = "p"
        else:
            marks[0] = True
            refined = fem.refine_tensor_hp_cells(
                topology, geometry, jnp.asarray((10,), dtype=jnp.int64)
            )
            target_topology, target_geometry, lineage = (
                refined.topology,
                refined.geometry,
                refined.lineage,
            )
            transfer_kind = "h-refinement"
        hp_plan = fem.close_finite_element_hp_decision(
            topology,
            initial.interfaces,
            fem.FiniteElementHPDecision(topology, degrees, marks, np.zeros_like(marks)),
        )
        target = fem.prepare_finite_element_hp_epoch(
            target_topology, target_geometry, "u", conformity="L2"
        )
        certificate = fem.certify_finite_element_hp_geometry(
            target.topology, target.geometry, target.interfaces
        )
        if not certificate.passed:
            raise RuntimeError("Native hp geometry admission failed.")
        preparation = perf_counter() - start
        compiled, values, condition, compilation, solve, compiler = compile_and_solve(
            target
        )
        start = perf_counter()
        transfer = fem.finite_element_hp_transfer_plan(
            initial, target, lineage, "u", transfer_kind
        )
        before = pack(initial, baseline_values, transfer.primal.shape[2])
        carried = transfer.apply_l2_projection(before)
        carried.block_until_ready()
        after = pack(target, values, transfer.primal.shape[1])
        conservation_error = (
            error_and_integral(target, unpack(target, carried), 7)[1]
            - error_and_integral(initial, baseline_values, 7)[1]
        )
        projection_evidence = transfer.l2_evidence
        if projection_evidence is None:
            raise RuntimeError(
                "The hp trial requires certified physical L2 projection evidence."
            )
        transaction = fem.FiniteElementHPTransaction(
            initial,
            target,
            lineage,
            p_transfers=(transfer,) if action is AdaptationAction.P else (),
            h_transfers=(transfer,) if action is AdaptationAction.H else (),
            conservation_error=conservation_error,
            geometry_valid=certificate.passed,
            admissible=projection_evidence.successful & jnp.all(jnp.isfinite(after)),
        )
        accepted = phx.solver.FiniteElementAcceptedState(
            (before,),
            0.0,
            0,
            topology.topology_id,
            initial.epoch_id,
            baseline_compiled.compilation_id,
        )
        rebinds: list[CompositionRebind] = []

        def capture(
            rebind: CompositionRebind, records: list[CompositionRebind] = rebinds
        ) -> CompositionRebind:
            records.append(rebind)
            return rebind

        auxiliary = (("previous-reaction-state", before * 0.5),) if with_history else ()
        integrator = (
            (("accumulated-reaction-state", before * 2.0),) if with_history else ()
        )

        executor = phx.solver.FiniteElementTopologyTransaction(
            _HPPhysicalCertifier(
                carried,
                jnp.asarray(error_and_integral(initial, baseline_values, 7)[1]),
                budget.tolerance,
            ),
            field_transfer=_SolvedFieldRefresh(after),
            composition_rebind=capture,
            history_transfer=_HPHistoryTransfer(transfer) if with_history else None,
        )
        staged = executor.execute_hp(
            accepted,
            transaction,
            auxiliary_state=auxiliary,
            integrator_state=integrator,
        )
        if not bool(staged.committed):
            raise RuntimeError(staged.diagnostics)
        derivative_evidence[hp_plan.decision_id] = hp_transfer_derivatives(
            transfer, before
        )
        transfer_seconds = perf_counter() - start
        start = perf_counter()
        trial_error = physical(target, values, "independent-Gauss-5", 5)
        _, independently_solved, _, _, _, reanalysis_compiler = compile_and_solve(target)
        compiler_evidence[hp_plan.decision_id] = {
            "trial": compiler,
            "independent_reanalysis": reanalysis_compiler,
        }
        reanalysis = physical(
            target, independently_solved, "independent-Gauss-7-recomputed", 7
        )
        reanalysis_seconds = perf_counter() - start
        start = perf_counter()
        rebind = rebinds[-1]
        refresh = fem.FiniteElementHPSolverRefreshPlan(initial, target)
        feasibility = RouteFeasibility(
            initial.epoch_id,
            target.epoch_id,
            transfer.transfer_id,
            hp_plan.decision_id,
            cell_families=("quadrilateral",),
            geometry_layout_id=refresh.geometry_layout_id,
            field_layouts=refresh.field_layouts,
            compiled_layout_id=compiled.compilation_id,
            geometry_certificate_id=certificate.evidence_id,
            topology_certificate_id=certificate.evidence_id,
            required_state_ids=rebind.source.entry_ids,
            state_dispositions=RouteFeasibility.rebind_dispositions(rebind),
            dofs=refresh.dofs,
            condition_estimate=condition,
        )
        decision_seconds = perf_counter() - start
        cost = MeasuredAdaptationCost(
            transfer.transfer_id,
            hp_plan.decision_id,
            preparation_seconds=preparation,
            compilation_seconds=compilation,
            solve_seconds=solve,
            transfer_seconds=transfer_seconds,
            reanalysis_seconds=reanalysis_seconds,
            decision_seconds=decision_seconds,
            peak_memory_bytes=peak_rss_bytes(),
        )
        alternatives.append(
            SolverAwareCandidate(
                action, hp_plan.decision_id, feasibility, trial_error, cost
            )
        )
        trials[hp_plan.decision_id] = (
            target,
            hp_plan,
            transaction,
            executor,
            accepted,
            compiled,
            reanalysis,
            values,
        )
    start = perf_counter()
    decision = SolverAwareDecision(
        baseline,
        budget,
        alternatives,
        observed_campaign_seconds=perf_counter() - workflow_started,
    )
    selection_seconds = perf_counter() - start
    if decision.selected is None:
        raise RuntimeError(str(decision.dispositions))
    target, plan, transaction, executor, accepted, compiled, reanalysis, values = trials[
        decision.selected.candidate_id
    ]
    decision.require_reanalysis(reanalysis)
    bound_plan = fem.close_finite_element_hp_decision(
        topology,
        initial.interfaces,
        plan,
        solver_decision=decision,
        source_revision_id=initial.epoch_id,
        target_revision_id=target.epoch_id,
    )
    bound = fem.FiniteElementHPTransaction(
        initial,
        target,
        transaction.lineage,
        p_transfers=transaction.p_transfers,
        h_transfers=transaction.h_transfers,
        conservation_error=transaction.conservation_error,
        geometry_valid=transaction.geometry_valid,
        admissible=transaction.admissible,
        hp_decision=bound_plan,
        reanalysis=reanalysis,
        compiled_layout_id=compiled.compilation_id,
    )
    result = executor.execute_hp(
        accepted,
        bound,
        auxiliary_state=(("previous-reaction-state", accepted.fields[0] * 0.5),)
        if with_history
        else (),
        integrator_state=(("accumulated-reaction-state", accepted.fields[0] * 2.0),)
        if with_history
        else (),
    )
    if (
        not bool(result.committed)
        or result.receipt is None
        or not result.receipt.published
    ):
        raise RuntimeError(result.diagnostics)
    actual_values = unpack(target, result.state.fields[0])
    actual_error = physical(target, actual_values, "committed-state-Gauss-7", 7)
    decision.require_reanalysis(actual_error)
    history_integral_errors: dict[str, float] = {}
    for entries, factor in (
        (result.auxiliary_state, 0.5),
        (result.integrator_state, 2.0),
    ):
        for name, history in entries:
            error = abs(
                error_and_integral(target, unpack(target, history), 7)[1]
                - factor * error_and_integral(initial, baseline_values, 7)[1]
            )
            if error >= 1.0e-10:
                raise RuntimeError("Physical hp history conservation failed.")
            history_integral_errors[name] = error
    if archive is not None:
        archive.mkdir(parents=True, exist_ok=True)
        phx.solver.write_finite_element_hp_epoch(archive / "epoch.npz", target)
        manifest = phx.solver.FiniteElementRestartManifest(
            result.state,
            auxiliary_state=result.auxiliary_state,
            integrator_state=result.integrator_state,
        )
        phx.solver.write_finite_element_restart(archive / "state.npz", manifest)
    transferred_integral = error_and_integral(
        target,
        unpack(
            target,
            (bound.p_transfers + bound.h_transfers)[0].apply_l2_projection(
                accepted.fields[0]
            ),
        ),
        7,
    )[1]
    transfer_integral_error = abs(
        transferred_integral - error_and_integral(initial, baseline_values, 7)[1]
    )
    integral = error_and_integral(target, values, 7)[1]
    total_observed_seconds = perf_counter() - workflow_started
    if total_observed_seconds > budget.maximum_wall_seconds:
        raise RuntimeError(
            "The complete hp physical workflow exceeded its declared wall budget."
        )
    return {
        "PDE": "u=x^2 (coercive reaction)",
        "baseline_L2_error": baseline.bound,
        "tolerance": budget.tolerance,
        "selected_action": decision.selected.action.value,
        "accepted_L2_error": actual_error.bound,
        "reanalysis_evidence": reanalysis.evidence_id,
        "actual_committed_state_evidence": actual_error.evidence_id,
        "composition_receipt": result.receipt.receipt_id,
        "geometry_order": 1,
        "source_revision": initial.epoch_id,
        "target_revision": target.epoch_id,
        "source_topology": initial.topology.topology_id,
        "target_topology": target.topology.topology_id,
        "decision_id": decision.decision_id,
        "p_candidate_degree": p_degree,
        "history_integral_errors": history_integral_errors,
        "transfer_id": (bound.p_transfers + bound.h_transfers)[0].transfer_id,
        "geometry_certificate": decision.selected.feasibility.geometry_certificate_id,
        "transfer_integral_error": transfer_integral_error,
        "geometry_error": 0.0,
        "integral": integral,
        "observed_total_seconds": total_observed_seconds,
        "baseline_phase_seconds": {
            "preparation": baseline_preparation_seconds,
            "compilation": baseline_compilation,
            "solve_and_condition": baseline_solve,
            "physical_error": baseline_reanalysis_seconds,
        },
        "baseline_condition": baseline_condition,
        "baseline_compiler": baseline_compiler,
        "selection_seconds": selection_seconds,
        "memory_scope": "CPU process-lifetime peak RSS",
        "trials": [
            {
                "action": c.action.value,
                "physical_error": c.error.bound,
                "phase_seconds": c.cost.phase_seconds,
                "peak_RSS_bytes": c.cost.peak_memory_bytes,
                "transfer_derivatives": derivative_evidence[c.candidate_id],
                "compiler": compiler_evidence[c.candidate_id],
                "condition": c.feasibility.condition_estimate,
                "dofs": c.feasibility.dofs,
                "compiled_layout": c.feasibility.compiled_layout_id,
            }
            for c in alternatives
        ],
        "claim": "Measured bounded routes only; no learned or performance superiority claim.",
    }


def run_metric() -> dict[str, Any]:
    """Price and physically reanalyse a native anisotropic planar metric event."""
    workflow_started = perf_counter()
    fem = phx.discretization.fem
    meshing = phx.meshing
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.7, 0.3))),
        np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32),
        vertex_global_ids=np.asarray((7, 3, 9, 1, 5), dtype=np.int64),
        cell_global_ids=np.asarray((10, 20, 30, 40), dtype=np.int64),
    )
    original = meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    field = fem.FiniteElementFieldSpec("u", fem.lagrange_element("triangle", 1))
    source_space = fem.FiniteElementPlan(
        mesh, field, coordinate_spec=original.geometry
    ).prepare()
    baseline_preparation_seconds = perf_counter() - workflow_started
    (
        source_compiled,
        source_values,
        baseline_condition,
        baseline_compilation,
        baseline_solve,
        baseline_compiler,
    ) = compile_and_solve(source_space)
    start = perf_counter()
    baseline_error, source_integral = error_and_integral(source_space, source_values, 7)
    baseline_reanalysis_seconds = perf_counter() - start
    baseline = PhysicalErrorEvidence(
        original.result_id,
        "reaction-L2-error",
        "baseline-independent-Duffy-7",
        field_error=baseline_error,
        geometry_error=0.0,
        algebraic_error=0.0,
        transfer_error=0.0,
    )
    budget = DecisionBudget(
        tolerance=0.025,
        maximum_wall_seconds=180.0,
        maximum_memory_bytes=8 * 1024**3,
        maximum_dofs=4096,
        maximum_condition=1000.0,
    )
    accepted = phx.solver.FiniteElementAcceptedState(
        (source_values,),
        0.0,
        0,
        mesh.topology_id,
        source_space.prepared_id,
        source_compiled.compilation_id,
    )
    campaign_started = perf_counter()
    proposal = meshing.MeshMetricProposal(
        original,
        meshing.mesh_proposal_scope(original, 0),
        np.broadcast_to(np.diag((1.0 / 0.15**2, 1.0 / 0.6**2)), (5, 2, 2)),
        proposer_id="analytic-reaction-x-anisotropy",
    )
    safety = meshing.MeshProposalSafetyPolicy(
        original,
        minimum_size=0.1,
        maximum_size=2.0,
        maximum_displacement=0.1,
        maximum_anisotropy=20.0,
        maximum_gradation=2.0,
    )
    prepared = meshing.prepare_mesh_proposal(original, proposal, safety)
    adaptation = prepared.adaptation
    if adaptation is None or not adaptation.status.converged or not prepared.admissible:
        raise RuntimeError(
            "The actual native metric route did not admit this physical trial."
        )
    target = adaptation.target
    target_space = fem.FiniteElementPlan(
        target.mesh,
        field,
        coordinate_spec=target.geometry,
    ).prepare()
    preparation_seconds = perf_counter() - campaign_started
    compiled, values, condition, compilation_seconds, solve_seconds, compiler = (
        compile_and_solve(target_space)
    )
    captured: list[CompositionRebind] = []

    def capture(rebind: CompositionRebind) -> CompositionRebind:
        captured.append(rebind)
        return rebind

    certify = _MetricPhysicalCertifier(
        target_space, jnp.asarray(source_integral), budget.tolerance
    )
    reanalysis = _MetricPhysicalReanalysis(
        target_space, target.result_id, baseline.objective_id
    )

    executor = phx.solver.FiniteElementTopologyTransaction(
        certify,
        fields=field,
        field_transfer=_SolvedFieldRefresh(values),
        composition_rebind=capture,
    )
    transition_preparation_started = perf_counter()
    preparation = executor.prepare_transition(
        accepted,
        mesh,
        adaptation,
        source=source_space,
        target=target_space,
    )
    preparation_seconds += perf_counter() - transition_preparation_started
    start = perf_counter()
    trial = executor.execute(
        accepted,
        mesh,
        adaptation,
        compiled_layout_id=compiled.compilation_id,
        preparation=preparation,
    )
    if not bool(trial.committed) or trial.receipt is None or not trial.receipt.published:
        raise RuntimeError(trial.diagnostics)
    transfer_seconds = perf_counter() - start
    start = perf_counter()
    error, _ = error_and_integral(target_space, values, 5)
    assessed = PhysicalErrorEvidence(
        target.result_id,
        baseline.objective_id,
        "candidate-independent-Duffy-5",
        field_error=error,
        geometry_error=0.0,
        algebraic_error=0.0,
        transfer_error=0.0,
    )
    _, independently_solved, _, _, _, reanalysis_compiler = compile_and_solve(
        target_space
    )
    actual = reanalysis(target.mesh, (independently_solved,), None, None)
    reanalysis_seconds = perf_counter() - start
    start = perf_counter()
    rebind = captured[-1]
    feasibility = RouteFeasibility(
        original.result_id,
        target.result_id,
        adaptation.route.value,
        adaptation.result_id,
        cell_families=("triangle",),
        geometry_layout_id=target.geometry.geometry_layout_id,
        field_layouts=tuple(
            (space.name, space.layout.layout_id) for space in target_space.field_spaces
        ),
        compiled_layout_id=compiled.compilation_id,
        geometry_certificate_id=target.audit.report_id,
        topology_certificate_id=target.audit.report_id,
        required_state_ids=rebind.source.entry_ids,
        state_dispositions=RouteFeasibility.rebind_dispositions(rebind),
        dofs=target_space.field_spaces[0].layout.size,
        condition_estimate=condition,
    )
    cost = MeasuredAdaptationCost(
        adaptation.result_id,
        adaptation.result_id,
        preparation_seconds=preparation_seconds,
        compilation_seconds=compilation_seconds,
        solve_seconds=solve_seconds,
        transfer_seconds=transfer_seconds,
        reanalysis_seconds=reanalysis_seconds,
        decision_seconds=perf_counter() - start,
        peak_memory_bytes=peak_rss_bytes(),
    )
    candidate = SolverAwareCandidate(
        AdaptationAction.METRIC, adaptation.result_id, feasibility, assessed, cost
    )
    decision = SolverAwareDecision(
        baseline,
        budget,
        (candidate,),
        observed_campaign_seconds=perf_counter() - workflow_started,
    )
    decision.require_reanalysis(actual)
    start = perf_counter()
    result = executor.execute(
        accepted,
        mesh,
        adaptation,
        decision=decision,
        reanalysis=reanalysis,
        compiled_layout_id=compiled.compilation_id,
        preparation=preparation,
    )
    publication_seconds = perf_counter() - start
    if (
        not bool(result.committed)
        or result.receipt is None
        or not result.receipt.published
    ):
        raise RuntimeError(result.diagnostics)

    def reject_reanalysis(
        candidate_mesh: Any, fields: Any, materials: Any, args: object
    ) -> PhysicalErrorEvidence:
        del candidate_mesh, fields, materials, args
        return PhysicalErrorEvidence(
            target.result_id,
            baseline.objective_id,
            "independent-rejected-physical-error",
            field_error=baseline.bound,
            geometry_error=0.0,
            algebraic_error=0.0,
            transfer_error=0.0,
        )

    start = perf_counter()
    refused = executor.execute(
        accepted,
        mesh,
        adaptation,
        decision=decision,
        reanalysis=reject_reanalysis,
        compiled_layout_id=compiled.compilation_id,
        preparation=preparation,
    )
    rejection_seconds = perf_counter() - start
    if (
        bool(refused.committed)
        or refused.state is not accepted
        or refused.mesh is not mesh
    ):
        raise RuntimeError(
            "Rejected physical metric reanalysis changed the accepted epoch."
        )
    transfer = result.transfers[0].transfer
    derivative_started = perf_counter()
    direction = jnp.linspace(-0.2, 0.3, source_values.size)
    dual = jnp.linspace(-0.4, 0.7, result.state.fields[0].size)
    _, tangent = jax.jvp(transfer.apply, (source_values,), (direction,))
    _, pullback = jax.vjp(transfer.apply, source_values)
    duality_error = float(
        jnp.abs(jnp.vdot(dual, tangent) - jnp.vdot(pullback(dual)[0], direction))
    )
    if duality_error > 1.0e-10:
        raise RuntimeError("Fixed metric transfer failed JVP/VJP duality.")
    epsilon = 1.0e-5
    finite_difference = np.asarray(
        (
            transfer.apply(source_values + epsilon * direction)
            - transfer.apply(source_values - epsilon * direction)
        )
        / (2.0 * epsilon)
    )
    jvp_error = float(np.max(np.abs(np.asarray(tangent) - finite_difference)))
    reference_adjoint = np.empty(source_values.size)
    for index in range(source_values.size):
        perturbation = jnp.zeros_like(source_values).at[index].set(epsilon)
        difference = (
            transfer.apply(source_values + perturbation)
            - transfer.apply(source_values - perturbation)
        ) / (2.0 * epsilon)
        reference_adjoint[index] = np.vdot(np.asarray(dual), np.asarray(difference))
    vjp_error = float(np.max(np.abs(np.asarray(pullback(dual)[0]) - reference_adjoint)))
    if max(jvp_error, vjp_error) > 1.0e-7:
        raise RuntimeError(
            "Fixed metric transfer derivatives disagree with independent finite differences."
        )
    derivative_seconds = perf_counter() - derivative_started
    final_error, integral = error_and_integral(target_space, result.state.fields[0], 7)
    total_observed_seconds = perf_counter() - workflow_started
    if total_observed_seconds > budget.maximum_wall_seconds:
        raise RuntimeError(
            "The complete native metric physical workflow exceeded its declared wall budget."
        )
    return {
        "action": "metric",
        "route": adaptation.route.value,
        "status": adaptation.status.value,
        "baseline_L2_error": baseline.bound,
        "accepted_L2_error": final_error,
        "tolerance": budget.tolerance,
        "integral_error": abs(integral - source_integral),
        "transfer_duality_error": duality_error,
        "physical_reanalysis_rollback": not bool(refused.committed),
        "transfer_JVP_finite_difference_error": jvp_error,
        "transfer_VJP_finite_difference_error": vjp_error,
        "derivative_seconds": derivative_seconds,
        "phase_seconds": cost.phase_seconds,
        "publication_seconds": publication_seconds,
        "rejected_reanalysis_seconds": rejection_seconds,
        "observed_total_seconds": total_observed_seconds,
        "baseline_phase_seconds": {
            "preparation": baseline_preparation_seconds,
            "compilation": baseline_compilation,
            "solve_and_condition": baseline_solve,
            "physical_error": baseline_reanalysis_seconds,
        },
        "baseline_condition": baseline_condition,
        "baseline_compiler": baseline_compiler,
        "trial_compiler": compiler,
        "independent_reanalysis_compiler": reanalysis_compiler,
        "peak_process_bytes": peak_rss_bytes(),
        "condition": condition,
        "dofs": feasibility.dofs,
        "source_revision": original.result_id,
        "target_revision": target.result_id,
        "decision_id": decision.decision_id,
        "receipt": result.receipt.receipt_id,
        "derivative_scope": "fixed qualified field-transfer operator only; metric acceptance is a stopped event",
        "claim": "Observed physical error/cost only; no learned or performance superiority.",
    }


def resume_hp(archive: Path) -> dict[str, Any]:
    """Cold consumer rebuilding the real hp space and independently re-solving."""
    started = perf_counter()
    epoch = phx.solver.read_finite_element_hp_epoch(archive / "epoch.npz")
    manifest = phx.solver.read_finite_element_restart(archive / "state.npz")
    if (
        manifest.state.prepared_id != epoch.epoch_id
        or manifest.state.topology_id != epoch.topology.topology_id
    ):
        raise RuntimeError("Archived hp state and space identify different epochs.")
    values = unpack(epoch, manifest.state.fields[0])
    error, integral = error_and_integral(epoch, values, 7)
    if error >= 0.025 or abs(integral - 1.0 / 3.0) >= 1.0e-10:
        raise RuntimeError("Cold accepted hp state fails the original physical gates.")
    histories: dict[str, float] = {}
    for entries, factor in (
        (manifest.auxiliary_state, 0.5),
        (manifest.integrator_state, 2.0),
    ):
        for name, history in entries:
            defect = abs(
                error_and_integral(epoch, unpack(epoch, history), 7)[1] - factor / 3.0
            )
            if defect >= 1.0e-10:
                raise RuntimeError(
                    "Cold hp history violates physical integral preservation."
                )
            histories[name] = defect
    preparation_seconds = perf_counter() - started
    compiled, solved, condition, compilation_seconds, solve_seconds, compiler = (
        compile_and_solve(epoch)
    )
    recomputed_error, recomputed_integral = error_and_integral(epoch, solved, 7)
    if recomputed_error >= 0.025 or abs(recomputed_integral - integral) >= 1.0e-10:
        raise RuntimeError("Independent cold PDE reanalysis rejected the accepted epoch.")
    return {
        "accepted_L2_error": error,
        "independently_recomputed_L2_error": recomputed_error,
        "history_integral_errors": histories,
        "integral": integral,
        "epoch_id": epoch.epoch_id,
        "transition_id": manifest.state.transition_id,
        "stored_compilation_id": manifest.state.compilation_id,
        "cold_compilation_id": compiled.compilation_id,
        "preparation_seconds": preparation_seconds,
        "compilation_seconds": compilation_seconds,
        "compiler": compiler,
        "solve_seconds": solve_seconds,
        "condition": condition,
        "peak_process_bytes": peak_rss_bytes(),
        "scope": "reconstructed numeric epoch and independently solved PDE; no compiled cache claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p-degree", type=int, default=2)
    parser.add_argument("--with-history", action="store_true")
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    if arguments.resume is not None:
        if arguments.metric or arguments.archive is not None or arguments.repeats != 1:
            parser.error(
                "--resume is a single hp cold consumer, not a new adaptation campaign"
            )
        samples = [resume_hp(arguments.resume)]
    else:
        if arguments.metric and arguments.archive is not None:
            parser.error(
                "--archive records hp epochs; metric mesh archives belong to the meshing owner"
            )
        samples = [
            run_metric()
            if arguments.metric
            else run(
                p_degree=arguments.p_degree,
                with_history=arguments.with_history,
                archive=arguments.archive,
            )
            for _ in range(arguments.repeats)
        ]
    print(
        json.dumps(
            samples[0]
            if arguments.repeats == 1
            else {
                "samples": samples,
                "scope": "ordered same-process observed phase samples; no cost uncertainty or superiority claim",
                "memory_scope": "shared process-lifetime peak RSS",
            },
            indent=2,
        )
    )
