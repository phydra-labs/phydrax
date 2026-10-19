# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Prepared hp publication with named histories, independent PDE reanalysis, and restart."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization.fem import FiniteElementHPTransferKind
from phydrax.lifecycle import CompositionRebind
from phydrax.meshing import (
    AdaptationAction,
    DecisionBudget,
    MeasuredAdaptationCost,
    RouteFeasibility,
    SolverAwareCandidate,
    SolverAwareDecision,
)
from tests._support.solver_campaigns_common import (
    compile_and_solve,
    error_and_integral,
    hp_transfer_derivatives,
    HPHistoryTransfer,
    HPPhysicalCertifier,
    pack,
    peak_rss_bytes,
    physical,
    SolvedFieldRefresh,
    unpack,
)


@dataclass(frozen=True)
class HPSource:
    initial: Any
    compiled: Any
    values: Any
    baseline: Any
    budget: DecisionBudget
    started: float


@dataclass(frozen=True)
class HPTarget:
    epoch: Any
    plan: Any
    lineage: Any
    certificate: Any
    kind: FiniteElementHPTransferKind
    preparation_seconds: float


@dataclass(frozen=True)
class HPStaging:
    target: HPTarget
    compiled: Any
    values: Any
    condition: float
    compilation_seconds: float
    solve_seconds: float
    transfer_seconds: float
    transaction: Any
    executor: Any
    accepted: Any
    rebind: CompositionRebind
    derivatives: dict[str, Any]


@dataclass(frozen=True)
class HPTrial:
    staging: HPStaging
    candidate: SolverAwareCandidate
    reanalysis: Any


def prepare_hp_source() -> HPSource:
    started = perf_counter()
    fem = phx.discretization.fem
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
    compiled, values, _, _, _ = compile_and_solve(initial)
    baseline = physical(initial, values, "independent-Gauss-5", 5)
    budget = DecisionBudget(
        tolerance=0.025,
        maximum_wall_seconds=180.0,
        maximum_memory_bytes=8 * 1024**3,
        maximum_dofs=64,
        maximum_condition=1000.0,
    )
    return HPSource(initial, compiled, values, baseline, budget, started)


def prepare_hp_target(
    source: HPSource, action: AdaptationAction, degree: int
) -> HPTarget:
    started = perf_counter()
    fem = phx.discretization.fem
    initial = source.initial
    topology, geometry = initial.topology, initial.geometry
    degrees = np.asarray(topology.cell_degrees).copy()
    marks = np.zeros((topology.capacity,), dtype=np.bool_)
    match action:
        case AdaptationAction.P:
            degrees[0, 0] = degree
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
            kind = "p"
        case AdaptationAction.H:
            marks[0] = True
            refined = fem.refine_tensor_hp_cells(
                topology, geometry, jnp.asarray((10,), dtype=jnp.int64)
            )
            target_topology, target_geometry, lineage = (
                refined.topology,
                refined.geometry,
                refined.lineage,
            )
            kind = "h-refinement"
        case _:
            raise ValueError("hp_campaign.action must be p or h; got unsupported action")
    plan = fem.close_finite_element_hp_decision(
        topology,
        initial.interfaces,
        fem.FiniteElementHPDecision(topology, degrees, marks, np.zeros_like(marks)),
    )
    epoch = fem.prepare_finite_element_hp_epoch(
        target_topology, target_geometry, "u", conformity="L2"
    )
    certificate = fem.certify_finite_element_hp_geometry(
        epoch.topology, epoch.geometry, epoch.interfaces
    )
    if not certificate.passed:
        raise RuntimeError(
            "hp_campaign.geometry must pass native admission; got refused certificate"
        )
    return HPTarget(epoch, plan, lineage, certificate, kind, perf_counter() - started)


def stage_hp_trial(source: HPSource, target: HPTarget, with_history: bool) -> HPStaging:
    fem = phx.discretization.fem
    initial, epoch = source.initial, target.epoch
    compiled, values, condition, compilation, solve = compile_and_solve(epoch)
    started = perf_counter()
    transfer = fem.finite_element_hp_transfer_plan(
        initial, epoch, target.lineage, "u", target.kind
    )
    before = pack(initial, source.values, transfer.primal.shape[2])
    carried = transfer.apply_l2_projection(before)
    carried.block_until_ready()
    after = pack(epoch, values, transfer.primal.shape[1])
    source_integral = error_and_integral(initial, source.values, 7)[1]
    conservation_error = (
        error_and_integral(epoch, unpack(epoch, carried), 7)[1] - source_integral
    )
    evidence = transfer.l2_evidence
    if evidence is None:
        raise RuntimeError(
            "hp_campaign.projection must retain certified physical L2 evidence; got None"
        )
    transaction = fem.FiniteElementHPTransaction(
        initial,
        epoch,
        target.lineage,
        p_transfers=(transfer,) if target.kind == "p" else (),
        h_transfers=(transfer,) if target.kind == "h-refinement" else (),
        conservation_error=conservation_error,
        geometry_valid=target.certificate.passed,
        admissible=evidence.successful & jnp.all(jnp.isfinite(after)),
    )
    accepted = phx.solver.FiniteElementAcceptedState(
        (before,),
        0.0,
        0,
        initial.topology.topology_id,
        initial.epoch_id,
        source.compiled.compilation_id,
    )
    rebinds: list[CompositionRebind] = []

    def capture(rebind: CompositionRebind) -> CompositionRebind:
        rebinds.append(rebind)
        return rebind

    executor = phx.solver.FiniteElementTopologyTransaction(
        HPPhysicalCertifier(
            carried, jnp.asarray(source_integral), source.budget.tolerance
        ),
        field_transfer=SolvedFieldRefresh(after),
        composition_rebind=capture,
        history_transfer=HPHistoryTransfer(transfer) if with_history else None,
    )
    staged = executor.execute_hp(
        accepted,
        transaction,
        auxiliary_state=(("previous-reaction-state", before * 0.5),)
        if with_history
        else (),
        integrator_state=(("accumulated-reaction-state", before * 2.0),)
        if with_history
        else (),
    )
    if not bool(staged.committed):
        raise RuntimeError(staged.diagnostics)
    derivatives = hp_transfer_derivatives(transfer, before)
    return HPStaging(
        target,
        compiled,
        values,
        condition,
        compilation,
        solve,
        perf_counter() - started,
        transaction,
        executor,
        accepted,
        rebinds[-1],
        derivatives,
    )


def price_hp_trial(
    source: HPSource, staged: HPStaging, action: AdaptationAction
) -> HPTrial:
    started = perf_counter()
    target = staged.target
    error = physical(target.epoch, staged.values, "independent-Gauss-5", 5)
    _, recomputed, _, _, _ = compile_and_solve(target.epoch)
    reanalysis = physical(target.epoch, recomputed, "independent-Gauss-7-recomputed", 7)
    reanalysis_seconds = perf_counter() - started
    started = perf_counter()
    refresh = phx.discretization.fem.FiniteElementHPSolverRefreshPlan(
        source.initial, target.epoch
    )
    feasibility = RouteFeasibility(
        source.initial.epoch_id,
        target.epoch.epoch_id,
        (staged.transaction.p_transfers + staged.transaction.h_transfers)[0].transfer_id,
        target.plan.decision_id,
        cell_families=("quadrilateral",),
        geometry_layout_id=refresh.geometry_layout_id,
        field_layouts=refresh.field_layouts,
        compiled_layout_id=staged.compiled.compilation_id,
        geometry_certificate_id=target.certificate.evidence_id,
        topology_certificate_id=target.certificate.evidence_id,
        required_state_ids=staged.rebind.source.entry_ids,
        state_dispositions=RouteFeasibility.rebind_dispositions(staged.rebind),
        dofs=refresh.dofs,
        condition_estimate=staged.condition,
    )
    cost = MeasuredAdaptationCost(
        feasibility.route_id,
        target.plan.decision_id,
        preparation_seconds=target.preparation_seconds,
        compilation_seconds=staged.compilation_seconds,
        solve_seconds=staged.solve_seconds,
        transfer_seconds=staged.transfer_seconds,
        reanalysis_seconds=reanalysis_seconds,
        decision_seconds=perf_counter() - started,
        peak_memory_bytes=peak_rss_bytes(),
    )
    candidate = SolverAwareCandidate(
        action, target.plan.decision_id, feasibility, error, cost
    )
    return HPTrial(staged, candidate, reanalysis)


def publish_hp_trial(
    source: HPSource, trial: HPTrial, decision: SolverAwareDecision, with_history: bool
) -> Any:
    fem = phx.discretization.fem
    staged = trial.staging
    target, transaction = staged.target, staged.transaction
    decision.require_reanalysis(trial.reanalysis)
    plan = fem.close_finite_element_hp_decision(
        source.initial.topology,
        source.initial.interfaces,
        target.plan,
        solver_decision=decision,
        source_revision_id=source.initial.epoch_id,
        target_revision_id=target.epoch.epoch_id,
    )
    bound = fem.FiniteElementHPTransaction(
        source.initial,
        target.epoch,
        transaction.lineage,
        p_transfers=transaction.p_transfers,
        h_transfers=transaction.h_transfers,
        conservation_error=transaction.conservation_error,
        geometry_valid=transaction.geometry_valid,
        admissible=transaction.admissible,
        hp_decision=plan,
        reanalysis=trial.reanalysis,
        compiled_layout_id=staged.compiled.compilation_id,
    )
    before = staged.accepted.fields[0]
    result = staged.executor.execute_hp(
        staged.accepted,
        bound,
        auxiliary_state=(("previous-reaction-state", before * 0.5),)
        if with_history
        else (),
        integrator_state=(("accumulated-reaction-state", before * 2.0),)
        if with_history
        else (),
    )
    if (
        not bool(result.committed)
        or result.receipt is None
        or not result.receipt.published
    ):
        raise RuntimeError(result.diagnostics)
    return result


def hp_history_errors(source: HPSource, epoch: Any, result: Any) -> dict[str, float]:
    errors: dict[str, float] = {}
    for entries, factor in (
        (result.auxiliary_state, 0.5),
        (result.integrator_state, 2.0),
    ):
        for name, history in entries:
            error = abs(
                error_and_integral(epoch, unpack(epoch, history), 7)[1]
                - factor * error_and_integral(source.initial, source.values, 7)[1]
            )
            if error >= 1.0e-10:
                raise RuntimeError(
                    "hp_campaign.history must preserve physical integrals; got excessive conservation defect"
                )
            errors[name] = error
    return errors


def run(
    *, p_degree: int = 2, with_history: bool = False, archive: Path | None = None
) -> dict[str, Any]:
    if p_degree < 2:
        raise ValueError(
            "hp_campaign.degree must increase the field degree; got a degree below two"
        )
    source = prepare_hp_source()
    trials = [
        price_hp_trial(
            source,
            stage_hp_trial(
                source, prepare_hp_target(source, action, p_degree), with_history
            ),
            action,
        )
        for action in (AdaptationAction.P, AdaptationAction.H)
    ]
    decision = SolverAwareDecision(
        source.baseline,
        source.budget,
        tuple(trial.candidate for trial in trials),
        observed_campaign_seconds=perf_counter() - source.started,
    )
    if decision.selected is None:
        raise RuntimeError(str(decision.dispositions))
    selected = next(trial for trial in trials if trial.candidate is decision.selected)
    result = publish_hp_trial(source, selected, decision, with_history)
    staged, epoch = selected.staging, selected.staging.target.epoch
    actual = physical(
        epoch, unpack(epoch, result.state.fields[0]), "committed-state-Gauss-7", 7
    )
    decision.require_reanalysis(actual)
    histories = hp_history_errors(source, epoch, result)
    if archive is not None:
        archive.mkdir(parents=True, exist_ok=True)
        phx.solver.write_finite_element_hp_epoch(archive / "epoch.npz", epoch)
        phx.solver.write_finite_element_restart(
            archive / "state.npz",
            phx.solver.FiniteElementRestartManifest(
                result.state,
                auxiliary_state=result.auxiliary_state,
                integrator_state=result.integrator_state,
            ),
        )
    transfer = (staged.transaction.p_transfers + staged.transaction.h_transfers)[0]
    transferred = transfer.apply_l2_projection(staged.accepted.fields[0])
    integral_error = abs(
        error_and_integral(epoch, unpack(epoch, transferred), 7)[1]
        - error_and_integral(source.initial, source.values, 7)[1]
    )
    if perf_counter() - source.started > source.budget.maximum_wall_seconds:
        raise RuntimeError(
            "hp_campaign.elapsed_seconds must fit the declared wall budget; got an exceeded deadline"
        )
    return {
        "baseline_L2_error": source.baseline.bound,
        "accepted_L2_error": actual.bound,
        "tolerance": source.budget.tolerance,
        "history_integral_errors": histories,
        "transfer_integral_error": integral_error,
        "trials": [
            {
                "action": trial.candidate.action.value,
                "phase_seconds": trial.candidate.cost.phase_seconds,
                "transfer_derivatives": trial.staging.derivatives,
            }
            for trial in trials
        ],
    }


def resume_hp(archive: Path) -> dict[str, Any]:
    epoch = phx.solver.read_finite_element_hp_epoch(archive / "epoch.npz")
    manifest = phx.solver.read_finite_element_restart(archive / "state.npz")
    if (
        manifest.state.prepared_id != epoch.epoch_id
        or manifest.state.topology_id != epoch.topology.topology_id
    ):
        raise RuntimeError(
            "hp_campaign.archive must bind state and space to one epoch; got inconsistent epoch IDs"
        )
    values = unpack(epoch, manifest.state.fields[0])
    error, integral = error_and_integral(epoch, values, 7)
    if error >= 0.025 or abs(integral - 1.0 / 3.0) >= 1.0e-10:
        raise RuntimeError(
            "hp_campaign.archive must pass the retained physical gates; got excessive solution or integral error"
        )
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
                    "hp_campaign.archive.history must preserve physical integrals; got excessive history defect"
                )
            histories[name] = defect
    _, solved, _, _, _ = compile_and_solve(epoch)
    recomputed_error, recomputed_integral = error_and_integral(epoch, solved, 7)
    if recomputed_error >= 0.025 or abs(recomputed_integral - integral) >= 1.0e-10:
        raise RuntimeError(
            "hp_campaign.archive.reanalysis must pass independent PDE gates; got excessive solution or integral error"
        )
    return {
        "accepted_L2_error": error,
        "independently_recomputed_L2_error": recomputed_error,
        "history_integral_errors": histories,
    }


def hp_campaign(p_degree: str) -> dict[str, Any]:
    return run(p_degree=int(p_degree), with_history=True)
