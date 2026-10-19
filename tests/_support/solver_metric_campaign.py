# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Native metric adaptation through public solver publication and physical rollback."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
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
from tests._support.solver_campaigns_common import (
    compile_and_solve,
    error_and_integral,
    MetricPhysicalCertifier,
    MetricPhysicalReanalysis,
    peak_rss_bytes,
    SolvedFieldRefresh,
)


@dataclass(frozen=True)
class MetricSource:
    original: Any
    space: Any
    values: Any
    integral: float
    baseline: PhysicalErrorEvidence
    budget: DecisionBudget
    accepted: Any
    started: float


@dataclass(frozen=True)
class MetricStaging:
    adaptation: Any
    space: Any
    values: Any
    compiled: Any
    condition: float
    preparation_seconds: float
    compilation_seconds: float
    solve_seconds: float
    transfer_seconds: float
    executor: Any
    preparation: Any
    reanalysis: Any
    rebind: CompositionRebind


def prepare_metric_source() -> MetricSource:
    started = perf_counter()
    fem = phx.discretization.fem
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.7, 0.3)), dtype=np.float64
        ),
        np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32),
        vertex_global_ids=np.asarray((7, 3, 9, 1, 5), dtype=np.int64),
        cell_global_ids=np.asarray((10, 20, 30, 40), dtype=np.int64),
    )
    original = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    field = fem.FiniteElementFieldSpec("u", fem.lagrange_element("triangle", 1))
    space = fem.FiniteElementPlan(
        mesh, field, coordinate_spec=original.geometry
    ).prepare()
    compiled, values, _, _, _ = compile_and_solve(space)
    error, integral = error_and_integral(space, values, 7)
    baseline = PhysicalErrorEvidence(
        original.result_id,
        "reaction-L2-error",
        "baseline-independent-Duffy-7",
        field_error=error,
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
        (values,), 0.0, 0, mesh.topology_id, space.prepared_id, compiled.compilation_id
    )
    return MetricSource(
        original, space, values, integral, baseline, budget, accepted, started
    )


def stage_metric_trial(source: MetricSource) -> MetricStaging:
    started = perf_counter()
    fem, meshing = phx.discretization.fem, phx.meshing
    original = source.original
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
            "metric_campaign.adaptation must admit the physical trial; got refused native metric route"
        )
    target = adaptation.target
    field = fem.FiniteElementFieldSpec("u", fem.lagrange_element("triangle", 1))
    space = fem.FiniteElementPlan(
        target.mesh, field, coordinate_spec=target.geometry
    ).prepare()
    preparation_seconds = perf_counter() - started
    compiled, values, condition, compilation_seconds, solve_seconds = compile_and_solve(
        space
    )
    captured: list[CompositionRebind] = []

    def capture(rebind: CompositionRebind) -> CompositionRebind:
        captured.append(rebind)
        return rebind

    certify = MetricPhysicalCertifier(
        space, jnp.asarray(source.integral), source.budget.tolerance
    )
    reanalysis = MetricPhysicalReanalysis(
        space, target.result_id, source.baseline.objective_id
    )
    executor = phx.solver.FiniteElementTopologyTransaction(
        certify,
        fields=field,
        field_transfer=SolvedFieldRefresh(values),
        composition_rebind=capture,
    )
    started = perf_counter()
    preparation = executor.prepare_transition(
        source.accepted,
        original.mesh,
        adaptation,
        source=source.space,
        target=space,
    )
    preparation_seconds += perf_counter() - started
    started = perf_counter()
    trial = executor.execute(
        source.accepted,
        original.mesh,
        adaptation,
        compiled_layout_id=compiled.compilation_id,
        preparation=preparation,
    )
    if not bool(trial.committed) or trial.receipt is None or not trial.receipt.published:
        raise RuntimeError(trial.diagnostics)
    return MetricStaging(
        adaptation,
        space,
        values,
        compiled,
        condition,
        preparation_seconds,
        compilation_seconds,
        solve_seconds,
        perf_counter() - started,
        executor,
        preparation,
        reanalysis,
        captured[-1],
    )


def price_metric_trial(
    source: MetricSource, staged: MetricStaging
) -> SolverAwareDecision:
    started = perf_counter()
    target = staged.adaptation.target
    error, _ = error_and_integral(staged.space, staged.values, 5)
    assessed = PhysicalErrorEvidence(
        target.result_id,
        source.baseline.objective_id,
        "candidate-independent-Duffy-5",
        field_error=error,
        geometry_error=0.0,
        algebraic_error=0.0,
        transfer_error=0.0,
    )
    _, recomputed, _, _, _ = compile_and_solve(staged.space)
    actual = staged.reanalysis(target.mesh, (recomputed,), None, None)
    reanalysis_seconds = perf_counter() - started
    started = perf_counter()
    feasibility = RouteFeasibility(
        source.original.result_id,
        target.result_id,
        staged.adaptation.route.value,
        staged.adaptation.result_id,
        cell_families=("triangle",),
        geometry_layout_id=target.geometry.geometry_layout_id,
        field_layouts=tuple(
            (space.name, space.layout.layout_id) for space in staged.space.field_spaces
        ),
        compiled_layout_id=staged.compiled.compilation_id,
        geometry_certificate_id=target.audit.report_id,
        topology_certificate_id=target.audit.report_id,
        required_state_ids=staged.rebind.source.entry_ids,
        state_dispositions=RouteFeasibility.rebind_dispositions(staged.rebind),
        dofs=staged.space.field_spaces[0].layout.size,
        condition_estimate=staged.condition,
    )
    cost = MeasuredAdaptationCost(
        staged.adaptation.result_id,
        staged.adaptation.result_id,
        preparation_seconds=staged.preparation_seconds,
        compilation_seconds=staged.compilation_seconds,
        solve_seconds=staged.solve_seconds,
        transfer_seconds=staged.transfer_seconds,
        reanalysis_seconds=reanalysis_seconds,
        decision_seconds=perf_counter() - started,
        peak_memory_bytes=peak_rss_bytes(),
    )
    candidate = SolverAwareCandidate(
        AdaptationAction.METRIC, staged.adaptation.result_id, feasibility, assessed, cost
    )
    decision = SolverAwareDecision(
        source.baseline,
        source.budget,
        (candidate,),
        observed_campaign_seconds=perf_counter() - source.started,
    )
    decision.require_reanalysis(actual)
    return decision


def publish_metric_trial(
    source: MetricSource, staged: MetricStaging, decision: SolverAwareDecision
) -> Any:
    result = staged.executor.execute(
        source.accepted,
        source.original.mesh,
        staged.adaptation,
        decision=decision,
        reanalysis=staged.reanalysis,
        compiled_layout_id=staged.compiled.compilation_id,
        preparation=staged.preparation,
    )
    if (
        not bool(result.committed)
        or result.receipt is None
        or not result.receipt.published
    ):
        raise RuntimeError(result.diagnostics)
    return result


def refuse_metric_reanalysis(
    source: MetricSource, staged: MetricStaging, decision: SolverAwareDecision
) -> Any:
    def reject(
        candidate_mesh: Any, fields: Any, materials: Any, args: object
    ) -> PhysicalErrorEvidence:
        del candidate_mesh, fields, materials, args
        return PhysicalErrorEvidence(
            staged.adaptation.target.result_id,
            source.baseline.objective_id,
            "independent-rejected-physical-error",
            field_error=source.baseline.bound,
            geometry_error=0.0,
            algebraic_error=0.0,
            transfer_error=0.0,
        )

    refused = staged.executor.execute(
        source.accepted,
        source.original.mesh,
        staged.adaptation,
        decision=decision,
        reanalysis=reject,
        compiled_layout_id=staged.compiled.compilation_id,
        preparation=staged.preparation,
    )
    if (
        bool(refused.committed)
        or refused.state is not source.accepted
        or refused.mesh is not source.original.mesh
    ):
        raise RuntimeError(
            "metric_campaign.rollback must preserve the accepted epoch; got changed state after rejected reanalysis"
        )
    return refused


def metric_transfer_derivatives(
    transfer: Any, values: Any, target_size: int
) -> dict[str, float]:
    direction = jnp.linspace(-0.2, 0.3, values.size)
    dual = jnp.linspace(-0.4, 0.7, target_size)
    _, tangent = jax.jvp(transfer.apply, (values,), (direction,))
    _, pullback = jax.vjp(transfer.apply, values)
    adjoint = pullback(dual)[0]
    duality = float(jnp.abs(jnp.vdot(dual, tangent) - jnp.vdot(adjoint, direction)))
    if duality > 1.0e-10:
        raise RuntimeError(
            "metric_campaign.duality must preserve the JVP/VJP pairing; got excessive duality defect"
        )
    epsilon = 1.0e-5
    finite_difference = np.asarray(
        (
            transfer.apply(values + epsilon * direction)
            - transfer.apply(values - epsilon * direction)
        )
        / (2.0 * epsilon)
    )
    jvp_error = float(np.max(np.abs(np.asarray(tangent) - finite_difference)))
    reference = np.empty(values.size, dtype=np.float64)
    for index in range(values.size):
        perturbation = jnp.zeros_like(values).at[index].set(epsilon)
        difference = (
            transfer.apply(values + perturbation) - transfer.apply(values - perturbation)
        ) / (2.0 * epsilon)
        reference[index] = np.vdot(np.asarray(dual), np.asarray(difference))
    vjp_error = float(np.max(np.abs(np.asarray(adjoint) - reference)))
    if max(jvp_error, vjp_error) > 1.0e-7:
        raise RuntimeError(
            "metric_campaign.derivatives must agree with independent finite differences; got excessive derivative defect"
        )
    return {
        "transfer_duality_error": duality,
        "transfer_JVP_finite_difference_error": jvp_error,
        "transfer_VJP_finite_difference_error": vjp_error,
    }


def run_metric() -> dict[str, Any]:
    source = prepare_metric_source()
    staged = stage_metric_trial(source)
    decision = price_metric_trial(source, staged)
    result = publish_metric_trial(source, staged, decision)
    refused = refuse_metric_reanalysis(source, staged, decision)
    derivatives = metric_transfer_derivatives(
        result.transfers[0].transfer, source.values, result.state.fields[0].size
    )
    error, integral = error_and_integral(staged.space, result.state.fields[0], 7)
    if perf_counter() - source.started > source.budget.maximum_wall_seconds:
        raise RuntimeError(
            "metric_campaign.elapsed_seconds must fit the declared wall budget; got an exceeded deadline"
        )
    return {
        "baseline_L2_error": source.baseline.bound,
        "accepted_L2_error": error,
        "tolerance": source.budget.tolerance,
        "integral_error": abs(integral - source.integral),
        "physical_reanalysis_rollback": not bool(refused.committed),
        **derivatives,
    }
