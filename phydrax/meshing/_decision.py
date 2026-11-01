#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from math import isfinite
from typing import final, Literal

import equinox as eqx
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, finite_real_scalar
from ..geometry.design._qualification import DesignQualificationEvidence
from ..geometry.design._schema import ParameterId
from ..lifecycle import CompositionRebind
from ..linalg import LinearSolveCheckEvidence
from ..typing import checked, parse


type ErrorQuantity = Literal["physical-error", "qoi"]


class AdaptationAction(StrEnum):
    H = "h"
    P = "p"
    METRIC = "metric"
    GEOMETRY_ORDER = "geometry-order"
    RELOCATION = "relocation"


def _bindings(
    values: Sequence[tuple[str, str]], name: str, /
) -> tuple[tuple[str, str], ...]:
    result = tuple(
        sorted(
            (canonical_identifier(key, name), canonical_identifier(value, name))
            for key, value in values
        )
    )
    if len({key for key, _ in result}) != len(result):
        raise ValueError(f"{name} must have unique identities.")
    return result


def _optional_identifier(value: str | None, name: str, /) -> str | None:
    return None if value is None else canonical_identifier(value, name)


def _error_matches(
    objective_id: str, quantity: ErrorQuantity, error: PhysicalErrorEvidence, /
) -> bool:
    return error.objective_id == objective_id and error.quantity == quantity


@final
class PhysicalErrorEvidence(StrictModule, NonTrainableState):
    """Nonnegative physical-error contributions bound to one analysis revision.

    Components are expressed in the identified quantity's units. Their sum is a
    conservative decision bound; signed QoI contributions cannot cancel.
    """

    revision_id: str = eqx.field(static=True)
    objective_id: str = eqx.field(static=True)
    estimator_id: str = eqx.field(static=True)
    quantity: ErrorQuantity = eqx.field(static=True)
    qoi_certificate_id: str | None = eqx.field(static=True)
    field_error: float = eqx.field(static=True)
    geometry_error: float = eqx.field(static=True)
    algebraic_error: float = eqx.field(static=True)
    transfer_error: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        revision_id: str,
        objective_id: str,
        estimator_id: str,
        /,
        *,
        field_error: float,
        geometry_error: float,
        algebraic_error: float,
        transfer_error: float,
        quantity: ErrorQuantity = "physical-error",
        qoi_certificate_id: str | None = None,
    ) -> None:
        revision = canonical_identifier(revision_id, "revision_id")
        objective = canonical_identifier(objective_id, "objective_id")
        estimator = canonical_identifier(estimator_id, "estimator_id")
        quantity_ = parse(quantity, ErrorQuantity, "quantity")
        certificate = _optional_identifier(qoi_certificate_id, "qoi_certificate_id")
        if quantity_ == "qoi" and certificate is None:
            raise ValueError("QoI error evidence requires a well-posedness certificate.")
        if quantity_ == "physical-error" and certificate is not None:
            raise ValueError("Physical-error evidence cannot carry a QoI certificate.")
        components = tuple(
            finite_real_scalar(value, name)
            for name, value in (
                ("field_error", field_error),
                ("geometry_error", geometry_error),
                ("algebraic_error", algebraic_error),
                ("transfer_error", transfer_error),
            )
        )
        if any(value < 0.0 for value in components):
            raise ValueError("Physical-error contributions must be nonnegative.")
        if not isfinite(sum(components)):
            raise ValueError("Physical-error contributions must have a finite sum.")
        self.revision_id = revision
        self.objective_id = objective
        self.estimator_id = estimator
        self.quantity = quantity_
        self.qoi_certificate_id = certificate
        (
            self.field_error,
            self.geometry_error,
            self.algebraic_error,
            self.transfer_error,
        ) = components
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "physical-error-evidence",
                "revision": revision,
                "objective": objective,
                "estimator": estimator,
                "quantity": quantity_,
                "qoi_certificate": certificate,
                "components": components,
            }
        )

    @property
    def bound(self) -> float:
        return (
            self.field_error
            + self.geometry_error
            + self.algebraic_error
            + self.transfer_error
        )


@final
class MeasuredAdaptationCost(StrictModule, NonTrainableState):
    """Measured candidate work; used only to rank feasible alternatives.

    Phase times include preparation, compilation, mesh/geometry execution, field
    transfer, and the candidate's physical reanalysis. No resource ceiling is
    inferred from these observations.
    """

    candidate_id: str = eqx.field(static=True)
    phase_seconds: tuple[float, ...] = eqx.field(static=True)
    cost_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidate_id: str,
        /,
        *,
        preparation_seconds: float,
        compilation_seconds: float,
        adaptation_seconds: float,
        transfer_seconds: float,
        reanalysis_seconds: float,
    ) -> None:
        candidate = canonical_identifier(candidate_id, "candidate_id")
        phases = tuple(
            finite_real_scalar(value, name)
            for name, value in (
                ("preparation_seconds", preparation_seconds),
                ("compilation_seconds", compilation_seconds),
                ("adaptation_seconds", adaptation_seconds),
                ("transfer_seconds", transfer_seconds),
                ("reanalysis_seconds", reanalysis_seconds),
            )
        )
        if any(value < 0.0 for value in phases):
            raise ValueError("Measured route phases must be nonnegative.")
        if not isfinite(sum(phases)) or sum(phases) <= 0.0:
            raise ValueError(
                "Measured route and reanalysis cost must be positive and finite."
            )
        self.candidate_id = candidate
        self.phase_seconds = phases
        self.cost_id = canonical_fingerprint(
            {
                "kind": "measured-mesh-adaptation-cost",
                "candidate": candidate,
                "phases": phases,
            }
        )

    @property
    def total_seconds(self) -> float:
        return sum(self.phase_seconds)


@final
class RouteFeasibility(StrictModule, NonTrainableState):
    """Identity-bound mesh, compiler, certificate, and state-transition evidence."""

    source_revision_id: str = eqx.field(static=True)
    target_revision_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)
    solver_operator_id: str = eqx.field(static=True)
    cell_families: tuple[str, ...] = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    field_layouts: tuple[tuple[str, str], ...] = eqx.field(static=True)
    transfer_plan_id: str | None = eqx.field(static=True)
    compiled_layout_id: str | None = eqx.field(static=True)
    geometry_certificate_id: str | None = eqx.field(static=True)
    topology_certificate_id: str | None = eqx.field(static=True)
    required_state_ids: tuple[str, ...] = eqx.field(static=True)
    state_dispositions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    issues: tuple[str, ...] = eqx.field(static=True)
    feasibility_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_revision_id: str,
        target_revision_id: str,
        route_id: str,
        candidate_id: str,
        /,
        *,
        cell_families: Sequence[str],
        geometry_layout_id: str,
        field_layouts: Sequence[tuple[str, str]],
        solver_operator_id: str,
        transfer_plan_id: str | None = None,
        compiled_layout_id: str | None,
        geometry_certificate_id: str | None,
        topology_certificate_id: str | None,
        required_state_ids: Sequence[str] = (),
        state_dispositions: Sequence[tuple[str, str]] = (),
        issues: Sequence[str] = (),
    ) -> None:
        source = canonical_identifier(source_revision_id, "source_revision_id")
        target = canonical_identifier(target_revision_id, "target_revision_id")
        route = canonical_identifier(route_id, "route_id")
        candidate = canonical_identifier(candidate_id, "candidate_id")
        solver_operator = canonical_identifier(solver_operator_id, "solver_operator_id")
        families = tuple(
            sorted(
                {canonical_identifier(value, "cell_families") for value in cell_families}
            )
        )
        geometry = canonical_identifier(geometry_layout_id, "geometry_layout_id")
        layouts = _bindings(field_layouts, "field_layouts")
        transfer = _optional_identifier(transfer_plan_id, "transfer_plan_id")
        required = tuple(
            sorted(
                canonical_identifier(value, "required_state_ids")
                for value in required_state_ids
            )
        )
        if not families or not layouts or len(set(required)) != len(required):
            raise ValueError(
                "Route feasibility requires cell families, field layouts, and unique state identities."
            )
        compiled = _optional_identifier(compiled_layout_id, "compiled_layout_id")
        geometry_certificate = _optional_identifier(
            geometry_certificate_id, "geometry_certificate_id"
        )
        topology_certificate = _optional_identifier(
            topology_certificate_id, "topology_certificate_id"
        )
        dispositions = _bindings(state_dispositions, "state_dispositions")
        disposition_ids = {key for key, _ in dispositions}
        if not disposition_ids.issubset(required):
            raise ValueError("State dispositions name an undeclared composition entry.")
        issue_set = {canonical_identifier(value, "issues") for value in issues}
        if compiled is None:
            issue_set.add("compiled-layout-unavailable")
        if geometry_certificate is None:
            issue_set.add("geometry-uncertified")
        if topology_certificate is None:
            issue_set.add("topology-uncertified")
        if required and transfer is None:
            issue_set.add("field-transfer-unavailable")
        if disposition_ids != set(required):
            issue_set.add("state-disposition-incomplete")
        issues_ = tuple(sorted(issue_set))
        self.source_revision_id = source
        self.target_revision_id = target
        self.route_id = route
        self.candidate_id = candidate
        self.solver_operator_id = solver_operator
        self.cell_families = families
        self.geometry_layout_id = geometry
        self.field_layouts = layouts
        self.transfer_plan_id = transfer
        self.compiled_layout_id = compiled
        self.geometry_certificate_id = geometry_certificate
        self.topology_certificate_id = topology_certificate
        self.required_state_ids = required
        self.state_dispositions = dispositions
        self.issues = issues_
        self.feasibility_id = canonical_fingerprint(
            {
                "kind": "mesh-adaptation-route-feasibility",
                "source": source,
                "target": target,
                "route": route,
                "candidate": candidate,
                "solver_operator": solver_operator,
                "families": families,
                "geometry_layout": geometry,
                "field_layouts": layouts,
                "transfer_plan": transfer,
                "compiled_layout": compiled,
                "geometry_certificate": geometry_certificate,
                "topology_certificate": topology_certificate,
                "required_states": required,
                "state_dispositions": dispositions,
                "issues": issues_,
            }
        )

    @staticmethod
    @checked
    def rebind_dispositions(rebind: CompositionRebind, /) -> tuple[tuple[str, str], ...]:
        """Content-address each disposition in the staged accepted composition."""
        transported = {
            entry_id: transport.transport_id
            for transport in rebind.transports
            for entry_id in transport.source_entry_ids
        }
        result: list[tuple[str, str]] = []
        for entry in rebind.source.entries:
            if entry.entry_id in rebind.invalidated:
                proof = canonical_fingerprint(
                    {"kind": "composition-invalidation", "entry": entry.record_id}
                )
            elif entry.entry_id in transported:
                proof = transported[entry.entry_id]
            else:
                proof = rebind.candidate.entry(entry.entry_id).record_id
            result.append((entry.entry_id, proof))
        return tuple(sorted(result))

    @checked
    def rebind_issues(self, rebind: CompositionRebind, /) -> tuple[str, ...]:
        issues: list[str] = []
        if rebind.source.entry_ids != self.required_state_ids:
            issues.append("state-disposition-source-mismatch")
        if self.rebind_dispositions(rebind) != self.state_dispositions:
            issues.append("state-disposition-mismatch")
        return tuple(issues)

    @checked
    def require_rebind(self, rebind: CompositionRebind, /) -> None:
        issues = self.rebind_issues(rebind)
        if issues:
            raise ValueError(
                "Route state dispositions must match the staged composition rebind: "
                + "; ".join(issues)
            )


@final
class MeshAdaptationCandidate(StrictModule, NonTrainableState):
    """One untrusted action proposal with route, solver, error, and cost evidence."""

    action: AdaptationAction = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)
    feasibility: RouteFeasibility
    error: PhysicalErrorEvidence
    cost: MeasuredAdaptationCost
    solve_check: LinearSolveCheckEvidence
    candidate_evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        action: AdaptationAction,
        candidate_id: str,
        feasibility: RouteFeasibility,
        error: PhysicalErrorEvidence,
        cost: MeasuredAdaptationCost,
        solve_check: LinearSolveCheckEvidence,
        /,
    ) -> None:
        action_ = parse(action, AdaptationAction, "action")
        candidate = canonical_identifier(candidate_id, "candidate_id")
        if (
            candidate != feasibility.candidate_id
            or candidate != cost.candidate_id
            or error.revision_id != feasibility.target_revision_id
        ):
            raise ValueError(
                "Candidate route, error, and measured cost must identify the same target."
            )
        self.action = action_
        self.candidate_id = candidate
        self.feasibility = feasibility
        self.error = error
        self.cost = cost
        self.solve_check = solve_check
        self.candidate_evidence_id = canonical_fingerprint(
            {
                "kind": "mesh-adaptation-candidate",
                "action": action_.value,
                "candidate": candidate,
                "feasibility": feasibility.feasibility_id,
                "error": error.evidence_id,
                "cost": cost.cost_id,
                "solver_operator": solve_check.operator_id,
                "solver_check": array_tree_fingerprint(
                    (
                        solve_check.status,
                        solve_check.true_residual_norm,
                        solve_check.residual_threshold,
                        solve_check.valid,
                    )
                ),
            }
        )

    @property
    def admissibility_issues(self) -> tuple[str, ...]:
        issues = set(self.feasibility.issues)
        if not bool(np.all(np.asarray(self.solve_check.valid))):
            issues.add("numerical-solve-unadmissible")
        if (
            self.solve_check.kind != "primal"
            or self.solve_check.operator_id != self.feasibility.solver_operator_id
        ):
            issues.add("solver-binding-mismatch")
        return tuple(sorted(issues))


@final
class MeshAdaptationObjective(StrictModule, NonTrainableState):
    """Scientific error/QoI target and deterministic ranking of feasible actions.

    The tolerance applies to a non-cancelling sum of physical contributions.
    Route feasibility and the solver's own numerical check are hard scientific
    gates; measured time only ranks alternatives that pass every gate.
    """

    objective_id: str = eqx.field(static=True)
    quantity: ErrorQuantity = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    objective_id_hash: str = eqx.field(static=True)

    def __init__(
        self,
        objective_id: str,
        tolerance: float,
        /,
        *,
        quantity: ErrorQuantity = "physical-error",
    ) -> None:
        objective = canonical_identifier(objective_id, "objective_id")
        quantity_ = parse(quantity, ErrorQuantity, "quantity")
        tolerance_ = finite_real_scalar(tolerance, "tolerance")
        if tolerance_ <= 0.0:
            raise ValueError("Mesh adaptation tolerance must be positive.")
        self.objective_id = objective
        self.quantity = quantity_
        self.tolerance = tolerance_
        self.objective_id_hash = canonical_fingerprint(
            {
                "kind": "mesh-adaptation-objective",
                "objective": objective,
                "quantity": quantity_,
                "tolerance": tolerance_,
            }
        )

    @checked
    def decide(
        self,
        baseline: PhysicalErrorEvidence,
        candidates: Sequence[MeshAdaptationCandidate],
        /,
    ) -> MeshAdaptationDecision:
        if not _error_matches(self.objective_id, self.quantity, baseline):
            raise ValueError("Baseline error evidence does not match the objective.")
        return MeshAdaptationDecision(self, baseline, candidates)


@final
class MeshAdaptationDecision(StrictModule, NonTrainableState):
    """Deterministic result of ranking admissible mesh/field-order alternatives."""

    objective: MeshAdaptationObjective
    baseline: PhysicalErrorEvidence
    candidates: tuple[MeshAdaptationCandidate, ...]
    selected: MeshAdaptationCandidate | None
    dispositions: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True)
    decision_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        objective: MeshAdaptationObjective,
        baseline: PhysicalErrorEvidence,
        candidates: Sequence[MeshAdaptationCandidate],
        /,
    ) -> None:
        if not _error_matches(objective.objective_id, objective.quantity, baseline):
            raise ValueError("Decision baseline does not match its objective.")
        candidates_ = tuple(candidates)
        if not all(isinstance(value, MeshAdaptationCandidate) for value in candidates_):
            raise TypeError("candidates must contain MeshAdaptationCandidate values.")
        candidates_ = tuple(sorted(candidates_, key=lambda value: value.candidate_id))
        candidate_ids = tuple(value.candidate_id for value in candidates_)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Decision candidates must have unique identities.")
        no_adaptation_needed = baseline.bound <= objective.tolerance
        dispositions: list[tuple[str, tuple[str, ...]]] = []
        feasible: list[MeshAdaptationCandidate] = []
        for candidate in candidates_:
            issues = set(candidate.admissibility_issues)
            if candidate.feasibility.source_revision_id != baseline.revision_id:
                issues.add("stale-source")
            if not _error_matches(
                objective.objective_id, objective.quantity, candidate.error
            ):
                issues.add("objective-mismatch")
            if candidate.error.bound > objective.tolerance:
                issues.add("physical-target-unmet")
            if candidate.error.bound >= baseline.bound:
                issues.add("physical-error-not-improved")
            if no_adaptation_needed:
                issues.add("baseline-already-meets-target")
            ordered = tuple(sorted(issues))
            dispositions.append((candidate.candidate_id, ordered))
            if not ordered:
                feasible.append(candidate)
        selected = (
            min(
                feasible, key=lambda value: (value.cost.total_seconds, value.candidate_id)
            )
            if feasible
            else None
        )
        dispositions_ = tuple(dispositions)
        selected_id = None if selected is None else selected.candidate_id
        self.objective = objective
        self.baseline = baseline
        self.candidates = candidates_
        self.selected = selected
        self.dispositions = dispositions_
        self.decision_id = canonical_fingerprint(
            {
                "kind": "mesh-adaptation-decision",
                "objective": objective.objective_id_hash,
                "baseline": baseline.evidence_id,
                "candidates": [value.candidate_evidence_id for value in candidates_],
                "selected": selected_id,
                "dispositions": dispositions_,
            }
        )

    @property
    def satisfied_without_adaptation(self) -> bool:
        return self.selected is None and self.baseline.bound <= self.objective.tolerance

    @checked
    def reanalysis_issues(
        self,
        actual: PhysicalErrorEvidence,
        solve_check: LinearSolveCheckEvidence,
        /,
        *,
        rebind: CompositionRebind | None = None,
    ) -> tuple[str, ...]:
        selected = self.selected
        if selected is None:
            return ("no-adaptation-selected",)
        issues: set[str] = set()
        feasibility = selected.feasibility
        if actual.revision_id != feasibility.target_revision_id:
            issues.add("reanalysis-target-mismatch")
        if not _error_matches(
            self.objective.objective_id, self.objective.quantity, actual
        ):
            issues.add("reanalysis-objective-mismatch")
        if actual.qoi_certificate_id != selected.error.qoi_certificate_id:
            issues.add("reanalysis-qoi-certificate-mismatch")
        if actual.estimator_id == selected.error.estimator_id:
            issues.add("independent-reanalysis-required")
        if actual.bound > self.objective.tolerance or actual.bound >= self.baseline.bound:
            issues.add("physical-reanalysis-rejected")
        if not bool(np.all(np.asarray(solve_check.valid))):
            issues.add("numerical-reanalysis-unadmissible")
        if (
            solve_check.kind != "primal"
            or solve_check.operator_id != feasibility.solver_operator_id
        ):
            issues.add("reanalysis-solver-binding-mismatch")
        if feasibility.required_state_ids and rebind is None:
            issues.add("composition-rebind-required")
        elif rebind is not None:
            issues.update(feasibility.rebind_issues(rebind))
        return tuple(sorted(issues))

    @checked
    def require_reanalysis(
        self,
        actual: PhysicalErrorEvidence,
        solve_check: LinearSolveCheckEvidence,
        /,
        *,
        rebind: CompositionRebind | None = None,
    ) -> MeshAdaptationCandidate:
        issues = self.reanalysis_issues(actual, solve_check, rebind=rebind)
        if issues:
            raise ValueError(
                "Selected adaptation requires independent, numerically admissible physical reanalysis: "
                + "; ".join(issues)
            )
        if self.selected is None:
            raise ValueError("No adaptation candidate was selected.")
        return self.selected


@final
class FixedEpochDerivativeEvidence(StrictModule, NonTrainableState):
    """Bind fixed-route geometry, PDE, and transfer derivatives to one epoch.

    This evidence rejects retopology, accepted-state, geometry-realization, and
    branch events. It gates owner-produced derivatives; it does not compute them.
    """

    epoch_id: str = eqx.field(static=True)
    parameter_id: ParameterId = eqx.field(static=True)
    routes: tuple[tuple[str, str], ...] = eqx.field(static=True)
    qualifications: tuple[DesignQualificationEvidence, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        epoch_id: str,
        parameter_id: ParameterId,
        /,
        *,
        geometry: DesignQualificationEvidence,
        pde: DesignQualificationEvidence,
        transfer: DesignQualificationEvidence,
        routes: Sequence[tuple[str, str]],
    ) -> None:
        epoch = canonical_identifier(epoch_id, "epoch_id")
        routes_ = _bindings(routes, "routes")
        route_table = dict(routes_)
        if set(route_table) != {"geometry", "pde", "transfer"}:
            raise ValueError(
                "Derivative routes must identify geometry, PDE, and transfer."
            )
        qualifications = (geometry, pde, transfer)
        if any(
            not isinstance(value, DesignQualificationEvidence) for value in qualifications
        ):
            raise TypeError("Each derivative stage requires DesignQualificationEvidence.")
        if any(value.numeric_revision_id != epoch for value in qualifications):
            raise ValueError(
                "Every derivative qualification must bind the same numeric epoch."
            )
        if any(
            not value.valid or not value.tier.is_derivative for value in qualifications
        ):
            raise ValueError(
                "Each fixed-route derivative stage must be valid and qualified."
            )
        if any(
            route_table[stage] != value.execution_plan_id
            for stage, value in zip(
                ("geometry", "pde", "transfer"), qualifications, strict=True
            )
        ):
            raise ValueError(
                "Derivative routes must match their qualified execution plans."
            )
        all_events = tuple(event for value in qualifications for event in value.event_ids)
        if not all_events:
            raise ValueError("Fixed-route derivatives require explicit event identities.")
        if any(
            not value.event_margins
            or len(value.event_ids) != len(value.event_margins)
            or any(margin <= 0.0 for margin in value.event_margins)
            for value in qualifications
        ):
            raise ValueError(
                "Fixed-route derivatives require positive margins from every event."
            )
        self.epoch_id = epoch
        self.parameter_id = parameter_id
        self.routes = routes_
        self.qualifications = qualifications
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "fixed-epoch-derivative-evidence",
                "epoch": epoch,
                "parameter": (parameter_id.feature_id, parameter_id.name),
                "routes": routes_,
                "qualifications": [value.evidence_id for value in qualifications],
            }
        )

    @checked
    def invalidation_issues(
        self,
        epoch_id: str,
        parameter_id: ParameterId,
        /,
        *,
        routes: Sequence[tuple[str, str]],
        event_margins: Sequence[tuple[str, float]],
        topology_event_id: str | None = None,
        geometry_realization_id: str | None = None,
    ) -> tuple[str, ...]:
        issues: set[str] = set()
        if epoch_id != self.epoch_id or parameter_id != self.parameter_id:
            issues.add("epoch-or-parameter-changed")
        if _bindings(routes, "routes") != self.routes:
            issues.add("fixed-route-changed")
        if topology_event_id is not None:
            canonical_identifier(topology_event_id, "topology_event_id")
            issues.add("stopped-topology-event")
        if geometry_realization_id is not None:
            canonical_identifier(geometry_realization_id, "geometry_realization_id")
            issues.add("stopped-geometry-realization-event")
        current: dict[str, float] = {}
        for name, value in event_margins:
            event = canonical_identifier(name, "event_margins")
            margin = float(value)
            if event in current:
                raise ValueError("Current event margins must have unique identities.")
            current[event] = margin
        expected = {
            name
            for qualification in self.qualifications
            for name in qualification.event_ids
        }
        if set(current) != expected:
            issues.add("event-margin-coverage")
        if any(not isfinite(value) or value <= 0.0 for value in current.values()):
            issues.add("event-boundary")
        if any(
            not qualification.valid or not qualification.tier.is_derivative
            for qualification in self.qualifications
        ):
            issues.add("owner-derivative-unqualified")
        return tuple(sorted(issues))

    @checked
    def require_valid(
        self,
        epoch_id: str,
        parameter_id: ParameterId,
        /,
        *,
        routes: Sequence[tuple[str, str]],
        event_margins: Sequence[tuple[str, float]],
        topology_event_id: str | None = None,
        geometry_realization_id: str | None = None,
    ) -> None:
        issues = self.invalidation_issues(
            epoch_id,
            parameter_id,
            routes=routes,
            event_margins=event_margins,
            topology_event_id=topology_event_id,
            geometry_realization_id=geometry_realization_id,
        )
        if issues:
            raise ValueError(
                "Fixed-epoch derivatives must remain on the qualified smooth branch: "
                + "; ".join(issues)
            )


__all__ = [
    "AdaptationAction",
    "ErrorQuantity",
    "FixedEpochDerivativeEvidence",
    "MeasuredAdaptationCost",
    "MeshAdaptationCandidate",
    "MeshAdaptationDecision",
    "MeshAdaptationObjective",
    "PhysicalErrorEvidence",
    "RouteFeasibility",
]
