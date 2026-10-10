#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host admission of finite, independently evaluated adaptation alternatives.

Estimator, native route, compiler, transfer, and reanalysis owners produce the
inputs. This owner neither estimates a PDE error nor predicts a measured cost.
A candidate denotes the complete prepared route to the requested tolerance,
not a promise that repeating a cheap local edit will eventually get there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import final, Literal, TYPE_CHECKING

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, finite_real_scalar, nonnegative_integer
from ..geometry.design._qualification import DesignQualificationEvidence
from ..typing import parse


if TYPE_CHECKING:
    from ..discretization.fem import FiniteElementDiscretization, FiniteElementFieldSpec
    from ..geometry import CompiledGeometry
    from ..lifecycle import CompositionRebind
    from ..solver import FiniteElementAcceptedState
    from ._adaptation import MeshAdaptationResult
    from ._contracts import SurfaceMeshingSpec
    from ._proposals import (
        LearnedMeshProposer,
        MeshProposalFeatures,
        MeshProposalTransaction,
    )
    from ._result import CellMeshingResult
    from .providers._native import NativeMeshingOptions, NativeMeshingPlan
    from .providers._native_sources import NativeImplicitSource


type ErrorQuantity = Literal["physical-error", "qoi"]


class AdaptationAction(StrEnum):
    H = "h"
    P = "p"
    METRIC = "metric"
    GEOMETRY_ORDER = "geometry-order"
    RELOCATION = "relocation"


def _nonnegative(value: float, name: str, /) -> float:
    result = finite_real_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


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
        raise ValueError(f"{name} must have unique scientific entry identities.")
    return result


@final
class PhysicalErrorEvidence(StrictModule, NonTrainableState):
    """Owner-produced objective assessment with independent physical contributions.

    Components use the units of the identified objective and account for
    pollution; signed QoI cancellation is not an error bound. The summed
    assessment does not promote residual indicators or quadrature diagnostics
    to rigorous bounds. Publication still requires independent physical
    reanalysis, and no estimator value establishes learned superiority.
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
        certificate = (
            None
            if qoi_certificate_id is None
            else canonical_identifier(qoi_certificate_id, "qoi_certificate_id")
        )
        if quantity_ == "qoi" and certificate is None:
            raise ValueError("QoI error requires its owning well-posedness certificate.")
        if quantity_ == "physical-error" and certificate is not None:
            raise ValueError(
                "A QoI certificate cannot identify a physical-error objective."
            )
        components = tuple(
            _nonnegative(value, name)
            for name, value in (
                ("field_error", field_error),
                ("geometry_error", geometry_error),
                ("algebraic_error", algebraic_error),
                ("transfer_error", transfer_error),
            )
        )
        if not isfinite(sum(components)):
            raise ValueError("The total physical-error bound must be finite.")
        self.revision_id, self.objective_id, self.estimator_id = (
            revision,
            objective,
            estimator,
        )
        self.quantity, self.qoi_certificate_id = quantity_, certificate
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
class DecisionBudget(StrictModule, NonTrainableState):
    tolerance: float = eqx.field(static=True)
    maximum_wall_seconds: float = eqx.field(static=True)
    maximum_memory_bytes: int = eqx.field(static=True)
    maximum_dofs: int = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    budget_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        tolerance: float,
        maximum_wall_seconds: float,
        maximum_memory_bytes: int,
        maximum_dofs: int,
        maximum_condition: float,
    ) -> None:
        tolerance_ = _nonnegative(tolerance, "tolerance")
        wall = _nonnegative(maximum_wall_seconds, "maximum_wall_seconds")
        memory = nonnegative_integer(maximum_memory_bytes, "maximum_memory_bytes")
        dofs = nonnegative_integer(maximum_dofs, "maximum_dofs")
        condition = _nonnegative(maximum_condition, "maximum_condition")
        if (
            tolerance_ <= 0.0
            or wall <= 0.0
            or memory == 0
            or dofs == 0
            or condition < 1.0
        ):
            raise ValueError(
                "Decision budgets must be positive and condition at least one."
            )
        self.tolerance, self.maximum_wall_seconds = tolerance_, wall
        self.maximum_memory_bytes, self.maximum_dofs, self.maximum_condition = (
            memory,
            dofs,
            condition,
        )
        self.budget_id = canonical_fingerprint(
            {
                "kind": "solver-aware-budget",
                "tolerance": tolerance_,
                "wall": wall,
                "memory": memory,
                "dofs": dofs,
                "condition": condition,
            }
        )


@final
class MeasuredAdaptationCost(StrictModule, NonTrainableState):
    """Observed complete-route cost, including compile/transfer/reanalysis.

    Zero compile time is valid only when the trial really reused a compiled
    layout. Peak bytes are measured for the trial, not an element-count guess.
    """

    sample_id: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)
    phase_seconds: tuple[float, ...] = eqx.field(static=True)
    peak_memory_bytes: int = eqx.field(static=True)
    cost_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_id: str,
        candidate_id: str,
        /,
        *,
        preparation_seconds: float,
        compilation_seconds: float,
        solve_seconds: float,
        transfer_seconds: float,
        reanalysis_seconds: float,
        decision_seconds: float,
        peak_memory_bytes: int,
    ) -> None:
        sample = canonical_identifier(sample_id, "sample_id")
        candidate = canonical_identifier(candidate_id, "candidate_id")
        phases = tuple(
            _nonnegative(value, name)
            for name, value in (
                ("preparation_seconds", preparation_seconds),
                ("compilation_seconds", compilation_seconds),
                ("solve_seconds", solve_seconds),
                ("transfer_seconds", transfer_seconds),
                ("reanalysis_seconds", reanalysis_seconds),
                ("decision_seconds", decision_seconds),
            )
        )
        memory = nonnegative_integer(peak_memory_bytes, "peak_memory_bytes")
        if not isfinite(sum(phases)) or sum(phases) <= 0.0 or memory == 0:
            raise ValueError(
                "A measured route requires positive total time and peak memory."
            )
        self.sample_id, self.candidate_id = sample, candidate
        self.phase_seconds, self.peak_memory_bytes = phases, memory
        self.cost_id = canonical_fingerprint(
            {
                "kind": "measured-adaptation-cost",
                "sample": sample,
                "candidate": candidate,
                "phases": phases,
                "peak_bytes": memory,
            }
        )

    @property
    def total_seconds(self) -> float:
        return sum(self.phase_seconds)


@final
class RouteFeasibility(StrictModule, NonTrainableState):
    """Exact prepared route/layout and every required state disposition.

    These are owner-produced identities, not learned capability flags. A failed
    native/compiler/transfer admission is represented by issues and is retained
    as a rejected alternative. State entries without transport/recompute proof
    cannot be selected, including materials, histories, RNG and solver artifacts.
    """

    source_revision_id: str = eqx.field(static=True)
    target_revision_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)
    cell_families: tuple[str, ...] = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    field_layouts: tuple[tuple[str, str], ...] = eqx.field(static=True)
    compiled_layout_id: str | None = eqx.field(static=True)
    geometry_certificate_id: str | None = eqx.field(static=True)
    topology_certificate_id: str | None = eqx.field(static=True)
    required_state_ids: tuple[str, ...] = eqx.field(static=True)
    state_dispositions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    dofs: int = eqx.field(static=True)
    condition_estimate: float = eqx.field(static=True)
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
        compiled_layout_id: str | None,
        geometry_certificate_id: str | None,
        topology_certificate_id: str | None,
        required_state_ids: Sequence[str],
        state_dispositions: Sequence[tuple[str, str]],
        dofs: int,
        condition_estimate: float,
        issues: Sequence[str] = (),
    ) -> None:
        source = canonical_identifier(source_revision_id, "source_revision_id")
        target = canonical_identifier(target_revision_id, "target_revision_id")
        route = canonical_identifier(route_id, "route_id")
        candidate = canonical_identifier(candidate_id, "candidate_id")
        families = tuple(
            sorted(
                {canonical_identifier(value, "cell_families") for value in cell_families}
            )
        )
        geometry = canonical_identifier(geometry_layout_id, "geometry_layout_id")
        layouts = _bindings(field_layouts, "field_layouts")
        required = tuple(
            sorted(
                canonical_identifier(value, "required_state_ids")
                for value in required_state_ids
            )
        )
        if not families or not layouts or len(set(required)) != len(required):
            raise ValueError(
                "Family/layout identities and unique state obligations are required."
            )
        dispositions = _bindings(state_dispositions, "state_dispositions")
        if not set(key for key, _ in dispositions).issubset(required):
            raise ValueError("State dispositions must name declared composition entries.")
        certificates = tuple(
            None if value is None else canonical_identifier(value, name)
            for name, value in (
                ("compiled_layout_id", compiled_layout_id),
                ("geometry_certificate_id", geometry_certificate_id),
                ("topology_certificate_id", topology_certificate_id),
            )
        )
        dofs_ = nonnegative_integer(dofs, "dofs")
        condition = _nonnegative(condition_estimate, "condition_estimate")
        if dofs_ == 0 or condition < 1.0:
            raise ValueError(
                "Route DOFs must be positive and measured condition at least one."
            )
        issues_ = set(canonical_identifier(value, "issues") for value in issues)
        for name, certificate in zip(
            (
                "compiled-layout-unavailable",
                "geometry-uncertified",
                "topology-uncertified",
            ),
            certificates,
            strict=True,
        ):
            if certificate is None:
                issues_.add(name)
        if set(required) != {key for key, _ in dispositions}:
            issues_.add("state-disposition-incomplete")
        self.source_revision_id, self.target_revision_id = source, target
        self.route_id, self.candidate_id = route, candidate
        self.cell_families, self.geometry_layout_id, self.field_layouts = (
            families,
            geometry,
            layouts,
        )
        (
            self.compiled_layout_id,
            self.geometry_certificate_id,
            self.topology_certificate_id,
        ) = certificates
        self.required_state_ids, self.state_dispositions = required, dispositions
        self.dofs, self.condition_estimate, self.issues = (
            dofs_,
            condition,
            tuple(sorted(issues_)),
        )
        self.feasibility_id = canonical_fingerprint(
            {
                "kind": "adaptation-route-feasibility",
                "source": source,
                "target": target,
                "route": route,
                "candidate": candidate,
                "families": families,
                "geometry": geometry,
                "fields": layouts,
                "certificates": certificates,
                "required": required,
                "dispositions": dispositions,
                "dofs": dofs_,
                "condition": condition,
                "issues": self.issues,
            }
        )

    @staticmethod
    def rebind_dispositions(rebind: CompositionRebind, /) -> tuple[tuple[str, str], ...]:
        """Content-address every actual disposition of the accepted composition."""
        from ..lifecycle import CompositionRebind

        if not isinstance(rebind, CompositionRebind):
            raise TypeError("Decision admission requires CompositionRebind.")
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
        return tuple(result)

    def require_rebind(self, rebind: CompositionRebind, /) -> None:
        if rebind.source.entry_ids != self.required_state_ids:
            raise ValueError("Decision omits or invents accepted composition entries.")
        if self.rebind_dispositions(rebind) != self.state_dispositions:
            raise ValueError(
                "Decision does not certify the actual staged state/artifact dispositions."
            )


@final
class SolverAwareCandidate(StrictModule, NonTrainableState):
    action: AdaptationAction = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)
    feasibility: RouteFeasibility
    error: PhysicalErrorEvidence
    cost: MeasuredAdaptationCost
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: AdaptationAction,
        candidate_id: str,
        feasibility: RouteFeasibility,
        error: PhysicalErrorEvidence,
        cost: MeasuredAdaptationCost,
        /,
    ) -> None:
        action_ = parse(action, AdaptationAction, "action")
        candidate = canonical_identifier(candidate_id, "candidate_id")
        if (
            not isinstance(feasibility, RouteFeasibility)
            or not isinstance(error, PhysicalErrorEvidence)
            or not isinstance(cost, MeasuredAdaptationCost)
        ):
            raise TypeError(
                "Candidate requires route, physical error, and measured cost evidence."
            )
        if (
            candidate != feasibility.candidate_id
            or candidate != cost.candidate_id
            or error.revision_id != feasibility.target_revision_id
        ):
            raise ValueError(
                "Candidate evidence must bind the identical prepared route and target."
            )
        self.action, self.candidate_id = action_, candidate
        self.feasibility, self.error, self.cost = feasibility, error, cost
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "solver-aware-candidate",
                "action": action_.value,
                "candidate": candidate,
                "feasibility": feasibility.feasibility_id,
                "error": error.evidence_id,
                "cost": cost.cost_id,
            }
        )


def _objective_matches(
    left: PhysicalErrorEvidence, right: PhysicalErrorEvidence, /
) -> bool:
    # A refined state space needs a new well-posedness certificate while retaining
    # the same physical QoI. Certificate identity is checked at its target epoch.
    return (left.objective_id, left.quantity) == (right.objective_id, right.quantity)


@final
class SolverAwareDecision(StrictModule, NonTrainableState):
    """Minimum observed cost to tolerance over admissible prepared alternatives.

    No selection means target failure, not a successful mesh-quality improvement.
    Trial costs include failed alternatives in the campaign budget; they cannot
    be hidden by selecting the nominally cheapest successful candidate.
    """

    source_revision_id: str = eqx.field(static=True)
    baseline: PhysicalErrorEvidence
    budget: DecisionBudget
    candidates: tuple[SolverAwareCandidate, ...]
    selected: SolverAwareCandidate | None
    dispositions: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True)
    observed_campaign_seconds: float = eqx.field(static=True)
    decision_id: str = eqx.field(static=True)

    def __init__(
        self,
        baseline: PhysicalErrorEvidence,
        budget: DecisionBudget,
        candidates: Sequence[SolverAwareCandidate],
        /,
        *,
        observed_campaign_seconds: float | None = None,
    ) -> None:
        if not isinstance(baseline, PhysicalErrorEvidence) or not isinstance(
            budget, DecisionBudget
        ):
            raise TypeError("Decision requires physical baseline and explicit budget.")
        alternatives = tuple(candidates)
        if any(
            not isinstance(candidate, SolverAwareCandidate) for candidate in alternatives
        ):
            raise TypeError("Decision alternatives must be SolverAwareCandidate values.")
        alternatives = tuple(
            sorted(alternatives, key=lambda candidate: candidate.candidate_id)
        )
        if len({candidate.candidate_id for candidate in alternatives}) != len(
            alternatives
        ):
            raise ValueError("Decision candidate identities must be unique.")
        dispositions: list[tuple[str, tuple[str, ...]]] = []
        admitted: list[SolverAwareCandidate] = []
        priced_seconds = sum(candidate.cost.total_seconds for candidate in alternatives)
        observed_seconds = (
            priced_seconds
            if observed_campaign_seconds is None
            else _nonnegative(observed_campaign_seconds, "observed_campaign_seconds")
        )
        if observed_seconds < priced_seconds:
            raise ValueError("Observed campaign time cannot omit priced candidate work.")
        campaign_overrun = observed_seconds > budget.maximum_wall_seconds
        for candidate in alternatives:
            feasibility, error, cost = (
                candidate.feasibility,
                candidate.error,
                candidate.cost,
            )
            issues = set(feasibility.issues)
            if feasibility.source_revision_id != baseline.revision_id:
                issues.add("stale-source")
            if not _objective_matches(baseline, error):
                issues.add("objective-mismatch")
            if error.bound > budget.tolerance:
                issues.add("physical-target-unmet")
            if error.bound >= baseline.bound:
                issues.add("physical-error-not-improved")
            if campaign_overrun:
                issues.add("campaign-wall-budget")
            if cost.peak_memory_bytes > budget.maximum_memory_bytes:
                issues.add("memory-budget")
            if feasibility.dofs > budget.maximum_dofs:
                issues.add("dof-budget")
            if feasibility.condition_estimate > budget.maximum_condition:
                issues.add("conditioning-budget")
            dispositions.append((candidate.candidate_id, tuple(sorted(issues))))
            if not issues:
                admitted.append(candidate)
        selected = (
            min(
                admitted,
                key=lambda candidate: (
                    candidate.cost.total_seconds,
                    candidate.cost.peak_memory_bytes,
                    candidate.candidate_id,
                ),
            )
            if admitted
            else None
        )
        self.source_revision_id, self.baseline, self.budget = (
            baseline.revision_id,
            baseline,
            budget,
        )
        self.candidates, self.selected, self.dispositions = (
            alternatives,
            selected,
            tuple(dispositions),
        )
        self.observed_campaign_seconds = observed_seconds
        self.decision_id = canonical_fingerprint(
            {
                "kind": "solver-aware-decision",
                "baseline": baseline.evidence_id,
                "budget": budget.budget_id,
                "candidates": [candidate.evidence_id for candidate in alternatives],
                "observed_campaign_seconds": observed_seconds,
                "selected": None if selected is None else selected.candidate_id,
                "dispositions": dispositions,
            }
        )

    def require_selected(
        self, source_revision_id: str, candidate_id: str, target_revision_id: str, /
    ) -> SolverAwareCandidate:
        selected = self.selected
        if selected is None:
            raise ValueError("No admissible route attains the declared physical target.")
        if (source_revision_id, candidate_id, target_revision_id) != (
            self.source_revision_id,
            selected.candidate_id,
            selected.feasibility.target_revision_id,
        ):
            raise ValueError(
                "Decision does not select this exact source, action, and target revision."
            )
        return selected

    def reanalysis_issues(self, actual: PhysicalErrorEvidence, /) -> tuple[str, ...]:
        if not isinstance(actual, PhysicalErrorEvidence):
            raise TypeError("Independent reanalysis must return PhysicalErrorEvidence.")
        selected = self.selected
        if selected is None:
            return ("decision-target-unmet",)
        issues: list[str] = []
        if (
            actual.revision_id != selected.feasibility.target_revision_id
            or not _objective_matches(self.baseline, actual)
            or actual.qoi_certificate_id != selected.error.qoi_certificate_id
        ):
            issues.append("reanalysis-identity-mismatch")
        if actual.estimator_id == selected.error.estimator_id:
            issues.append("independent-reanalysis-required")
        if actual.bound > self.budget.tolerance or actual.bound >= self.baseline.bound:
            issues.append("physical-reanalysis-rejected")
        return tuple(issues)

    def require_reanalysis(self, actual: PhysicalErrorEvidence, /) -> None:
        issues = self.reanalysis_issues(actual)
        if issues:
            raise ValueError("Decision reanalysis failed: " + "; ".join(issues))


@final
class FixedEpochDerivativeEvidence(StrictModule, NonTrainableState):
    """Composition of qualified geometry/PDE/transfer derivatives, never topology.

    Numerical JVP/VJP operators stay with their owners. This record gates their
    fixed-route composition and records the exact parameter/epoch identities.
    Acceptance, retopology, classification, Boolean and donor switches stop the
    route: consumers must prepare and qualify a new record after those events.
    """

    epoch_id: str = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)
    routes: tuple[tuple[str, str], ...] = eqx.field(static=True)
    qualifications: tuple[DesignQualificationEvidence, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        epoch_id: str,
        parameter_id: str,
        /,
        *,
        geometry: DesignQualificationEvidence,
        pde: DesignQualificationEvidence,
        transfer: DesignQualificationEvidence,
        routes: Sequence[tuple[str, str]],
    ) -> None:
        epoch = canonical_identifier(epoch_id, "epoch_id")
        parameter = canonical_identifier(parameter_id, "parameter_id")
        routes_ = _bindings(routes, "routes")
        qualifications = (geometry, pde, transfer)
        if any(
            not isinstance(value, DesignQualificationEvidence) for value in qualifications
        ):
            raise TypeError(
                "Fixed-route derivatives require their owner's qualification evidence."
            )
        route_table = dict(routes_)
        if set(route_table) != {"geometry", "pde", "transfer"}:
            raise ValueError(
                "Fixed-route evidence requires exactly geometry, PDE and transfer stages."
            )
        if any(
            qualification.numeric_revision_id != epoch for qualification in qualifications
        ):
            raise ValueError(
                "Fixed-route qualifications must bind the actual numeric epoch."
            )
        if any(
            route_table.get(stage) != qualification.execution_plan_id
            for stage, qualification in zip(
                ("geometry", "pde", "transfer"), qualifications, strict=True
            )
        ):
            raise ValueError(
                "Fixed-route stages must identify their exact qualified execution plans."
            )
        self.epoch_id, self.parameter_id, self.routes, self.qualifications = (
            epoch,
            parameter,
            routes_,
            qualifications,
        )
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "fixed-epoch-derivative-evidence",
                "epoch": epoch,
                "parameter": parameter,
                "routes": routes_,
                "qualifications": [value.evidence_id for value in qualifications],
            }
        )

    def invalidation_issues(
        self,
        epoch_id: str,
        parameter_id: str,
        /,
        *,
        routes: Sequence[tuple[str, str]],
        event_margins: Sequence[tuple[str, float]],
        topology_event_id: str | None = None,
        geometry_realization_id: str | None = None,
    ) -> tuple[str, ...]:
        issues: list[str] = []
        if epoch_id != self.epoch_id or parameter_id != self.parameter_id:
            issues.append("epoch-or-parameter-changed")
        if _bindings(routes, "routes") != self.routes:
            issues.append("fixed-route-changed")
        if topology_event_id is not None:
            canonical_identifier(topology_event_id, "topology_event_id")
            issues.append("stopped-topology-event")
        if geometry_realization_id is not None:
            canonical_identifier(geometry_realization_id, "geometry_realization_id")
            issues.append("stopped-geometry-realization-event")
        current = tuple(
            sorted(
                (canonical_identifier(name, "event_id"), float(value))
                for name, value in event_margins
            )
        )
        if len({name for name, _ in current}) != len(current):
            raise ValueError("Event margins must have unique event identities.")
        expected: dict[str, float] = {}
        for qualification in self.qualifications:
            if not qualification.valid or not qualification.tier.is_derivative:
                issues.append("owner-derivative-unqualified")
            if not qualification.event_ids or len(qualification.event_ids) != len(
                qualification.event_margins
            ):
                issues.append("owner-event-margins-missing")
            for name, margin in zip(
                qualification.event_ids, qualification.event_margins, strict=False
            ):
                expected[name] = min(expected.get(name, margin), margin)
        if {name for name, _ in current} != set(expected):
            issues.append("event-margin-coverage")
        if any(not isfinite(value) or value <= 0.0 for _, value in current) or any(
            value <= 0.0 for value in expected.values()
        ):
            issues.append("event-boundary")
        return tuple(sorted(set(issues)))

    def require_valid(
        self,
        epoch_id: str,
        parameter_id: str,
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
            raise ValueError("Fixed-epoch derivative invalidated: " + "; ".join(issues))


@dataclass(frozen=True, kw_only=True)
class NativeDesignSourceState:
    """Exact original design-source checkpoint, not a solver-template echo.

    The fixed implicit preparation, accepted fields and learned numerical model
    are retained as their actual owners. PDE executable caches and host recorder
    callbacks are deliberately absent; the cold consumer reprepares their owning
    FE problem from this source and field specification. Source1 has no authored
    constitutive history or carried RNG, so neither is fabricated here.
    """

    geometry: CompiledGeometry
    source: NativeImplicitSource
    specification: SurfaceMeshingSpec
    options: NativeMeshingOptions
    plan: NativeMeshingPlan
    initial: CellMeshingResult
    initial_state: FiniteElementAcceptedState
    source_fields: FiniteElementDiscretization
    accepted: CellMeshingResult
    accepted_state: FiniteElementAcceptedState
    target_fields: FiniteElementDiscretization
    field_specs: tuple[FiniteElementFieldSpec, ...]
    learned_proposer: LearnedMeshProposer
    learned_features: MeshProposalFeatures
    learned_transaction: MeshProposalTransaction
    decision: SolverAwareDecision
    adaptation: MeshAdaptationResult
    derivative: FixedEpochDerivativeEvidence
    physical: PhysicalErrorEvidence

    def __post_init__(self) -> None:
        if tuple(field.name for field in self.field_specs) != ("u", "density", "history"):
            raise ValueError(
                "Original design state requires its actual three declared fields."
            )
        for state, carrier, prepared in (
            (self.initial_state, self.initial, self.source_fields),
            (self.accepted_state, self.accepted, self.target_fields),
        ):
            if (
                state.topology_id != carrier.mesh.topology_id
                or state.prepared_id != prepared.prepared_id
                or len(state.fields) != len(prepared.field_spaces)
                or prepared.default_runtime.geometry_layout_id
                != carrier.geometry.geometry_layout_id
            ):
                raise ValueError(
                    "Archived accepted state differs from its actual source/field epoch."
                )
        if (
            self.source.source_revision != self.plan.source.source_revision
            or self.specification.specification_id
            != self.plan.specification.specification_id
            or self.options.options_id != self.plan.options.options_id
            or self.learned_features.source_result_id != self.initial.result_id
            or self.learned_transaction.source.result_id != self.initial.result_id
            or self.derivative.epoch_id != self.initial.result_id
        ):
            raise ValueError(
                "Original design checkpoint source, proposal or derivative ownership is stale."
            )
        self.decision.require_selected(
            self.initial.result_id,
            self.adaptation.result_id,
            self.accepted.result_id,
        )
        self.decision.require_reanalysis(self.physical)

    def validate_source_integrity(self) -> None:
        """Rebuild the fixed implicit preparation through its source owner."""
        from .._fingerprint import logical_array_value_collection_digest
        from .._model._structure import model_recipe_array_pairs
        from .providers._implicit import (
            prepare_adaptive_implicit_route,
            prepare_implicit_route,
        )

        self.__post_init__()
        renewed = self.learned_proposer.propose(self.initial, self.learned_features)
        recorded = self.learned_transaction.projection.proposal
        if (
            renewed.proposal_id != recorded.proposal_id
            or renewed.proposer_id != recorded.proposer_id
        ):
            raise ValueError(
                "Restored learned numerical model/features changed the original evaluated proposal."
            )
        policy = self.options.implicit_policy
        if policy is None:
            raise ValueError("Original implicit design source lost its owning policy.")
        from ..geometry.implicit import (
            AdaptiveImplicitSurfacePolicy,
            ImplicitSurfacePolicy,
        )

        if isinstance(policy, ImplicitSurfacePolicy):
            expected = prepare_implicit_route(self.source, self.specification, policy)
        elif isinstance(policy, AdaptiveImplicitSurfacePolicy):
            expected = prepare_adaptive_implicit_route(
                self.source, self.specification, policy
            )
        else:
            raise TypeError("Original implicit design policy changed type.")
        pairs = model_recipe_array_pairs(self.plan.prepared, expected)
        if pairs is None:
            raise ValueError(
                "Restored implicit preparation differs from its original source structure."
            )
        before = logical_array_value_collection_digest(
            {str(index): first for index, (first, _) in enumerate(pairs)},
        )
        after = logical_array_value_collection_digest(
            {str(index): second for index, (_, second) in enumerate(pairs)},
        )
        if before != after:
            raise ValueError(
                "Restored implicit preparation changed an original numerical source field."
            )


__all__ = [
    "AdaptationAction",
    "DecisionBudget",
    "ErrorQuantity",
    "FixedEpochDerivativeEvidence",
    "MeasuredAdaptationCost",
    "PhysicalErrorEvidence",
    "RouteFeasibility",
    "SolverAwareCandidate",
    "SolverAwareDecision",
    "NativeDesignSourceState",
]
