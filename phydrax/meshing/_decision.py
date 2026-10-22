#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Admit independently evaluated adaptation alternatives on the host.

Estimator, native route, compiler, transfer, and reanalysis owners provide the
evidence for each prepared route to the requested tolerance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import final, Literal, TYPE_CHECKING

import equinox as eqx

import phydrax.geometry._contracts as contracts_owner

from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._meshcore import NativeExecutionScope
from .._model._structure import model_recipe_array_pairs
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import (
    canonical_identifier,
    nonnegative_integer,
    nonnegative_real_scalar,
)
from ..geometry.design._qualification import DesignQualificationEvidence
from ..geometry.implicit import AdaptiveImplicitSurfacePolicy, ImplicitSurfacePolicy
from ..lifecycle import CompositionRebind
from ..typing import checked, parse


if TYPE_CHECKING:
    from ..discretization.fem import FiniteElementDiscretization, FiniteElementFieldSpec
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
        raise ValueError(
            f"Decision.{name} must have unique scientific entry identities; got repeated entry keys."
        )
    return result


@final
class PhysicalErrorEvidence(StrictModule, NonTrainableState):
    """Record objective assessments with independent physical contributions.

    Components use the identified objective's units and account for pollution.
    The bound sums physical contributions without signed QoI cancellation.
    Publication requires independent physical reanalysis.
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
            raise ValueError(
                "PhysicalErrorEvidence.qoi_certificate_id must identify a well-posedness certificate for QoI error; got None."
            )
        if quantity_ == "physical-error" and certificate is not None:
            raise ValueError(
                "PhysicalErrorEvidence.qoi_certificate_id must be absent for a physical-error objective; got a QoI certificate."
            )
        components = tuple(
            nonnegative_real_scalar(value, name)
            for name, value in (
                ("field_error", field_error),
                ("geometry_error", geometry_error),
                ("algebraic_error", algebraic_error),
                ("transfer_error", transfer_error),
            )
        )
        if not isfinite(sum(components)):
            raise ValueError(
                "PhysicalErrorEvidence.components must have a finite total physical-error bound; got a nonfinite sum."
            )
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
        tolerance_ = nonnegative_real_scalar(tolerance, "tolerance")
        wall = nonnegative_real_scalar(maximum_wall_seconds, "maximum_wall_seconds")
        memory = nonnegative_integer(maximum_memory_bytes, "maximum_memory_bytes")
        dofs = nonnegative_integer(maximum_dofs, "maximum_dofs")
        condition = nonnegative_real_scalar(maximum_condition, "maximum_condition")
        if (
            tolerance_ <= 0.0
            or wall <= 0.0
            or memory == 0
            or dofs == 0
            or condition < 1.0
        ):
            raise ValueError(
                "DecisionBudget.limits must be positive and condition at least one; got an invalid tolerance, wall-time limit, or condition bound."
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
    """Record observed route costs, including compilation, transfer, and reanalysis.

    Zero compilation time denotes reuse of a compiled layout. Peak memory is
    observed process-lifetime RSS evidence; it certifies neither the trial's
    peak nor an upper bound. Exceeding a memory budget refuses the route; a lower
    observation does not certify that unobserved memory stayed within the budget.
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
            nonnegative_real_scalar(value, name)
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
                f"MeasuredAdaptationCost.phase_seconds must have positive total time and peak_memory_bytes must be positive; got phases={phases!r}, peak_memory_bytes={memory}."
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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[MeasuredAdaptationCost], value: MeasuredAdaptationCost, /
    ) -> MeasuredAdaptationCost:
        """Rebuild measured costs from the six retained lifecycle phases."""
        if len(value.phase_seconds) != 6:
            raise ValueError(
                f"MeasuredAdaptationCost.phase_seconds must contain the six-entry phase table; got count={len(value.phase_seconds)}."
            )
        return cls(
            value.sample_id,
            value.candidate_id,
            preparation_seconds=value.phase_seconds[0],
            compilation_seconds=value.phase_seconds[1],
            solve_seconds=value.phase_seconds[2],
            transfer_seconds=value.phase_seconds[3],
            reanalysis_seconds=value.phase_seconds[4],
            decision_seconds=value.phase_seconds[5],
            peak_memory_bytes=value.peak_memory_bytes,
        )

    @property
    def total_seconds(self) -> float:
        return sum(self.phase_seconds)


@final
class RouteFeasibility(StrictModule, NonTrainableState):
    """Record prepared route/layout identities and required state dispositions.

    Failed native/compiler/transfer admission is recorded as issues and retained
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
                "RouteFeasibility.cell_families must contain family identities, field_layouts must contain layout identities, and required_state_ids must contain unique state obligations; got empty identities or repeated state obligations."
            )
        dispositions = _bindings(state_dispositions, "state_dispositions")
        if not set(key for key, _ in dispositions).issubset(required):
            raise ValueError(
                "RouteFeasibility.state_dispositions must name declared composition entries; got undeclared entry identities."
            )
        certificates = tuple(
            None if value is None else canonical_identifier(value, name)
            for name, value in (
                ("compiled_layout_id", compiled_layout_id),
                ("geometry_certificate_id", geometry_certificate_id),
                ("topology_certificate_id", topology_certificate_id),
            )
        )
        dofs_ = nonnegative_integer(dofs, "dofs")
        condition = nonnegative_real_scalar(condition_estimate, "condition_estimate")
        if dofs_ == 0 or condition < 1.0:
            raise ValueError(
                f"RouteFeasibility.dofs must be positive and condition_estimate must be at least one; got dofs={dofs_}, condition_estimate={condition!r}."
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
    @checked
    def rebind_dispositions(rebind: CompositionRebind, /) -> tuple[tuple[str, str], ...]:
        """Content-address each disposition of the accepted composition."""

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

    @checked
    def require_rebind(self, rebind: CompositionRebind, /) -> None:
        if rebind.source.entry_ids != self.required_state_ids:
            raise ValueError(
                "RouteFeasibility.required_state_ids must match accepted composition entries; got omitted or additional entries."
            )
        if self.rebind_dispositions(rebind) != self.state_dispositions:
            raise ValueError(
                "Decision.state_dispositions must match staged state/artifact dispositions; got a different disposition."
            )


@final
class SolverAwareCandidate(StrictModule, NonTrainableState):
    action: AdaptationAction = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)
    feasibility: RouteFeasibility
    error: PhysicalErrorEvidence
    cost: MeasuredAdaptationCost
    evidence_id: str = eqx.field(static=True)

    @checked
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
            candidate != feasibility.candidate_id
            or candidate != cost.candidate_id
            or error.revision_id != feasibility.target_revision_id
        ):
            raise ValueError(
                "SolverAwareCandidate.evidence must bind the identical prepared route and target; got a differing candidate or target revision."
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
    """Select the lowest observed cost to tolerance among admissible prepared routes.

    An absent selection records failure to attain the physical target.
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

    @checked
    def __init__(
        self,
        baseline: PhysicalErrorEvidence,
        budget: DecisionBudget,
        candidates: Sequence[SolverAwareCandidate],
        /,
        *,
        observed_campaign_seconds: float | None = None,
    ) -> None:
        alternatives = tuple(candidates)
        if any(
            not isinstance(candidate, SolverAwareCandidate) for candidate in alternatives
        ):
            raise TypeError(
                "SolverAwareDecision.candidates must contain SolverAwareCandidate values; got a different candidate kind."
            )
        alternatives = tuple(
            sorted(alternatives, key=lambda candidate: candidate.candidate_id)
        )
        if len({candidate.candidate_id for candidate in alternatives}) != len(
            alternatives
        ):
            raise ValueError(
                "SolverAwareDecision.candidates.candidate_id must be unique; got repeated candidate identities."
            )
        dispositions: list[tuple[str, tuple[str, ...]]] = []
        admitted: list[SolverAwareCandidate] = []
        priced_seconds = sum(candidate.cost.total_seconds for candidate in alternatives)
        observed_seconds = (
            priced_seconds
            if observed_campaign_seconds is None
            else nonnegative_real_scalar(
                observed_campaign_seconds, "observed_campaign_seconds"
            )
        )
        if observed_seconds < priced_seconds:
            raise ValueError(
                f"SolverAwareDecision.observed_campaign_seconds must include priced candidate work; got observed={observed_seconds!r}, priced={priced_seconds!r}."
            )
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
            raise ValueError(
                "SolverAwareDecision.selected must identify an admissible route attaining the declared physical target; got no selected route."
            )
        if (source_revision_id, candidate_id, target_revision_id) != (
            self.source_revision_id,
            selected.candidate_id,
            selected.feasibility.target_revision_id,
        ):
            raise ValueError(
                "SolverAwareDecision.selected must match the requested source, action, and target revision; got a different selected route."
            )
        return selected

    @checked
    def reanalysis_issues(self, actual: PhysicalErrorEvidence, /) -> tuple[str, ...]:
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

    @checked
    def require_reanalysis(self, actual: PhysicalErrorEvidence, /) -> None:
        issues = self.reanalysis_issues(actual)
        if issues:
            raise ValueError(
                "SolverAwareDecision.reanalysis must satisfy identity, independence, and physical-error requirements; got "
                + "; ".join(issues)
            )


@final
class FixedEpochDerivativeEvidence(StrictModule, NonTrainableState):
    """Bind geometry, PDE and transfer derivative qualifications to a fixed epoch.

    Numerical JVP/VJP operators stay with their owners. This record gates their
    fixed-route composition and records the declared parameter/epoch identities.
    Acceptance, retopology, classification, Boolean and donor switches stop the
    route: consumers must prepare and qualify a new record after those events.
    """

    epoch_id: str = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)
    routes: tuple[tuple[str, str], ...] = eqx.field(static=True)
    qualifications: tuple[DesignQualificationEvidence, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
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
        route_table = dict(routes_)
        if set(route_table) != {"geometry", "pde", "transfer"}:
            raise ValueError(
                "FixedEpochDerivativeEvidence.routes must identify exactly geometry, PDE and transfer stages; got missing or additional stages."
            )
        if any(
            qualification.numeric_revision_id != epoch for qualification in qualifications
        ):
            raise ValueError(
                f"FixedEpochDerivativeEvidence.qualifications must bind the numeric epoch; got a qualification outside epoch={epoch!r}."
            )
        if any(
            route_table.get(stage) != qualification.execution_plan_id
            for stage, qualification in zip(
                ("geometry", "pde", "transfer"), qualifications, strict=True
            )
        ):
            raise ValueError(
                "FixedEpochDerivativeEvidence.routes must identify their qualified execution plans; got a stage with a different execution plan."
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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[FixedEpochDerivativeEvidence], value: FixedEpochDerivativeEvidence, /
    ) -> FixedEpochDerivativeEvidence:
        """Rebuild derivative evidence from geometry, PDE and transfer qualifications."""
        if len(value.qualifications) != 3:
            raise ValueError(
                f"FixedEpochDerivativeEvidence.qualifications must contain all three stage qualifications; got count={len(value.qualifications)}."
            )
        return cls(
            value.epoch_id,
            value.parameter_id,
            geometry=value.qualifications[0],
            pde=value.qualifications[1],
            transfer=value.qualifications[2],
            routes=value.routes,
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
            raise ValueError(
                "FixedEpochDerivativeEvidence.event_margins must have unique event identities; got repeated event identities."
            )
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
            raise ValueError(
                "FixedEpochDerivativeEvidence.binding must retain its qualified epoch, routes, and event margins; got "
                + "; ".join(issues)
            )


@dataclass(frozen=True, kw_only=True)
class NativeDesignSourceState:
    """Retain design sources, accepted fields and learned model state for cold replay.

    The consumer prepares its FE problem from the source and field specification.
    PDE executable caches and host recorder callbacks are excluded from the checkpoint.
    """

    geometry: contracts_owner.CompiledGeometry
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
                "NativeDesignSourceState.field_specs must declare u, density, and history in order; got a different field inventory."
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
                    "NativeDesignSourceState.accepted_state must match its source topology and field epoch; got a different topology, preparation, field count, or geometry layout."
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
                "NativeDesignSourceState.bindings must match its source, plan, proposal, and derivative epoch; got stale bindings."
            )
        self.decision.require_selected(
            self.initial.result_id,
            self.adaptation.result_id,
            self.accepted.result_id,
        )
        self.decision.require_reanalysis(self.physical)

    def validate_source_integrity(self) -> None:
        """Rebuild the fixed implicit preparation through its source owner."""
        # Decision/source integrity ↔ implicit publication: prepare only during the integrity check.
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
                "NativeDesignSourceState.learned_transaction must retain the evaluated proposal identity; got a different proposer or proposal identity."
            )
        policy = self.options.implicit_policy
        if policy is None:
            raise ValueError(
                "NativeDesignSourceState.options.implicit_policy must retain the source preparation policy; got None."
            )

        if not isinstance(policy, ImplicitSurfacePolicy | AdaptiveImplicitSurfacePolicy):
            raise TypeError(
                f"NativeDesignSourceState.options.implicit_policy must be ImplicitSurfacePolicy or AdaptiveImplicitSurfacePolicy; got {type(policy).__name__}."
            )
        with NativeExecutionScope(self.specification.limits.native_limits()) as scope:
            match policy:
                case ImplicitSurfacePolicy():
                    expected = prepare_implicit_route(
                        self.source, self.specification, policy, scope=scope
                    )
                case AdaptiveImplicitSurfacePolicy():
                    expected = prepare_adaptive_implicit_route(
                        self.source, self.specification, policy, scope=scope
                    )
        pairs = model_recipe_array_pairs(self.plan.prepared, expected)
        if pairs is None:
            raise ValueError(
                "NativeDesignSourceState.plan.prepared must retain the source structure; got a different restored preparation structure."
            )
        before = logical_array_value_collection_digest(
            {str(index): first for index, (first, _) in enumerate(pairs)},
        )
        after = logical_array_value_collection_digest(
            {str(index): second for index, (_, second) in enumerate(pairs)},
        )
        if before != after:
            raise ValueError(
                "NativeDesignSourceState.plan.prepared must retain the numerical source fields; got different restored array content."
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
