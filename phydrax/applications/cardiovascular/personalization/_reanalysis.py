#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._frozendict import frozendict
from ....operators.mechanics import HyperelasticResponse
from .._case import CardiovascularCaseManifest
from .._execution import CardiovascularExecutionManifest
from .._quantities import CardiovascularQuantitySpec
from ..circulation._network import ConsistentInitializationResult
from ..electrophysiology._aliev_panfilov import AlievPanfilovCandidate
from ..hemodynamics._fixed_wall_lbm import FixedWallLBMCandidate
from ._surrogates import CardiacSurrogateProposal, GenerativeGeometryCandidate


def _identifier(value: str, name: str, /) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty.")
    return resolved


def _positive(value: float, name: str, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


@dataclass(frozen=True, slots=True)
class ElectrophysiologyReanalysisRoute:
    """Full native reaction/conduction route; never a string fidelity mode."""

    solver_id: str
    discretization_id: str
    cellular_model_id: str
    time_step_ms: float
    residual_tolerance: float = 1.0e-7
    constraint_tolerance: float = 1.0e-7
    route_id: str = field(init=False)

    def __post_init__(self) -> None:
        _finish_route(
            self,
            "electrophysiology",
            ("solver_id", "discretization_id", "cellular_model_id"),
            {"time_step_ms": _positive(self.time_step_ms, "time_step_ms")},
        )


@dataclass(frozen=True, slots=True)
class MechanicsReanalysisRoute:
    """Full native finite-deformation mechanics route."""

    solver_id: str
    discretization_id: str
    constitutive_model_id: str
    residual_tolerance: float = 1.0e-7
    constraint_tolerance: float = 1.0e-7
    route_id: str = field(init=False)

    def __post_init__(self) -> None:
        _finish_route(
            self,
            "mechanics",
            ("solver_id", "discretization_id", "constitutive_model_id"),
            {},
        )


@dataclass(frozen=True, slots=True)
class CirculationReanalysisRoute:
    """Full native closed-loop circulation route."""

    solver_id: str
    network_id: str
    time_step_ms: float
    residual_tolerance: float = 1.0e-7
    constraint_tolerance: float = 1.0e-7
    route_id: str = field(init=False)

    def __post_init__(self) -> None:
        _finish_route(
            self,
            "circulation",
            ("solver_id", "network_id"),
            {"time_step_ms": _positive(self.time_step_ms, "time_step_ms")},
        )


@dataclass(frozen=True, slots=True)
class HemodynamicsReanalysisRoute:
    """Full native vascular hemodynamics/FSI route."""

    solver_id: str
    discretization_id: str
    coupling_id: str
    residual_tolerance: float = 1.0e-7
    constraint_tolerance: float = 1.0e-7
    route_id: str = field(init=False)

    def __post_init__(self) -> None:
        _finish_route(
            self,
            "hemodynamics",
            ("solver_id", "discretization_id", "coupling_id"),
            {},
        )


NativeReanalysisRoute = (
    ElectrophysiologyReanalysisRoute
    | MechanicsReanalysisRoute
    | CirculationReanalysisRoute
    | HemodynamicsReanalysisRoute
)


def _finish_route(
    route: NativeReanalysisRoute,
    domain: str,
    identity_fields: Sequence[str],
    numeric: Mapping[str, float],
    /,
) -> None:
    identities: dict[str, str] = {}
    for name in identity_fields:
        value = _identifier(object.__getattribute__(route, name), name)
        object.__setattr__(route, name, value)
        identities[name] = value
    residual = _positive(
        object.__getattribute__(route, "residual_tolerance"), "residual_tolerance"
    )
    constraint = _positive(
        object.__getattribute__(route, "constraint_tolerance"), "constraint_tolerance"
    )
    object.__setattr__(route, "residual_tolerance", residual)
    object.__setattr__(route, "constraint_tolerance", constraint)
    for name, value in numeric.items():
        object.__setattr__(route, name, value)
    object.__setattr__(
        route,
        "route_id",
        canonical_fingerprint(
            {
                "kind": "cardiovascular-full-native-reanalysis-route",
                "domain": domain,
                "identities": identities,
                "numeric": dict(numeric),
                "residual_tolerance": residual,
                "constraint_tolerance": constraint,
            }
        ),
    )


class NativeDomain(StrEnum):
    ELECTROPHYSIOLOGY = "electrophysiology"
    MECHANICS = "mechanics"
    CIRCULATION = "circulation"
    HEMODYNAMICS = "hemodynamics"


@dataclass(frozen=True, slots=True)
class FullNativeReanalysisPlan:
    """The four explicit fidelity routes required before acceptance."""

    electrophysiology: ElectrophysiologyReanalysisRoute
    mechanics: MechanicsReanalysisRoute
    circulation: CirculationReanalysisRoute
    hemodynamics: HemodynamicsReanalysisRoute
    output_quantities: tuple[CardiovascularQuantitySpec, ...]
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        expected_types = (
            ElectrophysiologyReanalysisRoute,
            MechanicsReanalysisRoute,
            CirculationReanalysisRoute,
            HemodynamicsReanalysisRoute,
        )
        routes = self.routes
        if any(
            not isinstance(route, expected)
            for route, expected in zip(routes, expected_types, strict=True)
        ):
            raise TypeError(
                "Full native reanalysis requires all four distinct route types."
            )
        quantities = tuple(self.output_quantities)
        if not quantities or any(
            not isinstance(quantity, CardiovascularQuantitySpec)
            for quantity in quantities
        ):
            raise TypeError(
                "output_quantities must contain cardiovascular quantity specs."
            )
        names = tuple(quantity.name for quantity in quantities)
        if len(set(names)) != len(names):
            raise ValueError("Full reanalysis output quantity names must be unique.")
        object.__setattr__(self, "output_quantities", quantities)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-full-native-reanalysis-plan",
                    "routes": [route.route_id for route in routes],
                    "output_quantities": [
                        quantity.quantity_id for quantity in quantities
                    ],
                }
            ),
        )

    @property
    def routes(self) -> tuple[NativeReanalysisRoute, ...]:
        return (
            self.electrophysiology,
            self.mechanics,
            self.circulation,
            self.hemodynamics,
        )


@dataclass(frozen=True, slots=True)
class FullNativeReanalysisRequest:
    """A qualified proposal recast solely as initialization for authoritative solves."""

    case_manifest: CardiovascularCaseManifest
    plan: FullNativeReanalysisPlan
    proposal_id: str
    parameters: tuple[tuple[str, float], ...]
    initial_guess: Array
    geometry_coordinates_mm: Array
    topology_id: str
    request_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.case_manifest, CardiovascularCaseManifest):
            raise TypeError("case_manifest must be a CardiovascularCaseManifest.")
        if not isinstance(self.plan, FullNativeReanalysisPlan):
            raise TypeError("plan must be FullNativeReanalysisPlan.")
        proposal = _identifier(self.proposal_id, "proposal_id")
        parameters = tuple(
            sorted((str(name).strip(), float(value)) for name, value in self.parameters)
        )
        names = tuple(name for name, _ in parameters)
        if (
            not parameters
            or any(not name for name in names)
            or len(set(names)) != len(names)
            or any(not isfinite(value) for _, value in parameters)
        ):
            raise ValueError(
                "Reanalysis parameters must be a finite unique named record."
            )
        initial = jnp.asarray(self.initial_guess, dtype=float)
        geometry = jnp.asarray(self.geometry_coordinates_mm, dtype=float)
        if initial.size == 0 or bool(jnp.any(~jnp.isfinite(initial))):
            raise ValueError("Reanalysis initial guesses must be non-empty and finite.")
        if (
            geometry.ndim != 2
            or geometry.shape[0] == 0
            or bool(jnp.any(~jnp.isfinite(geometry)))
        ):
            raise ValueError(
                "Reanalysis geometry must be a finite non-empty point array."
            )
        topology = _identifier(self.topology_id, "topology_id")
        request_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-full-native-reanalysis-request",
                "case_manifest": self.case_manifest.manifest_id,
                "plan": self.plan.plan_id,
                "proposal": proposal,
                "parameters": parameters,
                "initial_guess": array_tree_fingerprint(initial),
                "geometry": array_tree_fingerprint(geometry),
                "topology": topology,
            }
        )
        object.__setattr__(self, "proposal_id", proposal)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "initial_guess", initial)
        object.__setattr__(self, "geometry_coordinates_mm", geometry)
        object.__setattr__(self, "topology_id", topology)
        object.__setattr__(self, "request_id", request_id)

    @classmethod
    def from_proposal(
        cls,
        case_manifest: CardiovascularCaseManifest,
        plan: FullNativeReanalysisPlan,
        proposal: CardiacSurrogateProposal,
        parameters: Mapping[str, float] | Sequence[tuple[str, float]],
        geometry: GenerativeGeometryCandidate,
        /,
    ) -> FullNativeReanalysisRequest:
        if not isinstance(proposal, CardiacSurrogateProposal):
            raise TypeError("proposal must be CardiacSurrogateProposal.")
        if not proposal.qualified_for_reanalysis or proposal.predicted_state is None:
            raise ValueError(
                "Only a qualified proposal may initialize full native reanalysis."
            )
        if proposal.geometry_evidence is None or not proposal.geometry_evidence.qualified:
            raise ValueError("Reanalysis requires qualified generated geometry evidence.")
        if not isinstance(geometry, GenerativeGeometryCandidate):
            raise TypeError("geometry must be GenerativeGeometryCandidate.")
        if proposal.geometry_evidence.candidate_id != geometry.candidate_id:
            raise ValueError(
                "Geometry does not match the proposal's qualification evidence."
            )
        if proposal.topology_id != geometry.topology_id:
            raise ValueError("Proposal and generated geometry topology IDs must match.")
        items = parameters.items() if isinstance(parameters, Mapping) else parameters
        return cls(
            case_manifest,
            plan,
            proposal.proposal_id,
            tuple(items),
            proposal.predicted_state,
            geometry.coordinates_mm,
            geometry.topology_id,
        )


@dataclass(frozen=True, slots=True, init=False)
class NativeDomainSolveReceipt:
    """Exact successful domain result bound to its solver execution manifest."""

    domain: NativeDomain
    reanalysis_route_id: str
    execution_manifest: CardiovascularExecutionManifest
    domain_result: object
    domain_result_id: str
    output_artifact_id: str
    receipt_id: str

    def __init__(
        self,
        domain: NativeDomain,
        reanalysis_route_id: str,
        execution_manifest: CardiovascularExecutionManifest,
        domain_result: object,
        /,
    ):
        if not isinstance(domain, NativeDomain):
            raise TypeError("domain must be a NativeDomain.")
        route = _identifier(reanalysis_route_id, "reanalysis_route_id")
        if not isinstance(execution_manifest, CardiovascularExecutionManifest):
            raise TypeError("execution_manifest must be CardiovascularExecutionManifest.")
        expected_type = {
            NativeDomain.ELECTROPHYSIOLOGY: AlievPanfilovCandidate,
            NativeDomain.MECHANICS: HyperelasticResponse,
            NativeDomain.CIRCULATION: ConsistentInitializationResult,
            NativeDomain.HEMODYNAMICS: FixedWallLBMCandidate,
        }[domain]
        if not isinstance(domain_result, expected_type):
            raise TypeError(
                f"{domain.value} receipts require exact {expected_type.__name__} results."
            )
        successful = {
            NativeDomain.ELECTROPHYSIOLOGY: lambda value: value.evidence.successful,
            NativeDomain.MECHANICS: lambda value: value.admissible,
            NativeDomain.CIRCULATION: lambda value: value.evidence.successful,
            NativeDomain.HEMODYNAMICS: lambda value: value.evidence.successful,
        }[domain](domain_result)
        if not bool(jnp.all(jnp.asarray(successful))):
            raise ValueError("Native domain receipts require successful solver evidence.")
        domain_result_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-native-domain-result",
                "domain": domain.value,
                "result": array_tree_fingerprint(domain_result),
            }
        )
        output_artifact_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-native-domain-output-artifact",
                "domain": domain.value,
                "execution_manifest": execution_manifest.manifest_id,
                "domain_result": domain_result_id,
            }
        )
        receipt_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-native-domain-solve-receipt",
                "domain": domain.value,
                "reanalysis_route": route,
                "execution_manifest": execution_manifest.manifest_id,
                "domain_result": domain_result_id,
                "output_artifact": output_artifact_id,
            }
        )
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "reanalysis_route_id", route)
        object.__setattr__(self, "execution_manifest", execution_manifest)
        object.__setattr__(self, "domain_result", domain_result)
        object.__setattr__(self, "domain_result_id", domain_result_id)
        object.__setattr__(self, "output_artifact_id", output_artifact_id)
        object.__setattr__(self, "receipt_id", receipt_id)

    @classmethod
    def from_electrophysiology(
        cls,
        route_id: str,
        execution_manifest: CardiovascularExecutionManifest,
        result: AlievPanfilovCandidate,
        /,
    ) -> NativeDomainSolveReceipt:
        return cls(NativeDomain.ELECTROPHYSIOLOGY, route_id, execution_manifest, result)

    @classmethod
    def from_mechanics(
        cls,
        route_id: str,
        execution_manifest: CardiovascularExecutionManifest,
        result: HyperelasticResponse,
        /,
    ) -> NativeDomainSolveReceipt:
        return cls(NativeDomain.MECHANICS, route_id, execution_manifest, result)

    @classmethod
    def from_circulation(
        cls,
        route_id: str,
        execution_manifest: CardiovascularExecutionManifest,
        result: ConsistentInitializationResult,
        /,
    ) -> NativeDomainSolveReceipt:
        return cls(NativeDomain.CIRCULATION, route_id, execution_manifest, result)

    @classmethod
    def from_hemodynamics(
        cls,
        route_id: str,
        execution_manifest: CardiovascularExecutionManifest,
        result: FixedWallLBMCandidate,
        /,
    ) -> NativeDomainSolveReceipt:
        return cls(NativeDomain.HEMODYNAMICS, route_id, execution_manifest, result)


def _receipt_identities_valid(receipt: NativeDomainSolveReceipt, /) -> bool:
    domain_result_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-native-domain-result",
            "domain": receipt.domain.value,
            "result": array_tree_fingerprint(receipt.domain_result),
        }
    )
    output_artifact_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-native-domain-output-artifact",
            "domain": receipt.domain.value,
            "execution_manifest": receipt.execution_manifest.manifest_id,
            "domain_result": domain_result_id,
        }
    )
    receipt_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-native-domain-solve-receipt",
            "domain": receipt.domain.value,
            "reanalysis_route": receipt.reanalysis_route_id,
            "execution_manifest": receipt.execution_manifest.manifest_id,
            "domain_result": domain_result_id,
            "output_artifact": output_artifact_id,
        }
    )
    return (
        receipt.domain_result_id == domain_result_id
        and receipt.output_artifact_id == output_artifact_id
        and receipt.receipt_id == receipt_id
    )


@dataclass(frozen=True, slots=True, init=False)
class NativeReanalysisCandidate:
    """Native fields and solver-owned receipts before the authority gate."""

    fields: frozendict[str, Array]
    field_quantity_ids: frozendict[str, str]
    receipts: tuple[NativeDomainSolveReceipt, ...]
    topology_id: str
    initialization_proposal_id: str
    truth_artifact_id: str
    candidate_id: str

    def __init__(
        self,
        fields: Mapping[str, ArrayLike],
        field_quantity_ids: Mapping[str, str],
        receipts: Sequence[NativeDomainSolveReceipt],
        /,
        *,
        topology_id: str,
        initialization_proposal_id: str,
    ):
        arrays = frozendict(
            {str(name): jnp.asarray(value) for name, value in fields.items()}
        )
        quantities = frozendict(
            {
                str(name): _identifier(value, "field quantity ID")
                for name, value in field_quantity_ids.items()
            }
        )
        if not arrays or any(
            not name or value.size == 0 for name, value in arrays.items()
        ):
            raise ValueError("Native reanalysis fields must be named and non-empty.")
        if set(arrays) != set(quantities):
            raise ValueError("Every native field must bind exactly one quantity ID.")
        solved = tuple(receipts)
        if any(not isinstance(item, NativeDomainSolveReceipt) for item in solved):
            raise TypeError("receipts must contain NativeDomainSolveReceipt values.")
        topology = _identifier(topology_id, "topology_id")
        proposal = _identifier(initialization_proposal_id, "initialization_proposal_id")
        truth_artifact_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-full-native-truth-artifact",
                "fields": {
                    name: array_tree_fingerprint(value) for name, value in arrays.items()
                },
                "field_quantity_ids": dict(quantities),
                "receipts": [receipt.receipt_id for receipt in solved],
                "topology": topology,
            }
        )
        candidate_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-native-reanalysis-candidate",
                "truth_artifact": truth_artifact_id,
                "proposal": proposal,
            }
        )
        object.__setattr__(self, "fields", arrays)
        object.__setattr__(self, "field_quantity_ids", quantities)
        object.__setattr__(self, "receipts", solved)
        object.__setattr__(self, "topology_id", topology)
        object.__setattr__(self, "initialization_proposal_id", proposal)
        object.__setattr__(self, "truth_artifact_id", truth_artifact_id)
        object.__setattr__(self, "candidate_id", candidate_id)


class ReanalysisStatus(StrEnum):
    ACCEPTED = "accepted"
    INCOMPLETE_SOLVE = "incomplete_solve"
    ROUTE_MISMATCH = "route_mismatch"
    ARTIFACT_MISMATCH = "artifact_mismatch"
    CONVERGENCE_FAILURE = "convergence_failure"
    TOPOLOGY_MISMATCH = "topology_mismatch"
    QUANTITY_MISMATCH = "quantity_mismatch"
    NUMERICAL_FAILURE = "numerical_failure"


@dataclass(frozen=True, slots=True)
class FullNativeReanalysisResult:
    """Fail-closed result whose accepted values can only come from native solves."""

    status: ReanalysisStatus
    accepted_fields: frozendict[str, Array] | None
    candidate: NativeReanalysisCandidate
    request_id: str
    learned_proposal_used_as_initialization: bool
    final_native_reanalysis: bool
    result_id: str

    @property
    def accepted(self) -> bool:
        return self.status is ReanalysisStatus.ACCEPTED


def run_full_native_reanalysis(
    request: FullNativeReanalysisRequest,
    solver: Callable[[FullNativeReanalysisRequest], NativeReanalysisCandidate],
    /,
) -> FullNativeReanalysisResult:
    """Run all native routes and accept only complete, converged, exact-contract output."""

    if not isinstance(request, FullNativeReanalysisRequest):
        raise TypeError("request must be FullNativeReanalysisRequest.")
    if not callable(solver):
        raise TypeError("solver must implement the full native reanalysis call contract.")
    candidate = solver(request)
    if not isinstance(candidate, NativeReanalysisCandidate):
        raise TypeError("Full native solvers must return NativeReanalysisCandidate.")
    expected_domains = tuple(NativeDomain)
    expected_routes = request.plan.routes
    expected_names = tuple(quantity.name for quantity in request.plan.output_quantities)
    expected_quantity_ids = {
        quantity.name: quantity.quantity_id for quantity in request.plan.output_quantities
    }
    if candidate.topology_id != request.topology_id:
        status = ReanalysisStatus.TOPOLOGY_MISMATCH
    elif candidate.initialization_proposal_id != request.proposal_id:
        status = ReanalysisStatus.INCOMPLETE_SOLVE
    elif (
        len(candidate.receipts) != len(expected_routes)
        or tuple(receipt.domain for receipt in candidate.receipts) != expected_domains
    ):
        status = ReanalysisStatus.INCOMPLETE_SOLVE
    elif any(
        receipt.reanalysis_route_id != route.route_id
        or receipt.execution_manifest.case_manifest_id
        != request.case_manifest.manifest_id
        or receipt.execution_manifest.analysis_plan_id != request.plan.plan_id
        or receipt.execution_manifest.topology_id != request.topology_id
        or receipt.execution_manifest.solver_policy_id != route.solver_id
        for receipt, route in zip(candidate.receipts, expected_routes, strict=True)
    ):
        status = ReanalysisStatus.ROUTE_MISMATCH
    elif any(not _receipt_identities_valid(receipt) for receipt in candidate.receipts):
        status = ReanalysisStatus.ARTIFACT_MISMATCH
    elif (
        set(candidate.fields) != set(expected_names)
        or dict(candidate.field_quantity_ids) != expected_quantity_ids
    ):
        status = ReanalysisStatus.QUANTITY_MISMATCH
    elif any(bool(jnp.any(~jnp.isfinite(value))) for value in candidate.fields.values()):
        status = ReanalysisStatus.NUMERICAL_FAILURE
    else:
        status = ReanalysisStatus.ACCEPTED
    accepted_fields = candidate.fields if status is ReanalysisStatus.ACCEPTED else None
    result_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-full-native-reanalysis-result",
            "request": request.request_id,
            "candidate": candidate.candidate_id,
            "status": status.value,
            "accepted_truth_artifact": (
                candidate.truth_artifact_id
                if status is ReanalysisStatus.ACCEPTED
                else None
            ),
        }
    )
    return FullNativeReanalysisResult(
        status,
        accepted_fields,
        candidate,
        request.request_id,
        True,
        status is ReanalysisStatus.ACCEPTED,
        result_id,
    )


__all__ = [
    "CirculationReanalysisRoute",
    "ElectrophysiologyReanalysisRoute",
    "FullNativeReanalysisPlan",
    "FullNativeReanalysisRequest",
    "FullNativeReanalysisResult",
    "HemodynamicsReanalysisRoute",
    "MechanicsReanalysisRoute",
    "NativeDomain",
    "NativeDomainSolveReceipt",
    "NativeReanalysisCandidate",
    "NativeReanalysisRoute",
    "ReanalysisStatus",
    "run_full_native_reanalysis",
]
