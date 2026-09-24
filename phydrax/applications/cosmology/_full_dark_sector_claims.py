#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent scientific claims for the full relativistic dark-sector closure.

No profile inherits production status from another profile.  Every claim binds
one exact unit contract, local ADM/tetrad frame snapshot, differentiation gate,
source-rights request, checkpoint product, and analysis-output product set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import equinox as eqx

from ..._differentiation import (
    DerivativeContract,
    DerivativeRoute,
    DerivativeSurface,
    GradientLevel,
    SurfaceDerivative,
)
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...qualification import (
    PromotionState,
    QualificationRoleTrust,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificClaimProfile,
    ScientificMetricCriterion,
    SupportTuple,
)
from ...qualification._registry import SupportValue
from ..relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)


FullDarkSectorClaimName: TypeAlias = Literal[
    "relativistic-stress-energy-pm",
    "einstein-vlasov-z4c",
    "dynamic-epoch-runtime",
    "fixed-multiplicity",
    "parton-shower",
    "hadronization",
    "quantum-uu",
    "thermal-qft",
    "coherent-qke",
    "off-shell-kb",
    "radiation-packet",
    "radiation-m1",
    "radiation-vet",
    "radiation-hierarchy",
    "fully-coupled-closure",
]
FullDarkSectorQualificationLevel: TypeAlias = Literal[
    "experimental",
    "numerically-qualified",
    "scientifically-qualified",
    "production",
]


class FullDarkSectorReferenceUse(StrictModule, NonTrainableState):
    """Exact requested-use rights bound into every full-closure claim."""

    commercial_use: bool = eqx.field(static=True)
    redistribution: bool = eqx.field(static=True)
    training_use: bool = eqx.field(static=True)
    export: bool = eqx.field(static=True)
    use_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        values = (commercial_use, redistribution, training_use, export)
        if any(type(value) is not bool for value in values):
            raise TypeError("Full dark-sector requested-use values must be Boolean.")
        (
            self.commercial_use,
            self.redistribution,
            self.training_use,
            self.export,
        ) = values
        self.use_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-reference-use",
                "commercial_use": commercial_use,
                "redistribution": redistribution,
                "training_use": training_use,
                "export": export,
            }
        )

    def rights_arguments(self, /) -> dict[str, bool]:
        return {
            "commercial_use": self.commercial_use,
            "redistribution": self.redistribution,
            "training_use": self.training_use,
            "export": self.export,
        }


@dataclass(frozen=True, slots=True)
class _MetricRequirement:
    metric_id: str
    direction: Literal["at_most", "at_least"]
    unit_id: str
    aggregation: Literal["pooled", "independent_unit_macro", "worst_stratum"]


@dataclass(frozen=True, slots=True)
class _DifferentiationSpec:
    """Profile derivative gate: first-order surfaces, no higher-order claim."""

    label: str
    surfaces: tuple[DerivativeSurface, ...]

    def contract(self) -> DerivativeContract:
        return DerivativeContract(
            (
                SurfaceDerivative(surface, GradientLevel.SMOOTH)
                for surface in self.surfaces
            ),
            route=DerivativeRoute.DIRECT,
        )


@dataclass(frozen=True, slots=True)
class _ClaimDefinition:
    name: FullDarkSectorClaimName
    capability: str
    governing_equation: str
    state_family: str
    solver: str
    differentiation: _DifferentiationSpec
    observables: tuple[str, ...]
    metrics: tuple[_MetricRequirement, ...]
    refusals: tuple[str, ...]
    checkpoint_product: str
    output_products: tuple[str, ...]

    @property
    def promotion_channel(self) -> str:
        return f"full-dark-sector.production.{self.name}"


_CONTINUOUS_FIXED = _DifferentiationSpec(
    "continuous-fixed-topology",
    (
        DerivativeSurface.INPUT,
        DerivativeSurface.MODEL_PARAMETER,
        DerivativeSurface.PHYSICAL_PARAMETER,
        DerivativeSurface.STORED_VALUES,
    ),
)
_COORDINATE_ONLY = _DifferentiationSpec(
    "coordinate-only-external-closure-stop-gradient",
    (DerivativeSurface.INPUT,),
)
_NONDIFFERENTIABLE = _DifferentiationSpec("nondifferentiable-discrete-runtime", ())


def _metric(
    metric_id: str,
    direction: Literal["at_most", "at_least"] = "at_most",
    unit_id: str = "dimensionless",
    aggregation: Literal[
        "pooled", "independent_unit_macro", "worst_stratum"
    ] = "worst_stratum",
) -> _MetricRequirement:
    return _MetricRequirement(metric_id, direction, unit_id, aggregation)


_COMMON_METRICS = (
    _metric("unit-frame-roundtrip-relative-error"),
    _metric("total-four-momentum-ledger-relative-error"),
    _metric("checkpoint-restart-exact-match", "at_least", "fraction"),
    _metric("source-rights-admission-fraction", "at_least", "fraction"),
)
_COMMON_REFUSALS = (
    "changed-physics-unit-frame-or-support-identity",
    "missing-or-inadmissible-source-rights",
    "checkpoint-compatibility-mismatch",
    "fixed-capacity-exhaustion-without-durable-continuation",
    "failed-numerical-or-physical-status",
    "undeclared-differentiation-path",
)


def _definition(
    name: FullDarkSectorClaimName,
    capability: str,
    governing_equation: str,
    state_family: str,
    solver: str,
    differentiation: _DifferentiationSpec,
    observables: tuple[str, ...],
    metrics: tuple[_MetricRequirement, ...],
    refusals: tuple[str, ...],
    checkpoint_product: str,
    output_products: tuple[str, ...],
) -> _ClaimDefinition:
    return _ClaimDefinition(
        name,
        capability,
        governing_equation,
        state_family,
        solver,
        differentiation,
        observables,
        (*_COMMON_METRICS, *metrics),
        (*_COMMON_REFUSALS, *refusals),
        checkpoint_product,
        output_products,
    )


_DEFINITIONS: dict[FullDarkSectorClaimName, _ClaimDefinition] = {
    value.name: value
    for value in (
        _definition(
            "relativistic-stress-energy-pm",
            "cosmology.dark-sector.relativistic-stress-energy-pm",
            "relativistic-vlasov-weak-field-einstein-pm",
            "on-shell-particles-and-adm-stress-projection",
            "conservative-particle-mesh-stress-deposition",
            _CONTINUOUS_FIXED,
            (
                "stress-energy-density",
                "stress-energy-momentum",
                "anisotropic-stress",
                "weak-field-metric-potentials",
            ),
            (
                _metric("mass-shell-relative-error"),
                _metric("stress-energy-conservation-relative-error"),
                _metric("pm-force-resolution-error"),
                _metric("fixed-membership-jvp-relative-error"),
            ),
            (
                "off-shell-or-past-directed-particle",
                "stress-projection-snapshot-mismatch",
                "unresolved-particle-mesh-force-scale",
                "particle-membership-or-mesh-derivative-requested",
            ),
            "relativistic-stress-energy-pm-checkpoint",
            ("stress-energy-projection", "weak-field-pm-snapshot"),
        ),
        _definition(
            "einstein-vlasov-z4c",
            "cosmology.dark-sector.einstein-vlasov-z4c",
            "einstein-vlasov-z4c",
            "adm-z4c-geometry-and-relativistic-distribution",
            "geodesic-vlasov-z4c-amr",
            _CONTINUOUS_FIXED,
            (
                "spacetime-constraints",
                "distribution-moments",
                "metric-wave-content",
                "vlasov-characteristics",
            ),
            (
                _metric("hamiltonian-constraint-relative-residual"),
                _metric("momentum-constraint-relative-residual"),
                _metric("z4c-constraint-damping-error"),
                _metric("vlasov-characteristic-residual"),
                _metric("amr-stress-transfer-conservation-error"),
                _metric("frozen-amr-epoch-jvp-relative-error"),
            ),
            (
                "inadmissible-adm-or-tetrad-snapshot",
                "constraint-growth-outside-qualified-envelope",
                "distribution-stress-snapshot-mismatch",
                "derivative-through-regrid-refinement-or-particle-topology",
            ),
            "einstein-vlasov-z4c-checkpoint",
            (
                "z4c-geometry-snapshot",
                "vlasov-distribution-snapshot",
                "relativistic-amr-topology-evidence",
            ),
        ),
        _definition(
            "dynamic-epoch-runtime",
            "cosmology.dark-sector.dynamic-epoch-runtime",
            "durable-finite-epoch-event-graph",
            "fixed-capacity-resident-epoch-and-durable-global-frontier",
            "transactional-event-graph-epoch-coordinator",
            _NONDIFFERENTIABLE,
            (
                "epoch-throughput",
                "frontier-backpressure",
                "global-lineage",
                "transactional-rollback",
            ),
            (
                _metric("whole-epoch-rollback-error"),
                _metric("epoch-chain-continuity-fraction", "at_least", "fraction"),
                _metric("global-id-collision-count", unit_id="count"),
                _metric("undeclared-work-loss-count", unit_id="count"),
                _metric("exact-resume-frontier-mismatch-count", unit_id="count"),
            ),
            (
                "unbounded-device-resident-state-requested",
                "epoch-parent-or-generation-regression",
                "nontransactional-partial-epoch-commit",
                "event-work-or-allocation-derivative-requested",
            ),
            "dark-sector-epoch-graph-checkpoint",
            (
                "event-graph-epoch-manifest",
                "deferred-work-frontier",
                "epoch-conservation-evidence",
            ),
        ),
        _definition(
            "fixed-multiplicity",
            "cosmology.dark-sector.fixed-multiplicity",
            "relativistic-fixed-multiplicity-s-matrix",
            "typed-incoming-and-outgoing-particle-event",
            "bounded-helicity-amplitude-and-invariant-phase-space",
            _CONTINUOUS_FIXED,
            (
                "differential-cross-section",
                "event-weight",
                "helicity-amplitude",
                "hard-process-kinematics",
            ),
            (
                _metric("ward-identity-relative-residual"),
                _metric("phase-space-normalization-relative-error"),
                _metric("cross-section-reference-relative-error"),
                _metric("hard-event-four-momentum-relative-defect"),
                _metric("fixed-multiplicity-amplitude-jvp-relative-error"),
            ),
            (
                "undeclared-spin-color-polarization-or-identical-normalization",
                "unsupported-variable-multiplicity",
                "gauge-identity-failure",
                "multiplicity-or-channel-choice-derivative-requested",
            ),
            "fixed-multiplicity-hard-process-checkpoint",
            ("hard-particle-event", "matrix-element-evidence"),
        ),
        _definition(
            "parton-shower",
            "cosmology.dark-sector.parton-shower",
            "ordered-dark-visible-parton-branching",
            "bounded-active-shower-event-with-durable-continuation",
            "model-specific-dark-shower-or-pinned-standard-model-provider",
            _NONDIFFERENTIABLE,
            (
                "exclusive-shower-event",
                "inclusive-splitting-spectrum",
                "jet-substructure",
                "shower-lineage",
            ),
            (
                _metric("shower-unitarity-relative-error"),
                _metric("splitting-kernel-reference-relative-error"),
                _metric("shower-cutoff-stability-error"),
                _metric("shower-four-momentum-relative-defect"),
                _metric("shower-seed-replay-mismatch-count", unit_id="count"),
            ),
            (
                "unversioned-or-unpinned-standard-model-shower-provider",
                "universal-dark-shower-claim",
                "undeclared-shower-cutoff-or-ordering-variable",
                "branch-choice-or-event-topology-derivative-requested",
            ),
            "parton-shower-epoch-checkpoint",
            ("showered-particle-event", "shower-lineage-evidence"),
        ),
        _definition(
            "hadronization",
            "cosmology.dark-sector.hadronization",
            "model-specific-confining-fragmentation-and-decay",
            "bounded-hadron-event-with-durable-continuation",
            "dark-hadronization-model-or-pinned-standard-model-provider",
            _NONDIFFERENTIABLE,
            (
                "exclusive-hadron-event",
                "identified-hadron-spectrum",
                "stable-particle-yield",
                "fragmentation-lineage",
            ),
            (
                _metric("hadronization-four-momentum-relative-defect"),
                _metric("conserved-quantum-number-defect"),
                _metric("fragmentation-reference-relative-error"),
                _metric("decay-branching-normalization-error"),
                _metric("hadronization-seed-replay-mismatch-count", unit_id="count"),
            ),
            (
                "unversioned-or-unpinned-standard-model-hadronization-provider",
                "universal-dark-hadronization-claim",
                "undeclared-confinement-spectrum-or-decay-ownership",
                "fragmentation-decay-or-species-topology-derivative-requested",
            ),
            "hadronization-epoch-checkpoint",
            ("hadronized-particle-event", "fragmentation-lineage-evidence"),
        ),
        _definition(
            "quantum-uu",
            "cosmology.dark-sector.quantum-uu",
            "quantum-uehling-uhlenbeck",
            "species-distribution-with-pauli-bose-occupancy",
            "conservative-quantum-collision-quadrature",
            _CONTINUOUS_FIXED,
            (
                "quantum-distribution",
                "collision-invariants",
                "entropy-production",
                "equilibrium-relaxation",
            ),
            (
                _metric("occupancy-bound-violation"),
                _metric("quantum-collision-invariant-relative-defect"),
                _metric("negative-entropy-production-magnitude"),
                _metric("quantum-equilibrium-relative-error"),
                _metric("fixed-quadrature-jvp-relative-error"),
            ),
            (
                "statistics-species-or-degeneracy-mismatch",
                "pauli-or-bose-occupancy-domain-violation",
                "unresolved-quantum-collision-quadrature",
                "channel-or-quadrature-topology-derivative-requested",
            ),
            "quantum-uu-checkpoint",
            ("quantum-distribution-snapshot", "quantum-collision-evidence"),
        ),
        _definition(
            "thermal-qft",
            "cosmology.dark-sector.thermal-qft",
            "finite-temperature-real-time-qft",
            "thermal-correlators-and-renormalized-self-energies",
            "kms-consistent-renormalized-thermal-integrals",
            _CONTINUOUS_FIXED,
            (
                "thermal-spectral-density",
                "thermal-rate",
                "equation-of-state",
                "renormalized-self-energy",
            ),
            (
                _metric("kms-relative-residual"),
                _metric("thermal-spectral-sum-rule-relative-error"),
                _metric("renormalization-scale-stability-error"),
                _metric("thermodynamic-identity-relative-error"),
                _metric("thermal-integral-jvp-relative-error"),
            ),
            (
                "undeclared-renormalization-scheme-or-scale",
                "kms-or-spectral-positivity-failure",
                "thermal-support-or-phase-regime-mismatch",
                "contour-channel-or-quadrature-topology-derivative-requested",
            ),
            "thermal-qft-checkpoint",
            ("thermal-qft-observables", "thermal-renormalization-evidence"),
        ),
        _definition(
            "coherent-qke",
            "cosmology.dark-sector.coherent-qke",
            "matrix-valued-coherent-quantum-kinetics",
            "hermitian-positive-species-density-matrices",
            "commutator-oscillation-and-collision-split",
            _CONTINUOUS_FIXED,
            (
                "density-matrix",
                "flavor-coherence",
                "conversion-probability",
                "coherent-collision-moments",
            ),
            (
                _metric("density-matrix-hermiticity-error"),
                _metric("minimum-density-matrix-eigenvalue", "at_least"),
                _metric("density-matrix-trace-relative-defect"),
                _metric("coherent-reference-relative-error"),
                _metric("fixed-basis-qke-jvp-relative-error"),
            ),
            (
                "coherence-basis-or-phase-convention-mismatch",
                "density-matrix-positivity-failure",
                "incoherent-boltzmann-substitution",
                "basis-channel-or-rank-topology-derivative-requested",
            ),
            "coherent-qke-checkpoint",
            ("coherent-density-matrix-snapshot", "qke-coherence-evidence"),
        ),
        _definition(
            "off-shell-kb",
            "cosmology.dark-sector.off-shell-kb",
            "two-time-kadanoff-baym",
            "spectral-and-statistical-green-functions-with-memory",
            "causal-conserving-self-energy-memory-integrator",
            _CONTINUOUS_FIXED,
            (
                "spectral-function",
                "statistical-propagator",
                "off-shell-rate",
                "memory-kernel",
            ),
            (
                _metric("kb-spectral-sum-rule-relative-error"),
                _metric("kb-causality-violation"),
                _metric("kb-conservation-relative-defect"),
                _metric("memory-window-truncation-error"),
                _metric("fixed-memory-kb-jvp-relative-error"),
            ),
            (
                "quasiparticle-on-shell-substitution",
                "noncausal-self-energy-or-memory-kernel",
                "unqualified-memory-window-truncation",
                "memory-window-grid-or-self-energy-topology-derivative-requested",
            ),
            "off-shell-kb-memory-checkpoint",
            ("kb-two-time-snapshot", "off-shell-spectral-evidence"),
        ),
        _definition(
            "radiation-packet",
            "cosmology.dark-sector.radiation-packet",
            "relativistic-monte-carlo-radiative-transfer",
            "weighted-radiation-packets-with-lineage",
            "bounded-packet-transport-and-interaction-epochs",
            _NONDIFFERENTIABLE,
            (
                "radiation-spectrum",
                "angular-intensity",
                "matter-radiation-exchange",
                "packet-lineage",
            ),
            (
                _metric("packet-moment-reference-relative-error"),
                _metric("packet-four-momentum-relative-defect"),
                _metric("packet-opacity-reference-relative-error"),
                _metric("packet-seed-replay-mismatch-count", unit_id="count"),
                _metric("packet-allocation-rollback-error"),
            ),
            (
                "unqualified-opacity-emissivity-or-scattering-source",
                "packet-lineage-or-weight-loss",
                "unbounded-device-packet-allocation",
                "interaction-choice-packet-allocation-or-seed-derivative-requested",
            ),
            "radiation-packet-epoch-checkpoint",
            ("radiation-packet-snapshot", "packet-interaction-evidence"),
        ),
        _definition(
            "radiation-m1",
            "cosmology.dark-sector.radiation-m1",
            "relativistic-two-moment-m1-radiation",
            "radiation-energy-and-flux-moments",
            "causal-hyperbolic-m1-with-source-coupling",
            _CONTINUOUS_FIXED,
            (
                "radiation-energy",
                "radiation-flux",
                "matter-radiation-exchange",
                "m1-closure",
            ),
            (
                _metric("m1-causality-violation"),
                _metric("m1-closure-reference-relative-error"),
                _metric("m1-conservation-relative-defect"),
                _metric("m1-source-coupling-relative-residual"),
                _metric("fixed-grid-m1-jvp-relative-error"),
            ),
            (
                "crossing-beam-regime-outside-m1-support",
                "superluminal-radiation-flux",
                "unqualified-stiff-source-integration",
                "mesh-regime-or-closure-choice-derivative-requested",
            ),
            "radiation-m1-checkpoint",
            ("radiation-m1-snapshot", "m1-closure-evidence"),
        ),
        _definition(
            "radiation-vet",
            "cosmology.dark-sector.radiation-vet",
            "relativistic-variable-eddington-tensor-radiation",
            "radiation-moments-and-eddington-tensor",
            "formal-angle-solve-coupled-to-moment-transport",
            _COORDINATE_ONLY,
            (
                "radiation-energy",
                "radiation-flux",
                "eddington-tensor",
                "angular-closure",
            ),
            (
                _metric("eddington-tensor-symmetry-error"),
                _metric("eddington-eigenvalue-bound-violation"),
                _metric("formal-angle-solver-relative-residual"),
                _metric("vet-source-staleness-error"),
                _metric("coordinate-only-vet-jvp-relative-error"),
            ),
            (
                "unversioned-or-unpinned-formal-solver-source",
                "nonrealizable-eddington-tensor",
                "stale-closure-outside-qualified-cadence",
                "derivative-through-external-formal-solver-or-angle-topology",
            ),
            "radiation-vet-checkpoint",
            ("radiation-vet-snapshot", "eddington-tensor-evidence"),
        ),
        _definition(
            "radiation-hierarchy",
            "cosmology.dark-sector.radiation-hierarchy",
            "relativistic-angular-moment-hierarchy",
            "fixed-order-radiation-multipoles",
            "collision-coupled-truncated-multipole-evolution",
            _CONTINUOUS_FIXED,
            (
                "radiation-multipoles",
                "anisotropic-stress",
                "angular-power",
                "hierarchy-tail",
            ),
            (
                _metric("multipole-tail-truncation-error"),
                _metric("hierarchy-collision-moment-relative-error"),
                _metric("hierarchy-conservation-relative-defect"),
                _metric("multipole-order-convergence-error"),
                _metric("fixed-order-hierarchy-jvp-relative-error"),
            ),
            (
                "unresolved-multipole-tail",
                "undeclared-hierarchy-closure",
                "collision-angular-support-mismatch",
                "multipole-order-or-closure-topology-derivative-requested",
            ),
            "radiation-hierarchy-checkpoint",
            ("radiation-hierarchy-snapshot", "hierarchy-truncation-evidence"),
        ),
        _definition(
            "fully-coupled-closure",
            "cosmology.dark-sector.fully-coupled-closure",
            "einstein-vlasov-quantum-field-radiation-event-closure",
            "composed-typed-dark-sector-subsystem-states",
            "transactional-multirate-full-dark-sector-coordinator",
            _NONDIFFERENTIABLE,
            (
                "spacetime-and-matter-observables",
                "dark-visible-event-yields",
                "quantum-thermal-coherence",
                "radiation-observables",
                "inference-ready-likelihood-products",
            ),
            (
                _metric("coupling-interface-relative-defect"),
                _metric("coupled-total-energy-relative-defect"),
                _metric("multirate-split-convergence-error"),
                _metric("whole-system-rollback-error"),
                _metric("cross-profile-restart-mismatch-count", unit_id="count"),
                _metric("locked-coupled-observable-relative-error"),
            ),
            (
                "missing-required-subprofile-production-promotion",
                "cross-profile-unit-frame-species-or-epoch-mismatch",
                "partial-subsystem-commit-or-restart",
                "derivative-through-discrete-event-regrid-allocation-or-provider",
            ),
            "fully-coupled-dark-sector-checkpoint",
            (
                "fully-coupled-dark-sector-snapshot",
                "coupling-conservation-evidence",
                "inference-observable-bundle",
            ),
        ),
    )
}

SUPPORTED_FULL_DARK_SECTOR_CLAIM_PROFILES = tuple(_DEFINITIONS)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def full_dark_sector_claim_metric_ids(
    profile_name: FullDarkSectorClaimName,
    /,
) -> tuple[str, ...]:
    """Return the complete frozen metric vocabulary for one independent profile."""
    if profile_name not in _DEFINITIONS:
        raise ValueError("Unknown full dark-sector claim profile.")
    return tuple(value.metric_id for value in _DEFINITIONS[profile_name].metrics)


def full_dark_sector_claim_criteria(
    profile_name: FullDarkSectorClaimName,
    thresholds: Mapping[str, float],
    /,
) -> tuple[ScientificMetricCriterion, ...]:
    """Build the exact criteria set from campaign-frozen threshold values."""
    if profile_name not in _DEFINITIONS:
        raise ValueError("Unknown full dark-sector claim profile.")
    if not isinstance(thresholds, Mapping):
        raise TypeError("thresholds must map metric IDs to frozen numeric bounds.")
    requirements = _DEFINITIONS[profile_name].metrics
    expected = {value.metric_id for value in requirements}
    if set(thresholds) != expected:
        missing = sorted(expected - set(thresholds))
        extra = sorted(set(thresholds) - expected)
        raise ValueError(
            f"Full dark-sector thresholds changed; missing={missing}, extra={extra}."
        )
    criteria: list[ScientificMetricCriterion] = []
    for requirement in requirements:
        threshold = float(thresholds[requirement.metric_id])
        lower = threshold if requirement.direction == "at_least" else None
        upper = threshold if requirement.direction == "at_most" else None
        criteria.append(
            ScientificMetricCriterion(
                requirement.metric_id,
                requirement.direction,
                lower,
                upper,
                requirement.unit_id,
                requirement.aggregation,
            )
        )
    return tuple(criteria)


def full_dark_sector_differentiation_contract(
    profile_name: FullDarkSectorClaimName,
    /,
) -> DerivativeContract:
    """Return the exact derivative gate for one profile."""
    if profile_name not in _DEFINITIONS:
        raise ValueError("Unknown full dark-sector claim profile.")
    return _DEFINITIONS[profile_name].differentiation.contract()


def _validate_criteria(
    definition: _ClaimDefinition,
    criteria: Sequence[ScientificMetricCriterion],
    campaign: ScientificCampaign,
    /,
) -> tuple[ScientificMetricCriterion, ...]:
    if not isinstance(campaign, ScientificCampaign):
        raise TypeError("campaign must be a ScientificCampaign.")
    if (
        not isinstance(criteria, Sequence)
        or isinstance(criteria, str)
        or any(not isinstance(value, ScientificMetricCriterion) for value in criteria)
    ):
        raise TypeError("criteria must contain ScientificMetricCriterion values.")
    values = tuple(criteria)
    expected = {value.metric_id: value for value in definition.metrics}
    by_name = {value.metric_id: value for value in values}
    if len(by_name) != len(values) or set(by_name) != set(expected):
        raise ValueError("Claim criteria do not exactly match the selected profile.")
    for name, requirement in expected.items():
        value = by_name[name]
        if (
            value.direction != requirement.direction
            or value.unit_id != requirement.unit_id
            or value.aggregation != requirement.aggregation
        ):
            raise ValueError(f"Criterion {name!r} changed its governed metadata.")
    if any(value.criterion_id not in campaign.criteria_ids for value in values):
        raise ValueError("Every criterion must be frozen by the scientific campaign.")
    return values


def _reference_binding(
    reference_artifacts: Sequence[ReferenceArtifactManifest],
    requested_use: FullDarkSectorReferenceUse,
    /,
) -> tuple[str, ...]:
    if (
        not isinstance(reference_artifacts, Sequence)
        or isinstance(reference_artifacts, str)
        or not reference_artifacts
        or any(
            not isinstance(value, ReferenceArtifactManifest)
            for value in reference_artifacts
        )
    ):
        raise TypeError(
            "reference_artifacts must contain governed ReferenceArtifactManifest values."
        )
    if not isinstance(requested_use, FullDarkSectorReferenceUse):
        raise TypeError("requested_use must be FullDarkSectorReferenceUse.")
    identifiers = tuple(
        sorted(
            artifact.require_rights(**requested_use.rights_arguments())
            for artifact in reference_artifacts
        )
    )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Reference artifact manifests must be unique.")
    return identifiers


def _support(
    definition: _ClaimDefinition,
    runtime_support: Mapping[str, SupportValue],
    unit_contract: RelativisticUnitContract,
    frame_plan: LocalRelativisticFramePlan,
    reference_manifest_ids: Sequence[str],
    requested_use: FullDarkSectorReferenceUse,
    /,
) -> SupportTuple:
    if not isinstance(runtime_support, Mapping) or not runtime_support:
        raise TypeError("runtime_support must be a non-empty exact support mapping.")
    if not isinstance(unit_contract, RelativisticUnitContract):
        raise TypeError("unit_contract must be a RelativisticUnitContract.")
    if not isinstance(frame_plan, LocalRelativisticFramePlan):
        raise TypeError("frame_plan must be a LocalRelativisticFramePlan.")
    if frame_plan.units.contract_id != unit_contract.contract_id:
        raise ValueError("Claim frame and unit-contract identities do not match.")
    differentiation = definition.differentiation.contract()
    reserved: dict[str, SupportValue] = {
        "profile": definition.name,
        "governing_equation": definition.governing_equation,
        "state_family": definition.state_family,
        "solver": definition.solver,
        "four_vector_convention": unit_contract.four_vector_convention,
        "unit_contract_id": unit_contract.contract_id,
        "scale_id": unit_contract.scale.scale_id,
        "relativity_convention_id": unit_contract.convention.convention_id,
        "frame_id": frame_plan.frame_id,
        "frame_realization_id": frame_plan.realization_id(),
        "frame_snapshot_token": int(frame_plan.frame_token),
        "tetrad_id": frame_plan.tetrad.tetrad_id,
        "geometry_lineage_id": frame_plan.geometry.geometry_lineage_id,
        "differentiation": definition.differentiation.label,
        "differentiation_contract_id": differentiation.contract_id,
        "fixed_capacity": True,
        "semantic_unboundedness": "durable-finite-epoch-chain",
        "production_inheritance": False,
        "automatic_regime_switching": False,
        "topology_events_differentiable": False,
        "external_products": "stop-gradient-unless-profile-explicit",
        "checkpoint_product": definition.checkpoint_product,
        "analysis_output_products": ",".join(definition.output_products),
        "checkpoint_output_separation": "strict",
        "source_rights": "manifest-required",
        "promotion_channel": definition.promotion_channel,
        "reference_manifest_ids": ",".join(reference_manifest_ids),
        "reference_requested_use_id": requested_use.use_id,
        "reference_commercial_use": requested_use.commercial_use,
        "reference_redistribution": requested_use.redistribution,
        "reference_training_use": requested_use.training_use,
        "reference_export": requested_use.export,
    }
    overlap = set(reserved) & set(runtime_support)
    if overlap:
        raise ValueError(
            "Runtime support cannot replace governed profile coordinates: "
            + ", ".join(sorted(overlap))
        )
    return SupportTuple(definition.capability, {**reserved, **dict(runtime_support)})


def full_dark_sector_claim_profile(
    profile_name: FullDarkSectorClaimName,
    campaign: ScientificCampaign,
    criteria: Sequence[ScientificMetricCriterion],
    condition_domain_ids: Sequence[str],
    runtime_support: Mapping[str, SupportValue],
    unit_contract: RelativisticUnitContract,
    frame_plan: LocalRelativisticFramePlan,
    /,
    *,
    reference_artifacts: Sequence[ReferenceArtifactManifest],
    requested_use: FullDarkSectorReferenceUse,
) -> ScientificClaimProfile:
    """Build one exact profile without inheriting evidence from any other profile."""
    if profile_name not in _DEFINITIONS:
        raise ValueError("Unknown full dark-sector claim profile.")
    definition = _DEFINITIONS[profile_name]
    criteria_ = _validate_criteria(definition, criteria, campaign)
    references = _reference_binding(reference_artifacts, requested_use)
    return ScientificClaimProfile(
        definition.capability,
        _support(
            definition,
            runtime_support,
            unit_contract,
            frame_plan,
            references,
            requested_use,
        ),
        definition.observables,
        condition_domain_ids,
        campaign.campaign_id,
        (
            "source-admission",
            "numerical-validity",
            "parameter-identifiability",
            "predictive-calibration",
            "locked-prediction",
            "external-transfer",
        ),
        criteria_,
        f"full-dark-sector-abstention:{profile_name}",
        definition.refusals,
        frozen_criteria_ids=campaign.criteria_ids,
    )


def _factory(profile_name: FullDarkSectorClaimName, *args, **kwargs):
    return full_dark_sector_claim_profile(profile_name, *args, **kwargs)


def relativistic_stress_energy_pm_claim_profile(*args, **kwargs):
    return _factory("relativistic-stress-energy-pm", *args, **kwargs)


def einstein_vlasov_z4c_claim_profile(*args, **kwargs):
    return _factory("einstein-vlasov-z4c", *args, **kwargs)


def dynamic_epoch_runtime_claim_profile(*args, **kwargs):
    return _factory("dynamic-epoch-runtime", *args, **kwargs)


def fixed_multiplicity_claim_profile(*args, **kwargs):
    return _factory("fixed-multiplicity", *args, **kwargs)


def parton_shower_claim_profile(*args, **kwargs):
    return _factory("parton-shower", *args, **kwargs)


def hadronization_claim_profile(*args, **kwargs):
    return _factory("hadronization", *args, **kwargs)


def quantum_uu_claim_profile(*args, **kwargs):
    return _factory("quantum-uu", *args, **kwargs)


def thermal_qft_claim_profile(*args, **kwargs):
    return _factory("thermal-qft", *args, **kwargs)


def coherent_qke_claim_profile(*args, **kwargs):
    return _factory("coherent-qke", *args, **kwargs)


def off_shell_kb_claim_profile(*args, **kwargs):
    return _factory("off-shell-kb", *args, **kwargs)


def radiation_packet_claim_profile(*args, **kwargs):
    return _factory("radiation-packet", *args, **kwargs)


def radiation_m1_claim_profile(*args, **kwargs):
    return _factory("radiation-m1", *args, **kwargs)


def radiation_vet_claim_profile(*args, **kwargs):
    return _factory("radiation-vet", *args, **kwargs)


def radiation_hierarchy_claim_profile(*args, **kwargs):
    return _factory("radiation-hierarchy", *args, **kwargs)


def fully_coupled_closure_claim_profile(*args, **kwargs):
    return _factory("fully-coupled-closure", *args, **kwargs)


def _definition_for_claim(claim: ScientificClaimProfile, /) -> _ClaimDefinition:
    if not isinstance(claim, ScientificClaimProfile):
        raise TypeError("claim must be a ScientificClaimProfile.")
    candidates = tuple(
        definition
        for definition in _DEFINITIONS.values()
        if definition.capability == claim.capability_name
    )
    if len(candidates) != 1:
        raise ValueError("Claim is not an exact full dark-sector profile.")
    definition = candidates[0]
    attributes = dict(claim.support.attributes)
    expected_metrics = tuple(sorted(value.metric_id for value in definition.metrics))
    expected_differentiation = definition.differentiation.contract()
    if (
        attributes.get("profile") != definition.name
        or attributes.get("promotion_channel") != definition.promotion_channel
        or attributes.get("production_inheritance") is not False
        or attributes.get("checkpoint_product") != definition.checkpoint_product
        or attributes.get("analysis_output_products")
        != ",".join(definition.output_products)
        or attributes.get("differentiation_contract_id")
        != expected_differentiation.contract_id
        or claim.observable_ids != tuple(sorted(definition.observables))
        or tuple(value.metric_id for value in claim.criteria) != expected_metrics
        or claim.abstention_policy_id != f"full-dark-sector-abstention:{definition.name}"
        or claim.invalidation_triggers != tuple(sorted(definition.refusals))
    ):
        raise ValueError("Full dark-sector claim metadata or support changed.")
    return definition


class PromotedFullDarkSectorClaim(StrictModule, NonTrainableState):
    """Verified signed decision for exactly one claim and derivative gate."""

    claim: ScientificClaimProfile
    promotion: PromotionState = eqx.field(static=True)
    differentiation: DerivativeContract
    qualification_level: FullDarkSectorQualificationLevel = eqx.field(static=True)
    unsupported_claims: tuple[str, ...] = eqx.field(static=True)
    refusal_reasons: tuple[str, ...] = eqx.field(static=True)
    reference_manifest_ids: tuple[str, ...] = eqx.field(static=True)
    requested_use_id: str = eqx.field(static=True)
    admitted: bool = eqx.field(static=True)
    decision_id: str = eqx.field(static=True)

    def __init__(
        self,
        claim: ScientificClaimProfile,
        promotion: PromotionState,
        differentiation: DerivativeContract,
        qualification_level: FullDarkSectorQualificationLevel,
        unsupported_claims: Sequence[str],
        refusal_reasons: Sequence[str],
        reference_manifest_ids: Sequence[str],
        requested_use_id: str,
        admitted: bool,
        /,
    ):
        if not isinstance(claim, ScientificClaimProfile):
            raise TypeError("claim must be ScientificClaimProfile.")
        if not isinstance(promotion, PromotionState):
            raise TypeError("promotion must be PromotionState.")
        if not isinstance(differentiation, DerivativeContract):
            raise TypeError("differentiation must be DerivativeContract.")
        definition = _definition_for_claim(claim)
        expected_differentiation = definition.differentiation.contract()
        if differentiation.contract_id != expected_differentiation.contract_id:
            raise ValueError("Promoted claim differentiation metadata changed.")
        expected_channel = f"{definition.promotion_channel}.{claim.claim_id}"
        if promotion.channel != expected_channel:
            raise ValueError("Promotion channel does not bind this exact claim ID.")
        if qualification_level not in (
            "experimental",
            "numerically-qualified",
            "scientifically-qualified",
            "production",
        ):
            raise ValueError("Unknown full dark-sector qualification level.")
        if type(admitted) is not bool:
            raise TypeError("admitted must be a Boolean.")
        unsupported = tuple(
            _identifier(value, "unsupported claim") for value in unsupported_claims
        )
        refusals = tuple(
            _identifier(value, "refusal reason") for value in refusal_reasons
        )
        references = tuple(
            _identifier(value, "reference manifest ID")
            for value in reference_manifest_ids
        )
        if (
            not unsupported
            or not references
            or len(set(unsupported)) != len(unsupported)
            or len(set(refusals)) != len(refusals)
            or len(set(references)) != len(references)
        ):
            raise ValueError(
                "Promotion unsupported claims, refusals, and references are invalid."
            )
        use = _identifier(requested_use_id, "requested use ID")
        admitted_ = admitted
        if admitted_ != (qualification_level == "production" and not refusals):
            raise ValueError("Full dark-sector production admission is inconsistent.")
        self.claim = claim
        self.promotion = promotion
        self.differentiation = differentiation
        self.qualification_level = qualification_level
        self.unsupported_claims = unsupported
        self.refusal_reasons = refusals
        self.reference_manifest_ids = references
        self.requested_use_id = use
        self.admitted = admitted_
        self.decision_id = canonical_fingerprint(
            {
                "kind": "promoted-full-dark-sector-claim",
                "claim": claim.claim_id,
                "promotion": promotion.state_id,
                "differentiation": differentiation.contract_id,
                "qualification_level": qualification_level,
                "unsupported_claims": list(unsupported),
                "refusal_reasons": list(refusals),
                "reference_manifest_ids": list(references),
                "requested_use_id": use,
                "admitted": admitted_,
            }
        )


def full_dark_sector_promotion_channel(claim: ScientificClaimProfile, /) -> str:
    """Return the signed channel binding every byte of one exact claim."""
    definition = _definition_for_claim(claim)
    return f"{definition.promotion_channel}.{claim.claim_id}"


def bind_full_dark_sector_promotion(
    claim: ScientificClaimProfile,
    promotion: PromotionState,
    trust: QualificationRoleTrust,
    differentiation: DerivativeContract,
    /,
    *,
    at_time: int,
    qualification_level: FullDarkSectorQualificationLevel,
    expected_index_id: str | None = None,
    reference_artifacts: Sequence[ReferenceArtifactManifest],
    requested_use: FullDarkSectorReferenceUse,
) -> PromotedFullDarkSectorClaim:
    """Verify source rights, derivative gate, and signed profile-specific promotion."""
    definition = _definition_for_claim(claim)
    if not isinstance(promotion, PromotionState):
        raise TypeError("promotion must be a PromotionState.")
    if not isinstance(trust, QualificationRoleTrust):
        raise TypeError("trust must be QualificationRoleTrust.")
    if not isinstance(differentiation, DerivativeContract):
        raise TypeError("differentiation must be a DerivativeContract.")
    expected_differentiation = definition.differentiation.contract()
    if differentiation.contract_id != expected_differentiation.contract_id:
        raise ValueError("Requested derivative gate does not exactly match the claim.")
    if qualification_level not in (
        "experimental",
        "numerically-qualified",
        "scientifically-qualified",
        "production",
    ):
        raise ValueError("Unknown full dark-sector qualification level.")
    expected_channel = full_dark_sector_promotion_channel(claim)
    if promotion.channel != expected_channel:
        raise ValueError("Promotion channel does not bind this exact claim ID.")
    promotion.verify(trust, at_time=int(at_time))
    references = _reference_binding(reference_artifacts, requested_use)
    attributes = dict(claim.support.attributes)
    if (
        attributes.get("reference_manifest_ids") != ",".join(references)
        or attributes.get("reference_requested_use_id") != requested_use.use_id
        or attributes.get("reference_commercial_use") is not requested_use.commercial_use
        or attributes.get("reference_redistribution") is not requested_use.redistribution
        or attributes.get("reference_training_use") is not requested_use.training_use
        or attributes.get("reference_export") is not requested_use.export
    ):
        raise ValueError("Promotion source manifests or requested-use rights changed.")
    active = promotion.action in ("promote", "rollback")
    if active:
        expected = _identifier(expected_index_id, "expected_index_id")
        if promotion.index_id != expected:
            raise ValueError("Promotion release index was substituted.")
    elif expected_index_id is not None:
        raise ValueError("A refusal or withdrawal cannot retain an expected index.")
    refusals: list[str] = []
    if qualification_level != "production":
        refusals.append(f"qualification-level:{qualification_level}")
    if not active:
        refusals.append(f"promotion-{promotion.action}:{promotion.reason}")
    return PromotedFullDarkSectorClaim(
        claim,
        promotion,
        differentiation,
        qualification_level,
        definition.refusals,
        tuple(refusals),
        references,
        requested_use.use_id,
        qualification_level == "production" and active,
    )


__all__ = [
    "FullDarkSectorClaimName",
    "FullDarkSectorQualificationLevel",
    "FullDarkSectorReferenceUse",
    "PromotedFullDarkSectorClaim",
    "SUPPORTED_FULL_DARK_SECTOR_CLAIM_PROFILES",
    "bind_full_dark_sector_promotion",
    "coherent_qke_claim_profile",
    "dynamic_epoch_runtime_claim_profile",
    "einstein_vlasov_z4c_claim_profile",
    "fixed_multiplicity_claim_profile",
    "full_dark_sector_claim_criteria",
    "full_dark_sector_claim_metric_ids",
    "full_dark_sector_claim_profile",
    "full_dark_sector_differentiation_contract",
    "full_dark_sector_promotion_channel",
    "fully_coupled_closure_claim_profile",
    "hadronization_claim_profile",
    "off_shell_kb_claim_profile",
    "parton_shower_claim_profile",
    "quantum_uu_claim_profile",
    "radiation_hierarchy_claim_profile",
    "radiation_m1_claim_profile",
    "radiation_packet_claim_profile",
    "radiation_vet_claim_profile",
    "relativistic_stress_energy_pm_claim_profile",
    "thermal_qft_claim_profile",
]
