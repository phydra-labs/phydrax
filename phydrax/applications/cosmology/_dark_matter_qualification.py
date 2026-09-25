#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent scientific claims for the supported dark-matter solver families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import equinox as eqx

from ..._differentiation import DerivativeContract, DerivativeRoute
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


DarkMatterClaimName = Literal[
    "periodic-wave",
    "rare-sidm-equal",
    "mixed-root-wave-particle",
    "mixed-root-wave-particle-gas",
    "wave-amr-periodic",
    "rare-sidm-differential",
    "rare-sidm-weighted",
    "frequent-sidm-angular",
    "sidm-fluid-spherical",
    "sidm-inelastic-2to2",
]
DarkMatterQualificationLevel = Literal[
    "experimental",
    "numerically-qualified",
    "scientifically-qualified",
    "production",
]


class DarkMatterReferenceUse(StrictModule, NonTrainableState):
    """Exact requested-use rights bound into one dark-matter claim."""

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
        if any(not isinstance(value, bool) for value in values):
            raise TypeError("Dark-matter requested-use values must be Boolean.")
        (
            self.commercial_use,
            self.redistribution,
            self.training_use,
            self.export,
        ) = values
        self.use_id = canonical_fingerprint(
            {
                "kind": "dark-matter-reference-use",
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
class _ClaimDefinition:
    name: DarkMatterClaimName
    capability: str
    governing_equation: str
    state_family: str
    geometry: str
    boundary: str
    solver: str
    differentiation: str
    observables: tuple[str, ...]
    metrics: tuple[_MetricRequirement, ...]
    refusals: tuple[str, ...]
    outputs: tuple[str, ...]

    @property
    def promotion_channel(self) -> str:
        return f"dark-matter.production.{self.name}"


def _metric(
    name: str,
    direction: Literal["at_most", "at_least"] = "at_most",
    unit: str = "dimensionless",
    aggregation: Literal[
        "pooled", "independent_unit_macro", "worst_stratum"
    ] = "worst_stratum",
) -> _MetricRequirement:
    return _MetricRequirement(name, direction, unit, aggregation)


_COMMON_METRICS = (
    _metric("source-and-unit-closure-error"),
    _metric("checkpoint-restart-exact-match", "at_least", "fraction"),
    _metric("analysis-restart-separation", "at_least", "fraction"),
)
_COMMON_REFUSALS = (
    "changed-physics-or-support-identity",
    "missing-or-inadmissible-source-rights",
    "checkpoint-compatibility-mismatch",
    "fixed-capacity-exhaustion",
    "failed-numerical-status",
)


def _definition(
    name: DarkMatterClaimName,
    capability: str,
    governing_equation: str,
    state_family: str,
    geometry: str,
    boundary: str,
    solver: str,
    differentiation: str,
    observables: tuple[str, ...],
    metrics: tuple[_MetricRequirement, ...],
    refusals: tuple[str, ...],
    outputs: tuple[str, ...],
) -> _ClaimDefinition:
    return _ClaimDefinition(
        name,
        capability,
        governing_equation,
        state_family,
        geometry,
        boundary,
        solver,
        differentiation,
        observables,
        (*_COMMON_METRICS, *metrics),
        (*_COMMON_REFUSALS, *refusals),
        outputs,
    )


_DEFINITIONS: dict[DarkMatterClaimName, _ClaimDefinition] = {
    value.name: value
    for value in (
        _definition(
            "periodic-wave",
            "cosmology.dark-matter.periodic-wave",
            "comoving-schrodinger-poisson",
            "complex-wavefunction",
            "fixed-tensor-fourier-grid",
            "periodic",
            "strang-split-spectral-poisson",
            "smooth-fixed-grid-accepted-branch",
            ("wave-density", "wave-potential", "wave-power", "wave-phase-current"),
            (
                _metric("wave-norm-relative-error"),
                _metric("poisson-relative-residual"),
                _metric("potential-zero-mode-absolute"),
                _metric("maximum-split-phase-radians", unit="radian"),
                _metric("de-broglie-nyquist-fraction"),
                _metric("fixed-grid-jvp-relative-error"),
            ),
            (
                "nonperiodic-or-curved-geometry",
                "unresolved-phase-or-de-broglie-scale",
                "topology-or-grid-change-during-derivative",
            ),
            ("wave-snapshot",),
        ),
        _definition(
            "rare-sidm-equal",
            "cosmology.dark-matter.rare-sidm-equal",
            "rare-isotropic-elastic-boltzmann-pm",
            "equal-mass-particles",
            "fixed-periodic-root-grid",
            "periodic",
            "pairwise-strang-collision-pm",
            "nondifferentiable-event-selection",
            ("particle-density", "velocity-dispersion", "collision-rate", "halo-core"),
            (
                _metric("particle-mass-balance-defect", unit="code-mass"),
                _metric("pair-momentum-defect", unit="code-momentum"),
                _metric("pair-kinetic-energy-defect", unit="code-energy"),
                _metric("maximum-pair-probability"),
                _metric("maximum-particle-probability"),
                _metric("minimum-knudsen-number", "at_least"),
                _metric("endpoint-disjoint-fraction", "at_least", "fraction"),
                _metric("seed-ensemble-observable-error"),
            ),
            (
                "unequal-active-microscopic-mass",
                "event-selection-derivative-requested",
                "probability-or-knudsen-envelope-violated",
            ),
            ("particle-snapshot",),
        ),
        _definition(
            "mixed-root-wave-particle",
            "cosmology.dark-matter.mixed-root-wave-particle",
            "shared-poisson-wave-particle",
            "wave-particle-composite",
            "fixed-periodic-root-grid",
            "periodic",
            "symmetric-shared-gravity-split",
            "smooth-fixed-grid-accepted-branch",
            ("component-power", "cross-power", "total-potential", "component-work"),
            (
                _metric("component-initial-cross-spectrum-error"),
                _metric("total-density-closure-error"),
                _metric("shared-poisson-relative-residual"),
                _metric("periodic-net-force-defect", unit="code-force"),
                _metric("component-only-limit-error"),
                _metric("whole-state-rollback-error"),
                _metric("coupled-time-convergence-error"),
                _metric("fixed-grid-jvp-relative-error"),
            ),
            (
                "uncorrelated-component-initial-conditions",
                "multiple-gravity-solves-per-time-level",
                "component-time-level-mismatch",
                "typed-mixed-wave-snapshot-unavailable",
            ),
            ("particle-snapshot", "common-gravity-snapshot"),
        ),
        _definition(
            "mixed-root-wave-particle-gas",
            "cosmology.dark-matter.mixed-root-wave-particle-gas",
            "shared-poisson-wave-particle-euler",
            "wave-particle-gas-composite",
            "fixed-periodic-root-grid",
            "periodic",
            "symmetric-shared-gravity-gas-split",
            "smooth-fixed-grid-accepted-branch",
            (
                "component-power",
                "cross-power",
                "total-potential",
                "component-work",
                "gas-thermodynamics",
            ),
            (
                _metric("component-initial-cross-spectrum-error"),
                _metric("total-density-closure-error"),
                _metric("shared-poisson-relative-residual"),
                _metric("periodic-net-force-defect", unit="code-force"),
                _metric("gas-mass-relative-error"),
                _metric("gas-positivity-violation"),
                _metric("whole-state-rollback-error"),
                _metric("coupled-time-convergence-error"),
                _metric("fixed-grid-jvp-relative-error"),
            ),
            (
                "uncorrelated-component-initial-conditions",
                "gas-positivity-failure",
                "undeclared-frozen-potential-subcycling",
                "component-time-level-mismatch",
                "typed-mixed-wave-snapshot-unavailable",
            ),
            (
                "particle-snapshot",
                "gas-snapshot",
                "common-gravity-snapshot",
            ),
        ),
        _definition(
            "wave-amr-periodic",
            "cosmology.dark-matter.wave-amr-periodic",
            "comoving-schrodinger-poisson-amr",
            "complex-wave-amr-hierarchy",
            "periodic-block-hierarchy",
            "periodic-composite",
            "weighted-cayley-composite-poisson",
            "continuous-state-on-frozen-amr-epoch",
            ("wave-density", "wave-current", "vortex-topology", "amr-interface-flux"),
            (
                _metric("weighted-norm-relative-error"),
                _metric("cayley-linear-residual"),
                _metric("composite-poisson-relative-residual"),
                _metric("coarse-fine-current-defect"),
                _metric("interface-reflection-error"),
                _metric("topology-transfer-defect"),
                _metric("vortex-preservation-fraction", "at_least", "fraction"),
                _metric("regrid-rollback-error"),
                _metric("frozen-epoch-jvp-relative-error"),
            ),
            (
                "derivative-through-regrid-or-vortex-topology",
                "phase-interpolation-through-node",
                "density-only-refinement-authority",
                "unqualified-level-subcycling",
                "typed-production-snapshot-unavailable",
            ),
            ("amr-topology-evidence",),
        ),
        _definition(
            "rare-sidm-differential",
            "cosmology.dark-matter.rare-sidm-differential",
            "differential-elastic-boltzmann-pm",
            "typed-species-particles",
            "fixed-periodic-root-grid",
            "periodic",
            "differential-kernel-pairwise-split",
            "nondifferentiable-event-and-angle-selection",
            ("collision-rate", "angular-moments", "halo-shape", "halo-offset"),
            (
                _metric("differential-kernel-normalization-error"),
                _metric("total-cross-section-moment-error"),
                _metric("transfer-cross-section-moment-error"),
                _metric("viscosity-cross-section-moment-error"),
                _metric("angular-sampler-moment-error"),
                _metric("forward-screening-resolution-error"),
                _metric("pair-conservation-defect"),
            ),
            (
                "unscreened-forward-divergence",
                "uniform-angle-surrogate-for-anisotropic-kernel",
                "isotropic-transfer-matched-merger-claim",
                "event-or-angle-derivative-requested",
                "typed-production-snapshot-unavailable",
            ),
            ("kernel-evidence",),
        ),
        _definition(
            "rare-sidm-weighted",
            "cosmology.dark-matter.rare-sidm-weighted",
            "weighted-rare-elastic-boltzmann-pm",
            "weighted-packet-particles",
            "fixed-periodic-root-grid",
            "periodic",
            "retained-subpacket-collision-split",
            "nondifferentiable-event-topology-and-resampling",
            ("collision-rate", "effective-sample-size", "packet-lineage", "halo-core"),
            (
                _metric("weighted-collision-rate-relative-error"),
                _metric("packet-mass-defect", unit="code-mass"),
                _metric("packet-momentum-defect", unit="code-momentum"),
                _metric("packet-kinetic-energy-defect", unit="code-energy"),
                _metric("lineage-restart-mismatch"),
                _metric("slot-exhaustion-rollback-error"),
                _metric("resampler-declared-moment-error"),
                _metric("packet-refinement-observable-error"),
            ),
            (
                "average-macro-mass-rate-formula",
                "unretained-immediate-velocity-merge",
                "unqualified-resampling",
                "particle-allocation-or-lineage-derivative-requested",
                "typed-production-snapshot-unavailable",
            ),
            ("packet-lineage-evidence",),
        ),
        _definition(
            "frequent-sidm-angular",
            "cosmology.dark-matter.frequent-sidm-angular",
            "small-angle-kramers-moyal-pm",
            "frequent-scattering-particles",
            "fixed-periodic-root-grid",
            "periodic",
            "pairwise-drag-transverse-diffusion",
            "fixed-pair-reparameterized-noise-only",
            ("angular-relaxation", "diffusion-moments", "halo-shape", "halo-offset"),
            (
                _metric("first-kramers-moyal-moment-error"),
                _metric("second-kramers-moyal-moment-error"),
                _metric("diffusion-negative-eigenvalue-magnitude"),
                _metric("pair-momentum-defect", unit="code-momentum"),
                _metric("drag-energy-ledger-defect", unit="code-energy"),
                _metric("split-angle-invariance-error"),
                _metric("drift-diffusion-step-limit-violation"),
            ),
            (
                "total-cross-section-only-frequent-model",
                "overlapping-or-gapped-angular-split",
                "automatic-rare-frequent-switch",
                "unfrozen-pair-noise-derivative-requested",
                "typed-production-snapshot-unavailable",
            ),
            ("diffusion-evidence",),
        ),
        _definition(
            "sidm-fluid-spherical",
            "cosmology.dark-matter.sidm-fluid-spherical",
            "spherical-gravothermal-conduction",
            "radial-gravothermal-fluid",
            "isolated-one-dimensional-radial-mass-mesh",
            "declared-inner-outer",
            "long-short-mean-free-path-conduction",
            "smooth-fixed-mesh-only",
            ("density-profile", "velocity-dispersion-profile", "gravothermal-phase"),
            (
                _metric("radial-mass-relative-error"),
                _metric("gravothermal-energy-ledger-error"),
                _metric("conductivity-reference-error"),
                _metric("knudsen-support-violation"),
                _metric("radial-resolution-convergence-error"),
                _metric("boundary-flux-closure-error"),
                _metric("particle-overlap-observable-error"),
                _metric("fixed-mesh-jvp-relative-error"),
            ),
            (
                "generic-three-dimensional-fluid-claim",
                "automatic-particle-fluid-switch",
                "uncalibrated-conductivity-interpolation",
                "model-selection-derivative-requested",
                "typed-production-snapshot-unavailable",
            ),
            ("gravothermal-evidence",),
        ),
        _definition(
            "sidm-inelastic-2to2",
            "cosmology.dark-matter.sidm-inelastic-2to2",
            "reversible-multistate-two-body-boltzmann-pm",
            "reactive-weighted-packets-radiation-ledger",
            "fixed-periodic-root-grid",
            "periodic",
            "thresholded-detailed-balance-reaction-split",
            "nondifferentiable-reaction-and-topology-selection",
            ("species-fraction", "recoil", "reaction-rate", "radiation-ledger"),
            (
                _metric("threshold-opening-error"),
                _metric("exothermic-recoil-error"),
                _metric("detailed-balance-relative-error"),
                _metric("conserved-charge-defect"),
                _metric("total-four-momentum-ledger-defect"),
                _metric("dynamic-mass-gravity-defect", unit="code-mass"),
                _metric("reaction-capacity-rollback-error"),
            ),
            (
                "missing-reverse-kernel-or-nonapplicability-reason",
                "nonrelativistic-kinematics-outside-validity",
                "dissipation-without-radiation-or-export-ledger",
                "reaction-choice-or-particle-allocation-derivative-requested",
                "typed-production-snapshot-unavailable",
            ),
            (
                "reaction-evidence",
                "dark-radiation-ledger",
            ),
        ),
    )
}


SUPPORTED_DARK_MATTER_CLAIM_PROFILES = tuple(_DEFINITIONS)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return normalized


def dark_matter_claim_metric_ids(
    profile_name: DarkMatterClaimName,
    /,
) -> tuple[str, ...]:
    """Return the complete frozen metric names required by one profile."""

    if profile_name not in _DEFINITIONS:
        raise ValueError("Unknown dark-matter claim profile.")
    return tuple(value.metric_id for value in _DEFINITIONS[profile_name].metrics)


def dark_matter_claim_criteria(
    profile_name: DarkMatterClaimName,
    thresholds: Mapping[str, float],
    /,
) -> tuple[ScientificMetricCriterion, ...]:
    """Build the exact profile metric set from campaign-frozen threshold values."""

    if profile_name not in _DEFINITIONS:
        raise ValueError("Unknown dark-matter claim profile.")
    if not isinstance(thresholds, Mapping):
        raise TypeError("thresholds must map metric IDs to frozen numeric bounds.")
    requirements = _DEFINITIONS[profile_name].metrics
    expected = {value.metric_id for value in requirements}
    if set(thresholds) != expected:
        missing = sorted(expected - set(thresholds))
        extra = sorted(set(thresholds) - expected)
        raise ValueError(
            f"Dark-matter criterion thresholds changed; missing={missing}, extra={extra}."
        )
    criteria = []
    for requirement in requirements:
        threshold = float(thresholds[requirement.metric_id])
        if requirement.direction == "at_most":
            lower, upper = None, threshold
        else:
            lower, upper = threshold, None
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


def _validate_criteria(
    definition: _ClaimDefinition,
    criteria: Sequence[ScientificMetricCriterion],
    campaign: ScientificCampaign,
    /,
) -> tuple[ScientificMetricCriterion, ...]:
    values = tuple(criteria)
    if any(not isinstance(value, ScientificMetricCriterion) for value in values):
        raise TypeError("criteria must contain ScientificMetricCriterion values.")
    by_name = {value.metric_id: value for value in values}
    expected = {value.metric_id: value for value in definition.metrics}
    if set(by_name) != set(expected):
        raise ValueError("Claim criteria must exactly match the profile metric set.")
    for name, requirement in expected.items():
        value = by_name[name]
        if (
            value.direction != requirement.direction
            or value.unit_id != requirement.unit_id
            or value.aggregation != requirement.aggregation
        ):
            raise ValueError(f"Criterion {name!r} changed its profile metadata.")
    if any(value.criterion_id not in campaign.criteria_ids for value in values):
        raise ValueError("Every dark-matter criterion must be frozen by the campaign.")
    return values


def _reference_binding(
    reference_artifacts: Sequence[ReferenceArtifactManifest],
    requested_use: DarkMatterReferenceUse,
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
    if not isinstance(requested_use, DarkMatterReferenceUse):
        raise TypeError("requested_use must be DarkMatterReferenceUse.")
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
    reference_manifest_ids: Sequence[str],
    requested_use: DarkMatterReferenceUse,
    /,
) -> SupportTuple:
    if not isinstance(runtime_support, Mapping) or not runtime_support:
        raise TypeError("runtime_support must be a nonempty exact support mapping.")
    reserved: dict[str, SupportValue] = {
        "profile": definition.name,
        "governing_equation": definition.governing_equation,
        "state_family": definition.state_family,
        "geometry": definition.geometry,
        "boundary": definition.boundary,
        "solver": definition.solver,
        "time_coordinate": "scale-factor",
        "differentiation": definition.differentiation,
        "fixed_capacity": True,
        "production_inheritance": False,
        "automatic_regime_switching": False,
        "topology_events_differentiable": False,
        "external_products": "stop-gradient",
        "checkpoint_contract": "runtime-checkpoint-envelope",
        "analysis_output_contract": "typed-snapshot-not-restart",
        "source_rights": "manifest-required",
        "promotion_channel": definition.promotion_channel,
        "reference_manifest_ids": ",".join(reference_manifest_ids),
        "reference_requested_use_id": requested_use.use_id,
        "reference_commercial_use": requested_use.commercial_use,
        "reference_redistribution": requested_use.redistribution,
        "reference_training_use": requested_use.training_use,
        "reference_export": requested_use.export,
        "output_products": ",".join(definition.outputs),
    }
    overlap = set(reserved) & set(runtime_support)
    if overlap:
        raise ValueError(
            "Runtime support cannot replace profile coordinates: "
            + ", ".join(sorted(overlap))
        )
    return SupportTuple(definition.capability, {**reserved, **dict(runtime_support)})


def _claim_profile(
    name: DarkMatterClaimName,
    campaign: ScientificCampaign,
    criteria: Sequence[ScientificMetricCriterion],
    condition_domain_ids: Sequence[str],
    runtime_support: Mapping[str, SupportValue],
    reference_artifacts: Sequence[ReferenceArtifactManifest],
    requested_use: DarkMatterReferenceUse,
    /,
) -> ScientificClaimProfile:
    if not isinstance(campaign, ScientificCampaign):
        raise TypeError("campaign must be a ScientificCampaign.")
    definition = _DEFINITIONS[name]
    criteria_ = _validate_criteria(definition, criteria, campaign)
    reference_ids = _reference_binding(reference_artifacts, requested_use)
    return ScientificClaimProfile(
        definition.capability,
        _support(
            definition,
            runtime_support,
            reference_ids,
            requested_use,
        ),
        definition.observables,
        condition_domain_ids,
        campaign.campaign_id,
        (
            "source-admission",
            "numerical-validity",
            "predictive-calibration",
            "locked-prediction",
            "external-transfer",
        ),
        criteria_,
        f"dark-matter-abstention:{name}",
        definition.refusals,
        frozen_criteria_ids=campaign.criteria_ids,
    )


def periodic_wave_claim_profile(
    campaign,
    criteria,
    condition_domain_ids,
    runtime_support,
    /,
    *,
    reference_artifacts,
    requested_use,
) -> ScientificClaimProfile:
    return _claim_profile(
        "periodic-wave",
        campaign,
        criteria,
        condition_domain_ids,
        runtime_support,
        reference_artifacts,
        requested_use,
    )


def rare_equal_sidm_claim_profile(
    campaign,
    criteria,
    condition_domain_ids,
    runtime_support,
    /,
    *,
    reference_artifacts,
    requested_use,
) -> ScientificClaimProfile:
    return _claim_profile(
        "rare-sidm-equal",
        campaign,
        criteria,
        condition_domain_ids,
        runtime_support,
        reference_artifacts,
        requested_use,
    )


def mixed_wave_particle_claim_profile(
    campaign,
    criteria,
    condition_domain_ids,
    runtime_support,
    /,
    *,
    reference_artifacts,
    requested_use,
) -> ScientificClaimProfile:
    return _claim_profile(
        "mixed-root-wave-particle",
        campaign,
        criteria,
        condition_domain_ids,
        runtime_support,
        reference_artifacts,
        requested_use,
    )


def mixed_wave_particle_gas_claim_profile(
    campaign,
    criteria,
    condition_domain_ids,
    runtime_support,
    /,
    *,
    reference_artifacts,
    requested_use,
) -> ScientificClaimProfile:
    return _claim_profile(
        "mixed-root-wave-particle-gas",
        campaign,
        criteria,
        condition_domain_ids,
        runtime_support,
        reference_artifacts,
        requested_use,
    )


def periodic_wave_amr_claim_profile(
    campaign,
    criteria,
    condition_domain_ids,
    runtime_support,
    /,
    *,
    reference_artifacts,
    requested_use,
) -> ScientificClaimProfile:
    return _claim_profile(
        "wave-amr-periodic",
        campaign,
        criteria,
        condition_domain_ids,
        runtime_support,
        reference_artifacts,
        requested_use,
    )


def differential_sidm_claim_profile(
    campaign,
    criteria,
    condition_domain_ids,
    runtime_support,
    /,
    *,
    reference_artifacts,
    requested_use,
) -> ScientificClaimProfile:
    return _claim_profile(
        "rare-sidm-differential",
        campaign,
        criteria,
        condition_domain_ids,
        runtime_support,
        reference_artifacts,
        requested_use,
    )


def weighted_sidm_claim_profile(
    campaign,
    criteria,
    condition_domain_ids,
    runtime_support,
    /,
    *,
    reference_artifacts,
    requested_use,
) -> ScientificClaimProfile:
    return _claim_profile(
        "rare-sidm-weighted",
        campaign,
        criteria,
        condition_domain_ids,
        runtime_support,
        reference_artifacts,
        requested_use,
    )


def frequent_sidm_claim_profile(
    campaign,
    criteria,
    condition_domain_ids,
    runtime_support,
    /,
    *,
    reference_artifacts,
    requested_use,
) -> ScientificClaimProfile:
    return _claim_profile(
        "frequent-sidm-angular",
        campaign,
        criteria,
        condition_domain_ids,
        runtime_support,
        reference_artifacts,
        requested_use,
    )


def gravothermal_sidm_claim_profile(
    campaign,
    criteria,
    condition_domain_ids,
    runtime_support,
    /,
    *,
    reference_artifacts,
    requested_use,
) -> ScientificClaimProfile:
    return _claim_profile(
        "sidm-fluid-spherical",
        campaign,
        criteria,
        condition_domain_ids,
        runtime_support,
        reference_artifacts,
        requested_use,
    )


def inelastic_sidm_claim_profile(
    campaign,
    criteria,
    condition_domain_ids,
    runtime_support,
    /,
    *,
    reference_artifacts,
    requested_use,
) -> ScientificClaimProfile:
    return _claim_profile(
        "sidm-inelastic-2to2",
        campaign,
        criteria,
        condition_domain_ids,
        runtime_support,
        reference_artifacts,
        requested_use,
    )


def _definition_for_claim(claim: ScientificClaimProfile, /) -> _ClaimDefinition:
    if not isinstance(claim, ScientificClaimProfile):
        raise TypeError("claim must be a ScientificClaimProfile.")
    candidates = tuple(
        definition
        for definition in _DEFINITIONS.values()
        if definition.capability == claim.capability_name
    )
    if len(candidates) != 1:
        raise ValueError("Claim is not an exact dark-matter profile.")
    definition = candidates[0]
    attributes = dict(claim.support.attributes)
    expected_metric_ids = tuple(value.metric_id for value in definition.metrics)
    if (
        attributes.get("profile") != definition.name
        or attributes.get("promotion_channel") != definition.promotion_channel
        or attributes.get("production_inheritance") is not False
        or claim.observable_ids != tuple(sorted(definition.observables))
        or tuple(value.metric_id for value in claim.criteria)
        != tuple(sorted(expected_metric_ids))
        or claim.abstention_policy_id != f"dark-matter-abstention:{definition.name}"
        or claim.invalidation_triggers != tuple(sorted(definition.refusals))
    ):
        raise ValueError("Dark-matter claim support or frozen claim metadata changed.")
    return definition


def _differentiation_contract() -> DerivativeContract:
    """Withhold derivative rights until typed campaign evidence is verified."""

    return DerivativeContract(route=DerivativeRoute.DIRECT)


class PromotedDarkMatterClaim(StrictModule, NonTrainableState):
    """Verified signed channel decision for exactly one scientific claim profile."""

    claim: ScientificClaimProfile
    promotion: PromotionState = eqx.field(static=True)
    differentiation: DerivativeContract
    qualification_level: DarkMatterQualificationLevel = eqx.field(static=True)
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
        qualification_level: DarkMatterQualificationLevel,
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
        use = _identifier(requested_use_id, "requested use ID")
        admitted_ = bool(admitted)
        if admitted_ != (qualification_level == "production" and not refusals):
            raise ValueError("Dark-matter production admission metadata is inconsistent.")
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
                "kind": "promoted-dark-matter-claim",
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


def dark_matter_promotion_channel(claim: ScientificClaimProfile, /) -> str:
    """Return the signed channel name bound to every byte of one exact claim."""

    definition = _definition_for_claim(claim)
    return f"{definition.promotion_channel}.{claim.claim_id}"


def bind_dark_matter_promotion(
    claim: ScientificClaimProfile,
    promotion: PromotionState,
    trust: QualificationRoleTrust,
    /,
    *,
    at_time: int,
    qualification_level: DarkMatterQualificationLevel,
    expected_index_id: str | None = None,
    reference_artifacts: Sequence[ReferenceArtifactManifest],
    requested_use: DarkMatterReferenceUse,
) -> PromotedDarkMatterClaim:
    """Verify and bind one profile-specific promotion without cross-profile inheritance."""

    definition = _definition_for_claim(claim)
    if not isinstance(promotion, PromotionState):
        raise TypeError("promotion must be a PromotionState.")
    if not isinstance(trust, QualificationRoleTrust):
        raise TypeError("trust must be QualificationRoleTrust.")
    if qualification_level not in (
        "experimental",
        "numerically-qualified",
        "scientifically-qualified",
        "production",
    ):
        raise ValueError("Unknown dark-matter qualification level.")
    expected_channel = dark_matter_promotion_channel(claim)
    if promotion.channel != expected_channel:
        raise ValueError("Promotion channel does not bind this exact claim ID.")
    promotion.verify(trust, at_time=int(at_time))
    reference_ids = _reference_binding(reference_artifacts, requested_use)
    attributes = dict(claim.support.attributes)
    if (
        attributes.get("reference_manifest_ids") != ",".join(reference_ids)
        or attributes.get("reference_requested_use_id") != requested_use.use_id
        or attributes.get("reference_commercial_use") is not requested_use.commercial_use
        or attributes.get("reference_redistribution") is not requested_use.redistribution
        or attributes.get("reference_training_use") is not requested_use.training_use
        or attributes.get("reference_export") is not requested_use.export
    ):
        raise ValueError("Promotion reference manifests or requested-use rights changed.")
    active = promotion.action in ("promote", "rollback")
    if active:
        expected = _identifier(expected_index_id, "expected_index_id")
        if promotion.index_id != expected:
            raise ValueError("Promotion release index was substituted.")
    elif expected_index_id is not None:
        raise ValueError("A refusal or withdrawal cannot retain an expected index.")
    refusals = []
    if qualification_level != "production":
        refusals.append(f"qualification-level:{qualification_level}")
    if not active:
        refusals.append(f"promotion-{promotion.action}:{promotion.reason}")
    return PromotedDarkMatterClaim(
        claim,
        promotion,
        _differentiation_contract(),
        qualification_level,
        definition.refusals,
        tuple(refusals),
        reference_ids,
        requested_use.use_id,
        qualification_level == "production" and active,
    )


__all__ = [
    "DarkMatterClaimName",
    "DarkMatterQualificationLevel",
    "DarkMatterReferenceUse",
    "PromotedDarkMatterClaim",
    "SUPPORTED_DARK_MATTER_CLAIM_PROFILES",
    "bind_dark_matter_promotion",
    "dark_matter_claim_criteria",
    "dark_matter_claim_metric_ids",
    "dark_matter_promotion_channel",
    "differential_sidm_claim_profile",
    "frequent_sidm_claim_profile",
    "gravothermal_sidm_claim_profile",
    "inelastic_sidm_claim_profile",
    "mixed_wave_particle_claim_profile",
    "mixed_wave_particle_gas_claim_profile",
    "periodic_wave_amr_claim_profile",
    "periodic_wave_claim_profile",
    "rare_equal_sidm_claim_profile",
    "weighted_sidm_claim_profile",
]
