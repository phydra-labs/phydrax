#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field

from ..._fingerprint import canonical_fingerprint
from ...artifacts import ArtifactManifest, ScientificArtifactEnvelope
from ...qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    CapabilityProfile,
    QualificationCriterion,
    QualificationEvidence,
    QualificationMatrix,
    ReferenceArtifactManifest,
    ReleaseGateEvidence,
    ReleaseIndex,
    ReleaseTrustPolicy,
    require_profile,
    SupportTuple,
)
from ._validity import (
    CIRCUIT_ECM_ENVELOPE,
    MARQUIS_2019_SPME_ENVELOPE,
    NEWMAN_DFN_ENVELOPE,
    SERIES_PACK_ENVELOPE,
)


_PROVIDER = "phydrax"
_CANDIDATE_VERSION = "candidate"

THERMAL_ECM_SUPPORT = SupportTuple(
    "battery.simulation",
    {
        "model_id": "battery:ecm:thermal-prescribed-current",
        "equation_form": "ode",
        "control": "prescribed-current",
        "circuit_connected": False,
    },
)
THERMAL_ECM_CANDIDATE = CapabilityProfile(
    "battery.thermal-ecm.candidate",
    _PROVIDER,
    _CANDIDATE_VERSION,
    (THERMAL_ECM_SUPPORT,),
    released=False,
)


def build_thermal_ecm_release_profile(
    bundle, /, *, trust_policy, at_time: int, expires_at: int
) -> CapabilityProfile:
    """Build only from authenticated typed causal records and complete coverage."""
    from ._release import build_battery_release

    return build_battery_release(
        THERMAL_ECM_CANDIDATE,
        bundle,
        trust_policy=trust_policy,
        at_time=at_time,
        expires_at=expires_at,
    ).profile


ISOTHERMAL_SPM_SUPPORT = SupportTuple(
    "battery.simulation",
    {
        "model_id": "battery:spm:isothermal-prescribed-current",
        "equation_form": "ode",
        "control": "prescribed-current",
    },
)
ISOTHERMAL_SPM_CANDIDATE = CapabilityProfile(
    "battery.isothermal-spm.candidate",
    _PROVIDER,
    _CANDIDATE_VERSION,
    (ISOTHERMAL_SPM_SUPPORT,),
    released=False,
)

MARQUIS_2019_SPME_SUPPORT = SupportTuple(
    "battery.simulation",
    {
        "model_id": "battery:spme:marquis-2019:isothermal-prescribed-current",
        "equation_form": "ode",
        "control": "prescribed-current-and-rest",
        "geometry": "planar-1d-asymptotic-electrolyte",
        "thermal": "isothermal",
        "chemistry": "lithium-ion-intercalation",
        "validity_envelope_id": MARQUIS_2019_SPME_ENVELOPE.envelope_id,
    },
)
MARQUIS_2019_SPME_CANDIDATE = CapabilityProfile(
    "battery.marquis-2019-spme.candidate",
    _PROVIDER,
    _CANDIDATE_VERSION,
    (MARQUIS_2019_SPME_SUPPORT,),
    released=False,
)

BROSA_PLANELLA_TSPME_SUPPORT = SupportTuple(
    "battery.simulation",
    {
        "model_id": "battery:tspme:brosa-planella:base-prescribed-current",
        "equation_form": "ode",
        "control": "prescribed-current",
    },
)
BROSA_PLANELLA_TSPME_CANDIDATE = CapabilityProfile(
    "battery.brosa-planella-tspme.candidate",
    _PROVIDER,
    _CANDIDATE_VERSION,
    (BROSA_PLANELLA_TSPME_SUPPORT,),
    released=False,
)

EMPIRICAL_AGEING_SUPPORT = SupportTuple(
    "battery.ageing",
    {
        "code_id": (
            "phydrax.applications.battery._ageing_empirical.advance_empirical_ageing"
        ),
        "coupling": "one-way-macrostep",
    },
)
EMPIRICAL_AGEING_CANDIDATE = CapabilityProfile(
    "battery.empirical-ageing.candidate",
    _PROVIDER,
    _CANDIDATE_VERSION,
    (EMPIRICAL_AGEING_SUPPORT,),
    released=False,
)

BROSA_PLANELLA_SPME_SEI_SUPPORT = SupportTuple(
    "battery.simulation",
    {
        "model_id": "battery:spme:sei:brosa-planella-widanage:isothermal",
        "equation_form": "ode",
        "control": "prescribed-current",
        "side_reaction": "sei-only",
    },
)
BROSA_PLANELLA_SPME_SEI_CANDIDATE = CapabilityProfile(
    "battery.brosa-planella-spme-sei.candidate",
    _PROVIDER,
    _CANDIDATE_VERSION,
    (BROSA_PLANELLA_SPME_SEI_SUPPORT,),
    released=False,
)

AFFINE_ECM_ESTIMATION_SUPPORT = SupportTuple(
    "battery.estimation",
    {
        "code_id": ("phydrax.applications.battery._estimation.estimate_exact_affine_ecm"),
        "model_id": "battery:ecm:thermal-prescribed-current",
        "estimator": "exact-affine-kalman-rts",
    },
)
AFFINE_ECM_ESTIMATION_CANDIDATE = CapabilityProfile(
    "battery.affine-ecm-estimation.candidate",
    _PROVIDER,
    _CANDIDATE_VERSION,
    (AFFINE_ECM_ESTIMATION_SUPPORT,),
    released=False,
)

BATTERY_OED_SUPPORT = SupportTuple(
    "battery.experimental-design",
    {
        "code_id": "phydrax.applications.battery._oed.evaluate_battery_oed",
        "design_space": "fixed-topology-current-amplitudes",
        "model_contract": "released-smooth",
    },
)
BATTERY_OED_CANDIDATE = CapabilityProfile(
    "battery.oed.candidate",
    _PROVIDER,
    _CANDIDATE_VERSION,
    (BATTERY_OED_SUPPORT,),
    released=False,
)

FIXED_HORIZON_CURRENT_CONTROL_SUPPORT = SupportTuple(
    "battery.current-control",
    {
        "code_id": (
            "phydrax.applications.battery._current_control.replay_battery_current_control"
        ),
        "horizon": "fixed",
        "control": "piecewise-constant-current",
    },
)
FIXED_HORIZON_CURRENT_CONTROL_CANDIDATE = CapabilityProfile(
    "battery.fixed-horizon-current-control.candidate",
    _PROVIDER,
    _CANDIDATE_VERSION,
    (FIXED_HORIZON_CURRENT_CONTROL_SUPPORT,),
    released=False,
)


DFN_ENTRY_SUPPORT = SupportTuple(
    "battery.expansion.dfn",
    {
        "source_model_id": ("battery:spme:marquis-2019:isothermal-prescribed-current"),
        "decision": "dfn-entry",
    },
)

NEWMAN_DFN_SUPPORT = SupportTuple(
    "battery.simulation",
    {
        "model_id": NEWMAN_DFN_ENVELOPE.model_id,
        "equation_form": "dae",
        "control": "prescribed-current-and-rest",
        "geometry": "planar-1d-spherical-particles",
        "thermal": "isothermal",
        "kinetics": "symmetric-butler-volmer",
        "side_reactions": "none",
        "circuit_connected": False,
        "validity_envelope_id": NEWMAN_DFN_ENVELOPE.envelope_id,
    },
)
NEWMAN_DFN_CANDIDATE = CapabilityProfile(
    "battery.newman-dfn.candidate",
    _PROVIDER,
    _CANDIDATE_VERSION,
    (NEWMAN_DFN_SUPPORT,),
)
CIRCUIT_ECM_SUPPORT = SupportTuple(
    "battery.simulation",
    {
        "model_id": CIRCUIT_ECM_ENVELOPE.model_id,
        "equation_form": "dae",
        "circuit_connected": True,
        "interface": "two-terminal-passive-sign",
        "thermal_boundary": "one-cell-linear-ambient",
        "validity_envelope_id": CIRCUIT_ECM_ENVELOPE.envelope_id,
    },
)
CIRCUIT_ECM_CANDIDATE = CapabilityProfile(
    "battery.circuit-ecm.candidate",
    _PROVIDER,
    _CANDIDATE_VERSION,
    (CIRCUIT_ECM_SUPPORT,),
)
SERIES_PACK_SUPPORT = SupportTuple(
    "battery.pack.simulation",
    {
        "model_id": SERIES_PACK_ENVELOPE.model_id,
        "equation_form": "dae",
        "cell_model_id": CIRCUIT_ECM_ENVELOPE.model_id,
        "electrical_topology": "single-series-string",
        "control": "prescribed-terminal-current-and-rest",
        "homogeneous_values": True,
        "interconnect": "uniform-positive-massless-resistors-fixed-heat-allocation",
        "thermal_topology": "ordered-nearest-neighbor-path-uniform-ambient",
        "balancing": "none",
        "estimation": "none",
        "bms_control": "none",
        "protection": "none",
        "validity_envelope_id": SERIES_PACK_ENVELOPE.envelope_id,
    },
)
SERIES_PACK_CANDIDATE = CapabilityProfile(
    "battery.homogeneous-series-pack.candidate",
    _PROVIDER,
    _CANDIDATE_VERSION,
    (SERIES_PACK_SUPPORT,),
)
SERIES_PACK_ENTRY_SUPPORT = SupportTuple(
    "battery.expansion.series-pack",
    {"source_model_id": CIRCUIT_ECM_ENVELOPE.model_id, "decision": "series-pack-entry"},
)

BATTERY_CANDIDATE_PROFILES = (
    THERMAL_ECM_CANDIDATE,
    ISOTHERMAL_SPM_CANDIDATE,
    MARQUIS_2019_SPME_CANDIDATE,
    BROSA_PLANELLA_TSPME_CANDIDATE,
    EMPIRICAL_AGEING_CANDIDATE,
    BROSA_PLANELLA_SPME_SEI_CANDIDATE,
    AFFINE_ECM_ESTIMATION_CANDIDATE,
    BATTERY_OED_CANDIDATE,
    FIXED_HORIZON_CURRENT_CONTROL_CANDIDATE,
    NEWMAN_DFN_CANDIDATE,
    CIRCUIT_ECM_CANDIDATE,
    SERIES_PACK_CANDIDATE,
)
BATTERY_RELEASE_COORDINATES = {
    THERMAL_ECM_SUPPORT.support_tuple_id: (
        "battery.thermal-ecm",
        "ecm-analytic-qualified",
    ),
    MARQUIS_2019_SPME_SUPPORT.support_tuple_id: (
        "battery.marquis-2019-spme",
        "numerical-qualified",
    ),
    NEWMAN_DFN_SUPPORT.support_tuple_id: ("battery.newman-dfn", "numerical-qualified"),
    CIRCUIT_ECM_SUPPORT.support_tuple_id: ("battery.circuit-ecm", "numerical-qualified"),
    SERIES_PACK_SUPPORT.support_tuple_id: (
        "battery.homogeneous-series-pack",
        "numerical-qualified",
    ),
}


PACK_SIGN_GATE = "battery.series-pack.sign-convention"
PACK_KCL_GATE = "battery.series-pack.kcl"
PACK_POWER_GATE = "battery.series-pack.power-balance"
PACK_THERMAL_GATE = "battery.series-pack.thermal-coupling"
SERIES_PACK_REQUIRED_GATES = (
    PACK_SIGN_GATE,
    PACK_KCL_GATE,
    PACK_POWER_GATE,
    PACK_THERMAL_GATE,
)


@dataclass(frozen=True, slots=True)
class BatteryExpansionGateDecision:
    """Immutable, content-addressed refusal or admission for conditional expansion."""

    eligible: bool
    conclusive: bool
    reason: str
    evidence_ids: tuple[str, ...]
    decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.eligible, bool) or not isinstance(self.conclusive, bool):
            raise TypeError("Expansion-gate dispositions must be booleans.")
        if self.eligible and not self.conclusive:
            raise ValueError("An eligible expansion-gate decision must be conclusive.")
        if (
            not isinstance(self.reason, str)
            or not self.reason
            or self.reason != self.reason.strip()
        ):
            raise ValueError("Expansion-gate reason must be a canonical identifier.")
        if not isinstance(self.evidence_ids, tuple) or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in self.evidence_ids
        ):
            raise TypeError("Expansion-gate evidence IDs must be canonical strings.")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Expansion-gate evidence IDs must be unique.")
        evidence_ids = tuple(sorted(self.evidence_ids))
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(
            self,
            "decision_id",
            canonical_fingerprint(
                {
                    "kind": "battery-expansion-gate-decision",
                    "eligible": self.eligible,
                    "conclusive": self.conclusive,
                    "reason": self.reason,
                    "evidence_ids": list(evidence_ids),
                }
            ),
        )


def validate_battery_candidate_profile(
    profile: CapabilityProfile, support_tuple: SupportTuple, /
) -> None:
    """Check exact known coordinates, never granting execution admission."""
    if not isinstance(profile, CapabilityProfile) or not isinstance(
        support_tuple, SupportTuple
    ):
        raise TypeError("Battery profile and support must use typed registry records.")
    CapabilityProfile.from_record(profile.to_record())
    SupportTuple.from_record(support_tuple.to_record())
    if tuple(item.support_tuple_id for item in profile.support_tuples) != (
        support_tuple.support_tuple_id,
    ):
        raise ValueError("Battery execution requires one exact support tuple.")
    if not profile.released:
        if profile.profile_id not in {
            item.profile_id for item in BATTERY_CANDIDATE_PROFILES
        }:
            raise ValueError("Unknown battery candidate profile.")
        return
    coordinates = BATTERY_RELEASE_COORDINATES.get(support_tuple.support_tuple_id)
    if (
        coordinates is None
        or (profile.name, profile.version) != coordinates
        or profile.provider != _PROVIDER
    ):
        raise ValueError("Unknown released battery profile coordinates.")
    if not profile.required_gates or not profile.release_evidence:
        raise ValueError(
            "Released battery profiles require authenticated scientific gates."
        )


def require_released_battery_profile(
    index: ReleaseIndex,
    profile_id: str,
    support_tuple: SupportTuple,
    trust_policy: ReleaseTrustPolicy,
    /,
    *,
    at_time: int,
) -> CapabilityProfile:
    """Delegate released-profile admission to the generic qualification registry."""
    if not support_tuple.capability.startswith("battery."):
        raise ValueError("Battery release lookup requires a battery support tuple.")
    return require_profile(
        index,
        profile_id,
        support_tuple,
        trust_policy,
        at_time=at_time,
    )


def artifact_identity(
    artifact: ArtifactManifest | ReferenceArtifactManifest | ScientificArtifactEnvelope,
    /,
) -> str:
    """Return the installed generic artifact's canonical identity without wrapping it."""
    if isinstance(artifact, ArtifactManifest):
        return artifact.manifest_id
    if isinstance(artifact, ReferenceArtifactManifest):
        return artifact.manifest_id
    if isinstance(artifact, ScientificArtifactEnvelope):
        return artifact.artifact_id
    raise TypeError("artifact must be a generic Phydrax artifact contract.")


__all__ = [
    "AFFINE_ECM_ESTIMATION_CANDIDATE",
    "AFFINE_ECM_ESTIMATION_SUPPORT",
    "ArtifactManifest",
    "BATTERY_OED_CANDIDATE",
    "BATTERY_OED_SUPPORT",
    "BROSA_PLANELLA_SPME_SEI_CANDIDATE",
    "BROSA_PLANELLA_SPME_SEI_SUPPORT",
    "BROSA_PLANELLA_TSPME_CANDIDATE",
    "BROSA_PLANELLA_TSPME_SUPPORT",
    "BatteryExpansionGateDecision",
    "CampaignObservationRecord",
    "CampaignStartRecord",
    "CapabilityProfile",
    "DFN_ENTRY_SUPPORT",
    "EMPIRICAL_AGEING_CANDIDATE",
    "EMPIRICAL_AGEING_SUPPORT",
    "FIXED_HORIZON_CURRENT_CONTROL_CANDIDATE",
    "FIXED_HORIZON_CURRENT_CONTROL_SUPPORT",
    "ISOTHERMAL_SPM_CANDIDATE",
    "ISOTHERMAL_SPM_SUPPORT",
    "MARQUIS_2019_SPME_CANDIDATE",
    "MARQUIS_2019_SPME_SUPPORT",
    "PACK_KCL_GATE",
    "PACK_POWER_GATE",
    "PACK_SIGN_GATE",
    "PACK_THERMAL_GATE",
    "QualificationCriterion",
    "QualificationEvidence",
    "QualificationMatrix",
    "ReferenceArtifactManifest",
    "ReleaseGateEvidence",
    "SERIES_PACK_REQUIRED_GATES",
    "ScientificArtifactEnvelope",
    "SupportTuple",
    "THERMAL_ECM_CANDIDATE",
    "THERMAL_ECM_SUPPORT",
    "build_thermal_ecm_release_profile",
    "artifact_identity",
    "require_released_battery_profile",
    "validate_battery_candidate_profile",
    "BATTERY_CANDIDATE_PROFILES",
    "BATTERY_RELEASE_COORDINATES",
    "CIRCUIT_ECM_CANDIDATE",
    "CIRCUIT_ECM_SUPPORT",
    "NEWMAN_DFN_CANDIDATE",
    "NEWMAN_DFN_SUPPORT",
    "SERIES_PACK_CANDIDATE",
    "SERIES_PACK_SUPPORT",
    "SERIES_PACK_ENTRY_SUPPORT",
]
