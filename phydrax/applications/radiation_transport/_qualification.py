#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ...qualification import (
    CampaignRole,
    CapabilityProfile,
    ScientificCampaign,
    ScientificCase,
    SupportTuple,
)


_SPECS = (
    (
        "radiation-transport.photon-diagnostic-xray",
        {
            "particle": "photon",
            "energy_unit": "electronvolt",
            "geometry": "dense-voxel-delta-tracking",
            "processes": "photoelectric-compton-rayleigh",
            "interaction_data": "source-pinned-mass-attenuation-plus-density",
            "deposition": "kerma-local-recoil-not-absorbed-dose",
        },
        (
            "photon-data-provenance",
            "photon-process-reference",
            "photon-energy-ledger",
            "detector-response",
        ),
    ),
    (
        "radiation-transport.discrete-ordinates",
        {
            "geometry": "one-dimensional-slab",
            "angular": "certified-gauss-legendre",
            "energy": "multigroup",
            "scattering": "isotropic-group-transfer",
            "acceleration": "optional-dsa",
        },
        ("sn-angular-moments", "sn-balance", "sn-reference-flux"),
    ),
    (
        "radiation-transport.charged-condensed-history",
        {
            "particles": "electron-and-positron",
            "geometry": "dense-voxel-boundary-limited",
            "physics": "stopping-multiple-scattering-bremsstrahlung-annihilation",
            "secondary_photons": "tallied-not-transported",
        },
        ("charged-range", "charged-energy-ledger", "charged-reference-transport"),
    ),
    (
        "radiation-transport.imc-ddmc",
        {
            "geometry": "one-dimensional-cells",
            "groups": "multigroup-packets",
            "coupling": "fleck-effective-absorption",
            "thick_limit": "ddmc-leakage",
        },
        ("imc-ddmc-energy-ledger", "imc-reference-wave", "ddmc-thick-limit"),
    ),
    (
        "radiation-transport.spectral-polarized-experiment",
        {
            "spectral": "correlated-k-or-prescribed-frequency",
            "transfer": "prescribed-rays",
            "polarization": "stokes-matrix-exponential",
            "sensor": "normalized-spectral-response",
        },
        ("correlated-k-reference", "polarized-transfer", "sensor-response"),
    ),
)

_REQUIRED_GATES = (
    "artifact-rights",
    "derivative-contract",
    "documentation-nonclaims",
    "lifecycle-restore",
    "resource-envelope",
    "runtime-distribution",
    "scientific-validation",
    "source-admission",
)


def radiation_transport_support_tuples() -> tuple[SupportTuple, ...]:
    return tuple(
        SupportTuple(capability, attributes) for capability, attributes, _ in _SPECS
    )


def radiation_transport_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_REQUIRED_GATES,
            released=False,
        )
        for support in radiation_transport_support_tuples()
    )


def _campaign(capability: str, criteria: tuple[str, ...], /) -> ScientificCampaign:
    slug = capability.replace(".", "-")
    calibration = ScientificCase(
        f"{slug}:calibration",
        f"{slug}:independent-calibration",
        f"{slug}:construct-calibration",
        f"{slug}:condition-calibration",
        f"{slug}:preparation-calibration",
        f"{slug}:batch-calibration",
        (f"source:{slug}:calibration",),
    )
    locked = ScientificCase(
        f"{slug}:locked",
        f"{slug}:independent-locked",
        f"{slug}:construct-locked",
        f"{slug}:condition-locked",
        f"{slug}:preparation-locked",
        f"{slug}:batch-locked",
        (f"source:{slug}:locked",),
    )
    return ScientificCampaign(
        (calibration, locked),
        (
            CampaignRole("calibration", (calibration.case_id,)),
            CampaignRole("locked_evaluation", (locked.case_id,)),
        ),
        preprocessing_source_ids=(calibration.case_id,),
        criteria_ids=criteria,
    )


def radiation_transport_candidate_campaigns() -> tuple[ScientificCampaign, ...]:
    return tuple(_campaign(capability, criteria) for capability, _, criteria in _SPECS)


__all__ = [
    "radiation_transport_candidate_campaigns",
    "radiation_transport_candidate_profiles",
    "radiation_transport_support_tuples",
]
