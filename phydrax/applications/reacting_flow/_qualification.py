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
        "reacting-flow.transport-properties",
        {
            "state": "all-species-ideal-mixture",
            "properties": "viscosity-conductivity-binary-diffusion",
            "routes": "power-law-kinetic-theory-certified-log-polynomial",
            "reuse": "accepted-state-error-bounded",
        },
        ("transport-reference-error", "diffusive-mass-enthalpy-closure"),
    ),
    (
        "reacting-flow.chemical-equilibrium",
        {
            "ensembles": "tp-tv-hp-uv-sp-sv",
            "phases": "ideal-gas-liquid-solid",
            "constraints": "elements-and-charge",
            "derivative": "fixed-active-phase-only",
        },
        ("equilibrium-balance", "equilibrium-reference-state"),
    ),
    (
        "reacting-flow.equilibrium-jumps",
        {
            "geometry": "normal-one-dimensional",
            "closure": "equilibrium-rankine-hugoniot",
            "branches": "shock-and-driven-cj",
        },
        ("jump-conservation", "jump-reference-state"),
    ),
    (
        "reacting-flow.chemical-explosive-modes",
        {
            "jacobian": "exact-mechanism-ad",
            "conservation": "nullspace-projected",
            "modes": "biorthogonal-tracked",
            "claim": "diagnostic-only",
        },
        ("cema-eigen-residual", "cema-reference-mode"),
    ),
    (
        "reacting-flow.low-mach-sdc",
        {
            "state": "face-velocity-rhoY-rhoh-p0-pi",
            "pressure": "constant-prescribed-or-closed",
            "time": "two-node-iterative-sdc",
            "transport": "periodic-mixture-averaged",
        },
        ("low-mach-conservation", "low-mach-eos-projection", "low-mach-order"),
    ),
    (
        "reacting-flow.amr-ale-production",
        {
            "amr": "post-reflux-synchronized-chemistry-no-level-subcycling",
            "ale": "fixed-connectivity-extensive-remap",
            "sources": "smooth-exact-work",
            "scheduling": "accepted-measured-chemistry-work",
        },
        ("amr-chemistry-conservation", "ale-gcl", "source-work-ledger"),
    ),
    (
        "reacting-flow.learned-chemical-transition",
        {
            "output": "stoichiometric-reaction-extents",
            "support": "feature-box-and-uncertainty",
            "fallback": "exact-mechanism-visible-per-lane",
            "derivative": "fixed-route-only",
        },
        ("learned-held-out-error", "learned-invariant-closure", "fallback-coverage"),
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


def reacting_flow_support_tuples() -> tuple[SupportTuple, ...]:
    return tuple(
        SupportTuple(capability, attributes) for capability, attributes, _ in _SPECS
    )


def reacting_flow_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_REQUIRED_GATES,
            released=False,
        )
        for support in reacting_flow_support_tuples()
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


def reacting_flow_candidate_campaigns() -> tuple[ScientificCampaign, ...]:
    return tuple(_campaign(capability, criteria) for capability, _, criteria in _SPECS)


__all__ = [
    "reacting_flow_candidate_campaigns",
    "reacting_flow_candidate_profiles",
    "reacting_flow_support_tuples",
]
