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
        "condensed-matter.superconductivity.fermionic-bdg",
        {
            "nambu": "c-k-then-cdagger-minus-k",
            "phs": "class-d-tau-x-k",
            "pairing": "delta-k-equals-minus-delta-transpose-minus-k",
            "capacity": "caller-explicit-hard-policy",
        },
        ("bdg-phs", "bdg-gap"),
    ),
    (
        "condensed-matter.superconductivity.finite-channel-mean-field",
        {
            "channels": "finite-caller-supplied-antisymmetric",
            "ensemble": "fixed-mu-or-fixed-filling-distinct",
            "gauge": "positive-real-anchor",
            "capacity": "caller-explicit-hard-policy",
        },
        ("gap-self-consistency", "free-energy-closure"),
    ),
    (
        "condensed-matter.superconductivity.class-d-chern",
        {
            "dimension": 2,
            "manifold": "gapped-negative-energy",
            "topology": "canonical-periodic-first-chern",
            "refinement": "required",
            "capacity": "caller-explicit-hard-policy",
        },
        ("chern-quantization", "chern-refinement"),
    ),
    (
        "condensed-matter.superconductivity.london-thin-film",
        {
            "geometry": "planar-triangle-film",
            "state": "stream-function-sheet-current",
            "screening": "pearl-length",
            "constraints": "holes-fluxoids-terminal-currents",
        },
        ("london-current-continuity", "london-fluxoid", "london-inductance"),
    ),
    (
        "condensed-matter.superconductivity.gauge-covariant-gl",
        {
            "field": "complex-order-parameter-and-u1-links",
            "mesh": "fixed-simplicial",
            "gauge": "local-u1-exact",
            "dynamics": "static",
        },
        ("gl-gauge-invariance", "gl-vortex-flux", "gl-stationarity"),
    ),
    (
        "condensed-matter.superconductivity.tdgl",
        {
            "field": "complex-order-parameter-and-u1-links",
            "time": "transactional-gradient-flow",
            "events": "vortex-branch-explicit",
        },
        ("tdgl-energy-dissipation", "tdgl-restart"),
    ),
    (
        "condensed-matter.superconductivity.quasiclassical-equilibrium",
        {
            "dimension": 2,
            "spin": "degenerate-singlet",
            "boundary": "specular",
            "solver": "matsubara-riccati-self-consistent",
        },
        ("riccati-normalization", "bulk-gap", "quasiclassical-free-energy"),
    ),
    (
        "condensed-matter.superconductivity.retarded-spectroscopy",
        {
            "axis": "real-energy",
            "input": "converged-quasiclassical-branch",
            "observable": "dos-ldos",
        },
        ("retarded-causality", "spectral-gap"),
    ),
    (
        "condensed-matter.superconductivity.engineering-cable-quench",
        {
            "geometry": "one-dimensional-component-graph",
            "electric": "current-sharing-circuit",
            "thermal": "solid-contact-coolant",
            "events": "quench-and-protection",
        },
        ("cable-current-closure", "cable-energy-ledger", "quench-reference"),
    ),
    (
        "condensed-matter.superconductivity.fidelity-bridges",
        {
            "ladder": "bdg-quasiclassical-gl-london-cable",
            "claim": "evidence-not-automatic-parameter-inference",
        },
        ("bridge-common-support", "bridge-asymptotic-error"),
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


def superconductivity_support_tuples() -> tuple[SupportTuple, ...]:
    return tuple(
        SupportTuple(capability, attributes) for capability, attributes, _ in _SPECS
    )


def superconductivity_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_REQUIRED_GATES,
            released=False,
        )
        for support in superconductivity_support_tuples()
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


def superconductivity_candidate_campaigns() -> tuple[ScientificCampaign, ...]:
    return tuple(_campaign(capability, criteria) for capability, _, criteria in _SPECS)


__all__ = [
    "superconductivity_candidate_campaigns",
    "superconductivity_candidate_profiles",
    "superconductivity_support_tuples",
]
