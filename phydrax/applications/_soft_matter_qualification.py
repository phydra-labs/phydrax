#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Application-leaf profiles for implemented soft-matter compositions."""

from __future__ import annotations

from ..qualification import (
    CampaignRole,
    CapabilityProfile,
    ScientificCampaign,
    ScientificCase,
    SupportTuple,
)


_SPECS = (
    (
        "soft-matter.passive-phase-field",
        {
            "models": "allen-cahn-and-cahn-hilliard",
            "discretization": "finite-element",
            "transaction": "energy-mass-evidence-atomic-rollback",
        },
    ),
    (
        "soft-matter.binary-free-energy-lbm",
        {
            "model": "bounded-model-h",
            "state": "hydrodynamic-and-conservative-phase-populations",
            "transaction": "conservation-energy-population-atomic",
        },
    ),
    (
        "soft-matter.passive-nematic-mac",
        {
            "model": "passive-beris-edwards",
            "coupling": "periodic-mac-work-adjoint",
            "activity": False,
        },
    ),
    (
        "soft-matter.atomistic-observables-protocols",
        {
            "observables": "structure-factor-msd-vacf-diffusion",
            "protocols": "lj-binary-glass-wca-colloid-linear-polymer",
            "kinetic_coarse_fidelity": False,
        },
    ),
    (
        "soft-matter.discrete-path-thermodynamics",
        {
            "input": "canonical-path-buffer",
            "weights": "separately-normalized-forward-reverse",
            "evidence": "first-law-detailed-balance-fluctuation-reversal-ess",
        },
    ),
    (
        "soft-matter.polymer-particle-kremer-grest",
        {
            "potential": "fene-energy-shifted-wca-optional-bending",
            "integrator": "baoab",
            "architecture": "explicit-realized-linear-chains",
            "hydrodynamic_interactions": False,
        },
    ),
    (
        "soft-matter.polymer-prism-hnc",
        {
            "dimension": "three-dimensional-isotropic",
            "closure": "hnc",
            "transform": "interior-dst-i",
            "root": "anderson-fixed-point",
        },
    ),
    (
        "soft-matter.polymer-prism-piecewise-closures",
        {
            "closures": "percus-yevick-msa-martynov-sarkisov",
            "hard_core": "explicit-mask",
            "continuation": "density-scale",
            "derivatives": "smooth-fixed-branch-only",
        },
    ),
    (
        "soft-matter.polymer-scft-fixed-cell",
        {
            "chain_model": "gaussian",
            "architectures": "linear-and-acyclic-branched",
            "domain": "periodic-fourier",
            "contour": "strang-2-or-richardson-strang-4",
        },
    ),
    (
        "soft-matter.polymer-scft-advanced",
        {
            "continuation": "interaction-scale",
            "derivatives": "implicit-fixed-branch",
            "cell": "isotropic-differentiable-scale",
            "symmetry": "declared-finite-group-projection-and-defect",
        },
    ),
    (
        "soft-matter.polymer-fts-partial-saddle",
        {
            "fields": "real-exchange",
            "constraint": "conditional-incompressibility-saddle",
            "noise": "addressed-fixed-step",
            "complex_langevin": False,
        },
    ),
    (
        "soft-matter.polymer-fts-complex-langevin",
        {
            "fields": "complex",
            "action": "holomorphic-polymer-field-action",
            "runtime": "bounded-complex-langevin",
            "claim": "finite-trajectory-only",
        },
    ),
    (
        "soft-matter.polymer-construction",
        {
            "input": "explicit-material-recipe",
            "lowering": "stable-id-atomistic-topology",
            "adapters": "loss-accounted-admitted-artifacts",
            "inference_from_coordinates": False,
        },
    ),
    (
        "soft-matter.polymer-reaction-epochs",
        {
            "nonperiodic": "fixed-particle-connectivity-rewrite",
            "periodic": "explicit-molecule-image-winding",
            "transaction": "immutable-replacement-and-ledger",
            "kinetics": False,
        },
    ),
    (
        "soft-matter.chromatin-atomistic-coupling",
        {
            "relation": "loop-extrusion",
            "coordinates": "live-stable-particle-map",
            "force": "symmetric-spring-kick",
            "checkpoint": "joint-replay",
        },
    ),
    (
        "soft-matter.overdamped-atomistic",
        {
            "mobility": "constant-diagonal",
            "noise": "stable-particle-addressed",
            "constraints": False,
            "hydrodynamic_interactions": False,
        },
    ),
    (
        "soft-matter.generalized-langevin-atomistic",
        {
            "transition": "fixed-discrete-memory",
            "fdt": "covariance-identity",
            "noise": "stable-particle-addressed",
            "constraints": False,
        },
    ),
    (
        "soft-matter.polymer-equilibrium-rheology",
        {
            "route": "green-kubo-shear-stress",
            "uncertainty": "block-standard-error",
            "stationarity": "split-window",
            "driven_flow": False,
        },
    ),
)
_GATES = (
    "scientific-validation",
    "resource-envelope",
    "lifecycle-restore",
    "runtime-distribution",
    "documentation-nonclaims",
)


def soft_matter_support_tuples() -> tuple[SupportTuple, ...]:
    return tuple(
        SupportTuple(capability, attributes) for capability, attributes in _SPECS
    )


def soft_matter_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_GATES,
            released=False,
        )
        for support in soft_matter_support_tuples()
    )


def _campaign(capability: str, /) -> ScientificCampaign:
    slug = capability.replace(".", "-")
    calibration = ScientificCase(
        f"{slug}-calibration",
        f"{slug}-calibration-unit",
        capability,
        "analytic-thermodynamic-control",
        f"{slug}-calibration-preparation",
        f"{slug}-calibration-batch",
        (f"source:{slug}:analytic",),
    )
    locked = ScientificCase(
        f"{slug}-locked",
        f"{slug}-locked-unit",
        capability,
        "independent-transaction-control",
        f"{slug}-locked-preparation",
        f"{slug}-locked-batch",
        (f"source:{slug}:independent",),
    )
    return ScientificCampaign(
        (calibration, locked),
        (
            CampaignRole("calibration", (calibration.case_id,)),
            CampaignRole("locked_evaluation", (locked.case_id,)),
        ),
        criteria_ids=(
            "thermodynamic-identity",
            "transactional-refusal",
            "resource-admission",
        ),
    )


def soft_matter_candidate_campaigns() -> tuple[ScientificCampaign, ...]:
    return tuple(_campaign(capability) for capability, _ in _SPECS)


__all__ = [
    "soft_matter_candidate_campaigns",
    "soft_matter_candidate_profiles",
    "soft_matter_support_tuples",
]
