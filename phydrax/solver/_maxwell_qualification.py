#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased Maxwell far-field, antenna, dispersion-audit, and moving-charge
capability declarations."""

from __future__ import annotations

from ..qualification import CapabilityProfile, SupportTuple


_REQUIRED_GATES = (
    "hertzian-dipole-pattern-and-energy",
    "grid-convergence",
    "nested-surface-agreement",
    "surface-poynting-energy-balance",
    "admissibility-refusals",
    "documentation-nonclaims",
)

_ANTENNA_REQUIRED_GATES = (
    "one-way-forward-backward-ratio",
    "paraxial-gaussian-waist-and-gouy-phase",
    "antenna-work-equals-injected-energy",
    "moving-antenna-doppler",
    "moving-antenna-second-order-leakage",
    "declared-magnetic-charge-three-dimensional-beam",
    "laser-envelope-adapter-round-trip",
    "admissibility-refusals",
    "documentation-nonclaims",
)

_DISPERSION_REQUIRED_GATES = (
    "vacuum-yee-analytic-dispersion",
    "audit-equals-runtime-phase-advance",
    "ade-continuum-limit",
    "courant-boundary",
    "appleton-hartree-continuum-and-cutoffs",
    "negative-index-band",
    "cherenkov-regime-masks",
    "passivity-and-power-balance",
    "admissibility-refusals",
    "documentation-nonclaims",
)


def maxwell_far_field_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "electromagnetics.maxwell-far-field",
            {
                "acquisition": "time-integral-trapezoid-positive-exponent",
                "surface": "structured-huygens-box-or-tetrahedral-face-set",
                "exterior": "declared-lossless-homogeneous",
                "admissibility": "no-cpml-overlap-no-surface-current",
                "outputs": "field-spectrum-coherency-stokes-spectral-energy",
            },
        ),
    )


def maxwell_far_field_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_REQUIRED_GATES,
            released=False,
        )
        for support in maxwell_far_field_support_tuples()
    )


def maxwell_antenna_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "electromagnetics.one-way-antenna",
            {
                "source": "sampled-plane-electric-and-magnetic-sheet",
                "placement": "smoothed-tfsf-quadratic-bspline-pairs",
                "medium": "declared-lossless-homogeneous-on-sheet-support",
                "motion": "normal-boost-in-scale-vacuum-with-convective-currents",
                "magnetic_charge": "declared-and-tracked-not-projected",
                "adapters": "pulse-envelope-focused-gaussian-openpmd-laser-envelope",
                "outputs": "cochain-currents-work-ledger-support-evidence",
            },
        ),
    )


def maxwell_antenna_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_ANTENNA_REQUIRED_GATES,
            released=False,
        )
        for support in maxwell_antenna_support_tuples()
    )


def maxwell_dispersion_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "electromagnetics.discrete-dispersion-audit",
            {
                "update": "executed-compatible-leapfrog-one-step-bloch-map",
                "grid": "uniform-structured-axes-declared-homogeneous-region",
                "media": "diagonal-lorentz-drude-magnetic-poles-magnetized-cold-plasma",
                "admissibility": "local-update-no-magnetic-projection-verified-stencil",
                "outputs": "multipliers-numerical-frequencies-cherenkov-regime-masks",
            },
        ),
    )


def maxwell_dispersion_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_DISPERSION_REQUIRED_GATES,
            released=False,
        )
        for support in maxwell_dispersion_support_tuples()
    )


_MOVING_CHARGE_COMMON_GATES = (
    "coincident-neutral-start-and-freeze",
    "continuity-gauss-and-power-ledger",
    "admissibility-refusals",
    "documentation-nonclaims",
)

_MOVING_CHARGE_GATES = {
    "electromagnetics.moving-charge-cherenkov": (
        "frank-tamm-spectral-power-per-length",
        "cone-angle-audit-and-continuum",
        "below-threshold-null",
        "reversed-cherenkov-backward-flow",
        "time-domain-equals-frequency-domain",
    ),
    "electromagnetics.moving-charge-transition": (
        "vacuum-orbit-trajectory-radiation-second-order",
        "pec-image-pair-trajectory-radiation",
        "ginzburg-frank-perfect-conductor",
        "matched-interface-null",
    ),
    "electromagnetics.moving-charge-smith-purcell": (
        "smith-purcell-wavelength-relation",
        "time-domain-equals-frequency-domain",
        "diffraction-radiation-rigorous-reference",
    ),
}


def maxwell_moving_charge_support_tuples() -> tuple[SupportTuple, ...]:
    shared = {
        "source": "prescribed-point-charges-charge-conserving-whitney-current",
        "start": "coincident-neutral-compensating-charge",
        "stop": "freeze-after-trajectory-or-boundary-exit",
        "media": "linear-dispersive-lossy-heterogeneous-plasma-negative-index",
        "boundaries": "cpml-pec-pmc-impedance-and-interior-conductor-supports",
        "outputs": "fields-observers-power-ledger-constraint-evidence",
    }
    geometry = {
        "electromagnetics.moving-charge-cherenkov": "homogeneous-medium-uniform-motion",
        "electromagnetics.moving-charge-transition": "planar-interface-or-conductor",
        "electromagnetics.moving-charge-smith-purcell": "periodic-grating-or-aperture",
    }
    return tuple(
        SupportTuple(capability, {**shared, "geometry": geometry[capability]})
        for capability in _MOVING_CHARGE_GATES
    )


def maxwell_moving_charge_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=(
                *_MOVING_CHARGE_GATES[support.capability],
                *_MOVING_CHARGE_COMMON_GATES,
            ),
            released=False,
        )
        for support in maxwell_moving_charge_support_tuples()
    )


_FREQUENCY_MOVING_CHARGE_GATES = (
    "analytic-uniform-motion-field-maxwell-and-gauss",
    "frank-tamm-power-per-length",
    "line-charge-cherenkov-analytic",
    "below-threshold-null",
    "whitney-continuity-and-uniform-field-work",
    "scattered-field-equals-total-field",
    "ginzburg-frank-transition-radiation",
    "smith-purcell-wavelength-and-intensity",
    "krylov-direct-agreement-and-convergence-evidence",
    "admissibility-refusals",
    "documentation-nonclaims",
)


def maxwell_frequency_moving_charge_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "electromagnetics.frequency-moving-charge",
            {
                "source": "exact-whitney-edge-integral-uniform-motion",
                "formulation": "total-field-or-scattered-field-analytic-incident",
                "incident": "homogeneous-isotropic-dispersive-k0-k1-outgoing-hankel",
                "domain": "periodic-commensurate-path-or-clipped-with-open-endpoints",
                "solver": "cochain-krylov-or-sparse-direct-with-cfs-stretching",
                "fourier_modal": "moving-line-charge-bloch-and-point-charge-ky-quadrature",
                "outputs": "field-phasors-power-ledger-branch-and-domain-evidence",
            },
        ),
    )


def maxwell_frequency_moving_charge_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_FREQUENCY_MOVING_CHARGE_GATES,
            released=False,
        )
        for support in maxwell_frequency_moving_charge_support_tuples()
    )


__all__ = [
    "maxwell_antenna_candidate_profiles",
    "maxwell_antenna_support_tuples",
    "maxwell_dispersion_candidate_profiles",
    "maxwell_dispersion_support_tuples",
    "maxwell_far_field_candidate_profiles",
    "maxwell_far_field_support_tuples",
    "maxwell_frequency_moving_charge_candidate_profiles",
    "maxwell_frequency_moving_charge_support_tuples",
    "maxwell_moving_charge_candidate_profiles",
    "maxwell_moving_charge_support_tuples",
]
