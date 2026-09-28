#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased vacuum trajectory-radiation capability declarations."""

from __future__ import annotations

from ..qualification import CapabilityProfile, SupportTuple


_REQUIRED_GATES = {
    "electromagnetics.vacuum-trajectory-radiation": (
        "larmor-and-schott-harmonics",
        "lienard-total-power",
        "retardation-stability",
        "route-agreement-within-reported-floor",
        "streaming-equals-offline",
        "boost-covariance",
        "derivative-contract",
        "resource-envelope",
        "documentation-nonclaims",
    ),
    "electromagnetics.near-zone-lienard-wiechert": (
        "heaviside-uniform-motion",
        "born-hyperbolic-motion",
        "hermite-convergence-order",
        "far-zone-equals-trajectory-waveform",
        "gauss-flux",
        "history-causality-exclusion-evidence",
        "chunk-invariance",
        "derivative-contract",
        "resource-envelope",
        "documentation-nonclaims",
    ),
    "electromagnetics.kinetic-dispersion": (
        "landau-damping-roots",
        "bernstein-perpendicular-dispersion",
        "whistler-anisotropy-growth",
        "cyclotron-maser-analytic-growth",
        "cold-limit-equals-stix",
        "analytic-continuation-and-parallel-sign",
        "harmonic-truncation-evidence",
        "shkarofsky-dnestrovskii-absorption",
        "root-failure-status",
        "documentation-nonclaims",
    ),
    "electromagnetics.magnetobremsstrahlung": (
        "bekefi-thermal-cyclotron-harmonics",
        "razin-single-particle-spectrum",
        "ordinary-extraordinary-polarization",
        "kirchhoff-detailed-balance",
        "mny96-equation-31-codata-2022",
        "harmonic-set-and-truncation-evidence",
        "continuous-route-error-versus-harmonic-sum",
        "evanescent-and-resonance-cone-status",
        "free-free-born-gaunt-and-gray-means",
        "documentation-nonclaims",
    ),
    "electromagnetics.plasma-rays": (
        "linear-ramp-cutoff-reflection",
        "appleton-hartree-turning-point",
        "implicit-midpoint-symplectic-second-order-energy",
        "graded-index-kick-drift-kick-agreement",
        "rotation-measure-integral",
        "ray-refractive-index-invariant",
        "twisted-field-mode-coupling-limits",
        "thermal-kirchhoff-saturation",
        "launch-and-coupling-refusals",
        "documentation-nonclaims",
    ),
}


def electromagnetic_radiation_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "electromagnetics.vacuum-trajectory-radiation",
            {
                "medium": "vacuum",
                "zone": "far-field",
                "fourier_convention": "exp-minus-i-omega-t-one-sided-energy",
                "routes": "segment-exact-segment-hermite-node-gridded-type3",
                "coherence": "coherent-incoherent-gaussian-or-tabulated-form-factor",
                "precision": "float64-only",
                "outputs": "field-spectrum-coherency-stokes-spectral-energy-waveform",
            },
        ),
        SupportTuple(
            "electromagnetics.near-zone-lienard-wiechert",
            {
                "medium": "vacuum",
                "zone": "near-and-far-zone-time-domain",
                "sources": "point-charges-outside-exclusion-radius",
                "retarded_solve": "index-bisection-then-toms748-implicit-derivative",
                "interpolation": "hermite-cubic-or-hermite-quintic",
                "history": "refuse-or-inertial-extrapolation",
                "precision": "float64-only",
                "outputs": "electric-magnetic-velocity-acceleration-fields-retarded-times",
            },
        ),
        SupportTuple(
            "electromagnetics.kinetic-dispersion",
            {
                "medium": "hot-magnetized-multispecies-plasma",
                "distributions": "drifting-bi-maxwellian-or-isotropic-maxwellian-weakly-relativistic",
                "susceptibility": "nonrelativistic-harmonic-sum-or-lowest-flr-shkarofsky",
                "continuation": "landau-contour-all-im-omega-both-parallel-signs",
                "weak_growth": "relativistic-resonance-ellipse-arbitrary-gyrotropic-cold-modes",
                "roots": "vector-newton-branch-continuation-with-failure-status",
                "precision": "float64-only",
            },
        ),
        SupportTuple(
            "electromagnetics.magnetobremsstrahlung",
            {
                "medium": "cold-magnetized-multispecies-plasma-collisionless-modes",
                "distributions": "juttner-power-law-kappa-tabulated-gyrotropic",
                "routes": "exact-harmonic-sum-or-exact-bessel-continuous-harmonic",
                "harmonics": "resonance-curve-intersection-including-anomalous-doppler",
                "polarization": "mode-resolved-stokes-with-cold-plasma-faraday",
                "thermal_routes": "mny96-stokes-i-si-and-born-free-free-gray-means",
                "precision": "float64-only",
            },
        ),
        SupportTuple(
            "electromagnetics.plasma-rays",
            {
                "medium": "collisionless-cold-magnetized-multispecies-analytic-profiles",
                "hamiltonian": "stix-quartic-group-time-normalized-mode-fixed-at-launch",
                "integrator": "implicit-midpoint-symplectic-or-separable-kick-drift-kick",
                "transfer": "stokes-over-ray-index-squared-weak-or-strong-mode-coupling",
                "coefficients": "mode-resolved-magnetobremsstrahlung-per-segment",
                "exclusions": "vacuum-mode-degeneracy-airy-turning-phase-collisional-rays",
                "precision": "float64-only",
            },
        ),
    )


def electromagnetic_radiation_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_REQUIRED_GATES[support.capability],
            released=False,
        )
        for support in electromagnetic_radiation_support_tuples()
    )


__all__ = [
    "electromagnetic_radiation_candidate_profiles",
    "electromagnetic_radiation_support_tuples",
]
