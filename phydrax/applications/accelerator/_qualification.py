#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased accelerator capability declarations."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


_INSERTION_DEVICE_GATES = (
    "undulator-resonance-and-odd-harmonics",
    "harmonic-line-width",
    "matched-termination-exit-orbit",
    "exit-delay-convention",
    "bend-field-integral",
    "tabulated-map-interpolation-bound",
    "field-support-refusal",
    "resource-envelope",
    "documentation-nonclaims",
)

_RING_RADIATION_GATES = (
    "weak-focusing-closed-form-integrals",
    "isomagnetic-fodo-hill-equation-integrals",
    "published-radiation-constants-equilibrium",
    "photon-spectrum-mellin-moments",
    "zero-radiation-symplectic-identity",
    "classical-loss-per-turn",
    "partitioned-damping-rates",
    "stochastic-equilibrium-convergence",
    "identity-addressed-emission",
    "unsupported-lattice-refusal",
    "resource-envelope",
    "documentation-nonclaims",
)

_FEL_AVERAGED_GATES = (
    "one-dimensional-cold-growth-rate",
    "one-dimensional-energy-spread-dispersion",
    "ming-xie-gain-length",
    "saturation-power-scale",
    "quiet-start-and-fawley-shot-noise",
    "harmonic-bessel-coupling",
    "particle-field-energy-ledger",
    "taper-and-focusing-transfer",
    "per-slice-wake-loss",
    "identity-addressed-loading",
    "unsupported-configuration-refusal",
    "documentation-nonclaims",
)


def _support_gates() -> tuple[tuple[SupportTuple, tuple[str, ...]], ...]:
    return (
        (
            SupportTuple(
                "accelerator.insertion-device-radiation",
                {
                    "fields": "planar-helical-undulator-wiggler-dipole-bend-tabulated-3d",
                    "field_model": "exact-vacuum-analytic-or-multilinear-table",
                    "frame": "straight-cartesian-map-frame",
                    "tracking": "lab-time-relativistic-push-per-lane-arrival",
                    "coordinates": "accelerator-convention-s-to-t",
                    "radiation": "vacuum-trajectory-radiation-far-field",
                    "precision": "float64-only",
                },
            ),
            _INSERTION_DEVICE_GATES,
        ),
        (
            SupportTuple(
                "accelerator.ring-radiation",
                {
                    "lattice": "periodic-cell-drift-quadrupole-sector-bend-thin-rf",
                    "bends": "horizontal-normal-entry-sector-optional-gradient",
                    "optics": "first-order-uncoupled-periodic",
                    "integrals": "sands-i1-i5-gauss-kronrod",
                    "radiation": "classical-ultrarelativistic-collinear-emission",
                    "spectrum": "synchrotron-function-photon-number-table",
                    "tracking": "sliced-bend-kicks-classical-or-poisson-photons",
                    "randomness": "particle-identity-addressed",
                    "precision": "float64-only",
                },
            ),
            _RING_RADIATION_GATES,
        ),
        (
            SupportTuple(
                "accelerator.fel-averaged",
                {
                    "model": "kmr-period-averaged-time-independent-slices",
                    "lattice": "planar-or-helical-modules-stepwise-taper-thin-quad-breaks",
                    "harmonics": "odd-planar-bessel-coupled-or-helical-fundamental",
                    "field": "one-dimensional-or-angular-spectrum-grid",
                    "integration": "strang-betatron-diffraction-rk4-source",
                    "loading": "quiet-start-fawley-shot-noise-identity-addressed",
                    "collective": "per-slice-longitudinal-wake-loss",
                    "precision": "float64-only",
                },
            ),
            _FEL_AVERAGED_GATES,
        ),
    )


def accelerator_support_tuples() -> tuple[SupportTuple, ...]:
    return tuple(support for support, _ in _support_gates())


def accelerator_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=gates,
            released=False,
        )
        for support, gates in _support_gates()
    )


__all__ = ["accelerator_candidate_profiles", "accelerator_support_tuples"]
