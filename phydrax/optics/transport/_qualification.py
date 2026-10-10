#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased optical-photon transport capability declarations."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


_OPTICAL_PHOTON_TRANSPORT_GATES = (
    "identity-addressed-launch-and-order-batch-invariance",
    "jones-vector-unit-norm-and-transverse",
    "roulette-expected-weight-and-closed-ledger",
    "time-of-flight-and-wavelength-carriage",
    "tissue-scenarios-rebaselined",
    "wiscombe-lorenz-mie-cases-and-optical-theorem",
    "mie-rayleigh-small-sphere-limit",
    "rayleigh-polarized-dipole-sampling",
    "spectral-table-interpolation-and-support",
    "wavelength-shifting-quanta-energy-timing",
    "polarized-fresnel-maxwell-boundary-conditions",
    "total-internal-reflection-phase-and-ellipticity",
    "unified-surface-branch-probabilities",
    "photodetection-qe-transit-spread-and-dark-counts",
    "hit-bank-order-invariance",
    "geant4-optical-oracle",
    "documentation-nonclaims",
)

_CHARGED_STEP_SOURCE_GATES = (
    "frank-tamm-water-yield",
    "cherenkov-threshold-and-cone",
    "cherenkov-polarization",
    "dispersive-band-quadrature",
    "birks-mean-and-scintillation-timing",
    "step-subdivision-and-parent-order-invariance",
    "atomic-capacity-refusal",
    "charged-step-to-detector-hits",
    "geant4-optical-oracle",
    "documentation-nonclaims",
)


def _support_gates() -> tuple[tuple[SupportTuple, tuple[str, ...]], ...]:
    return (
        (
            SupportTuple(
                "optics.transport.optical-photon-transport",
                {
                    "state": "position-direction-jones-wavelength-time-weight-identity",
                    "geometry": "triangle-media-fixed-capacity",
                    "media": "spectral-absorption-rayleigh-lorenz-mie-hg-wavelength-shift",
                    "surfaces": "polarized-fresnel-and-unified-finishes",
                    "variance_reduction": "roulette-and-expected-split-branching",
                    "detection": "qe-collection-transit-spread-spe-dark-count-hit-banks",
                    "randomness": "sample-address-identity-keyed",
                    "precision": "float64-only",
                },
            ),
            _OPTICAL_PHOTON_TRANSPORT_GATES,
        ),
        (
            SupportTuple(
                "optics.transport.charged-step-optical-sources",
                {
                    "processes": "cherenkov-scintillation",
                    "cherenkov_model": "dispersive-frank-tamm-linear-speed-steps",
                    "scintillation_model": "birks-multi-exponential-tabulated-spectra",
                    "inputs": "charged-step-banks-and-charged-trajectories",
                    "counts": "poisson-identity-addressed",
                    "allocation": "fixed-capacity-atomic-refusal",
                    "precision": "float64-only",
                },
            ),
            _CHARGED_STEP_SOURCE_GATES,
        ),
    )


def optical_transport_support_tuples() -> tuple[SupportTuple, ...]:
    return tuple(support for support, _ in _support_gates())


def optical_transport_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            (support,),
            required_gates=gates,
            released=False,
        )
        for support, gates in _support_gates()
    )


__all__ = [
    "optical_transport_candidate_profiles",
    "optical_transport_support_tuples",
]
