#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased optical-photon transport capability declarations."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


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


def optical_transport_support_tuples() -> tuple[SupportTuple, ...]:
    return (
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
    )


def optical_transport_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_CHARGED_STEP_SOURCE_GATES,
            released=False,
        )
        for support in optical_transport_support_tuples()
    )


__all__ = [
    "optical_transport_candidate_profiles",
    "optical_transport_support_tuples",
]
