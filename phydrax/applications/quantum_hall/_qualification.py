#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Candidate capability profiles for native quantum Hall workflows."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


_REQUIRED_GATES = (
    "scientific-validation",
    "resource-envelope",
    "lifecycle-restore",
    "runtime-distribution",
    "documentation-nonclaims",
)


def quantum_hall_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "quantum-hall.lattice.chern",
            {"models": "haldane-hofstadter", "route": "overlap-bundle"},
        ),
        SupportTuple(
            "quantum-hall.lattice.time-reversal-z2",
            {"model": "kane-mele", "route": "rank-two-wilson-flow"},
        ),
        SupportTuple(
            "quantum-hall.sphere.projected-ed",
            {"geometry": "haldane-sphere", "route": "matrix-free-fixed-lz"},
        ),
        SupportTuple(
            "quantum-hall.sphere.landau-level-mixing-vmc",
            {"geometry": "monopole-sphere", "spin": "fully-polarized"},
        ),
        SupportTuple(
            "quantum-hall.cylinder.finite-dmrg",
            {"geometry": "finite-cylinder", "symmetry": "abelian-charge"},
        ),
        SupportTuple(
            "quantum-hall.topology.real-space-bott",
            {"geometry": "finite-periodic", "disorder": "prescribed"},
        ),
        SupportTuple(
            "quantum-hall.transport.multiterminal-coherent",
            {"contacts": "periodic-principal-layer", "route": "landauer-buttiker"},
        ),
        SupportTuple(
            "quantum-hall.material.finite-width",
            {"interaction": "subband-coulomb-form-factor", "units": "si"},
        ),
        SupportTuple(
            "quantum-hall.landau-level.projected-higher",
            {"geometry": "haldane-sphere", "levels": "explicit-manifold"},
        ),
        SupportTuple(
            "quantum-hall.landau-level.mixing-effective",
            {"route": "second-order-schrieffer-wolff", "body_order": "two-three"},
        ),
        SupportTuple(
            "quantum-hall.landau-level.mixing-explicit",
            {"route": "finite-manifold-ed", "sector": "abelian-direct"},
        ),
        SupportTuple(
            "quantum-hall.sphere.multicomponent",
            {"components": "spin-valley-layer", "route": "component-explicit"},
        ),
        SupportTuple(
            "quantum-hall.torus.projected-ed",
            {"geometry": "magnetic-torus", "sector": "modular-momentum"},
        ),
        SupportTuple(
            "quantum-hall.disk.edge",
            {"geometry": "disk", "confinement": "explicit"},
        ),
        SupportTuple(
            "quantum-hall.cylinder.infinite-vumps",
            {"geometry": "infinite-cylinder", "route": "uniform-vumps"},
        ),
        SupportTuple(
            "quantum-hall.response.intrinsic-sheet-hall",
            {"dimension": "two", "units": "siemens"},
        ),
        SupportTuple(
            "quantum-hall.transport.probes",
            {"routes": "voltage-dephasing", "constraint": "zero-current"},
        ),
        SupportTuple(
            "quantum-hall.transport.scba-keldysh",
            {"route": "local-fock-scba", "conservation": "collision-balance"},
        ),
        SupportTuple(
            "quantum-hall.transport.markovian-open",
            {"route": "finite-liouvillian", "bath": "explicit"},
        ),
        SupportTuple(
            "quantum-hall.transport.localized-network",
            {"route": "conservative-rate-network", "claim": "mesoscopic"},
        ),
    )


def quantum_hall_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_REQUIRED_GATES,
            released=False,
        )
        for support in quantum_hall_support_tuples()
    )


__all__ = ["quantum_hall_candidate_profiles", "quantum_hall_support_tuples"]
