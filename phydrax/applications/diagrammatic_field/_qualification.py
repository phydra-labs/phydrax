#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased fermionic diagrammatic-field profiles."""

from ...qualification import CapabilityProfile, SupportTuple


def _candidate(name, attributes, gates):
    support = SupportTuple(name, attributes)
    profile = CapabilityProfile(
        f"{name}.candidate",
        "phydrax",
        "candidate",
        (support,),
        required_gates=gates,
        released=False,
    )
    return support, profile


LATTICE_PARQUET_SUPPORT, LATTICE_PARQUET_CANDIDATE = _candidate(
    "diagrammatic-field.lattice-parquet",
    {
        "bands": "single",
        "channels": "ph-direct-ph-crossed-pp",
        "frequency_bank": "finite-cyclic",
    },
    ("crossing", "parquet-residual", "schwinger-dyson", "resource-envelope"),
)
SIGN_FREE_CTINT_SUPPORT, SIGN_FREE_CTINT_CANDIDATE = _candidate(
    "diagrammatic-field.sign-free-ct-int-control",
    {
        "model": "half-filled-bipartite-repulsive-hubbard",
        "determinant_order": "low-order",
    },
    ("proposal-ratio", "sign-free-symmetry", "raw-chain", "locked-small-control"),
)
LOW_ORDER_DIAGRAM_MC_SUPPORT, LOW_ORDER_DIAGRAM_MC_CANDIDATE = _candidate(
    "diagrammatic-field.low-order-fermion-diagram-monte-carlo",
    {"catalog": "finite-caller-supplied", "proposal": "reversible-metropolis-hastings"},
    ("proposal-ratio", "detailed-balance", "order-evidence", "raw-sign-evidence"),
)


__all__ = [
    "LATTICE_PARQUET_CANDIDATE",
    "LATTICE_PARQUET_SUPPORT",
    "LOW_ORDER_DIAGRAM_MC_CANDIDATE",
    "LOW_ORDER_DIAGRAM_MC_SUPPORT",
    "SIGN_FREE_CTINT_CANDIDATE",
    "SIGN_FREE_CTINT_SUPPORT",
]
