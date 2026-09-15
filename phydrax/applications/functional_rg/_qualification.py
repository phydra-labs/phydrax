#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased frontier functional-RG profile."""

from ...qualification import CapabilityProfile, SupportTuple


FERMION_PATCH_FRG_SUPPORT = SupportTuple(
    "functional-rg.fermionic-fermi-surface-patch",
    {
        "dimension": 2,
        "bands": "single",
        "spin_symmetry": "su2",
        "frequency": "static-vertex-finite-matsubara-loop",
        "regulator": "additive-linearized-energy-shell",
    },
)
FERMION_PATCH_FRG_CANDIDATE = CapabilityProfile(
    "functional-rg.fermionic-fermi-surface-patch.candidate",
    "phydrax",
    "candidate",
    (FERMION_PATCH_FRG_SUPPORT,),
    required_gates=("crossing", "regulator", "locked-small-control", "resource-envelope"),
    released=False,
)


__all__ = ["FERMION_PATCH_FRG_CANDIDATE", "FERMION_PATCH_FRG_SUPPORT"]
