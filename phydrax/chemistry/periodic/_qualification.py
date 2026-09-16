#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact unreleased profiles for implemented periodic chemistry capabilities."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple
from .._production_qualification import periodic_chemistry_support_tuples
from ._transport_support import periodic_transport_support_tuples


_REQUIRED_GATES = (
    "scientific-validation",
    "resource-envelope",
    "lifecycle-restore",
    "runtime-distribution",
    "documentation-nonclaims",
)


def _profile_name(support: SupportTuple, /) -> str:
    attributes = dict(support.attributes)
    suffix = (
        f".{attributes['boundary']}"
        if support.capability == "chemistry.periodic.finite.realization"
        else ""
    )
    return f"{support.capability}{suffix}.profile"


def periodic_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return one evidence-free profile per exact implemented support tuple."""

    supports = (
        *periodic_chemistry_support_tuples(),
        *periodic_transport_support_tuples(),
    )
    return tuple(
        CapabilityProfile(
            _profile_name(support),
            "phydrax",
            "candidate",
            (support,),
            required_gates=_REQUIRED_GATES,
            released=False,
        )
        for support in supports
    )


__all__ = ["periodic_candidate_profiles"]
