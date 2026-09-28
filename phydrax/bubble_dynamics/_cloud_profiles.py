#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Candidate capability profile for bubble clouds and far-field emission."""

from __future__ import annotations

from ..qualification import CapabilityProfile, SupportTuple


def bubble_cloud_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return the unreleased bubble-cloud candidate; it graduates only via its campaign."""
    return (
        CapabilityProfile(
            "bubble-dynamics.cloud.profile",
            "phydrax",
            "candidate",
            (
                SupportTuple(
                    "bubble-dynamics.cloud",
                    {
                        "coupling": "implicit-incompressible,retarded-neutral-delay",
                        "routes": "dense-cholesky,fmm-conjugate-gradients",
                        "species": "static-groups-batched-parameters",
                        "translation": "added-mass-bjerknes",
                        "emission": "far-field-monopole-dense-output",
                        "interfaces": "smooth-only",
                    },
                ),
            ),
            required_gates=(
                "two-bubble-normal-modes",
                "bjerknes-sign-rule",
                "dense-direct-reference",
                "fmm-dense-agreement",
                "retarded-incompressible-limit",
                "work-identity",
                "public-workflow",
            ),
        ),
    )


__all__ = ["bubble_cloud_candidate_profiles"]
