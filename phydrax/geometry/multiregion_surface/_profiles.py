#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Candidate capability profiles owned by multiregion surface geometry."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


_SPECS = (
    (
        "multiregion-surface.geometry",
        {
            "topology": "labeled-non-manifold-triangles",
            "validation": "exact-predicates-bvh",
            "volumes": "signed-finite-regions",
            "periodic": False,
        },
    ),
    (
        "multiregion-surface.label-field-extraction",
        {
            "source": "uniform-or-sparse-hard-label-grid",
            "algorithm": "freudenthal-multilabel-marching-tetrahedra",
            "identity": "stable-label-to-explicit-region",
            "junctions": "non-manifold-pair-labeled",
            "validation": "exact-host-collision-certified",
            "authority": "one-way-seed-or-repair",
        },
    ),
    (
        "multiregion-surface.remeshing",
        {
            "events": "edge-split,edge-collapse,edge-flip",
            "transaction": "exact-guards-ccd-full-validation",
            "transfer": "sparse-conservative-sheet-and-region",
            "derivatives-across-event": False,
        },
    ),
    (
        "multiregion-surface.topology-events",
        {
            "events": "t1-pop,pinch,merge,region-split,burst",
            "search": "bounded-candidate-certified",
            "transaction": "lineage-conservative-transfer-atomic-rollback",
            "derivatives-across-event": False,
        },
    ),
)

_GATES = (
    "analytic-control",
    "topology-evidence",
    "qualification-campaign",
    "public-workflow",
    "documentation-nonclaims",
)


def multiregion_surface_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return all unreleased profiles owned by multiregion surface geometry."""
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, attributes),),
            required_gates=_GATES,
        )
        for name, attributes in _SPECS
    )


__all__ = ["multiregion_surface_candidate_profiles"]
