#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased research capability profiles for medical-image physics."""

from __future__ import annotations

from collections.abc import Mapping

from ..qualification import CapabilityProfile, SupportTuple


_SPECS: dict[str, tuple[str, Mapping[str, str], tuple[str, ...]]] = {
    "imaging.ct.hu-material-calibration": (
        "diagnostic-photon-material-calibration",
        {
            "input": "governed-ct-number-image",
            "output": "density-and-material-fractions",
            "support": "closed-calibration-interval-no-extrapolation",
            "scope": "research-only",
        },
        ("source-admission", "unit-validity", "numerical-validity"),
    ),
    "imaging.ct.material-basis-polychromatic": (
        "diagnostic-photon-forward-model",
        {
            "transport": "primary-only-material-basis-beer-lambert",
            "acquisition": "per-view-source-filter-bowtie-aec-detector",
            "scatter": "external-explicit-label-only",
            "scope": "research-only",
        },
        (
            "source-admission",
            "numerical-validity",
            "differentiation",
            "locked-reference",
        ),
    ),
    "imaging.dicom.enhanced-ct": (
        "dicom-medical-image-admission",
        {
            "sop-profile": "enhanced-ct-regular-grid",
            "operation": "read-only-normalization",
            "coordinate-system": "lps-voxel-centre",
            "scope": "research-only-deidentified",
        },
        (
            "source-admission",
            "deidentification",
            "semantic-round-trip",
            "coordinate-validity",
        ),
    ),
    "imaging.dicom.legacy-ct": (
        "dicom-medical-image-admission",
        {
            "sop-profile": "legacy-ct-image-series-regular-grid",
            "operation": "read-only-normalization",
            "coordinate-system": "lps-voxel-centre",
            "scope": "research-only-deidentified",
        },
        (
            "source-admission",
            "deidentification",
            "semantic-round-trip",
            "coordinate-validity",
        ),
    ),
    "imaging.dicom.nuclear-medicine-counts": (
        "dicom-medical-image-admission",
        {
            "sop-profile": "nuclear-medicine-counts-image",
            "quantity": "stored-counts-not-activity",
            "scope": "research-only-deidentified",
        },
        (
            "source-admission",
            "deidentification",
            "semantic-round-trip",
            "coordinate-validity",
        ),
    ),
    "imaging.dicom.pet-activity-concentration": (
        "dicom-medical-image-admission",
        {
            "sop-profile": "pet-activity-concentration-image-series",
            "quantity": "activity-concentration-explicit-calibration",
            "scope": "research-only-deidentified",
        },
        (
            "source-admission",
            "deidentification",
            "semantic-round-trip",
            "coordinate-validity",
        ),
    ),
    "imaging.dicom.rt-dose-linked-plan": (
        "dicom-radiotherapy-admission",
        {
            "sop-profile": "rt-dose-linked-plan",
            "reference-graph": "closed",
            "operation": "read-only-normalization",
            "scope": "research-only-deidentified",
        },
        (
            "source-admission",
            "deidentification",
            "semantic-round-trip",
            "reference-closure",
        ),
    ),
    "imaging.dicom.rt-dose-unlinked": (
        "dicom-radiotherapy-admission",
        {
            "sop-profile": "rt-dose-unlinked",
            "reference-graph": "unlinked-explicit",
            "operation": "read-only-normalization",
            "scope": "research-only-deidentified",
        },
        (
            "source-admission",
            "deidentification",
            "semantic-round-trip",
            "coordinate-validity",
        ),
    ),
    "imaging.dicom.rt-plan-metadata": (
        "dicom-radiotherapy-admission",
        {
            "sop-profile": "rt-plan-metadata",
            "operation": "identity-and-reference-admission-only",
            "delivery-model": "not-supported",
            "scope": "research-only-deidentified",
        },
        (
            "source-admission",
            "deidentification",
            "semantic-round-trip",
            "reference-closure",
        ),
    ),
    "imaging.dicom.rt-structure-closed-planar": (
        "dicom-radiotherapy-admission",
        {
            "sop-profile": "rt-structure-set-closed-planar",
            "operation": "contour-admission-no-rasterization",
            "scope": "research-only-deidentified",
        },
        (
            "source-admission",
            "deidentification",
            "semantic-round-trip",
            "coordinate-validity",
        ),
    ),
}


def imaging_candidate_profile(name: str, /) -> CapabilityProfile:
    """Return one exact unreleased medical-imaging capability profile."""

    if name not in _SPECS:
        raise ValueError("Unknown imaging candidate profile.")
    capability, attributes, gates = _SPECS[name]
    return CapabilityProfile(
        name,
        "phydrax",
        "candidate",
        (SupportTuple(capability, attributes),),
        required_gates=gates,
        released=False,
    )


def imaging_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return all candidate profiles in deterministic name order."""

    return tuple(imaging_candidate_profile(name) for name in sorted(_SPECS))


__all__ = ["imaging_candidate_profile", "imaging_candidate_profiles"]
