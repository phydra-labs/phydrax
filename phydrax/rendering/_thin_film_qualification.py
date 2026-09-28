#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ..qualification import CapabilityProfile, SupportTuple


def thin_film_appearance_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    colorimetry = SupportTuple(
        "rendering.spectral-colorimetry",
        {
            "observer": "cie-1931-2-degree-analytic-or-hash-verified-table",
            "illuminant": "caller-array-or-hash-verified-host-resource",
            "encoding": "iec-61966-2-1-srgb",
            "evidence": "white-neutral-error-and-gamut-excess",
            "nonclaims": "no-bundled-cie-data-no-chromatic-adaptation",
        },
    )
    appearance = SupportTuple(
        "rendering.thin-film-appearance",
        {
            "inputs": "thickness-normal-view-light-and-explicit-support-arrays",
            "geometry": "specular-half-vector-or-uniform-environment",
            "composition": "thin-film-interference-colorimetry-and-surface-fields",
            "evidence": "per-sample-stage-and-support-status",
            "nonclaims": (
                "no-roughness-scattering-border-thickness-or-film-state-certification"
            ),
        },
    )
    return (
        CapabilityProfile(
            colorimetry.capability,
            "phydrax",
            "candidate",
            (colorimetry,),
            required_gates=(
                "srgb-transfer-endpoints",
                "d65-unit-reflector-neutral-white",
                "observer-fit-error-vs-cie-table",
                "equal-energy-white",
            ),
            released=False,
        ),
        CapabilityProfile(
            appearance.capability,
            "phydrax",
            "candidate",
            (appearance,),
            required_gates=(
                "interference-colorimetry-composition",
                "geometry-and-facing-status",
                "rendered-color-ramp-smoke",
                "plateau-sheet-slot-colors-and-state-nonmutation",
            ),
            released=False,
        ),
    )


__all__ = ["thin_film_appearance_candidate_profiles"]
