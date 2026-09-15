#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Leakage-controlled candidate campaigns for periodic electronic transport."""

from __future__ import annotations

from ...qualification import CampaignRole, ScientificCampaign, ScientificCase


def _case(
    case_id: str,
    construct_id: str,
    condition_id: str,
    role: str,
    source_manifest_id: str,
) -> ScientificCase:
    return ScientificCase(
        case_id,
        f"independent:{case_id}",
        construct_id,
        condition_id,
        f"preparation:{case_id}:{role}",
        f"batch:{case_id}:{role}",
        (source_manifest_id,),
    )


def _campaign(
    calibration: ScientificCase,
    locked: ScientificCase,
    criteria_ids: tuple[str, ...],
) -> ScientificCampaign:
    return ScientificCampaign(
        (calibration, locked),
        (
            CampaignRole("calibration", (calibration.case_id,)),
            CampaignRole("locked_evaluation", (locked.case_id,)),
        ),
        preprocessing_source_ids=(calibration.case_id,),
        criteria_ids=criteria_ids,
    )


def periodic_transport_candidate_campaigns() -> tuple[ScientificCampaign, ...]:
    """Return predeclared Kubo and Boltzmann campaigns; no release is implied."""

    kubo = _campaign(
        _case(
            "periodic-kubo-calibration-two-level",
            "periodic-independent-particle-kubo",
            "gapped-two-level-f-sum",
            "calibration",
            "reference:analytic-two-level-kubo",
        ),
        _case(
            "periodic-kubo-locked-anisotropic-gauge",
            "periodic-independent-particle-kubo",
            "anisotropic-degenerate-gauge-rotation",
            "locked",
            "reference:independent-anisotropic-kubo",
        ),
        (
            "retarded-fourier-and-charge-current-sign",
            "regular-response-passivity",
            "independent-diamagnetic-f-sum",
            "drude-distribution-separation",
            "gauge-energy-shift-invariance",
            "degenerate-subspace-unitary-covariance",
            "conserved-collinear-spin-closure",
            "linewidth-is-not-relaxation-time",
        ),
    )
    boltzmann = _campaign(
        _case(
            "periodic-boltzmann-calibration-parabolic-band",
            "periodic-constant-tau-boltzmann",
            "isotropic-parabolic-thermoelectric",
            "calibration",
            "reference:analytic-parabolic-boltzmann",
        ),
        _case(
            "periodic-boltzmann-locked-anisotropic",
            "periodic-constant-tau-boltzmann",
            "anisotropic-gauge-shifted-full-rank",
            "locked",
            "reference:independent-anisotropic-boltzmann",
        ),
        (
            "positive-scalar-relaxation-time",
            "linear-relaxation-time-scaling",
            "gauge-energy-shift-invariance",
            "kelvin-onsager-reciprocity",
            "electrical-and-thermal-passivity",
            "cartesian-rank-refusal",
            "no-linewidth-to-relaxation-inference",
        ),
    )
    return kubo, boltzmann


__all__ = ["periodic_transport_candidate_campaigns"]
