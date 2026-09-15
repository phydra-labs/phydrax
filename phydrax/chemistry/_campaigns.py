#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Leakage-controlled candidate campaigns for bounded chemistry capability promotion."""

from __future__ import annotations

from ..qualification import CampaignRole, ScientificCampaign, ScientificCase


def _case(
    case_id: str,
    construct_id: str,
    condition_id: str,
    role_suffix: str,
    source_manifest_id: str,
) -> ScientificCase:
    return ScientificCase(
        case_id,
        f"independent:{case_id}",
        construct_id,
        condition_id,
        f"preparation:{case_id}:{role_suffix}",
        f"batch:{case_id}:{role_suffix}",
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


def candidate_chemistry_qualification_campaigns() -> tuple[ScientificCampaign, ...]:
    """Return fixed candidate memberships; campaigns alone are not release evidence."""

    molecular = _campaign(
        _case(
            "chemistry-molecular-calibration-h2",
            "molecular-electronic-structure",
            "equilibrium-closed-shell",
            "calibration",
            "reference:analytic-sto3g-h2",
        ),
        _case(
            "chemistry-molecular-locked-open-shell",
            "molecular-electronic-structure",
            "stretched-and-open-shell",
            "locked",
            "reference:independent-open-shell-cc",
        ),
        (
            "energy-absolute-error",
            "force-central-difference-closure",
            "stationary-response-residual",
            "electron-and-spin-count",
        ),
    )
    excited = _campaign(
        _case(
            "chemistry-excited-calibration-two-level",
            "excited-response-and-vibronic",
            "isolated-roots",
            "calibration",
            "reference:analytic-two-level-response",
        ),
        _case(
            "chemistry-excited-locked-crossing",
            "excited-response-and-vibronic",
            "degenerate-crossing",
            "locked",
            "reference:independent-crossing-manifold",
        ),
        (
            "eigenpair-residual",
            "biorthogonality-or-symplectic-normalization",
            "state-overlap-continuity",
            "spectral-area-closure",
            "trajectory-norm-and-energy-ledger",
        ),
    )
    reaction_embedding = _campaign(
        _case(
            "chemistry-multiscale-calibration-polarizable-dimer",
            "reaction-and-multiscale",
            "fixed-partition",
            "calibration",
            "reference:analytic-polarizable-dimer",
        ),
        _case(
            "chemistry-multiscale-locked-adaptive-path",
            "reaction-and-multiscale",
            "adaptive-partition-crossing",
            "locked",
            "reference:independent-adaptive-qmmm-path",
        ),
        (
            "force-energy-closure",
            "mutual-polarization-residual",
            "partition-of-unity",
            "topology-epoch-identity",
            "forward-reverse-path-endpoints",
        ),
    )
    periodic = _campaign(
        _case(
            "chemistry-periodic-calibration-one-band",
            "periodic-electronic-and-lattice",
            "gamma-insulator",
            "calibration",
            "reference:analytic-one-band-crystal",
        ),
        _case(
            "chemistry-periodic-locked-metal-phonon",
            "periodic-electronic-and-lattice",
            "spin-metal-and-nonanalytic-phonon",
            "locked",
            "reference:independent-periodic-electronic-phonon",
        ),
        (
            "ewald-energy-force-stress",
            "kpoint-electron-and-spin-count",
            "scf-free-energy-residual",
            "acoustic-sum-rule",
            "berry-loop-unitarity",
            "quasiparticle-and-bse-residual",
        ),
    )
    return molecular, excited, reaction_embedding, periodic


__all__ = ["candidate_chemistry_qualification_campaigns"]
