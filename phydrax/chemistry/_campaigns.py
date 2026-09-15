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
    periodic = periodic_chemistry_qualification_campaigns()
    return molecular, excited, reaction_embedding, *periodic


def _periodic_campaign(
    capability: str,
    slug: str,
    calibration_condition: str,
    locked_condition: str,
    criteria_ids: tuple[str, ...],
) -> ScientificCampaign:
    return _campaign(
        _case(
            f"chemistry-{slug}-calibration",
            capability,
            calibration_condition,
            "calibration",
            f"reference:analytic-{slug}",
        ),
        _case(
            f"chemistry-{slug}-locked",
            capability,
            locked_condition,
            "locked",
            f"reference:independent-{slug}",
        ),
        criteria_ids,
    )


def periodic_chemistry_qualification_campaigns() -> tuple[ScientificCampaign, ...]:
    """Return one fixed campaign per physics-local periodic capability family."""

    specifications = (
        (
            "chemistry.periodic.pencil.orthonormal",
            "periodic-pencil-orthonormal",
            "analytic-chain",
            "oblique-complex-multiorbital",
            ("hermiticity-residual", "translation-reversal", "gauge-covariance"),
        ),
        (
            "chemistry.periodic.pencil.generalized",
            "periodic-pencil-generalized",
            "analytic-two-orbital-overlap",
            "oblique-complex-multiorbital-overlap",
            (
                "hermiticity-residual",
                "overlap-positive-definite",
                "cross-k-metric-covariance",
            ),
        ),
        (
            "chemistry.periodic.spectrum.bands",
            "periodic-spectrum-bands",
            "analytic-chain-dispersion",
            "oblique-multiband-pencil",
            ("eigenpair-residual", "band-gauge-covariance"),
        ),
        (
            "chemistry.periodic.spectrum.dos-pdos",
            "periodic-spectrum-dos-pdos",
            "analytic-density-of-states",
            "metric-projector-spectrum",
            ("dos-state-count", "pdos-projector-sum"),
        ),
        (
            "chemistry.periodic.spectrum.fermi-surface",
            "periodic-spectrum-fermi-surface",
            "single-sheet-metal",
            "anisotropic-multisheet-metal",
            ("mesh-connectivity", "lifshitz-and-unresolved-refusal"),
        ),
        (
            "chemistry.periodic.topology.wilson-zak",
            "periodic-topology-wilson-zak",
            "ssh-loop",
            "gauge-rotated-multiband-loop",
            ("gap-evidence", "link-conditioning", "gauge-invariant-spectrum"),
        ),
        (
            "chemistry.periodic.topology.first-chern",
            "periodic-topology-first-chern",
            "analytic-two-band-insulator",
            "gauge-rotated-oblique-insulator",
            ("gap-evidence", "link-conditioning", "mesh-refinement-stability"),
        ),
        (
            "chemistry.periodic.finite.realization",
            "periodic-finite-realization",
            "analytic-open-chain",
            "twisted-oblique-slab",
            ("matrix-free-dense-parity", "boundary-phase-covariance"),
        ),
        (
            "chemistry.interchange.wannier90-hr",
            "wannier90-hr",
            "analytic-hermitian-record",
            "complex-degenerate-translation-record",
            ("degeneracy-once", "translation-reversal", "malformed-input-refusal"),
        ),
        (
            "chemistry.interchange.wannier90-mmn",
            "wannier90-mmn",
            "analytic-neighbor-overlap",
            "complex-connected-mesh-overlap",
            (
                "connectivity-identity",
                "raw-overlap-preservation",
                "malformed-input-refusal",
            ),
        ),
        (
            "chemistry.periodic.electrostatics.ewald-neutral",
            "periodic-ewald-neutral",
            "madelung-calibration",
            "neutral-triclinic-cell",
            (
                "energy-reference-error",
                "force-stress-directional-closure",
                "shell-convergence",
            ),
        ),
        (
            "chemistry.periodic.electrostatics.ewald-background",
            "periodic-ewald-background",
            "charged-cell-homogeneous-background",
            "charged-triclinic-cell",
            (
                "background-energy-reference",
                "force-stress-directional-closure",
                "shell-convergence",
            ),
        ),
        (
            "chemistry.periodic.pseudopotential.gth-components",
            "periodic-gth-components",
            "analytic-local-component",
            "nonlocal-projector-components",
            ("component-reference-error", "source-manifest-identity"),
        ),
        (
            "chemistry.periodic.scf.ao-hubbard-orthonormal-insulator-restricted",
            "periodic-scf-ao-hubbard-orthonormal-insulator-restricted",
            "one-band-gapped-chain",
            "oblique-multiorbital-insulator",
            ("electron-count", "commutator-residual", "energy-ledger"),
        ),
        (
            "chemistry.periodic.scf.ao-hubbard-generalized-insulator-restricted",
            "periodic-scf-ao-hubbard-generalized-insulator-restricted",
            "two-orbital-overlap-insulator",
            "oblique-generalized-insulator",
            (
                "metric-electron-count",
                "generalized-commutator-residual",
                "energy-ledger",
            ),
        ),
        (
            "chemistry.periodic.scf.ao-hubbard-orthonormal-metal-restricted",
            "periodic-scf-ao-hubbard-orthonormal-metal-restricted",
            "one-band-finite-temperature-metal",
            "anisotropic-multiband-metal",
            ("electron-count", "free-energy-stationarity", "smearing-refinement"),
        ),
        (
            "chemistry.periodic.scf.ao-hubbard-orthonormal-metal-collinear",
            "periodic-scf-ao-hubbard-orthonormal-metal-collinear",
            "spin-split-finite-temperature-metal",
            "anisotropic-collinear-metal",
            (
                "electron-and-spin-count",
                "free-energy-stationarity",
                "commutator-residual",
            ),
        ),
        (
            "chemistry.periodic.scf.gamma-gdf-rhf",
            "periodic-scf-gamma-gdf-rhf",
            "supplied-integral-two-electron-cell",
            "oblique-multiorbital-supplied-integrals",
            ("electron-count", "factorization-residual", "energy-ledger"),
        ),
        (
            "chemistry.periodic.scf.provider-scalar-relativistic",
            "periodic-scf-provider-scalar-relativistic",
            "exact-provider-bound-crystal",
            "independent-provider-bound-crystal",
            ("request-result-identity", "provider-provenance", "refinement-closure"),
        ),
        (
            "chemistry.periodic.derivatives.stationary-force-stress",
            "periodic-stationary-force-stress",
            "stationary-energy-direction",
            "oblique-free-energy-cell-direction",
            (
                "force-directional-closure",
                "stress-directional-closure",
                "ledger-completeness",
            ),
        ),
        (
            "chemistry.periodic.scf.gamma-local-gth-lda-x",
            "periodic-scf-gamma-local-gth-lda-x",
            "gamma-insulator-local-only",
            "independent-local-gth-crystal",
            ("electron-count", "energy-reference-error", "grid-factor-refinement"),
        ),
        (
            "chemistry.periodic.many-body.diagonal-self-energy",
            "periodic-many-body-diagonal-self-energy",
            "analytic-diagonal-root",
            "provider-kernel-multiband-root",
            ("quasiparticle-root-residual", "provider-provenance"),
        ),
        (
            "chemistry.periodic.many-body.supplied-bse",
            "periodic-many-body-supplied-bse",
            "analytic-transition-kernel",
            "provider-kernel-multiband-exciton",
            (
                "eigenpair-residual",
                "transition-order-identity",
                "provider-provenance",
            ),
        ),
    )
    return tuple(_periodic_campaign(*specification) for specification in specifications)


__all__ = [
    "candidate_chemistry_qualification_campaigns",
    "periodic_chemistry_qualification_campaigns",
]
