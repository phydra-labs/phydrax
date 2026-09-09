#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Narrow candidate capability profiles for experimentally qualified biophysics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..qualification import CapabilityProfile, SupportTuple


_PROFILE_SPECS: dict[str, tuple[str, Mapping[str, str], tuple[str, ...]]] = {
    "protein.stability.megascale-natural-small-domain.v1": (
        "protein.mutation-stability-prediction",
        {
            "assay": "cDNA-display-proteolysis",
            "cohort": "megascale-natural-small-domain",
            "generalization": "held-out-domain-family-and-background",
            "observable": "mutation-stability-kcal-per-mol",
            "scope": "single-substitutions-only",
        },
        (
            "source-admission",
            "measurement-calibration",
            "parameter-identifiability",
            "predictive-calibration",
            "locked-prediction",
        ),
    ),
    "nucleic.strand-displacement.rna-to-dna.declared-condition.v1": (
        "nucleic.strand-displacement",
        {
            "chemistry": "RNA-invader-DNA-substrate",
            "generalization": "held-out-sequence-family-and-preparation",
            "observable": "raw-fluorescence-time-trace",
            "reporter": "independently-calibrated",
            "scope": "declared-temperature-buffer-concentration",
        },
        (
            "source-admission",
            "measurement-calibration",
            "parameter-identifiability",
            "predictive-calibration",
            "locked-prediction",
        ),
    ),
    "protein.coordinate-proposal.fixed-construct-standard-chemistry.v1": (
        "protein-coordinate-proposal",
        {
            "chemistry": "explicitly-supported-standard-residues",
            "equilibrium": "not-claimed",
            "generalization": "fixed-construct-only",
            "observable": "decoded-cartesian-coordinate-proposal",
            "qualification": "unfiltered-full-geometry",
        },
        (
            "source-admission",
            "numerical-validity",
            "chemical-validity",
        ),
    ),
    "rna.ensemble.adenine-riboswitch.declared-protocol.v1": (
        "rna-conditional-ensemble-inference",
        {
            "construct": "adenine-riboswitch",
            "generalization": "held-out-condition-and-perturbation",
            "observable": "mapped-mutation-profile",
            "protocol": "declared-chemical-mapping",
            "state-support": "finite-declared-hypotheses",
        },
        (
            "source-admission",
            "measurement-calibration",
            "parameter-identifiability",
            "predictive-calibration",
            "locked-prediction",
        ),
    ),
}


def biophysical_candidate_profile(
    name: str,
    /,
    *,
    version: str = "candidate",
    dependencies: Sequence[str] = (),
) -> CapabilityProfile:
    """Return one unreleased, evidence-free profile for an exact scientific scope."""
    if name not in _PROFILE_SPECS:
        known = ", ".join(sorted(_PROFILE_SPECS))
        raise ValueError(
            f"Unknown biophysical capability profile {name!r}; expected one of {known}."
        )
    capability, attributes, required_gates = _PROFILE_SPECS[name]
    support = SupportTuple(capability, attributes)
    return CapabilityProfile(
        name,
        "phydrax",
        version,
        (support,),
        dependencies=dependencies,
        required_gates=required_gates,
        released=False,
    )


def biophysical_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return all narrow candidate profiles in deterministic name order."""
    return tuple(biophysical_candidate_profile(name) for name in sorted(_PROFILE_SPECS))


__all__ = ["biophysical_candidate_profile", "biophysical_candidate_profiles"]
