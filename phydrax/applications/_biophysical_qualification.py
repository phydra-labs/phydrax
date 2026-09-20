#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Narrow candidate capability profiles for experimentally qualified biophysics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..qualification import CapabilityProfile, SupportTuple


_PROFILE_SPECS: dict[str, tuple[str, Mapping[str, str], tuple[str, ...]]] = {
    "protein.stability.megascale-natural-small-domain.canonical": (
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
    "nucleic.strand-displacement.rna-to-dna.declared-condition.canonical": (
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
    "protein.coordinate-proposal.fixed-construct-standard-chemistry.canonical": (
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
    "rna.ensemble.adenine-riboswitch.declared-protocol.canonical": (
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
    "radiation.blood-dose.ctmc-deterministic": (
        "radiation-circulating-blood-dose",
        {
            "circulation": "finite-state-continuous-time-markov-chain",
            "integration": "exact-occupation-block-exponential",
            "dose-rate": "piecewise-constant-absorbed-water-or-medium",
            "scope": "research-only-no-biological-response",
        },
        (
            "source-admission",
            "unit-validity",
            "numerical-validity",
            "physiological-validity",
        ),
    ),
    "radiation.blood-dose.ctmc-stochastic": (
        "radiation-circulating-blood-dose",
        {
            "circulation": "finite-state-continuous-time-markov-chain",
            "realization": "exact-ssa-bounded-events",
            "dose-rate": "piecewise-constant-absorbed-water-or-medium",
            "scope": "research-only-no-biological-response",
        },
        (
            "source-admission",
            "unit-validity",
            "numerical-validity",
            "replay-validity",
            "locked-reference",
        ),
    ),
    "radiation.external-score.mcgpu-raw": (
        "radiation-external-score-admission",
        {
            "provider-profile": "mcgpu-raw-config",
            "execution": "forbidden-import-only",
            "uncertainty": "native-correlation-preserved",
            "scope": "research-only",
        },
        (
            "source-admission",
            "semantic-round-trip",
            "coordinate-validity",
            "locked-reference",
        ),
    ),
    "radiation.external-score.moqui-array": (
        "radiation-external-score-admission",
        {
            "provider-profile": "moqui-npz-or-embedded-mha",
            "execution": "forbidden-import-only",
            "uncertainty": "native-correlation-preserved",
            "scope": "research-only",
        },
        (
            "source-admission",
            "semantic-round-trip",
            "coordinate-validity",
            "locked-reference",
        ),
    ),
    "radiation.external-score.openxraymc-hdf5": (
        "radiation-external-score-admission",
        {
            "provider-profile": "openxraymc-hdf5",
            "execution": "forbidden-import-only",
            "uncertainty": "native-correlation-preserved",
            "scope": "research-only",
        },
        (
            "source-admission",
            "semantic-round-trip",
            "coordinate-validity",
            "locked-reference",
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
