from collections.abc import Mapping
from pathlib import Path

import pytest

from phydrax.atomistic._capability_closure import (
    atomistic_closure_matrices,
    atomistic_closure_requirement_names,
    EXTERNAL_PREFIX,
    NATIVE_UNQUALIFIED,
    OBSERVED,
)
from phydrax.qualification import (
    builtin_capability_catalog,
    builtin_omniphysics_closure_matrices,
    builtin_source_absorption_ledger,
    CapabilityClosureMatrix,
    CapabilityGapResolution,
    ClosureDisposition,
    ClosureState,
    validate_closure_catalog,
)


_ROOT = Path(__file__).resolve().parents[3]
_LEDGER = tuple(f"atomistic.c{index:02d}" for index in range(1, 23))


def _resolutions(
    matrix: CapabilityClosureMatrix,
) -> dict[str, CapabilityGapResolution]:
    by_id = {value.requirement_id: value for value in matrix.resolutions}
    return {
        value.requirement: by_id[value.requirement_id] for value in matrix.requirements
    }


def _matrix(
    family: str, evidence: Mapping[str, tuple[str, ...]] | None = None
) -> CapabilityClosureMatrix:
    return {value.family: value for value in atomistic_closure_matrices(evidence)}[family]


def test_every_ledger_row_is_classified_through_the_builtin_aggregation() -> None:
    catalog = builtin_capability_catalog()
    matrices = builtin_omniphysics_closure_matrices(catalog)
    atomistic = tuple(value for value in matrices if value.family in _LEDGER)
    assert tuple(value.family for value in atomistic) == _LEDGER
    assert validate_closure_catalog(catalog, matrices) == ()
    sources = {value.source_id for value in builtin_source_absorption_ledger().sources}
    for matrix in atomistic:
        assert matrix.classified
        assert matrix.state is ClosureState.CLASSIFIED
        assert not matrix.implementation_closed
        assert not matrix.release_closed
        for requirement in matrix.requirements:
            assert requirement.requirement.startswith(f"atomistic-{matrix.family[-3:]}-")
            assert requirement.required_evidence_dimensions
            assert set(requirement.source_ids) <= sources
            for path in requirement.required_benchmarks:
                assert (_ROOT / path).is_file()


def test_unobserved_rows_separate_unqualified_and_external() -> None:
    names = set(atomistic_closure_requirement_names())
    seen: set[str] = set()
    for matrix in atomistic_closure_matrices():
        for name, resolution in _resolutions(matrix).items():
            seen.add(name)
            assert resolution.capability_ids
            assert not resolution.evidence_ids
            assert not resolution.provider_ids
            assert not resolution.release_authorized
            assert resolution.disposition is not ClosureDisposition.IMPLEMENTED
            if resolution.disposition is ClosureDisposition.CANDIDATE:
                assert resolution.rationale == NATIVE_UNQUALIFIED
            else:
                assert resolution.disposition is ClosureDisposition.MISSING
                assert resolution.rationale.startswith(EXTERNAL_PREFIX)
    assert seen == names


def test_external_prerequisites_and_legitimate_boundaries_stay_explicit() -> None:
    rationales = {
        name: resolution.rationale
        for matrix in atomistic_closure_matrices()
        for name, resolution in _resolutions(matrix).items()
    }
    assert rationales["atomistic-c02-pretrained-chemical-accuracy"] == (
        f"{EXTERNAL_PREFIX}admitted-pretrained-weights-and-reference-data"
    )
    assert rationales["atomistic-c08-d3-element-table-fidelity"].startswith(
        EXTERNAL_PREFIX
    )
    assert rationales["atomistic-c12-physical-multi-host-execution"].startswith(
        EXTERNAL_PREFIX
    )
    assert rationales["atomistic-c22-signed-release-authorization"].startswith(
        EXTERNAL_PREFIX
    )
    # Special-site algebra exists natively; it is not demanded as a new sampler.
    assert (
        rationales["atomistic-c20-special-site-expansion-constrained-continuation"]
        == NATIVE_UNQUALIFIED
    )


@pytest.mark.parametrize(
    "name",
    [
        "atomistic-c04-tiled-reference-route",
        "atomistic-c12-pencil-reciprocal-distribution",
        "atomistic-c12-domain-dispersion-reciprocal-terms",
        "atomistic-c13-stage-by-domain-campaign",
        "atomistic-c19-distributed-generation-quotas",
        "atomistic-c20-reconstructible-packing-recipe",
    ],
)
def test_required_native_paths_are_unqualified_not_prerequisites(name: str) -> None:
    resolution = {
        key: value
        for matrix in atomistic_closure_matrices()
        for key, value in _resolutions(matrix).items()
    }[name]
    assert resolution.disposition is ClosureDisposition.CANDIDATE
    assert resolution.rationale == NATIVE_UNQUALIFIED
    assert not resolution.evidence_ids


def test_observed_evidence_resolves_only_the_named_obligation() -> None:
    name = "atomistic-c05-segmented-route-reduction"
    artifact = "sha256:" + "a" * 64
    matrix = _matrix("atomistic.c05", {name: (artifact,)})
    resolutions = _resolutions(matrix)
    resolved = resolutions[name]
    requirement = next(
        value for value in matrix.requirements if value.requirement == name
    )
    assert resolved.disposition is ClosureDisposition.IMPLEMENTED
    assert resolved.rationale == OBSERVED
    assert resolved.evidence_ids == (artifact,)
    assert resolved.actual_depth == requirement.minimum_depth
    assert not resolved.release_authorized
    other = resolutions["atomistic-c05-particle-grid-spline-transfer"]
    assert other.disposition is ClosureDisposition.CANDIDATE
    assert not matrix.implementation_closed


def test_complete_observed_row_closes_implementation_but_never_release() -> None:
    family = "atomistic.c05"
    names = tuple(_resolutions(_matrix(family)))
    evidence = {name: (f"sha256:{index:064x}",) for index, name in enumerate(names)}
    matrix = _matrix(family, evidence)
    assert matrix.implementation_closed
    assert matrix.state is ClosureState.IMPLEMENTATION_CLOSED
    assert not matrix.release_closed


def test_external_prerequisite_needs_its_own_observed_evidence() -> None:
    names = tuple(_resolutions(_matrix("atomistic.c08")))
    native = {
        name: ("sha256:" + "b" * 64,)
        for name in names
        if name != "atomistic-c08-d3-element-table-fidelity"
    }
    matrix = _matrix("atomistic.c08", native)
    assert not matrix.implementation_closed
    assert (
        _resolutions(matrix)["atomistic-c08-d3-element-table-fidelity"].disposition
        is ClosureDisposition.MISSING
    )


@pytest.mark.parametrize(
    "evidence",
    [
        {"atomistic-c99-unknown": ("sha256:" + "c" * 64,)},
        {"atomistic-c05-segmented-route-reduction": ()},
    ],
)
def test_unknown_or_empty_evidence_is_refused(
    evidence: dict[str, tuple[str, ...]],
) -> None:
    with pytest.raises(ValueError, match="nonempty known"):
        atomistic_closure_matrices(evidence)


def test_string_evidence_is_refused() -> None:
    with pytest.raises(TypeError, match="sequence of identifiers"):
        atomistic_closure_matrices(
            {"atomistic-c05-segmented-route-reduction": "sha256:" + "d" * 64}
        )
