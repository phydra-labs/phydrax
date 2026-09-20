import phydrax as phx
from phydrax.qualification import CapabilityDepth, ClosureState


def test_exact_candidates_close_implementation_without_claiming_release():
    catalog = phx.qualification.builtin_capability_catalog()
    matrices = phx.qualification.builtin_omniphysics_closure_matrices(catalog)
    assert len(matrices) == 21
    assert phx.qualification.validate_closure_catalog(catalog, matrices) == ()
    assert all(matrix.classified for matrix in matrices)
    assert all(matrix.implementation_closed for matrix in matrices)
    assert all(not matrix.release_closed for matrix in matrices)
    assert all(matrix.state is ClosureState.IMPLEMENTATION_CLOSED for matrix in matrices)
    reduced = {
        "correlation",
        "frequency",
        "phoresis",
        "process-systems",
        "system-modeling",
    }
    for matrix in matrices:
        implementation_depth = max(
            requirement.minimum_depth for requirement in matrix.requirements
        )
        if matrix.family in reduced:
            assert implementation_depth is CapabilityDepth.REDUCED_SYSTEM
        else:
            assert implementation_depth >= CapabilityDepth.SPATIAL_SINGLE_PHYSICS
    assert all(
        not resolution.release_authorized
        for matrix in matrices
        for resolution in matrix.resolutions
    )


def test_promoted_candidate_tuples_are_exact_and_single_host_only():
    catalog = phx.qualification.builtin_capability_catalog()
    declarations = tuple(
        value
        for value in catalog.declarations
        if value.domain_maturity == "implementation-qualified-candidate"
    )
    assert len(declarations) == 21
    for declaration in declarations:
        assert declaration.disposition.value == "candidate"
        assert len(declaration.profiles) == 1
        profile = declaration.profiles[0]
        assert not profile.released
        assert len(profile.support_tuples) == 1
        attributes = dict(profile.support_tuples[0].attributes)
        assert attributes["depth"] in {
            "reduced-system",
            "spatial-single-physics",
            "spatial-coupled",
        }
        assert attributes["execution"] == "single-host-cpu-float64"
        assert "no-distributed-hardware-evidence" in declaration.nonclaims


def test_source_ledger_pins_and_reviews_every_reference():
    ledger = phx.qualification.builtin_source_absorption_ledger()
    assert phx.qualification.validate_source_coverage(ledger) == ()
    assert all(
        source.revision not in ("unpinned", "unresolved") for source in ledger.sources
    )
    assert all(source.archive_digest != "unresolved" for source in ledger.sources)
    assert all(source.licence_digest != "unresolved" for source in ledger.sources)
    assert all(len(source.relevant_documents) >= 2 for source in ledger.sources)


def test_production_evidence_runs_controls_refinements_and_applications():
    evidence = phx.qualification.builtin_omniphysics_qualification_evidence()
    assert evidence.to_record()["passed"]
    assert len(evidence.controls) == 21
    assert all(control.passed for control in evidence.controls)
    assert all(campaign.passed for campaign in evidence.refinements)
    assert all(validation.passed for validation in evidence.applications)
    assert all(provider.executed for provider in evidence.providers)
    assert evidence.to_record()["distributed_qualified"] == any(
        provider.qualifies_distributed for provider in evidence.providers
    )
