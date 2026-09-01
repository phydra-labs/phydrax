#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import phydrax as phx


class _Validator:
    def validate(self, token: str, /) -> phx.service.ValidatedPrincipal:
        assert token == "token"
        return phx.service.ValidatedPrincipal(
            "user",
            "tenant",
            "issuer",
            "audience",
            "client",
            "token-id",
            frozenset(
                {
                    "service:submit",
                    "service:status",
                    "service:execute",
                    "service:usage",
                }
            ),
            0,
            2**31,
        )


def _analysis_and_execution():
    analysis = phx.lifecycle.AnalysisPlan(
        "analysis",
        "provider-plan",
        "discretization",
        ("field-layout",),
    )
    execution = phx.lifecycle.ExecutionPlan(
        "execution",
        "cpu",
        "float64",
        "direct",
    )
    return analysis, execution


def test_lifecycle_model_round_trip(tmp_path: Path):
    values = np.arange(4.0)
    digest = phx.lifecycle.payload_digest(values)
    revision = phx.lifecycle.NumericRevision(digest, label="fixture")
    manifest = phx.lifecycle.ModelManifest(
        "model",
        "analysis",
        revision.revision_id,
        {"state": digest},
    )

    archive = phx.lifecycle.create(
        tmp_path / "model.zip",
        manifest=manifest,
        arrays={"state": values},
    )
    query = phx.lifecycle.query(archive, fields=("state",))

    assert phx.lifecycle.list_fields(archive) == ("state",)
    np.testing.assert_array_equal(query.fields[0].values, values)


def test_unit_square_nurbs_has_local_geometry_certificate():
    from phydrax.discretization import iga
    from phydrax.discretization.iga._certificate import (
        CertificateDisposition,
        certify_tensor_nurbs,
    )
    from phydrax.discretization.iga._volume import TensorNURBSVolume

    grid = iga.BSplineGrid.open_uniform(2, 1)
    coordinates = grid.greville_abscissae
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    geometry = iga.NURBSGeometryState(
        np.stack((xx, yy), axis=-1),
        np.ones((grid.coefficient_count, grid.coefficient_count)),
    )
    plan = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        geometry,
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    )

    certificate = certify_tensor_nurbs(
        TensorNURBSVolume("unit-square", plan.basis, geometry)
    )

    assert certificate.disposition is CertificateDisposition.PASS
    assert certificate.cells


def test_spline_de_rham_complex_squares_to_zero():
    from phydrax.discretization import iga
    from phydrax.discretization.iga._compatible import SplineDeRhamComplex

    grid = iga.BSplineGrid.open_uniform(2, 2)
    complex_ = SplineDeRhamComplex((grid, grid))

    np.testing.assert_allclose(complex_.d_squared_defects, 0.0, atol=1.0e-13)


def test_thb_certificate_and_goal_marking_are_deterministic():
    from phydrax.discretization.iga._adaptive import DWREstimate, QoICertificate
    from phydrax.discretization.iga._identity import OverlayCellId
    from phydrax.discretization.iga._thb import THBHierarchy, THBLevel

    hierarchy = THBHierarchy(
        (
            THBLevel(0, "coarse", (True,), (True, False)),
            THBLevel(1, "fine", (True, True), (False, True, True)),
        ),
        (np.asarray(((1.0, 0.0), (0.5, 0.5), (0.0, 1.0))),),
    )
    certificate = hierarchy.certify()
    cells = (
        OverlayCellId("overlay", (0,)),
        OverlayCellId("overlay", (1,)),
    )
    estimate = DWREstimate(
        cells,
        (0.25, -0.75),
        pollution=(("algebraic", 0.0), ("quadrature", 0.0)),
    )

    assert certificate.passed
    assert estimate.mark_dorfler(0.5) == (cells[1],)
    assert QoICertificate(
        "integral",
        "H1",
        continuity_bound=1.0,
        frechet_differentiable=True,
        trace_regular=True,
    ).passed
    assert not QoICertificate(
        "point",
        "H1",
        continuity_bound=1.0,
        frechet_differentiable=True,
        trace_regular=True,
        point_evaluation=True,
    ).passed


def test_capability_names_reject_pascal_case_namespaces():
    with pytest.raises(ValueError, match="lowercase dotted"):
        phx.qualification.SupportTuple(
            "IGA.Core.Tensor",
            {"backend": "cpu"},
        )


def test_capability_registry_signs_and_requires_profile():
    support = phx.qualification.SupportTuple(
        "iga.tensor",
        {"dimension": 2, "backend": "cpu", "precision": "float64"},
    )
    evidence = phx.qualification.ReleaseGateEvidence(
        "core",
        passed=True,
        evidence_ids=("evidence",),
        reviewer_id="reviewer",
        issued_at=1,
        expires_at=100,
    )
    profile = phx.qualification.CapabilityProfile(
        "iga.tensor",
        "phydrax",
        "canonical",
        (support,),
        required_gates=("core",),
        release_evidence=(evidence,),
        released=True,
    )
    signer = phx.qualification.HMACSHA256ReleaseSigner("signer", b"secret")
    trust = phx.qualification.HMACSHA256TrustPolicy(
        {"signer": b"secret"},
        maximum_index_age=100,
        maximum_evidence_age=100,
    )
    index = phx.qualification.ReleaseIndex.sign((profile,), signer, issued_at=1)

    selected = phx.qualification.require_profile(
        index,
        profile.profile_id,
        support,
        trust,
        at_time=2,
    )

    assert selected.profile_id == profile.profile_id


def test_linear_pod_rom_executes_and_audits_truth():
    cases = tuple(
        phx.rom.ROMCaseSpec(f"case-{index}", (("mu", float(index)),))
        for index in range(3)
    )

    def truth(case: phx.rom.ROMCaseSpec) -> phx.rom.TruthSample:
        mu = dict(case.parameters)["mu"]
        operator = np.eye(2)
        state = np.asarray((1.0, mu))
        return phx.rom.TruthSample(
            state,
            f"truth-{case.case_id}",
            operator=operator,
            rhs=state,
            dual_norm_inverse=np.eye(2),
            stability_lower_bound=1.0,
        )

    corpus = phx.rom.create_corpus(
        cases,
        truth,
        truth_model_id="truth-model",
        truth_model_revision="revision",
        split=phx.rom.CorpusSplit(("case-0", "case-1", "case-2")),
    )
    artifact = phx.rom.train_profile(corpus, phx.rom.LinearPODProfile(2))
    evaluation = phx.rom.evaluate(artifact, cases[1], truth_model=truth)
    audit = phx.rom.audit_against_truth(evaluation, truth(cases[1]))

    assert evaluation.source == "rom"
    assert audit.relative_state_error < 1.0e-12


def test_in_process_service_executes_authorized_provider():
    analysis, execution = _analysis_and_execution()
    service = phx.service.InProcessReferenceService(
        _Validator(),
        phx.service.ScopeTenantAuthorizer(),
        {
            "tenant": phx.service.TenantQuota(
                active_jobs=2,
                cpu_cores=4,
                memory_bytes=2**30,
                gpu_count=0,
                retained_artifact_bytes=2**20,
            )
        },
    )
    service.register_provider(
        "iga.tensor",
        lambda submission, context: phx.service.ProviderResult(("result",)),
    )
    submission = phx.service.JobSubmission(
        analysis,
        execution,
        "numeric-revision",
        "iga.tensor",
        {},
        phx.service.ResourceRequest(cpu_cores=1, memory_bytes=1024),
    )

    queued = service.submit("token", submission)
    completed = service.execute("token", queued.job_id)

    assert completed.state is phx.service.JobState.SUCCEEDED
    assert completed.run_record.result_ids == ("result",)
