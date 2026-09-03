#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run a bounded research-only cardiovascular composition through public APIs."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np

from phydrax.applications import cardiovascular as cardio
from phydrax.discretization import (
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)
from phydrax.qualification import HMACSHA256TrustPolicy, SupportTuple


def _anatomy():
    coordinates = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ]
    )
    tetrahedra = np.asarray(
        [
            (0, 1, 3, 7),
            (0, 3, 2, 7),
            (0, 2, 6, 7),
            (0, 6, 4, 7),
            (0, 4, 5, 7),
            (0, 5, 1, 7),
        ],
        dtype=np.int32,
    )
    mesh = CellMesh.from_tetrahedra(coordinates, tetrahedra)
    points = coordinates[np.asarray(mesh.connectivity.faces)]
    role_faces = {
        "endocardium": np.flatnonzero(np.all(points[..., 0] == 0.0, axis=1)),
        "epicardium": np.flatnonzero(np.all(points[..., 0] == 1.0, axis=1)),
        "apex": np.flatnonzero(np.all(points[..., 2] == 0.0, axis=1)),
        "base": np.flatnonzero(np.all(points[..., 2] == 1.0, axis=1)),
        "posterior": np.flatnonzero(np.all(points[..., 1] == 0.0, axis=1)),
        "anterior": np.flatnonzero(np.all(points[..., 1] == 1.0, axis=1)),
    }
    shared = tuple(
        (first, second)
        for first in ("endocardium", "epicardium")
        for second in ("apex", "base", "posterior", "anterior")
    ) + tuple(
        (first, second)
        for first in ("apex", "base")
        for second in ("posterior", "anterior")
    )
    profile = cardio.anatomy.CardiacBoundaryProfile(
        "example-unit-cube",
        required_roles=tuple(role_faces),
        connected_roles=tuple(role_faces),
        disjoint_closure_pairs=(
            ("endocardium", "epicardium"),
            ("apex", "base"),
            ("posterior", "anterior"),
        ),
        shared_closure_pairs=shared,
        exhaustive=True,
    )
    roles = cardio.anatomy.CardiacBoundaryRoles(mesh, role_faces, profile=profile)
    fields = (
        cardio.anatomy.HarmonicCoordinatePlan(
            mesh,
            roles,
            (
                cardio.anatomy.HarmonicCoordinateSpec(
                    "transmural", "endocardium", "epicardium"
                ),
                cardio.anatomy.HarmonicCoordinateSpec("longitudinal", "apex", "base"),
            ),
        )
        .prepare(numeric_version="example")
        .solve()
        .commit()
    )
    microstructure = (
        cardio.anatomy.VentricularMicrostructurePlan("transmural", "longitudinal")
        .prepare(fields)
        .build()
        .commit()
    )
    if not bool(fields.evidence.all_successful & microstructure.evidence.all_successful):
        raise RuntimeError("Anatomy or microstructure evidence was not successful.")
    return mesh, fields, microstructure


def _ep_runtime(mesh, microstructure):
    finite_element = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec("activation", lagrange_element("tetrahedron", 1)),
    ).prepare()
    diffusivity = cardio.electrophysiology.CellwiseDiffusivity.from_fibers(
        microstructure.fiber, 0.2, 0.05
    )
    reaction = cardio.electrophysiology.AlievPanfilovParameters(
        0.05, 0.15, 8.0, 0.002, 0.2, 0.3, 12.9
    )
    return cardio.electrophysiology.PhenomenologicalMonodomainPlan(
        finite_element,
        diffusivity,
        reaction,
        pulses=(cardio.electrophysiology.CellStimulusPulse((0,), 0.0, 0.04, 2.0),),
    ).prepare(0.01)


def _blocked_release(case):
    support = SupportTuple(
        "cardiovascular.workflow",
        {
            "data_classification": "non-phi",
            "deployment": "local",
            "fidelity_route": "monodomain-lumped-circulation",
            "precision": "float64",
            "regulated_device": False,
        },
    )
    exclusions = (
        "clinical-decision-support",
        "diagnosis",
        "regulated-medical-device",
        "treatment",
    )
    claims = cardio.CardiovascularClaimsMatrix(
        (
            cardio.CardiovascularClaimDecision(
                support,
                cardio.CardiovascularClaimStatus.TECHNICAL_SUPPORT_CANDIDATE,
                ("research-grade-forward-simulation",),
                exclusions,
                "Scope is limited to qualified engineering simulation.",
            ),
        )
    )
    roles = cardio.CardiovascularReviewRoles(
        "author",
        "technical-reviewer",
        "validation-reviewer",
        "security-reviewer",
        "release-approver",
    )
    profile = cardio.CardiovascularCommercialSupportProfile(
        "cardiovascular.local-non-phi",
        "qualification-example",
        support,
        claims,
        cardio.CardiovascularResourcePolicy(3600, 2**30, 2**28, 1),
        cardio.CardiovascularPrivacyPolicy(maximum_retention_days=7),
        cardio.CardiovascularSecurityPolicy(
            authorized_reviewer_ids=(
                roles.technical_reviewer_id,
                roles.validation_reviewer_id,
                roles.security_reviewer_id,
            ),
            trusted_signer_ids=(roles.technical_reviewer_id,),
        ),
        cardio.CardiovascularUsePolicy(
            "Local, non-PHI engineering research and evaluation."
        ),
    )
    release_case = cardio.CardiovascularCaseManifest(
        case.case_id,
        case.anatomy_id,
        case.model_id,
        case.protocol_id,
        profile.profile_id,
        case.release_id,
        case.build_id,
        case.sbom_id,
        metadata={
            "data_classification": "non-phi",
            "intended_use": "engineering-evaluation",
        },
    )
    bundle = cardio.CardiovascularQualificationBundle(
        support,
        (),
        cardio.CardiovascularArtifactSet(()),
        (),
        (),
        roles,
        release_case,
        (),
    )
    trust = HMACSHA256TrustPolicy(
        {"example-untrusted-signer": b"example-key"},
        maximum_index_age=100,
        maximum_evidence_age=100,
    )
    candidate = cardio.evaluate_cardiovascular_release_candidate(
        profile, bundle, trust, {}, at_time=20
    )
    if candidate.qualified or candidate.commercial_ready or not candidate.blockers:
        raise RuntimeError("Incomplete commercial evidence was not rejected.")
    return candidate


def main() -> None:
    voltage = cardio.cardiovascular_quantity("transmembrane_potential")
    if cardio.CARDIOVASCULAR_QUANTITIES[voltage.name] is not voltage:
        raise RuntimeError("Quantity identity did not resolve canonically.")

    mesh, fields, microstructure = _anatomy()
    runtime = _ep_runtime(mesh, microstructure)
    case = cardio.CardiovascularCaseManifest(
        "example-case",
        fields.fields_id,
        runtime.plan.plan_id,
        "example-pacing",
        "research-only",
        "unreleased",
        "example-build",
        "example-sbom",
        metadata={"purpose": "research-use-only", "data_classification": "synthetic"},
    )
    repeated_case = cardio.CardiovascularCaseManifest(
        "example-case",
        fields.fields_id,
        runtime.plan.plan_id,
        "example-pacing",
        "research-only",
        "unreleased",
        "example-build",
        "example-sbom",
        metadata={"data_classification": "synthetic", "purpose": "research-use-only"},
    )
    if case.manifest_id != repeated_case.manifest_id:
        raise RuntimeError("Equivalent case manifests did not share an identity.")

    initial = runtime.initialize(
        jnp.zeros(runtime.plan.node_count), jnp.zeros(runtime.plan.node_count)
    )
    uninterrupted = cardio.electrophysiology.run_monodomain_steps(runtime, initial, 4)
    if not bool(uninterrupted.successful):
        raise RuntimeError("Phenomenological monodomain integration failed.")
    activation, _ = runtime.split(uninterrupted.state)
    observation_plan = cardio.electrophysiology.ActivationObservationPlan(
        runtime.plan.node_count, (0,), threshold=1.0e-9
    )
    observation_state = cardio.electrophysiology.initialize_activation_observation(
        observation_plan, runtime.split(initial)[0], time_ms=0.0
    )
    observation_candidate = cardio.electrophysiology.evaluate_activation_observation(
        observation_plan,
        observation_state,
        activation,
        float(uninterrupted.state.time_ms),
    )
    observation_state = cardio.electrophysiology.commit_activation_observation(
        observation_candidate, observation_state
    )
    activation_result = cardio.electrophysiology.activation_observation_result(
        observation_plan, observation_state
    )
    if not bool(observation_candidate.evidence.successful & activation_result.successful):
        raise RuntimeError("Activation evidence was not successful.")

    with TemporaryDirectory() as directory:
        prefix = cardio.electrophysiology.run_monodomain_steps(runtime, initial, 2)
        checkpoint = Path(directory) / "cardiovascular-example.phx"
        cardio.electrophysiology.write_monodomain_checkpoint(
            runtime, prefix.state, checkpoint
        )
        restored = cardio.electrophysiology.read_monodomain_checkpoint(
            runtime, checkpoint
        )
        resumed = cardio.electrophysiology.run_monodomain_steps(runtime, restored, 2)
    if resumed.state_id != uninterrupted.state_id or not np.array_equal(
        np.asarray(resumed.state.values), np.asarray(uninterrupted.state.values)
    ):
        raise RuntimeError("Checkpoint replay did not reproduce the uninterrupted state.")

    pressure = jnp.asarray([1.0, 3.0, 3.0, 1.0, 1.0])
    volume = jnp.asarray([3.0, 3.0, 1.0, 1.0, 3.0])
    pv_loop = cardio.observations.PressureVolumeLoopPlan(
        cardio.observations.TimeBase.uniform("example-pv", 5, 1.0),
        pressure_reference_kpa=0.0,
        reference_configuration="synthetic absolute chamber pressure",
        loop_id="example-pv-loop",
    ).evaluate(pressure, volume)
    circulation_work = cardio.circulation.pressure_volume_work(pressure, volume)
    if not bool(pv_loop.evidence.successful) or not np.allclose(
        np.asarray(pv_loop.external_work_mg_mm2_per_ms2), np.asarray(circulation_work)
    ):
        raise RuntimeError("Pressure-volume evidence disagreed with circulation work.")

    release_candidate = _blocked_release(case)
    print(
        json.dumps(
            {
                "case_manifest_id": case.manifest_id,
                "quantity_id": voltage.quantity_id,
                "microstructure_id": microstructure.microstructure_id,
                "replay_state_id": resumed.state_id,
                "activation_time_ms": float(activation_result.activation_times_ms[0]),
                "pv_work_mJ": float(pv_loop.external_work_mj),
                "commercial_ready": release_candidate.commercial_ready,
                "commercial_blockers": list(release_candidate.blockers),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
