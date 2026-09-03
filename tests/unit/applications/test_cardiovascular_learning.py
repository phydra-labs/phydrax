#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.applications.cardiovascular._case import CardiovascularCaseManifest
from phydrax.applications.cardiovascular._execution import (
    CardiovascularCapacityManifest,
    CardiovascularExecutionManifest,
    CardiovascularSerialExecution,
)
from phydrax.applications.cardiovascular._quantities import cardiovascular_quantity
from phydrax.applications.cardiovascular.circulation._components import (
    PressureSource,
    Resistance,
)
from phydrax.applications.cardiovascular.circulation._network import (
    CirculationNetwork,
    initialize_consistent_state,
    prepare_consistent_initialization,
    PressureFlowConnection,
)
from phydrax.applications.cardiovascular.electrophysiology._aliev_panfilov import (
    AlievPanfilovParameters,
    AlievPanfilovState,
    evaluate_aliev_panfilov,
)
from phydrax.applications.cardiovascular.hemodynamics._domain import (
    FixedWallLumenRegion,
    HemodynamicsScaling,
)
from phydrax.applications.cardiovascular.hemodynamics._fixed_wall_lbm import (
    FixedWallLBMPlan,
)
from phydrax.applications.cardiovascular.hemodynamics._ports import (
    CirculationPortBinding,
    FlowTerminalPort,
    PressureTerminalPort,
    TerminalDirection,
    TerminalFace,
    TerminalPortValues,
)
from phydrax.applications.cardiovascular.hemodynamics._rheology import (
    NewtonianRheology,
)
from phydrax.applications.cardiovascular.mechanics._materials import (
    FiniteBulkCardiacMaterial,
)
from phydrax.applications.cardiovascular.personalization._cohorts import (
    adapt_complete_truth_to_rom,
    batch_fixed_topology_cohort,
    CardiovascularCohortSplit,
    CardiovascularTruthCase,
    CohortCaseStatus,
    DeidentifiedCohortIdentity,
    OODSplitPolicy,
    prepare_learning_cohort,
    SiteSplitPolicy,
    split_cardiovascular_cohort,
    SubjectSplitPolicy,
)
from phydrax.applications.cardiovascular.personalization._random_fields import (
    CanonicalCardiacCoordinates,
    CanonicalCoordinateAxis,
    CardiacRandomFieldRecipe,
)
from phydrax.applications.cardiovascular.personalization._reanalysis import (
    CirculationReanalysisRoute,
    ElectrophysiologyReanalysisRoute,
    FullNativeReanalysisPlan,
    FullNativeReanalysisRequest,
    HemodynamicsReanalysisRoute,
    MechanicsReanalysisRoute,
    NativeDomain,
    NativeDomainSolveReceipt,
    NativeReanalysisCandidate,
    ReanalysisStatus,
    run_full_native_reanalysis,
)
from phydrax.applications.cardiovascular.personalization._surrogates import (
    assess_surrogate_input,
    CardiacSurrogateCalibration,
    CardiacSurrogateProposalManifest,
    FixedTopologyReferenceGeometry,
    GenerativeGeometryCandidate,
    GeometryCandidateStatus,
    GeometryQualificationPolicy,
    propose_cardiac_surrogate,
    qualify_generative_geometry,
    SurrogateInputStatus,
    SurrogateProposalStatus,
    SurrogateRefusalPolicy,
)
from phydrax.discretization import (
    D3Q19,
    LatticeBoltzmannPlan,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.nn.operator import (
    FunctionSamples,
    OperatorAxis,
    OperatorBatch,
    OperatorTargetBatch,
)
from phydrax.rom import TruthSample


def _manifest(index: int) -> CardiovascularCaseManifest:
    return CardiovascularCaseManifest(
        f"case-{index}",
        f"anatomy-{index}",
        f"model-{index}",
        f"protocol-{index}",
        f"support-{index}",
        f"release-{index}",
        f"build-{index}",
        f"sbom-{index}",
    )


def _subject(index: int) -> DeidentifiedCohortIdentity:
    return DeidentifiedCohortIdentity(
        f"group-{index}",
        "deidentification-policy-v1",
        f"deidentification-receipt-{index}",
    )


def _truth_case(
    index: int,
    parameter: float,
    /,
    *,
    site: str = "site-a",
    ood_tags: tuple[str, ...] = (),
    probability_mass: float = 1.0,
) -> CardiovascularTruthCase:
    nodes = jnp.linspace(0.0, 1.0, 4)
    axis = OperatorAxis(
        "canonical",
        nodes,
        quadrature_weights=jnp.asarray([1.0, 2.0, 2.0, 1.0]) / 6.0,
    )
    source_values = parameter + nodes
    target_values = 2.0 * source_values
    batch = OperatorBatch(
        inputs={"parameter_field": FunctionSamples(values=source_values, axes=(axis,))},
        queries={"state": FunctionSamples(values=None, axes=(axis,))},
    )
    targets = OperatorTargetBatch.from_arrays({"voltage": target_values}, batch)
    return CardiovascularTruthCase(
        case_id=f"case-{index}",
        subject_identity=_subject(index),
        site_id=site,
        topology_id="topology-fixed",
        parameters=(("conductivity", parameter), ("stiffness", parameter + 1.0)),
        probability_mass=probability_mass,
        status=CohortCaseStatus.COMPLETE,
        case_manifest=_manifest(index),
        operator_batch=batch,
        operator_targets=targets,
        truth_sample=TruthSample(target_values, f"truth-{index}"),
        execution_manifest_id=f"execution-{index}",
        ood_tags=ood_tags,
        acquisition_order=float(index),
    )


def _cohort_cases() -> tuple[CardiovascularTruthCase, ...]:
    complete = tuple(
        _truth_case(
            index,
            0.25 * index,
            site="site-z" if index == 6 else "site-a",
            ood_tags=("rare-geometry",) if index == 6 else (),
        )
        for index in range(7)
    )
    invalid = CardiovascularTruthCase(
        case_id="case-7",
        subject_identity=_subject(7),
        site_id="site-a",
        topology_id="topology-fixed",
        parameters=(("conductivity", 0.2), ("stiffness", 1.2)),
        probability_mass=1.0,
        status=CohortCaseStatus.INCOMPLETE_SOLVE,
        case_manifest=_manifest(7),
    )
    return (*complete, invalid)


def test_fixed_topology_cohort_split_and_preprocessing_are_leakage_safe():
    cases = _cohort_cases()
    batched = batch_fixed_topology_cohort(cases)

    assert batched.dataset.size == 7
    assert batched.valid_probability == pytest.approx(7.0 / 8.0)
    assert batched.invalid_probability == pytest.approx(1.0 / 8.0)
    assert float(jnp.sum(batched.conditional_probability)) == pytest.approx(1.0)

    policy = OODSplitPolicy(("rare-geometry",), seed=19)
    first = split_cardiovascular_cohort(cases, policy)
    second = split_cardiovascular_cohort(cases, policy)
    assert first == second
    assert first.ood_test_ids == ("case-6",)
    assert set(first.all_ids) == {f"case-{index}" for index in range(7)}

    prepared = prepare_learning_cohort(cases, first)
    train_cases = {case.case_id: case for case in cases}
    expected_input_mean = jnp.mean(
        jnp.stack(
            [
                train_cases[case_id].operator_batch.input("parameter_field").values
                for case_id in first.train_ids
            ]
        )
    )
    assert jnp.allclose(
        prepared.normalization.input_values["parameter_field"].mean,
        expected_input_mean,
    )
    assert prepared.features.training_case_ids == first.train_ids
    assert prepared.ood_test is not None and prepared.ood_test.size == 1

    rom = adapt_complete_truth_to_rom(
        cases,
        first,
        truth_model_id="cardiac-native-stack",
        truth_model_revision="build-1",
    )
    assert len(rom.cases) == 7
    assert rom.manifest.split_id == rom.split.split_id
    assert {case.sample.truth_artifact_id for case in rom.cases} == {
        f"truth-{index}" for index in range(7)
    }

    site_split = split_cardiovascular_cohort(
        cases, SiteSplitPolicy(("site-z",), calibration_fraction=0.25, seed=3)
    )
    assert site_split.ood_test_ids == ("case-6",)
    assert "case-6" not in site_split.train_ids + site_split.calibration_ids

    repeated_subject = (
        cases[0],
        replace(cases[1], subject_identity=cases[0].subject_identity),
        *cases[2:],
    )
    subject_policy = SubjectSplitPolicy(0.6, 0.2, seed=11)
    subject_split = split_cardiovascular_cohort(repeated_subject, subject_policy)
    assert subject_split == split_cardiovascular_cohort(repeated_subject, subject_policy)
    partitions = (
        subject_split.train_ids,
        subject_split.calibration_ids,
        subject_split.interpolation_test_ids,
    )
    assert any(
        "case-0" in partition and "case-1" in partition for partition in partitions
    )

    leaked = CardiovascularCohortSplit(
        ("case-0", "case-2", "case-3"),
        ("case-1", "case-4"),
        ("case-5",),
        ("case-6",),
        "manual-leaking-split",
    )
    with pytest.raises(ValueError, match="leaks one deidentified subject"):
        prepare_learning_cohort(repeated_subject, leaked)

    with pytest.raises(ValueError, match="PHI or linkable"):
        DeidentifiedCohortIdentity(
            "patient-123",
            "deidentification-policy-v1",
            "deidentification-receipt-phi",
        )


def test_canonical_random_field_has_covariance_and_exact_replay():
    coordinates = CanonicalCardiacCoordinates(
        jnp.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.5],
                [0.0, 1.0],
                [0.5, 0.0],
                [0.5, 0.5],
                [0.5, 1.0],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],
            ]
        ),
        jnp.ones((9,)) / 9.0,
        (
            CanonicalCoordinateAxis("transmural", 0.0, 1.0),
            CanonicalCoordinateAxis("apicobasal", 0.0, 1.0),
        ),
        "topology-fixed",
    )
    recipe = CardiacRandomFieldRecipe(
        "myocardial-strain-heterogeneity",
        cardiovascular_quantity("strain"),
        0.0,
        0.2,
        (0.35, 0.5),
        5,
    )
    field = recipe.instantiate(coordinates)
    realization = field.realize(jr.key(4), sample_count=4096)
    sample = field.sample(realization)
    replay = field.sample(realization)
    diagnostics = field.diagnostics(realization)

    assert sample.values.shape == (4096, 9)
    assert jnp.array_equal(sample.values, replay.values)
    assert diagnostics.replay_exact
    assert diagnostics.coefficient_covariance_relative_error < 0.08
    assert diagnostics.pointwise_variance_relative_error < 0.08
    assert jnp.allclose(field.covariance.matrix, field.covariance.matrix.T)


def _qualified_geometry():
    reference = FixedTopologyReferenceGeometry(
        jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        jnp.asarray([[0, 1, 2, 3]]),
        "topology-fixed",
    )
    candidate_coordinates = 1.02 * reference.coordinates_mm
    candidate = GenerativeGeometryCandidate(
        candidate_coordinates,
        "topology-fixed",
        "generator-artifact",
        motion_coordinates_mm=jnp.stack(
            [1.01 * candidate_coordinates, 1.02 * candidate_coordinates]
        ),
    )
    evidence = qualify_generative_geometry(
        reference,
        candidate,
        GeometryQualificationPolicy(
            minimum_cell_measure_ratio=0.5,
            maximum_displacement_mm=1.0,
            maximum_motion_increment_mm=1.0,
        ),
    )
    return reference, candidate, evidence


def test_calibrated_surrogate_refuses_ood_and_native_reanalysis_is_only_authority():
    cases = _cohort_cases()
    split = split_cardiovascular_cohort(
        cases, OODSplitPolicy(("rare-geometry",), seed=19)
    )
    prepared = prepare_learning_cohort(cases, split)
    preprocessing = prepared.features
    calibration_count = len(split.calibration_ids)
    location = jnp.zeros((calibration_count, 3))
    raw_scale = jnp.full((calibration_count, 3), 0.2)
    target = jnp.tile(jnp.asarray([[0.10, -0.10, 0.20]]), (calibration_count, 1))
    calibration = CardiacSurrogateCalibration.fit(
        location,
        raw_scale,
        target,
        prepared,
        alpha=0.5,
    )
    assert calibration.calibration_case_ids == split.calibration_ids
    with pytest.raises(ValueError, match="exactly match"):
        CardiacSurrogateCalibration.fit(
            jnp.zeros((calibration_count + 1, 3)),
            jnp.ones((calibration_count + 1, 3)),
            jnp.ones((calibration_count + 1, 3)),
            prepared,
            alpha=0.5,
        )
    quantity = cardiovascular_quantity("pressure")
    manifest = CardiacSurrogateProposalManifest(
        "trained-operator-artifact",
        "operator-contract",
        "truth-corpus",
        calibration.split_id,
        preprocessing.preprocessing_id,
        prepared.preparation_id,
        calibration.calibration_id,
        "topology-fixed",
        (quantity,),
    )
    case_by_id = {case.case_id: case for case in cases}
    supported_case = case_by_id[split.train_ids[0]]
    reference, geometry, geometry_evidence = _qualified_geometry()
    assert geometry_evidence.status is GeometryCandidateStatus.QUALIFIED
    inverted = GenerativeGeometryCandidate(
        reference.coordinates_mm[jnp.asarray([0, 2, 1, 3])],
        "topology-fixed",
        "generator-inverted",
    )
    inverted_evidence = qualify_generative_geometry(
        reference, inverted, GeometryQualificationPolicy()
    )
    assert inverted_evidence.status is GeometryCandidateStatus.INVERTED_OR_DEGENERATE

    proposal = propose_cardiac_surrogate(
        manifest,
        supported_case.parameters,
        jnp.zeros((3,)),
        jnp.full((3,), 0.2),
        preprocessing,
        calibration,
        SurrogateRefusalPolicy(2.0),
        topology_id="topology-fixed",
        geometry_evidence=geometry_evidence,
    )
    assert proposal.status is SurrogateProposalStatus.QUALIFIED_FOR_REANALYSIS
    assert not proposal.accepted

    preflight = assess_surrogate_input(
        manifest,
        (("conductivity", 100.0), ("stiffness", 200.0)),
        preprocessing,
        topology_id="topology-fixed",
    )
    assert preflight.status is SurrogateInputStatus.OOD_REFUSAL

    refused = propose_cardiac_surrogate(
        manifest,
        (("conductivity", 100.0), ("stiffness", 200.0)),
        jnp.zeros((3,)),
        jnp.full((3,), 0.2),
        preprocessing,
        calibration,
        SurrogateRefusalPolicy(2.0),
        topology_id="topology-fixed",
        geometry_evidence=geometry_evidence,
    )
    assert refused.status is SurrogateProposalStatus.OOD_REFUSAL
    assert not refused.accepted

    plan = FullNativeReanalysisPlan(
        ElectrophysiologyReanalysisRoute("ep-native", "ep-mesh", "cell-model", 0.02),
        MechanicsReanalysisRoute("mechanics-native", "mechanics-mesh", "material-model"),
        CirculationReanalysisRoute("circulation-native", "network", 0.1),
        HemodynamicsReanalysisRoute("flow-native", "flow-mesh", "fsi-coupling"),
        (quantity,),
    )
    request = FullNativeReanalysisRequest.from_proposal(
        _manifest(20), plan, proposal, supported_case.parameters, geometry
    )

    electrophysiology = evaluate_aliev_panfilov(
        AlievPanfilovParameters(0.05, 0.1, 8.0, 0.01, 0.2, 0.3, 1.0),
        AlievPanfilovState(jnp.asarray([0.2]), jnp.asarray([0.01])),
        activation_source_per_ms=jnp.asarray([0.1]),
    )
    passive_material = FiniteBulkCardiacMaterial(
        lambda deformation: (
            0.5 * jnp.sum((deformation - jnp.eye(3, dtype=deformation.dtype)) ** 2)
        ),
        100.0,
        energy_id="test-passive-material",
    )
    mechanics = passive_material.evaluate(1.01 * jnp.eye(3))
    circulation_network = CirculationNetwork(
        (PressureSource("pump", 10.0), Resistance("load", 2.0)),
        (
            PressureFlowConnection("pump", "outlet", "load", "inlet"),
            PressureFlowConnection("load", "outlet", "pump", "inlet"),
        ),
    )
    circulation = initialize_consistent_state(
        prepare_consistent_initialization(circulation_network)
    )
    grid = TensorGridPlan(
        tuple(UniformCellAxisSpec(count) for count in (6, 4, 4)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, 0.0), (6.0, 4.0, 4.0))))
    terminal_component = Resistance("terminal_resistance", 1.0)
    fixed_wall = FixedWallLBMPlan(
        LatticeBoltzmannPlan(grid, D3Q19()).prepare(),
        HemodynamicsScaling(
            1.0,
            1.0,
            1.06,
            reference_velocity_mm_per_ms=0.02,
        ),
        FixedWallLumenRegion(jnp.ones((6, 4, 4), dtype=bool)),
        (
            FlowTerminalPort(
                "inlet",
                TerminalFace("x", "lower", TerminalDirection.INTO_LUMEN),
                CirculationPortBinding(terminal_component, "inlet"),
            ),
            PressureTerminalPort(
                "outlet",
                TerminalFace("x", "upper", TerminalDirection.OUT_OF_LUMEN),
                CirculationPortBinding(terminal_component, "outlet"),
            ),
        ),
        NewtonianRheology(0.004, maximum_shear_rate_per_ms=1.0),
    ).prepare()
    hemodynamics = fixed_wall.candidate(
        fixed_wall.initialize_state(),
        TerminalPortValues(jnp.zeros((2,)), jnp.zeros((2,))),
    )
    domain_results = (electrophysiology, mechanics, circulation, hemodynamics)
    receipt_adapters = (
        NativeDomainSolveReceipt.from_electrophysiology,
        NativeDomainSolveReceipt.from_mechanics,
        NativeDomainSolveReceipt.from_circulation,
        NativeDomainSolveReceipt.from_hemodynamics,
    )
    receipts = []
    for domain, route, domain_result, receipt_adapter in zip(
        NativeDomain, plan.routes, domain_results, receipt_adapters, strict=True
    ):
        execution = CardiovascularExecutionManifest(
            case_manifest_id=request.case_manifest.manifest_id,
            analysis_plan_id=request.plan.plan_id,
            numeric_revision_id=f"numeric-{domain.value}",
            topology_id=request.topology_id,
            solver_policy_id=route.solver_id,
            precision_policy_id="float64",
            backend="jax-native",
            capacity=CardiovascularCapacityManifest(
                maximum_cohort_cases=1,
                maximum_state_values=4096,
                maximum_checkpoint_arrays=16,
                maximum_checkpoint_bytes=1_000_000,
                maximum_macro_steps=1,
                maximum_scheduled_steps=1,
                maximum_events=0,
                maximum_partitions=1,
            ),
            route=CardiovascularSerialExecution(),
        )
        receipts.append(receipt_adapter(route.route_id, execution, domain_result))

    def native_candidate(receipt_count: int = 4) -> NativeReanalysisCandidate:
        return NativeReanalysisCandidate(
            {quantity.name: hemodynamics.macroscopic.gauge_pressure_kpa},
            {quantity.name: quantity.quantity_id},
            receipts[:receipt_count],
            topology_id="topology-fixed",
            initialization_proposal_id=proposal.proposal_id,
        )

    accepted = run_full_native_reanalysis(request, lambda _: native_candidate())
    assert accepted.status is ReanalysisStatus.ACCEPTED
    assert accepted.accepted
    assert accepted.final_native_reanalysis
    assert jnp.array_equal(
        accepted.accepted_fields[quantity.name],
        hemodynamics.macroscopic.gauge_pressure_kpa,
    )
    assert not jnp.array_equal(
        accepted.accepted_fields[quantity.name], proposal.predicted_state
    )

    incomplete = run_full_native_reanalysis(
        request, lambda _: native_candidate(receipt_count=3)
    )
    assert incomplete.status is ReanalysisStatus.INCOMPLETE_SOLVE
    assert incomplete.accepted_fields is None
    assert not incomplete.accepted
