#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax.numpy as jnp
import jax.random as jr

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
    batch_fixed_topology_cohort,
    CardiovascularTruthCase,
    CohortCaseStatus,
    DeidentifiedCohortIdentity,
    OODSplitPolicy,
    prepare_learning_cohort,
    split_cardiovascular_cohort,
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
    run_full_native_reanalysis,
)
from phydrax.applications.cardiovascular.personalization._surrogates import (
    assess_surrogate_input,
    CardiacSurrogateCalibration,
    CardiacSurrogateProposalManifest,
    FixedTopologyReferenceGeometry,
    GenerativeGeometryCandidate,
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


def _case(index: int, parameter: float, *, ood: bool = False) -> CardiovascularTruthCase:
    axis = OperatorAxis(
        "canonical",
        jnp.linspace(0.0, 1.0, 4),
        quadrature_weights=jnp.full((4,), 0.25),
    )
    values = parameter + axis.nodes
    batch = OperatorBatch(
        inputs={"parameter_field": FunctionSamples(values=values, axes=(axis,))},
        queries={"state": FunctionSamples(values=None, axes=(axis,))},
    )
    target = 2.0 * values
    return CardiovascularTruthCase(
        f"case-{index}",
        _subject(index),
        "external-site" if ood else "development-site",
        "topology-fixed",
        (("conductivity", parameter), ("stiffness", parameter + 1.0)),
        1.0,
        CohortCaseStatus.COMPLETE,
        _manifest(index),
        batch,
        OperatorTargetBatch.from_arrays({"voltage": target}, batch),
        TruthSample(target, f"truth-{index}"),
        f"execution-{index}",
        ("rare-geometry",) if ood else (),
        float(index),
    )


cases = tuple(_case(index, 0.2 * index, ood=index == 6) for index in range(7))
invalid = CardiovascularTruthCase(
    "case-7",
    _subject(7),
    "development-site",
    "topology-fixed",
    (("conductivity", 0.5), ("stiffness", 1.5)),
    1.0,
    CohortCaseStatus.INCOMPLETE_SOLVE,
    _manifest(7),
)
all_cases = (*cases, invalid)
cohort = batch_fixed_topology_cohort(all_cases)
split = split_cardiovascular_cohort(
    all_cases, OODSplitPolicy(("rare-geometry",), seed=41)
)
prepared = prepare_learning_cohort(all_cases, split)
replayed_split = split_cardiovascular_cohort(
    all_cases, OODSplitPolicy(("rare-geometry",), seed=41)
)

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
    jnp.full((9,), 1.0 / 9.0),
    (
        CanonicalCoordinateAxis("transmural", 0.0, 1.0),
        CanonicalCoordinateAxis("apicobasal", 0.0, 1.0),
    ),
    "topology-fixed",
)
random_field = CardiacRandomFieldRecipe(
    "strain-heterogeneity",
    cardiovascular_quantity("strain"),
    0.0,
    0.2,
    (0.4, 0.5),
    5,
).instantiate(coordinates)
realization = random_field.realize(jr.key(17), sample_count=4096)
random_diagnostics = random_field.diagnostics(realization)
random_replay = jnp.array_equal(
    random_field.sample(realization).values,
    random_field.sample(realization).values,
)

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
generated = GenerativeGeometryCandidate(
    1.01 * reference.coordinates_mm,
    "topology-fixed",
    "qualified-generator-artifact",
    motion_coordinates_mm=jnp.stack(
        (1.02 * reference.coordinates_mm, 1.03 * reference.coordinates_mm)
    ),
)
geometry_evidence = qualify_generative_geometry(
    reference,
    generated,
    GeometryQualificationPolicy(
        minimum_cell_measure_ratio=0.5,
        maximum_displacement_mm=1.0,
        maximum_motion_increment_mm=1.0,
    ),
)

calibration_count = len(split.calibration_ids)
calibration = CardiacSurrogateCalibration.fit(
    jnp.zeros((calibration_count, 3)),
    jnp.full((calibration_count, 3), 0.2),
    jnp.tile(jnp.asarray([[0.10, -0.10, 0.20]]), (calibration_count, 1)),
    prepared,
    alpha=0.5,
)
features = prepared.features
supported_case = {case.case_id: case for case in cases}[split.train_ids[0]]
quantity = cardiovascular_quantity("pressure")
proposal_manifest = CardiacSurrogateProposalManifest(
    "qualified-operator",
    "operator-contract",
    "truth-corpus",
    calibration.split_id,
    features.preprocessing_id,
    prepared.preparation_id,
    calibration.calibration_id,
    "topology-fixed",
    (quantity,),
)
proposal = propose_cardiac_surrogate(
    proposal_manifest,
    supported_case.parameters,
    jnp.zeros((3,)),
    jnp.full((3,), 0.2),
    features,
    calibration,
    SurrogateRefusalPolicy(2.0),
    topology_id="topology-fixed",
    geometry_evidence=geometry_evidence,
)
ood_preflight = assess_surrogate_input(
    proposal_manifest,
    (("conductivity", 100.0), ("stiffness", 200.0)),
    features,
    topology_id="topology-fixed",
)
ood_proposal = propose_cardiac_surrogate(
    proposal_manifest,
    (("conductivity", 100.0), ("stiffness", 200.0)),
    jnp.zeros((3,)),
    jnp.full((3,), 0.2),
    features,
    calibration,
    SurrogateRefusalPolicy(2.0),
    topology_id="topology-fixed",
    geometry_evidence=geometry_evidence,
)

plan = FullNativeReanalysisPlan(
    ElectrophysiologyReanalysisRoute("ep-native", "ep-mesh", "cell-model", 0.02),
    MechanicsReanalysisRoute("mechanics-native", "mechanics-mesh", "material-model"),
    CirculationReanalysisRoute("circulation-native", "closed-loop", 0.1),
    HemodynamicsReanalysisRoute("flow-native", "flow-mesh", "fsi-coupling"),
    (quantity,),
)
request = FullNativeReanalysisRequest.from_proposal(
    _manifest(20), plan, proposal, supported_case.parameters, generated
)


def _native_solver(current: FullNativeReanalysisRequest) -> NativeReanalysisCandidate:
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
        energy_id="qualification-passive-material",
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
    discretization = LatticeBoltzmannPlan(grid, D3Q19()).prepare()
    terminal_component = Resistance("terminal_resistance", 1.0)
    fixed_wall = FixedWallLBMPlan(
        discretization,
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
        NativeDomain, current.plan.routes, domain_results, receipt_adapters, strict=True
    ):
        execution = CardiovascularExecutionManifest(
            case_manifest_id=current.case_manifest.manifest_id,
            analysis_plan_id=current.plan.plan_id,
            numeric_revision_id=f"qualification-{domain.value}",
            topology_id=current.topology_id,
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
    return NativeReanalysisCandidate(
        {quantity.name: hemodynamics.macroscopic.gauge_pressure_kpa},
        {quantity.name: quantity.quantity_id},
        receipts,
        topology_id=current.topology_id,
        initialization_proposal_id=current.proposal_id,
    )


reanalysis = run_full_native_reanalysis(request, _native_solver)
metrics = {
    "deterministic_split": split == replayed_split,
    "train_only_preprocessing": prepared.features.training_case_ids == split.train_ids,
    "invalid_probability": cohort.invalid_probability,
    "random_replay_exact": bool(random_replay and random_diagnostics.replay_exact),
    "coefficient_covariance_relative_error": random_diagnostics.coefficient_covariance_relative_error,
    "pointwise_variance_relative_error": random_diagnostics.pointwise_variance_relative_error,
    "geometry_qualified": geometry_evidence.qualified,
    "supported_proposal_only": proposal.qualified_for_reanalysis
    and not proposal.accepted,
    "ood_refused": (
        ood_preflight.status is SurrogateInputStatus.OOD_REFUSAL
        and ood_proposal.status is SurrogateProposalStatus.OOD_REFUSAL
    ),
    "native_reanalysis_accepted": reanalysis.accepted
    and reanalysis.final_native_reanalysis,
    "accepted_differs_from_learned": bool(
        not jnp.array_equal(
            reanalysis.accepted_fields[quantity.name], proposal.predicted_state
        )
    ),
}
passed = bool(
    metrics["deterministic_split"]
    and metrics["train_only_preprocessing"]
    and abs(metrics["invalid_probability"] - 0.125) < 1.0e-12
    and metrics["random_replay_exact"]
    and metrics["coefficient_covariance_relative_error"] < 0.08
    and metrics["pointwise_variance_relative_error"] < 0.08
    and metrics["geometry_qualified"]
    and metrics["supported_proposal_only"]
    and metrics["ood_refused"]
    and metrics["native_reanalysis_accepted"]
    and metrics["accepted_differs_from_learned"]
)
print(
    json.dumps(
        {"campaign": "cardiovascular-learning", "passed": passed, **metrics}, indent=2
    )
)
if not passed:
    raise SystemExit(1)
