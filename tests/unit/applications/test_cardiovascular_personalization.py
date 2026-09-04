#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._likelihoods import GaussianLikelihood
from phydrax.applications.cardiovascular._quantities import CardiovascularQuantitySpec
from phydrax.applications.cardiovascular.observations._metadata import ObservationRecord
from phydrax.applications.cardiovascular.personalization._design import (
    check_directional_derivative,
    ExperimentDesignCandidate,
    ExperimentDesignCriterion,
    ExperimentDesignPlan,
    fisher_local_diagnostics,
    ForwardAdjointEvidence,
    ProfileLikelihoodPlan,
    SensitivitySVDPlan,
)
from phydrax.applications.cardiovascular.personalization._inverse import (
    ElectrophysiologyInverseProblem,
)
from phydrax.applications.cardiovascular.personalization._likelihood import (
    GaussianModelDiscrepancy,
    LinearNuisanceModel,
    ModalityLikelihoodChannel,
    ModalityObservation,
    MultimodalLikelihoodPlan,
    ReferenceGauge,
)
from phydrax.applications.cardiovascular.personalization._parameters import (
    CardiacParameterSchema,
    CardiacParameterSpec,
    CardiacParameterSupport,
    CardiacSubsystem,
    ParameterIdentifiability,
)
from phydrax.applications.cardiovascular.personalization._validation import (
    ClinicalResearchContext,
    ClinicalResearchValidationPlan,
    ClinicalResearchValidationRecord,
)
from phydrax.optim import Bounds, OptimizationTermination, ReducedAdjoint
from phydrax.units import ONE
from phydrax.uq import (
    IdentityBijector,
    Normal,
    SigmoidIntervalBijector,
    Uniform,
)


def _dimensionless_quantity(name: str = "cardiac_gain") -> CardiovascularQuantitySpec:
    return CardiovascularQuantitySpec(name, "strain", ONE)


def _scalar_schema(
    *, subsystem: CardiacSubsystem = CardiacSubsystem.ELECTROPHYSIOLOGY
) -> CardiacParameterSchema:
    support = CardiacParameterSupport(0.1, 4.0)
    return CardiacParameterSchema(
        (
            CardiacParameterSpec(
                "conduction_gain",
                _dimensionless_quantity(),
                SigmoidIntervalBijector(0.1, 4.0),
                support,
                Uniform(0.1, 4.0),
                subsystem,
                identifiability=ParameterIdentifiability.PRIMARY,
            ),
        ),
        schema_id="ep.synthetic",
    )


def test_parameter_schema_preserves_units_transforms_support_prior_and_owner():
    schema = _scalar_schema()
    physical = (jnp.asarray(2.0),)
    raw = schema.unconstrain(physical)

    np.testing.assert_allclose(schema.constrain(raw)[0], physical[0], rtol=1e-6)
    assert bool(schema.contains(physical))
    assert np.isfinite(np.asarray(schema.log_prior(physical)))
    assert schema.fields[0].quantity.kernel_unit == "1"
    assert schema.fields[0].subsystem is CardiacSubsystem.ELECTROPHYSIOLOGY
    assert schema.fields[0].identifiability is ParameterIdentifiability.PRIMARY

    with pytest.raises(ValueError, match="outside"):
        schema.unconstrain((jnp.asarray(5.0),))


def test_multimodal_likelihood_masks_gauge_covariance_nuisance_and_discrepancy():
    record = ObservationRecord(
        "ecg-1",
        "ecg",
        jnp.asarray([1.0, 2.0, 4.0]),
        jnp.asarray([True, True, True]),
        "electric_potential",
        "mV",
        frame_id="patient-lps",
        timebase_id="ecg-time",
        asset_id="ecg-asset",
    )
    voltage = ModalityObservation.from_record(record)
    assert voltage.frame_id == "patient-lps"
    assert voltage.timebase_id == "ecg-time"
    assert voltage.asset_id == "ecg-asset"
    with pytest.raises(ValueError, match="gauged Gaussian"):
        ModalityLikelihoodChannel(
            voltage,
            likelihood=GaussianLikelihood(0.2),
            gauge=ReferenceGauge(0),
        )
    gauged = ModalityLikelihoodChannel.correlated_gaussian(
        voltage,
        0.04 * jnp.eye(3),
        gauge=ReferenceGauge(0),
    )
    shifted_prediction = jnp.asarray([11.0, 12.0, 14.0])
    gauge_result = gauged.evaluate(shifted_prediction)
    np.testing.assert_allclose(gauge_result.residual, 0.0, atol=1e-7)
    assert bool(gauge_result.successful)

    strain = ModalityObservation(
        "strain-1",
        "strain",
        jnp.asarray([0.4, 99.0, 0.8]),
        jnp.asarray([True, False, True]),
        "strain",
        "1",
    )
    nuisance = LinearNuisanceModel(jnp.ones((3, 1)), Normal(jnp.zeros(1), jnp.ones(1)))
    discrepancy = GaussianModelDiscrepancy(
        jnp.asarray([0.05, 0.0, -0.05]),
        jnp.asarray([[0.03], [0.0], [0.02]]),
    )
    with pytest.raises(ValueError, match="off-diagonal covariance"):
        ModalityLikelihoodChannel(
            strain,
            likelihood=GaussianLikelihood(0.1),
            discrepancy=discrepancy,
        )
    correlated = ModalityLikelihoodChannel.correlated_gaussian(
        strain,
        0.01 * jnp.eye(3),
        nuisance=nuisance,
        discrepancy=discrepancy,
    )
    prediction = jnp.asarray([0.25, -20.0, 0.75])
    result = correlated.evaluate(prediction, nuisance_values=jnp.asarray([0.1]))
    np.testing.assert_allclose(result.residual, 0.0, atol=1e-7)
    assert bool(result.successful)

    full = MultimodalLikelihoodPlan((gauged, correlated)).prepare()
    combined = full.evaluate(
        (shifted_prediction, prediction),
        nuisance_values=(None, jnp.asarray([0.1])),
    )
    held_out = full.plan.held_out(("strain",)).prepare().evaluate((shifted_prediction,))
    assert bool(combined.successful)
    np.testing.assert_allclose(
        held_out.log_likelihood,
        combined.channel_results[0].log_likelihood,
        rtol=1e-6,
    )
    assert full.plan.plan_id != full.plan.held_out(("strain",)).plan_id


def test_subsystem_inverse_recovers_synthetic_parameter_from_multiple_starts():
    observation = ModalityObservation(
        "activation-1",
        "activation_time",
        jnp.asarray([2.0]),
        jnp.asarray([True]),
        "time",
        "ms",
    )
    likelihood = MultimodalLikelihoodPlan(
        (ModalityLikelihoodChannel(observation, likelihood=GaussianLikelihood(0.1)),)
    ).prepare()
    base_schema = _scalar_schema()
    fixed_field = CardiacParameterSpec(
        "fixed_reference",
        _dimensionless_quantity("fixed_reference"),
        IdentityBijector(),
        CardiacParameterSupport(-10.0, 10.0),
        Normal(0.0, 1.0),
        CardiacSubsystem.ELECTROPHYSIOLOGY,
        identifiability=ParameterIdentifiability.FIXED,
    )
    schema = CardiacParameterSchema(
        (*base_schema.fields, fixed_field),
        schema_id="ep.synthetic.with-fixed-reference",
    )
    inverse = ElectrophysiologyInverseProblem(
        schema,
        likelihood,
        lambda state, physical, args: state - physical[0],
        lambda state, physical, args: (state.reshape((1,)),),
        fixed_topology=lambda state, physical, args: jnp.asarray(True),
        problem_id="ep-synthetic-recovery",
    )
    result = inverse.solve_multistart(
        jnp.asarray([1.0]),
        ((jnp.asarray(0.5), jnp.asarray(1.25)), (jnp.asarray(3.5), jnp.asarray(1.25))),
        method=ReducedAdjoint(),
        termination=OptimizationTermination(
            absolute_optimality=2.0e-5,
            relative_optimality=0.0,
            maximum_steps=80,
        ),
    )

    assert bool(result.successful)
    assert bool(result.best.evidence.state_accepted)
    assert bool(result.best.evidence.adjoint_accepted)
    np.testing.assert_allclose(result.best.physical_parameters[0], 2.0, atol=3e-3)
    np.testing.assert_allclose(result.best.physical_parameters[1], 1.25, atol=0.0)
    assert len(result.best.state_design.design) == 1
    assert len(result.results) == 2
    assert int(result.best_index) in (0, 1)
    with pytest.raises(ValueError, match="identical across every multi-start"):
        inverse.solve_multistart(
            jnp.asarray([1.0]),
            (
                (jnp.asarray(0.5), jnp.asarray(1.25)),
                (jnp.asarray(3.5), jnp.asarray(1.5)),
            ),
            method=ReducedAdjoint(),
        )


def test_inverse_routes_reject_monolithic_cross_subsystem_parameter_blocks():
    observation = ModalityObservation(
        "activation-2",
        "activation_time",
        jnp.asarray([1.0]),
        jnp.asarray([True]),
        "time",
        "ms",
    )
    likelihood = MultimodalLikelihoodPlan(
        (ModalityLikelihoodChannel(observation, likelihood=GaussianLikelihood(1.0)),)
    ).prepare()
    mechanics_schema = _scalar_schema(subsystem=CardiacSubsystem.PASSIVE_MECHANICS)

    with pytest.raises(ValueError, match="cannot own"):
        ElectrophysiologyInverseProblem(
            mechanics_schema,
            likelihood,
            lambda state, physical, args: state - physical[0],
            lambda state, physical, args: (state.reshape((1,)),),
            fixed_topology=lambda state, physical, args: jnp.asarray(True),
        )


def test_sensitivity_svd_fisher_and_profile_expose_confounding():
    sensitivity = SensitivitySVDPlan(
        lambda parameters, args: jnp.asarray(
            [parameters[0] + parameters[1], 2.0 * (parameters[0] + parameters[1])]
        ),
        jnp.ones(2),
        jnp.ones(2),
        relative_rank_tolerance=1e-7,
    ).evaluate(jnp.asarray([1.0, 1.0]))

    assert bool(sensitivity.confounded)
    assert int(sensitivity.rank) == 1
    assert int(sensitivity.nullity) == 1
    null_direction = sensitivity.nullspace_basis[:, 0]
    np.testing.assert_allclose(jnp.sum(null_direction), 0.0, atol=2e-6)

    fisher = fisher_local_diagnostics(
        sensitivity.jacobian,
        relative_rank_tolerance=1e-7,
    )
    assert bool(fisher.confounded)
    assert int(fisher.rank) == 1
    assert bool(fisher.observation_precision_evidence.accepted)
    assert bool(fisher.prior_information_evidence.accepted)
    with pytest.raises(ValueError, match="symmetric"):
        fisher_local_diagnostics(
            sensitivity.jacobian,
            observation_precision=jnp.asarray([[1.0, 1.0], [0.0, 1.0]]),
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        fisher_local_diagnostics(
            sensitivity.jacobian,
            prior_information=jnp.asarray([[1.0, 0.0], [0.0, -1.0]]),
        )

    profile = ProfileLikelihoodPlan(
        lambda parameters, args: (
            (parameters[0] - 2.0) ** 2 + (parameters[1] - parameters[0]) ** 2
        ),
        0,
        jnp.asarray([1.0, 2.0, 3.0]),
        bounds=Bounds(jnp.asarray([-4.0, -4.0]), jnp.asarray([4.0, 4.0])),
        termination=OptimizationTermination(maximum_steps=60),
    ).evaluate(jnp.asarray([0.0, 0.0]))
    assert bool(profile.all_successful)
    assert int(jnp.argmin(profile.delta_objective)) == 1
    np.testing.assert_allclose(
        profile.optimized_parameters[:, 1], profile.grid, atol=2e-3
    )


def test_derivative_check_and_experiment_design_require_accepted_evidence():
    derivative = check_directional_derivative(
        lambda values: jnp.sum(jnp.sin(values) ** 2),
        jnp.asarray([0.2, -0.4]),
        jnp.asarray([1.0, 0.5]),
        step=2e-3,
        relative_tolerance=2e-4,
    )
    assert bool(derivative.accepted)

    accepted = ForwardAdjointEvidence(True, True, True, True)
    rejected = ForwardAdjointEvidence(True, False, True, True)
    candidates = (
        ExperimentDesignCandidate(
            "pace-site-x", jnp.asarray([[1.0, 0.0]]), jnp.eye(1), accepted
        ),
        ExperimentDesignCandidate(
            "pace-site-y", jnp.asarray([[0.0, 1.0]]), jnp.eye(1), accepted
        ),
        ExperimentDesignCandidate(
            "uncertified-site",
            jnp.asarray([[10.0, 10.0]]),
            jnp.eye(1),
            rejected,
        ),
    )
    design = (
        ExperimentDesignPlan(
            candidates,
            0.1 * jnp.eye(2),
            criterion=ExperimentDesignCriterion.D_OPTIMAL,
            maximum_experiments=2,
            budget=2.0,
        )
        .prepare()
        .select()
    )

    assert bool(design.successful)
    assert int(design.selected_count) == 2
    assert not bool(design.selected_mask[2])
    assert set(np.asarray(design.selected_indices).tolist()) == {0, 1}


def test_clinical_research_validation_is_governed_complete_and_fail_closed():
    context = ClinicalResearchContext(
        "Does the model preserve a prespecified held-out endpoint?",
        "Retrospective multi-site methods study",
        "Software-method evaluation only",
        "protocol-17",
        irb_id="irb-42",
        deidentification_ids=("deid-pipeline-3",),
        data_rights_ids=("dua-9",),
    )
    plan = ClinicalResearchValidationPlan(
        context,
        "development-2023",
        "calibration-2024q1",
        "validation-2024q2",
        site_holdout_ids=("site-c", "site-d"),
        temporal_holdout_id="temporal-2025",
        endpoint_definition="Prespecified waveform error at 100 ms",
        comparator_definition="Frozen baseline model build",
        subgroup_definitions=("sex-stratum", "age-stratum"),
        ood_definition="Acquisition and anatomy support outside training bounds",
        failure_analysis_plan="Enumerate nonconvergence, OOD, and missing-modality failures",
        acceptance_criteria=("Endpoint interval reported", "All failures retained"),
    )
    record = ClinicalResearchValidationRecord(
        "validation-record-1",
        plan,
        "execution-immutable-1",
        calibration_results={"endpoint_error": 0.12},
        validation_results={"endpoint_error": 0.15},
        site_holdout_results={
            "site-c": {"endpoint_error": 0.16},
            "site-d": {"endpoint_error": 0.14},
        },
        temporal_holdout_results={"temporal-2025": {"endpoint_error": 0.17}},
        subgroup_results={
            "sex-stratum": {"maximum_error": 0.19},
            "age-stratum": {"maximum_error": 0.21},
        },
        ood_results={"detected": 4, "evaluated": 4},
        failure_analysis_results={"nonconvergence": 2, "missing_modality": 1},
    )
    evidence = record.evaluate()
    assert evidence.record_complete
    assert evidence.site_holdout_complete
    assert evidence.temporal_holdout_complete
    assert evidence.subgroup_analysis_complete
    with pytest.raises(TypeError):
        record.site_holdout_results["site-c"]["endpoint_error"] = 0.0

    with pytest.raises(ValueError, match="separate"):
        ClinicalResearchValidationPlan(
            context,
            "development-2023",
            "calibration-2024q1",
            "calibration-2024q1",
            site_holdout_ids=("site-c",),
            temporal_holdout_id="temporal-2025",
            endpoint_definition="Endpoint",
            comparator_definition="Comparator",
            subgroup_definitions=("subgroup",),
            ood_definition="OOD",
            failure_analysis_plan="Failure analysis",
            acceptance_criteria=("Criterion",),
        )
    with pytest.raises(ValueError, match="protected health information"):
        ClinicalResearchContext(
            "Question",
            "Context",
            "Research only",
            "protocol-18",
            waiver_id="waiver-2",
            deidentification_ids=("deid-2",),
            data_rights_ids=("dua-2",),
            contains_phi=True,
        )
    with pytest.raises(ValueError, match="clinical decision"):
        ClinicalResearchValidationRecord(
            "invalid-claim",
            plan,
            "execution-2",
            calibration_results={},
            validation_results={},
            site_holdout_results={},
            temporal_holdout_results={},
            subgroup_results={},
            ood_results={},
            failure_analysis_results={},
            clinical_decision_claim=True,
        )
