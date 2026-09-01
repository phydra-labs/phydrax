#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


mop = phx.applications.solid_mechanics.operator


class _IdentityMechanicsOperator(phx.nn.operator.AbstractOperatorModel):
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self):
        self.in_size = "scalar"
        self.out_size = "scalar"

    def __call_operator_batch__(self, batch, *, key=None):
        del key
        values = batch.input("source").values
        assert values is not None
        return values

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("DeepONet")


def _spec(*, upper=4.0):
    return mop.MechanicsParameterSpec(
        (
            mop.MechanicsParameterField(
                "length",
                lower=0.5,
                upper=upper,
                role="geometry",
                unit="m",
            ),
            mop.MechanicsParameterField(
                "family",
                role="material",
                kind="categorical",
                support=("base", "reinforced"),
            ),
            mop.MechanicsParameterField(
                "reinforcement",
                role="material",
                lower=0.0,
                upper=1.0,
                active_when={"family": ("reinforced",)},
            ),
        )
    )


def _realization(spec, length, family="base", *, weight=None, case_id=None):
    values = {"length": length, "family": family}
    if family == "reinforced":
        values["reinforcement"] = 0.25
    return mop.MechanicsParameterRealization(
        spec,
        values,
        probability_weight=weight,
        case_id=case_id,
    )


def _distribution(lengths, weights, *, distribution_id):
    spec = _spec()
    realizations = tuple(
        _realization(
            spec,
            length,
            weight=weight,
            case_id=f"{distribution_id}-{index}",
        )
        for index, (length, weight) in enumerate(zip(lengths, weights, strict=True))
    )
    return mop.MechanicsParameterDistribution(
        spec,
        realizations,
        distribution_id=distribution_id,
    )


def _geometry(realization):
    length = realization.values["length"]
    return mop.MechanicsGeometryMap(
        lambda xi, parameters: parameters.values["length"] * xi,
        lambda xi, parameters: jnp.broadcast_to(
            jnp.asarray([[parameters.values["length"]]]),
            xi.shape[:-1] + (1, 1),
        ),
        reference_domain_id="unit-interval",
        physical_domain_id="bar",
        geometry_id=f"bar-{float(length)}",
        boundary_correspondence={"left": "x=0", "right": "x=L"},
    )


def _operator_case(realization, geometry):
    reference = jnp.asarray([[0.0], [1.0]])
    coordinates = geometry.map_coordinates(reference, realization)
    weights = geometry.volume_weights(
        reference,
        jnp.asarray([0.5, 0.5]),
        realization,
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.full((2,), realization.values["length"]),
        coordinates=coordinates,
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=weights,
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={"source": source},
        queries={"domain": query},
    )
    return phx.nn.operator.OperatorCase(
        batch,
        phx.nn.operator.OperatorTargetBatch.from_arrays({}, batch),
        provenance=phx.nn.operator.OperatorCaseProvenance(realization.case_id),
    )


def _builder(distribution, reduction, *, split):
    return mop.MechanicsCaseBuilder(
        distribution,
        _geometry,
        _operator_case,
        reduction=reduction,
        mechanics_problem_id="bar-energy",
        material_id="linear-material",
        load_id="dead-load",
        boundary_condition_id="fixed-left",
        spatial_realization_id="two-point-volume-rule",
        validity=lambda realization, geometry, case: realization.values["length"] > 0,
        split_fingerprint=split,
    )


def _adapter():
    return mop.OperatorTrialFieldAdapter(
        ("output",),
        adapter_id="bar-displacement",
        field_domain_ids={"output": "bar"},
    )


def _energy_term():
    return mop.MechanicsCaseFunctional(
        "internal_energy",
        lambda fields, prediction, batch, case: jnp.sum(
            batch.query("domain").weights() * fields["output"] ** 2
        ),
        kind="energy",
        query_name="domain",
        functional_id="quadratic-bar-energy",
    )


def _problem(distribution, reduction, *, split="train"):
    return mop.ConservativeMechanicsOperatorProblem(
        _builder(distribution, reduction, split=split),
        _adapter(),
        (_energy_term(),),
        problem_id="conditional-bar",
    )


def _prediction(problem, values):
    batch = problem.batch()
    return phx.nn.operator.OperatorPrediction.from_field(
        "output",
        jnp.asarray(values),
        "domain",
        batch.query("domain"),
        spec=phx.nn.operator.OperatorOutputSpec("scalar"),
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def _task():
    return phx.nn.operator.OperatorTask(
        "mechanics-conditional-bar",
        dimension_basis=("length",),
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "source",
                role="source",
                source_name="source",
            ),
            phx.nn.operator.OperatorFieldSpec(
                "output",
                role="target",
                query_name="domain",
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "domain",
                geometry_kind="point_cloud",
                coordinate_components=("x",),
                coordinate_dimensions=((1.0,),),
            ),
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="coincident",
            query_is_fixed=False,
        ),
    )


def _trained(problem):
    return phx.nn.operator.training.TrainedOperator(
        _IdentityMechanicsOperator(),
        _task(),
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        provenance=dict(problem.metadata),
        artifact_id="immutable-mechanics-operator",
    )


def test_parameter_law_preserves_hierarchy_correlations_and_split_identity():
    spec = _spec()
    assert spec.contains({"length": 1.0, "family": "base"})
    assert not spec.contains({"length": 1.0, "family": "base", "reinforcement": 0.2})
    reinforced = _realization(spec, 1.5, "reinforced", weight=0.4, case_id="r")
    base = _realization(spec, 1.0, weight=0.6, case_id="b")
    distribution = mop.MechanicsParameterDistribution(spec, (base, reinforced))
    assert distribution.weight_kind == "probability"
    assert jnp.allclose(distribution.normalized_weights, jnp.asarray([0.6, 0.4]))
    assert distribution.normalized_weight("r") == pytest.approx(0.4)
    same_point = mop.MechanicsParameterRealization(
        spec,
        base.values,
        probability_weight=1.0,
        case_id="held-out-alias",
    )
    leaked = mop.MechanicsParameterDistribution(spec, (same_point,))
    with pytest.raises(ValueError, match="not held-out"):
        distribution.assert_disjoint(leaked)


def test_case_risk_reduces_complete_cases_and_reports_batch_max_semantics():
    values = jnp.asarray([1.0, 4.0, 4.0])
    weights = jnp.asarray([0.5, 0.25, 0.25])
    weighted = phx.nn.operator.training.MechanicsCaseReduction("weighted_mean")
    assert weighted(values, probability_weights=weights) == pytest.approx(2.5)
    cvar = phx.nn.operator.training.MechanicsCaseReduction("cvar", alpha=0.5)
    tail = cvar.evaluate(values, probability_weights=weights)
    assert tail.value == pytest.approx(4.0)
    assert tail.tail_mass == pytest.approx(0.5)
    maximum = phx.nn.operator.training.MechanicsCaseReduction("max").evaluate(values)
    assert maximum.value == pytest.approx(4.0)
    assert maximum.batch_dependent
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="cannot be dropped"):
        invalid = weighted(values, probability_weights=weights, valid=(True, False, True))
        jax.block_until_ready(invalid)


def test_geometry_and_trial_adapter_keep_physical_measure_and_domain_explicit():
    realization = _realization(_spec(), 2.0, weight=1.0, case_id="geometry")
    geometry = _geometry(realization)
    reference = jnp.asarray([[0.0], [0.5], [1.0]])
    assert jnp.allclose(
        geometry.map_coordinates(reference, realization).reshape(-1),
        jnp.asarray([0.0, 1.0, 2.0]),
    )
    assert jnp.allclose(
        geometry.volume_weights(reference, jnp.ones((3,)), realization),
        2.0,
    )
    inverted = mop.MechanicsGeometryMap(
        lambda xi, parameters: -xi,
        lambda xi, parameters: jnp.broadcast_to(
            jnp.asarray([[-1.0]]), xi.shape[:-1] + (1, 1)
        ),
        reference_domain_id="reference",
        physical_domain_id="inverted",
        geometry_id="inverted",
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="orientation"):
        jacobian = inverted.jacobian(reference, realization)
        jax.block_until_ready(jacobian)


def test_energy_loss_integrates_each_geometry_before_parameter_risk():
    reduction = phx.nn.operator.training.MechanicsCaseReduction("weighted_mean")
    distribution = _distribution((1.0, 2.0), (0.25, 0.75), distribution_id="train")
    problem = _problem(distribution, reduction)
    prediction = _prediction(problem, ((1.0, 1.0), (2.0, 2.0)))
    result = mop.ExpectedMechanicsEnergyLoss(problem, reduction).evaluate(
        prediction,
        problem.batch(),
    )
    assert jnp.allclose(result.cases.values, jnp.asarray([1.0, 8.0]))
    assert result.value == pytest.approx(6.25)
    assert result.cases.measure_ids["internal_energy"][0]
    assert result.cases.measure_ids["internal_energy"][1]
    assert (
        result.cases.measure_ids["internal_energy"][0]
        != result.cases.measure_ids["internal_energy"][1]
    )


def test_residual_and_mixed_problems_retain_named_nonpotential_blocks():
    reduction = phx.nn.operator.training.MechanicsCaseReduction("weighted_mean")
    distribution = _distribution((1.0,), (1.0,), distribution_id="blocks")
    builder = _builder(distribution, reduction, split="train")
    residual = mop.MechanicsCaseFunctional(
        "equilibrium",
        lambda fields, prediction, batch, case: jnp.sum(
            batch.query("domain").weights() * fields["output"] ** 2
        ),
        kind="residual",
        query_name="domain",
        functional_id="equilibrium-residual",
    )
    residual_problem = mop.MechanicsResidualOperatorProblem(
        builder,
        _adapter(),
        (residual,),
        problem_id="bar-residual",
    )
    residual_prediction = _prediction(residual_problem, ((2.0, 2.0),))
    residual_result = mop.MechanicsResidualLoss(
        residual_problem,
        reduction,
    ).evaluate(residual_prediction, residual_problem.batch())
    assert residual_result.cases.formulation == "residual"

    primal = mop.OperatorTrialFieldAdapter(("output",), adapter_id="primal")
    dual = mop.OperatorTrialFieldAdapter(("pressure",), adapter_id="dual")
    equilibrium = mop.MechanicsCaseFunctional(
        "equilibrium",
        lambda fields, prediction, batch, case: jnp.sum(
            batch.query("domain").weights() * fields["output"] ** 2
        ),
        kind="mixed_block",
        query_name="domain",
        functional_id="mixed-equilibrium",
    )
    gauge = mop.MechanicsCaseFunctional(
        "pressure_gauge",
        lambda fields, prediction, batch, case: (
            jnp.sum(batch.query("domain").weights() * fields["pressure"]) ** 2
        ),
        kind="gauge",
        query_name="domain",
        functional_id="pressure-zero-mean",
    )
    mixed = mop.MixedMechanicsOperatorProblem(
        builder,
        primal,
        dual,
        (equilibrium,),
        problem_id="mixed-bar",
        gauge_blocks=(gauge,),
    )
    batch = mixed.batch()
    prediction = phx.nn.operator.OperatorPrediction(
        {
            "output": phx.nn.operator.OperatorFieldBatch(
                jnp.ones((1, 2)),
                query_name="domain",
                spec=phx.nn.operator.OperatorOutputSpec("scalar"),
            ),
            "pressure": phx.nn.operator.OperatorFieldBatch(
                jnp.asarray(((1.0, -1.0),)),
                query_name="domain",
                spec=phx.nn.operator.OperatorOutputSpec("scalar"),
            ),
        },
        batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )
    mixed_result = mop.MixedMechanicsLoss(mixed, reduction).evaluate(prediction, batch)
    assert tuple(mixed_result.cases.term_values) == (
        "equilibrium",
        "pressure_gauge",
    )
    assert mixed_result.cases.formulation == "mixed"


def test_support_qualification_and_adaptation_are_explicitly_separate():
    reduction = phx.nn.operator.training.MechanicsCaseReduction("weighted_mean")
    training = _distribution((1.0, 2.0), (0.5, 0.5), distribution_id="training")
    held_out = _distribution((1.5, 3.0), (0.5, 0.5), distribution_id="held-out")
    problem = _problem(training, reduction)
    trained = _trained(problem)
    held_out_builder = _builder(held_out, reduction, split="held-out")
    qualification = mop.MechanicsOperatorQualification(
        training,
        held_out_builder,
        (
            mop.MechanicsQualificationMetric(
                "field_energy",
                lambda prediction, batch, case: jnp.sum(
                    batch.query("domain").weights()
                    * prediction.field("output").values ** 2
                ),
                query_name="domain",
                unit="m^3",
                metric_id="held-out-field-energy",
            ),
        ),
        reduction,
        required_metadata=problem.metadata,
        qualification_id="held-out-bars",
    )
    evidence = mop.qualify_mechanics_operator(trained, qualification)
    assert evidence.case_ids == ("held-out-0", "held-out-1")
    assert all(item.supported for item in evidence.support)
    assert evidence.observed_worst_case["field_energy"] == pytest.approx(27.0)

    case = held_out_builder.build(0)
    inference = mop.infer_mechanics_operator(
        trained,
        case,
        training.spec,
        required_metadata=problem.metadata,
    )
    assert inference.inference_kind == "amortized"
    assert inference.prediction is not None

    policy = mop.MechanicsFineTuningPolicy(
        phx.nn.operator.training.BoundedResidualAdaptationPolicy(
            iterations=2,
            learning_rate=0.1,
            maximum_update_norm=1.0,
            gradient_clip_norm=1.0,
        ),
        allowed_observable_ids=("equilibrium-residual",),
        residual_objective_id="context-residual",
        policy_id="two-step-context",
    )
    adapted = mop.fine_tune_mechanics_operator(
        trained,
        case,
        jnp.asarray([0.0]),
        lambda context: jnp.sum((context - 1.0) ** 2),
        policy=policy,
        support_spec=training.spec,
        required_metadata=problem.metadata,
        jit=False,
    )
    assert adapted.adaptation_kind == "bounded_residual_context"
    assert adapted.base_operator is trained
    assert not isinstance(adapted, phx.nn.operator.training.TrainedOperator)

    wider = _spec(upper=8.0)
    outside = _realization(wider, 6.0, weight=1.0, case_id="outside")
    status = mop.assess_mechanics_support(training.spec, outside)
    assert status.status == "out_of_support"
    outside_distribution = mop.MechanicsParameterDistribution(
        wider,
        (outside,),
        distribution_id="outside-design",
    )
    outside_case = _builder(
        outside_distribution,
        reduction,
        split="outside",
    ).build(0)
    refused = mop.infer_mechanics_operator(
        trained,
        outside_case,
        training.spec,
        required_metadata=problem.metadata,
    )
    assert refused.support.status == "out_of_support"
    assert refused.prediction is None
