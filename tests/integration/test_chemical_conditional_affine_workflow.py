#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def _mechanism():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A", "B", "C"),
        (
            phx.equations.ChemicalPhaseKind.GAS,
            phx.equations.ChemicalPhaseKind.GAS,
            phx.equations.ChemicalPhaseKind.GAS,
        ),
        jnp.asarray((1.0, 1.0, 3.0)),
        ("X", "Y"),
        jnp.asarray(((1, 0, 2), (0, 1, 1)), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
        gas_standard_pressure=101325.0,
    )
    thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.full((3,), 10.0),
        jnp.zeros((3,)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=2000.0,
    )
    return phx.equations.ChemicalMechanismIR(
        "association",
        schema,
        thermodynamics,
        (
            phx.equations.ChemicalReactionSpec(
                "2A+B<->C",
                {"A": 2.0, "B": 1.0},
                {"C": 1.0},
                phx.equations.ArrheniusRatePlan(2.0),
                reverse_rate=phx.equations.ArrheniusRatePlan(0.5),
            ),
        ),
    ).prepare()


def _model():
    mechanism = _mechanism()
    chemistry = phx.equations.ChemicalConditionalAffinePlan(("B", "C"), ("A",)).prepare(
        mechanism
    )
    latent = 4
    driver = phx.nn.operator.architectures.DeepONet(
        branch=phx.nn.models.MLP(
            in_size=3,
            out_size=latent,
            width_size=6,
            depth=2,
            key=jr.key(1),
        ),
        trunk=phx.nn.models.MLP(
            in_size=1,
            out_size=latent,
            width_size=6,
            depth=2,
            key=jr.key(2),
        ),
        coord_dim=1,
        latent_size=latent,
        out_size=1,
        source_key="state",
    )
    scaling = phx.nn.operator.architectures.ChemicalConditionalAffineScaling(
        jnp.ones((3,)),
        jnp.ones((1,)),
        1.0,
        driver_output_transform="softplus",
    )
    return phx.nn.operator.architectures.ChemicalConditionalAffineOperator(
        chemistry,
        driver,
        scaling,
    )


def _task():
    return phx.nn.operator.OperatorTask(
        "chemical-conditional-affine-local-transition",
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "state",
                channels=3,
                role="both",
                source_name="state",
                query_name="time",
                component_names=("A", "B", "C"),
            ),
            phx.nn.operator.OperatorFieldSpec(
                "temperature",
                role="source",
                source_name="temperature",
            ),
            phx.nn.operator.OperatorFieldSpec(
                "pressure",
                role="source",
                source_name="pressure",
            ),
            phx.nn.operator.OperatorFieldSpec(
                "driver_targets",
                channels=1,
                role="source",
                source_name="driver_targets",
                required=False,
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "time",
                geometry_kind="point_cloud",
                coordinate_components=("duration",),
            ),
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="independent",
            query_is_fixed=False,
            rollout_steps=2,
        ),
    )


def _batch(duration=1.0e-4):
    return phx.nn.operator.OperatorBatch(
        inputs={
            "state": phx.nn.operator.FunctionSamples(values=jnp.asarray((2.0, 1.0, 0.0))),
            "temperature": phx.nn.operator.FunctionSamples(values=jnp.asarray(500.0)),
            "pressure": phx.nn.operator.FunctionSamples(values=jnp.asarray(101325.0)),
        },
        queries={
            "time": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=jnp.asarray(((duration,),)),
            )
        },
    )


def _trained():
    return phx.nn.operator.training.TrainedOperator(
        _model(),
        _task(),
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "state"},
        artifact_id="conditional-affine-workflow",
        provenance={"dataset": "manufactured reversible association"},
    )


def test_conditional_affine_artifact_and_discrete_system_roundtrip(tmp_path):
    trained = _trained()
    expected = trained.predict(_batch()).field("state").values
    destination = phx.nn.operator.training.save_operator_artifact(tmp_path, trained)
    restored = phx.nn.operator.training.load_trained_operator(destination)
    actual = restored.predict(_batch()).field("state").values

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    state_layout = phx.dynamics.StateLayout((3,), component_names=("A", "B", "C"))
    input_layout = phx.dynamics.InputLayout(
        (2,),
        component_names=("temperature", "pressure"),
        roles=("parameter", "parameter"),
    )
    transition = phx.nn.operator.adapters.TrainedChemicalConditionalAffineTransition(
        restored,
        state_layout=state_layout,
        input_layout=input_layout,
        minimum_duration=1.0e-8,
        maximum_duration=1.0e-2,
    )
    system = transition.discrete_system()
    context = phx.dynamics.DiscreteStepContext(0.0, 1.0e-4, 0)
    state = jnp.asarray((2.0, 1.0, 0.0))
    result = system(
        context,
        state,
        inputs=jnp.asarray((500.0, 101325.0)),
    )
    evidence = transition.evaluate_with_evidence(
        state,
        jnp.asarray(1.0e-4),
        jnp.asarray((500.0, 101325.0)),
    )

    np.testing.assert_allclose(result, actual[0], rtol=1e-12, atol=1e-12)
    assert evidence.successful[0]
    np.testing.assert_allclose(evidence.element_residual, 0.0, atol=1e-12)
