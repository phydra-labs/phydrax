#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax.nn.operator.engine import AbstractOperatorModel


class _InitialADriver(AbstractOperatorModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 3
        self.out_size = 1

    def __call_operator_batch__(self, batch, /, *, key=None):
        del key
        state = batch.input("state").values
        assert state is not None
        query = batch.require_single_query()
        value = state[..., 0]
        return jnp.broadcast_to(
            value.reshape(batch.case_shape + (1, 1)),
            batch.case_shape + query.sample_shape + (1,),
        )

    def __call__(self, x, /, *, key=None):
        return self.__call_operator_batch__(x, key=key)


class _QueryDriver(AbstractOperatorModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 3
        self.out_size = 1

    def __call_operator_batch__(self, batch, /, *, key=None):
        del key
        return batch.require_single_query().coordinates_array(case_shape=batch.case_shape)

    def __call__(self, x, /, *, key=None):
        return self.__call_operator_batch__(x, key=key)


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
        jnp.asarray((10.0, 10.0, 10.0)),
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


def _chemistry():
    return phx.equations.ChemicalConditionalAffinePlan(("B", "C"), ("A",)).prepare(
        _mechanism()
    )


def _batch(duration=(0.0, 0.01), *, mask=None):
    return phx.nn.operator.OperatorBatch(
        inputs={
            "state": phx.nn.operator.FunctionSamples(
                values=jnp.asarray((2.0, 1.0, 0.0), dtype=jnp.float64)
            ),
            "temperature": phx.nn.operator.FunctionSamples(
                values=jnp.asarray(500.0, dtype=jnp.float64)
            ),
            "pressure": phx.nn.operator.FunctionSamples(
                values=jnp.asarray(101325.0, dtype=jnp.float64)
            ),
        },
        queries={
            "time": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=jnp.asarray(duration, dtype=jnp.float64)[:, None],
                mask=None if mask is None else jnp.asarray(mask),
            )
        },
    )


def _scaling():
    return phx.nn.operator.architectures.ChemicalConditionalAffineScaling(
        jnp.ones((3,), dtype=jnp.float64),
        jnp.ones((1,), dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )


def test_operator_uses_auxiliary_driver_and_returns_authoritative_state():
    model = phx.nn.operator.architectures.ChemicalConditionalAffineOperator(
        _chemistry(),
        _InitialADriver(),
        _scaling(),
    )
    batch = _batch()

    drivers = model.predict_drivers(batch, key=jr.key(0))
    result = model.evaluate_transition(batch, key=jr.key(1))
    values = model.__call_operator_batch__(batch, key=jr.key(2))

    np.testing.assert_allclose(drivers, ((2.0,), (2.0,)))
    np.testing.assert_array_equal(values[0], batch.input("state").values)
    np.testing.assert_allclose(values, result.candidate_state)
    np.testing.assert_array_equal(result.successful, (True, True))
    np.testing.assert_allclose(result.element_residual, 0.0, atol=1e-13)


def test_operator_queries_driver_at_scaled_midpoint_and_honors_masks():
    model = phx.nn.operator.architectures.ChemicalConditionalAffineOperator(
        _chemistry(),
        _QueryDriver(),
        _scaling(),
    )
    batch = _batch((0.2, 0.4), mask=(True, False))

    drivers = model.predict_drivers(batch, key=jr.key(0))
    values = model.__call_operator_batch__(batch, key=jr.key(1))

    np.testing.assert_allclose(drivers[:, 0], jnp.log1p(jnp.asarray((0.1, 0.0))))
    np.testing.assert_array_equal(values[1], jnp.zeros((3,)))


def test_stoichiometric_rate_correction_is_positive_and_identity_initialized():
    context = phx.nn.models.MLP(
        in_size=4,
        out_size=2,
        hidden_sizes=(),
        key=jr.key(1),
    )
    species = phx.nn.models.MLP(
        in_size=2,
        out_size=3,
        hidden_sizes=(),
        key=jr.key(2),
    )
    correction = phx.nn.operator.architectures.StoichiometricRateCorrection(
        context,
        species,
        jnp.asarray(((1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        _mechanism().net_stoichiometry,
        log_multiplier_bound=3.0,
    )

    identity = correction(
        jnp.asarray((2.0, 1.0, 0.0)),
        jnp.asarray(0.1),
        key=jr.key(3),
    )
    active = eqx.tree_at(lambda value: value.strength, correction, jnp.asarray(1.0))
    multiplier = active(
        jnp.asarray((2.0, 1.0, 0.0)),
        jnp.asarray(0.1),
        key=jr.key(4),
    )

    np.testing.assert_array_equal(identity, jnp.ones((1,)))
    assert jnp.all(jnp.isfinite(multiplier) & (multiplier > 0.0))
    assert multiplier.shape == (1,)


def test_operator_catalog_declares_research_local_transition():
    status = phx.nn.operator.catalog.operator_architecture_status(
        "ChemicalConditionalAffineOperator"
    )

    assert status.tier == "research"
    assert status.capabilities.autoregressive_rollout
    assert status.capabilities.source_geometries == ("abstract", "point_cloud")


def test_staged_losses_use_driver_and_teacher_forced_paths():
    model = phx.nn.operator.architectures.ChemicalConditionalAffineOperator(
        _chemistry(),
        _InitialADriver(),
        _scaling(),
    )
    base_batch = _batch((0.01,))
    drivers = model.predict_drivers(base_batch, key=jr.key(6))
    query = base_batch.query("time")
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            **dict(base_batch.inputs),
            "driver_targets": phx.nn.operator.FunctionSamples(
                values=drivers,
                coordinates=query.coordinates,
                mask=query.mask,
            ),
        },
        queries=base_batch.queries,
    )
    prediction = model.predict_prevalidated(batch, key=jr.key(5))
    targets = phx.nn.operator.OperatorTargetBatch.from_arrays(
        {"state": prediction.field("output").values},
        batch,
    )
    context = phx.nn.operator.training.OperatorLossContext(
        prediction,
        batch,
        targets,
        prediction,
        batch,
        targets,
    )
    driver_loss = phx.nn.operator.training.ChemicalConditionalAffineDriverLoss()
    teacher_loss = phx.nn.operator.training.ChemicalConditionalAffineTeacherForcedLoss()

    driver_value = driver_loss(
        model,
        prediction,
        batch,
        targets,
        key=jr.key(7),
        step=jnp.asarray(0),
        training=True,
        context=context,
    )
    teacher_value = teacher_loss(
        model,
        prediction,
        batch,
        targets,
        key=jr.key(8),
        step=jnp.asarray(0),
        training=True,
        context=context,
    )

    np.testing.assert_allclose(driver_value, 0.0, atol=1e-14)
    np.testing.assert_allclose(teacher_value, 0.0, atol=1e-14)
    assert driver_loss.fingerprint != teacher_loss.fingerprint
