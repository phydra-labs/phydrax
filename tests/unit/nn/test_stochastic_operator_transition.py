import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array

import phydrax as phx


class _BrownianFieldOperator(phx.nn.operator.AbstractProbabilisticOperatorModel):
    scale_power: Array
    uncertainty_source: str
    in_size: str
    out_size: str

    def __init__(self, scale_power=0.5, uncertainty_source="process"):
        self.scale_power = jnp.asarray(scale_power, dtype=float)
        self.uncertainty_source = str(uncertainty_source)
        self.in_size = "scalar"
        self.out_size = "scalar"

    @property
    def operator_output_specs(self):
        return {"output": phx.nn.operator.OperatorOutputSpec("scalar")}

    def distribution(self, batch, /, *, key=None):
        del key
        state = batch.input("state").values
        duration = batch.input("duration").values
        forcing = batch.input("forcing").values
        return phx.nn.operator.GaussianOperatorDistribution(
            mean=state + duration * forcing,
            scale=duration**self.scale_power,
            factors=None,
            query=batch.require_single_query(),
            output_spec=phx.nn.operator.OperatorOutputSpec("scalar"),
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
            uncertainty_source=self.uncertainty_source,
        )


class _AdditiveDriverOperator(eqx.Module):
    def __call__(self, batch, /, *, key=None):
        del key
        return (
            batch.input("state").values
            + batch.input("driver").values
            + batch.input("duration").values * batch.input("forcing").values
        )


def _transition_batch(*, cases=2, size=4, driver=False, forcing=0.0):
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, size, endpoint=False),
        quadrature_weights=jnp.full((size,), 1.0 / size),
        periodic=True,
    )
    values = jnp.zeros((cases, size))
    inputs = {
        "state": phx.nn.operator.FunctionSamples(values=values, axes=(axis,)),
        "duration": phx.nn.operator.FunctionSamples(values=jnp.ones_like(values), axes=(axis,)),
        "forcing": phx.nn.operator.FunctionSamples(
            values=jnp.full_like(values, forcing),
            axes=(axis,),
        ),
    }
    if driver:
        inputs["driver"] = phx.nn.operator.FunctionSamples(values=values, axes=(axis,))
    return phx.nn.operator.OperatorBatch(
        inputs=inputs,
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
    )


def _marginal_law(*, forcing=0.0, uncertainty_source="process"):
    batch = _transition_batch(forcing=forcing)
    spec = phx.nn.operator.training.OperatorTransitionSpec(phx.nn.operator.OperatorOutputSpec("scalar"))
    return phx.nn.operator.training.OperatorMarginalTransition(
        _BrownianFieldOperator(uncertainty_source=uncertainty_source),
        batch,
        spec,
        process_id="brownian-field",
    )


def _pathwise_law(*, forcing=0.0):
    batch = _transition_batch(driver=True, forcing=forcing)
    spec = phx.nn.operator.training.OperatorTransitionSpec(
        phx.nn.operator.OperatorOutputSpec("scalar"),
        driver_bindings=(
            phx.nn.operator.training.OperatorDriverBinding(
                "driver",
                "wiener",
                kind="wiener",
                quantity="increment",
            ),
        ),
    )
    return phx.nn.operator.training.OperatorPathwiseTransition(
        _AdditiveDriverOperator(),
        batch,
        spec,
        process_id="driven-brownian-field",
    )


def test_marginal_transition_is_process_distribution_and_preserves_forcing():
    law = _marginal_law(forcing=0.25)
    state = jnp.zeros((2, 4))
    distribution = law.marginal_transition(state, t0=0.1, t1=0.5)

    assert isinstance(distribution, phx.nn.operator.training.OperatorProcessDistribution)
    assert distribution.uncertainty_source == "process"
    assert distribution.batch_shape == (2,)
    assert distribution.event_shape == (4,)
    assert jnp.allclose(distribution.location, 0.1)
    assert distribution.sample(jr.key(0), (7,)).shape == (7, 2, 4)

    invalid = _marginal_law(uncertainty_source="observation")
    with pytest.raises(ValueError, match="process uncertainty"):
        invalid.marginal_transition(state, t0=0.0, t1=0.2)


def test_marginal_rollout_replays_and_exports_process_uncertainty():
    law = _marginal_law()
    times = jnp.asarray([0.0, 0.2, 0.5])
    first = phx.nn.operator.training.marginal_operator_rollout(
        law,
        times,
        key=jr.key(1),
        num_realizations=32,
    )
    replay = phx.nn.operator.training.marginal_operator_rollout(
        law,
        times,
        key=jr.key(1),
        num_realizations=32,
    )
    independent = phx.nn.operator.training.marginal_operator_rollout(
        law,
        times,
        key=jr.key(2),
        num_realizations=32,
    )
    shifted_initial = phx.nn.operator.training.marginal_operator_rollout(
        law,
        times,
        key=jr.key(1),
        num_realizations=32,
        initial_state=jnp.ones((2, 4)),
    )

    assert first.kind == "marginal"
    assert not first.is_pathwise
    assert first.states.shape == (2, 32, 3, 4)
    assert jnp.array_equal(first.states, replay.states)
    assert not jnp.array_equal(first.states, independent.states)
    assert first.trajectory.realizations == (None, None)
    assert first.metadata["pathwise"] is False
    assert "marginal_chain_id" in first.metadata
    assert (
        first.metadata["marginal_chain_id"]
        != shifted_initial.metadata["marginal_chain_id"]
    )

    predictive = first.to_predictive()
    assert predictive.samples.dims == ("case", "__phydra_uq_process", "time", "x")
    assert predictive.sample_axes[0].source == "process"
    assert predictive.mean().data.shape == (2, 3, 4)


def test_pathwise_rollout_reuses_one_wiener_field_and_satisfies_cocycle():
    law = _pathwise_law(forcing=0.3)
    driver = phx.stochastic.WienerRealization(
        jr.key(3),
        (4,),
        support=(0.0, 1.0),
        sample_shape=(16,),
        tolerance=1e-3,
        label="operator-driver",
    )
    coarse = phx.nn.operator.training.pathwise_operator_rollout(
        law,
        driver,
        jnp.asarray([0.0, 0.5]),
    )
    fine = phx.nn.operator.training.pathwise_operator_rollout(
        law,
        driver,
        jnp.asarray([0.0, 0.2, 0.5]),
    )
    expected = driver.increments(jnp.asarray(0.0), jnp.asarray(0.5)) + 0.15

    assert coarse.kind == "pathwise"
    assert coarse.is_pathwise
    assert coarse.states.shape == (2, 16, 2, 4)
    assert jnp.allclose(coarse.states[:, :, -1], expected[None])
    assert jnp.allclose(coarse.states[:, :, -1], fine.states[:, :, -1])
    assert all(
        realization.realization_id == driver.realization_id
        for realization in coarse.trajectory.realizations
    )
    assert coarse.metadata["coupling_id"] == driver.coupling_id
    assert coarse.metadata["driver_case_mode"] == "shared"

    case_driver = phx.stochastic.WienerRealization(
        jr.key(7),
        (2, 4),
        support=(0.0, 1.0),
        sample_shape=(8,),
    )
    case_specific = phx.nn.operator.training.pathwise_operator_rollout(
        law,
        case_driver,
        jnp.asarray([0.0, 0.5]),
    )
    case_increments = case_driver.increments(
        jnp.asarray(0.0),
        jnp.asarray(0.5),
    )
    assert case_specific.metadata["driver_case_mode"] == "case_specific"
    assert jnp.allclose(
        case_specific.states[:, :, -1],
        jnp.moveaxis(case_increments, 1, 0) + 0.15,
    )

    first_increment = driver.increments(jnp.asarray(0.0), jnp.asarray(0.2))
    second_increment = driver.increments(jnp.asarray(0.2), jnp.asarray(0.5))
    cocycle = phx.stochastic.cocycle_objective(
        law,
        jnp.zeros((2, 4)),
        t0=0.0,
        tmid=0.2,
        t1=0.5,
        first_driver_segment=first_increment,
        second_driver_segment=second_increment,
    )
    assert jnp.allclose(cocycle, 0.0, atol=1e-12)

    wrong_driver = phx.stochastic.WienerRealization(
        jr.key(4),
        (3,),
        support=(0.0, 1.0),
        sample_shape=(2,),
    )
    with pytest.raises(ValueError, match="Driver noise shape"):
        phx.nn.operator.training.pathwise_operator_rollout(
            law,
            wrong_driver,
            jnp.asarray([0.0, 0.5]),
        )


def test_operator_transition_objectives_are_exactly_shaped_jittable_and_differentiable():
    law = _marginal_law()
    states = jnp.zeros((3, 2, 4))
    times = jnp.asarray([0.0, 0.2, 0.5])
    chain = phx.nn.operator.training.operator_markov_chain_nll(
        law,
        states,
        times,
        reduction="none",
    )
    direct = phx.nn.operator.training.direct_operator_horizon_nll(
        law,
        states[0],
        states[1:],
        initial_time=0.0,
        target_times=times[1:],
        reduction="none",
    )
    expected_chain = 2.0 * jnp.log(2.0 * jnp.pi * jnp.asarray([0.2, 0.3]))
    expected_direct = 2.0 * jnp.log(2.0 * jnp.pi * jnp.asarray([0.2, 0.5]))

    assert chain.shape == (2, 2)
    assert direct.shape == (2, 2)
    assert jnp.allclose(chain[:, 0], expected_chain)
    assert jnp.allclose(direct[:, 0], expected_direct)

    def semigroup_objective(candidate):
        return phx.stochastic.semigroup_objective(
            candidate,
            states[0],
            t0=0.0,
            tmid=0.2,
            t1=0.5,
            key=jr.key(5),
            num_samples=32,
        )

    value = eqx.filter_jit(semigroup_objective)(law)
    gradient = eqx.filter_grad(semigroup_objective)(law)
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient.model.scale_power)

    generator = phx.nn.operator.training.operator_weak_generator_objective(
        law,
        states[0],
        time=0.0,
        step=0.01,
        observable=lambda value: jnp.mean(value**2, axis=-1),
        generator_observable=lambda value, time: jnp.ones(value.shape[:-1]),
        key=jr.key(6),
        num_samples=4096,
    )
    assert generator < 5e-3
