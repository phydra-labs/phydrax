import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array

import phydrax as phx


class _BrownianTransition(phx.nn.operator.AbstractProbabilisticOperatorModel):
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
        state = batch.input("state").values
        duration = batch.input("duration").values[..., 0]
        scale = jnp.broadcast_to(
            duration[..., None] ** self.scale_power,
            state.shape,
        )
        return phx.nn.operator.GaussianOperatorDistribution(
            mean=state,
            scale=scale,
            factors=None,
            query=batch.require_single_query(),
            output_spec=phx.nn.operator.OperatorOutputSpec("scalar"),
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
            uncertainty_source=self.uncertainty_source,
        )


def _transition_batch(cases=2, size=4):
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, size, endpoint=False),
        quadrature_weights=jnp.full((size,), 1.0 / size),
        periodic=True,
    )
    state = jnp.zeros((cases, size))
    duration = jnp.full((cases, size), 0.1)
    return phx.nn.operator.OperatorBatch(
        inputs={
            "state": phx.nn.operator.FunctionSamples(values=state, axes=(axis,)),
            "duration": phx.nn.operator.FunctionSamples(values=duration, axes=(axis,)),
        },
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
    )


def _condition(batch, duration):
    values = jnp.broadcast_to(
        jnp.asarray(duration)[..., None],
        batch.input("duration").values.shape,
    )
    return eqx.tree_at(lambda item: item.inputs["duration"].values, batch, values)


def _advance(batch, values):
    return eqx.tree_at(lambda item: item.inputs["state"].values, batch, values)


def test_energy_distance_is_zero_for_identical_ensembles_and_chunk_invariant():
    left = jr.normal(jr.key(0), (17, 3, 4))
    right = left + 0.6
    identical = phx.uq.energy_distance(left, left)
    dense = phx.uq.energy_distance(left, right)
    chunked = phx.uq.energy_distance(left, right, chunk_size=5)

    assert jnp.allclose(identical, 0.0, atol=1e-12)
    assert dense > 0.0
    assert jnp.allclose(chunked, dense, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="equal event shapes"):
        phx.uq.energy_distance(left, right[..., :-1])
    with pytest.raises(ValueError, match="beta"):
        phx.uq.energy_distance(left, right, beta=0.0)


def test_operator_ensemble_energy_distance_respects_query_measure_and_geometry():
    batch = _transition_batch(cases=2, size=4)
    left_samples = jr.normal(jr.key(1), (8, 2, 4))
    right_samples = left_samples + jnp.asarray([0.0, 0.2, 0.4, 0.8])
    left = phx.uq.operator_predictive_from_samples(
        left_samples,
        batch,
        phx.nn.operator.OperatorOutputSpec("scalar"),
        sample_axes=(phx.uq.SampleAxis("left", "process"),),
        field_name="output",
        query_name="query",
    )
    right = phx.uq.operator_predictive_from_samples(
        right_samples,
        batch,
        phx.nn.operator.OperatorOutputSpec("scalar"),
        sample_axes=(phx.uq.SampleAxis("right", "process"),),
        field_name="output",
        query_name="query",
    )
    quadrature = phx.uq.operator_ensemble_energy_distance(
        left,
        right,
        measure="quadrature",
        chunk_size=3,
        reduction="none",
    )
    uniform = phx.uq.operator_ensemble_energy_distance(
        left,
        right,
        measure="uniform",
        reduction="none",
    )

    assert quadrature.shape == (2,)
    assert uniform.shape == (2,)
    assert jnp.all(quadrature > 0.0)
    assert jnp.all(uniform > 0.0)

    shifted_batch = _transition_batch(cases=2, size=5)
    shifted = phx.uq.operator_predictive_from_samples(
        jr.normal(jr.key(2), (8, 2, 5)),
        shifted_batch,
        phx.nn.operator.OperatorOutputSpec("scalar"),
        sample_axes=(phx.uq.SampleAxis("shifted", "process"),),
        field_name="output",
        query_name="query",
    )
    with pytest.raises(ValueError, match="physical output contract"):
        phx.uq.operator_ensemble_energy_distance(left, shifted)


def test_distributional_semigroup_recognizes_brownian_composition_and_key_replay():
    batch = _transition_batch(cases=2, size=4)
    objective = phx.nn.operator.training.DistributionalSemigroupObjective(
        num_samples=32,
        chunk_size=8,
        reduction="mean",
        key_mode="fold_in",
    )
    consistent = objective(
        _BrownianTransition(0.5),
        batch,
        0.3,
        0.4,
        _condition,
        _advance,
        key=jr.key(9),
    )
    replay = objective(
        _BrownianTransition(0.5),
        batch,
        0.3,
        0.4,
        _condition,
        _advance,
        key=jr.key(9),
    )
    inconsistent = objective(
        _BrownianTransition(1.0),
        batch,
        0.3,
        0.4,
        _condition,
        _advance,
        key=jr.key(9),
    )

    assert jnp.isfinite(consistent)
    assert jnp.array_equal(consistent, replay)
    assert consistent < inconsistent
    compiled = eqx.filter_jit(
        lambda candidate: objective(
            candidate,
            batch,
            0.3,
            0.4,
            _condition,
            _advance,
            key=jr.key(9),
        )
    )(_BrownianTransition(0.5))
    gradient = eqx.filter_grad(
        lambda candidate: objective(
            candidate,
            batch,
            0.3,
            0.4,
            _condition,
            _advance,
            key=jr.key(9),
        )
    )(_BrownianTransition(0.5))
    assert jnp.isfinite(compiled)
    assert jnp.isfinite(gradient.scale_power)
    with pytest.raises(ValueError, match="requires a PRNG key"):
        objective(
            _BrownianTransition(),
            batch,
            0.3,
            0.4,
            _condition,
            _advance,
        )
    with pytest.raises(ValueError, match="process distributions"):
        objective(
            _BrownianTransition(uncertainty_source="observation"),
            batch,
            0.3,
            0.4,
            _condition,
            _advance,
            key=jr.key(3),
        )


def test_sinkhorn_distributional_semigroup_recognizes_process_composition():
    batch = _transition_batch(cases=2, size=4)
    objective = phx.nn.operator.training.SinkhornDistributionalSemigroupObjective(
        num_samples=8,
        epsilon=1.0,
        reduction="mean",
        key_mode="fold_in",
    )
    consistent = objective(
        _BrownianTransition(0.5),
        batch,
        0.3,
        0.4,
        _condition,
        _advance,
        key=jr.key(19),
    )
    replay = objective(
        _BrownianTransition(0.5),
        batch,
        0.3,
        0.4,
        _condition,
        _advance,
        key=jr.key(19),
    )
    inconsistent = objective(
        _BrownianTransition(0.0),
        batch,
        0.3,
        0.4,
        _condition,
        _advance,
        key=jr.key(19),
    )
    gradient = eqx.filter_grad(
        lambda candidate: objective(
            candidate,
            batch,
            0.3,
            0.4,
            _condition,
            _advance,
            key=jr.key(19),
        )
    )(_BrownianTransition(0.5))

    assert jnp.isfinite(consistent)
    assert jnp.array_equal(consistent, replay)
    assert consistent < inconsistent
    assert jnp.isfinite(gradient.scale_power)
    with pytest.raises(ValueError, match="requires a PRNG key"):
        objective(
            _BrownianTransition(),
            batch,
            0.3,
            0.4,
            _condition,
            _advance,
        )
