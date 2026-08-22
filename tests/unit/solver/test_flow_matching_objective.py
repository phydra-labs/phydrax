import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx


class _ConstantVelocity(eqx.Module):
    value: jnp.ndarray

    def __call__(self, state, time):
        del time
        return jnp.broadcast_to(self.value, state.shape)


class _ContextVelocity(eqx.Module):
    def __call__(self, state, time, condition):
        del state, time
        return condition


def _velocity_function(model, dimension, *, with_context=False):
    state = phx.domain.HyperRectangle(
        jnp.full((dimension,), -20.0),
        jnp.full((dimension,), 20.0),
        label="x",
    )
    domain = state @ phx.domain.TimeInterval(0.0, 1.0)
    if with_context:
        condition = phx.domain.HyperRectangle(
            jnp.full((dimension,), -20.0),
            jnp.full((dimension,), 20.0),
            label="condition",
        )
        domain = domain @ condition
        return domain.Function("x", "t", "condition")(model)
    return domain.Function("x", "t")(model)


def _translated_endpoints(*, count=32, dimension=2, offset=None):
    source = jr.normal(jr.key(1), (count, dimension))
    displacement = (
        jnp.arange(1.0, dimension + 1.0) if offset is None else jnp.asarray(offset)
    )
    return phx.transport.EndpointCouplingSample(
        source=source,
        target=source + displacement,
        source_indices=jnp.arange(count),
        target_indices=jnp.arange(count),
        valid=jnp.ones((count,), dtype=bool),
        log_weights=jnp.zeros((count,)),
        coupling_id="paired-translation",
        provenance="unit-test",
    )


def test_flow_matching_loss_is_zero_for_exact_conditional_velocity():
    endpoints = _translated_endpoints(offset=jnp.asarray([2.0, -1.0]))
    interpolant = phx.transport.LinearEndpointInterpolant((2,))
    term = phx.terms.FlowMatchingTerm("velocity", endpoints, interpolant)
    velocity = _velocity_function(_ConstantVelocity(jnp.asarray([2.0, -1.0])), 2)

    loss = eqx.filter_jit(
        lambda function: term.loss({"velocity": function}, key=jr.key(4))
    )(velocity)
    diagnostics = term.diagnostics({"velocity": velocity}, key=jr.key(4))

    assert jnp.allclose(loss, 0.0)
    assert jnp.allclose(diagnostics.root_mean_squared_component_error, 0.0)
    assert diagnostics.finite
    assert diagnostics.valid_fraction == 1.0


def test_flow_matching_passes_pair_aligned_context_to_velocity_field():
    target = jnp.asarray([[1.0, 2.0], [-3.0, 4.0], [2.5, -1.5]])
    endpoints = phx.transport.independent_endpoint_coupling(
        jnp.zeros((1, 2)),
        target,
        jr.key(8),
        num_pairs=12,
        target_context={"condition": target},
    )
    term = phx.terms.FlowMatchingTerm(
        "velocity",
        endpoints,
        phx.transport.LinearEndpointInterpolant((2,)),
    )
    velocity = _velocity_function(_ContextVelocity(), 2, with_context=True)

    assert jnp.allclose(term.loss({"velocity": velocity}, key=jr.key(9)), 0.0)


def test_flow_matching_term_trains_constant_translation_field():
    endpoints = _translated_endpoints(count=64, dimension=2, offset=[1.5, -0.5])
    velocity = _velocity_function(_ConstantVelocity(jnp.zeros((2,))), 2)
    term = phx.terms.FlowMatchingTerm(
        "velocity",
        endpoints,
        phx.transport.LinearEndpointInterpolant((2,)),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"velocity": velocity},
        terms=(term,),
    )

    trained = solver.solve(
        num_iter=60,
        optim=optax.adam(0.1),
        jit=True,
        keep_best=False,
        log_every=0,
    )
    learned = trained.functions["velocity"].func(jnp.zeros((2,)), jnp.asarray(0.5))

    assert jnp.allclose(learned, jnp.asarray([1.5, -0.5]), atol=5e-2)


def test_resampled_endpoint_provider_runs_once_per_optimizer_update():
    calls = []

    def provider(key):
        calls.append(key)
        return _translated_endpoints(count=16, dimension=1, offset=[1.0])

    velocity = _velocity_function(_ConstantVelocity(jnp.zeros((1,))), 1)
    term = phx.terms.FlowMatchingTerm(
        "velocity",
        provider,
        phx.transport.LinearEndpointInterpolant((1,)),
        sampling_mode="resample",
    )
    solver = phx.solver.FunctionalSolver(
        functions={"velocity": velocity},
        terms=(term,),
    )
    solver.solve(
        num_iter=4,
        optim=optax.sgd(0.05),
        jit=True,
        keep_best=False,
        log_every=0,
    )

    assert len(calls) == 4
