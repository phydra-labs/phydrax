import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


class _AdditiveFieldOperator(eqx.Module):
    def __call__(self, batch, /, *, key=None):
        del key
        return (
            batch.input("state").values
            + batch.input("driver").values
            + batch.input("duration").values * batch.input("forcing").values
        )


class _ScalarPathwiseTransition(phx.stochastic.AbstractPathwiseTransition):
    state_shape: tuple[int, ...] = eqx.field(static=True)
    driver_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(self):
        self.state_shape = ()
        self.driver_shape = ()
        self.process_id = "scalar-pathwise"

    def pathwise_transition(self, state, /, *, t0, t1, driver_increment):
        del t0, t1
        return state + driver_increment

    def combine_driver_segments(self, first, second, /):
        return first + second


def test_differential_transition_adapts_ode_and_sde_solvers():
    ode = phx.stochastic.DifferentialTransitionKernel(
        lambda time, state, context: state * context.args,
        state_shape=(1,),
        process_id="exponential",
    )
    context = phx.stochastic.StateSpaceStepContext.empty(args=1.0)
    ode_sample = ode.sample(jr.key(0), jnp.asarray([1.0]), 0.0, 1.0, context)
    wiener_term = phx.solver.WienerTerm(
        "w",
        lambda time, state, context: (
            jnp.ones(state.shape + (1,)) + 0.0 * context.step_index
        ),
        (1,),
        structure="additive",
    )
    sde = phx.stochastic.DifferentialTransitionKernel(
        lambda time, state, context: jnp.zeros_like(state) * context.args,
        state_shape=(1,),
        process_id="brownian",
        wiener_terms=(wiener_term,),
        dt0=0.01,
        wiener_tolerance=1e-4,
    )
    first = sde.sample(jr.key(1), jnp.asarray([0.0]), 0.0, 0.1, context)
    replay = sde.sample(jr.key(1), jnp.asarray([0.0]), 0.0, 0.1, context)

    assert ode_sample.valid
    assert jnp.allclose(ode_sample.values, jnp.exp(1.0), rtol=1e-5)
    assert first.valid
    assert jnp.array_equal(first.values, replay.values)
    traced = jax.jit(
        lambda start, end: ode.sample(
            jr.key(10),
            jnp.asarray([1.0]),
            start,
            end,
            context,
        )
    )(jnp.asarray(0.0), jnp.asarray(1.0))
    assert traced.valid
    assert jnp.allclose(traced.values, jnp.exp(1.0), rtol=1e-5)


def test_jump_and_hybrid_transition_adapters_preserve_solver_status():
    process = phx.stochastic.JumpProcess(
        lambda time, state, context: jnp.asarray([1.0]) + 0.0 * context.step_index,
        lambda state, channel, mark, context: (
            state + jnp.asarray([1.0]) + 0.0 * context.step_index
        ),
        state_shape=(1,),
        num_channels=1,
        process_id="counting",
    )
    jump = phx.stochastic.JumpTransitionKernel(process, max_events_per_channel=16)
    hybrid = phx.stochastic.JumpDifferentialTransitionKernel(
        lambda time, state, context: jnp.ones_like(state) + 0.0 * context.step_index,
        process,
        state_shape=(1,),
        max_events_per_channel=16,
    )
    context = phx.stochastic.StateSpaceStepContext.empty()
    jump_sample = jump.sample(jr.key(2), jnp.asarray([0.0]), 0.0, 1.0, context)
    hybrid_sample = hybrid.sample(jr.key(3), jnp.asarray([0.0]), 0.0, 1.0, context)

    assert jump_sample.valid
    assert hybrid_sample.valid
    assert jump_sample.status == phx.stochastic.JUMP_SUCCESS
    assert hybrid_sample.status == phx.stochastic.JUMP_SUCCESS
    traced_jump = jax.jit(
        lambda start, end: jump.sample(
            jr.key(20),
            jnp.asarray([0.0]),
            start,
            end,
            context,
        )
    )(jnp.asarray(0.0), jnp.asarray(1.0))
    traced_hybrid = jax.jit(
        lambda start, end: hybrid.sample(
            jr.key(21),
            jnp.asarray([0.0]),
            start,
            end,
            context,
        )
    )(jnp.asarray(0.0), jnp.asarray(1.0))
    assert traced_jump.status == phx.stochastic.JUMP_SUCCESS
    assert traced_hybrid.status == phx.stochastic.JUMP_SUCCESS
    assert hybrid_sample.values[0] >= 1.0


def test_scalar_transition_batches_keep_per_member_validity():
    transition = phx.stochastic.PathwiseTransitionKernel(
        _ScalarPathwiseTransition(),
        lambda _key, _t0, _t1, _context: jnp.asarray([1.0, 1.0, 1.0]),
    )
    sample = transition.sample(
        jr.key(30),
        jnp.asarray([0.0, jnp.nan, 2.0]),
        0.0,
        1.0,
        phx.stochastic.StateSpaceStepContext.empty(),
    )

    assert sample.values.shape == (3,)
    assert jnp.array_equal(sample.valid, jnp.asarray([True, False, True]))
    assert jnp.array_equal(sample.status, jnp.asarray([0, 1, 0]))

    scalar_process = phx.stochastic.JumpProcess(
        lambda _time, state, _context: jnp.asarray([jnp.where(state < 2, 1.0, 0.0)]),
        lambda state, _channel, _mark, _context: jnp.minimum(state + 1, 2),
        state_shape=(),
        num_channels=1,
        process_id="scalar-finite-birth",
    )
    scalar_transition = phx.stochastic.FiniteStateTransitionKernel(
        phx.solver.finite_state_generator(
            scalar_process,
            jnp.asarray([0, 1, 2]),
        )
    )
    scalar_sample = scalar_transition.sample(
        jr.key(31),
        jnp.asarray([0, 1]),
        0.0,
        1.0,
        phx.stochastic.StateSpaceStepContext.empty(),
    )
    assert scalar_sample.values.shape == (2,)
    assert scalar_sample.valid.shape == (2,)


def test_finite_state_transition_has_exact_normalized_mass_and_filters():
    process = phx.stochastic.JumpProcess(
        lambda time, state, context: jnp.where(
            state[0] < 2, jnp.asarray([1.0]), jnp.asarray([0.0])
        ),
        lambda state, channel, mark, context: jnp.minimum(state + 1, 2),
        state_shape=(1,),
        num_channels=1,
        process_id="finite-birth",
    )
    states = jnp.asarray([[0], [1], [2]])
    generator = phx.solver.finite_state_generator(process, states)
    transition = phx.stochastic.FiniteStateTransitionKernel(generator)
    context = phx.stochastic.StateSpaceStepContext.empty()
    probabilities = jnp.exp(
        jnp.stack(
            [
                transition.log_prob(state, jnp.asarray([0]), 0.0, 1.0, context)
                for state in states
            ]
        )
    )
    prior = phx.stochastic.CategoricalStatePrior(
        states,
        jnp.asarray([1.0, 0.0, 0.0]),
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([1.0]),
        jnp.asarray([[1.0]]),
        case_ids=("chain",),
        sequence_id="finite-chain",
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="finite-chain"
    )
    problem = phx.stochastic.StateSpaceProblem(
        model, observations, initial_time=0.0, problem_id="finite-chain"
    )
    result = phx.uq.bootstrap_particle_filter(jr.key(4), problem, num_particles=64)

    assert jnp.allclose(jnp.sum(probabilities), 1.0)
    assert result.successful
    assert jnp.all(jnp.isin(result.particles, states))


def test_operator_pathwise_transition_filters_complete_fields():
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, 4, endpoint=False),
        quadrature_weights=jnp.full((4,), 0.25),
        periodic=True,
    )
    zeros = jnp.zeros((4,))
    inputs = {
        "state": phx.nn.operator.FunctionSamples(values=zeros, axes=(axis,)),
        "duration": phx.nn.operator.FunctionSamples(values=jnp.ones((4,)), axes=(axis,)),
        "forcing": phx.nn.operator.FunctionSamples(values=zeros, axes=(axis,)),
        "driver": phx.nn.operator.FunctionSamples(values=zeros, axes=(axis,)),
    }
    batch = phx.nn.operator.OperatorBatch(
        inputs=inputs,
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
    )
    spec = phx.nn.operator.training.OperatorTransitionSpec(
        phx.nn.operator.OperatorOutputSpec("scalar"),
        driver_bindings=(
            phx.nn.operator.training.OperatorDriverBinding(
                "driver", "wiener", kind="wiener", quantity="increment"
            ),
        ),
    )
    law = phx.nn.operator.training.OperatorPathwiseTransition(
        _AdditiveFieldOperator(), batch, spec, process_id="field-driver"
    )
    transition = phx.stochastic.PathwiseTransitionKernel(
        law,
        lambda key, t0, t1, context: (
            jnp.sqrt(t1 - t0) * jr.normal(key, (4,)) + 0.0 * context.step_index
        ),
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((4,)), jnp.eye(4), state_shape=(4,), prior_id="field-prior"
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.full((1, 4), 0.25),
        jnp.asarray([[0.2]]),
        state_shape=(4,),
        observation_shape=(1,),
    )
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5]),
        jnp.asarray([[0.0]]),
        case_ids=("field",),
        sequence_id="field-observation",
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="operator-state-space"
    )
    problem = phx.stochastic.StateSpaceProblem(
        model, observations, initial_time=0.0, problem_id="operator-state-space"
    )
    result = phx.uq.bootstrap_particle_filter(jr.key(5), problem, num_particles=16)

    assert result.particles.shape == (1, 16, 4)
    assert result.successful
    assert result.problem.model.transition.process_id == "field-driver"
