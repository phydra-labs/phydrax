import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax._model import AbstractArrayModel


class _LinearField(AbstractArrayModel):
    scale: jax.Array = eqx.field(init=False)
    in_size: int = 1
    out_size: int = 1

    def __init__(self, scale):
        self.scale = jnp.asarray(scale)

    def __call__(self, state, /, *, key=None):
        del key
        return self.scale * state


def _scalar_kernel(scale=-0.5, diffusion=0.3):
    system = phx.dynamics.continuous_model_system(
        _LinearField(scale),
        state_layout=phx.dynamics.StateLayout((1,)),
        system_id="linear-drift",
    )
    noise = phx.solver.WienerTerm(
        "brownian",
        lambda time, state, args: jnp.full((1, 1), diffusion),
        (1,),
        basis_id="scalar-brownian",
    )
    return phx.stochastic.EulerMaruyamaTransitionKernel(
        system,
        (noise,),
        state_shape=(1,),
        noise_shape=(1,),
        process_id="linear-diffusion",
    )


def test_wiener_coefficient_matrix_flattens_only_state_and_noise_axes():
    term = phx.solver.WienerTerm(
        "matrix-state",
        lambda time, state, args: jnp.arange(8.0).reshape((2, 2, 2)),
        (2,),
    )
    state = jnp.zeros((2, 2))

    matrix = jax.jit(term.coefficient_matrix)(0.0, state, None)

    assert matrix.shape == (4, 2)
    assert jnp.array_equal(matrix, jnp.arange(8.0).reshape((4, 2)))


def test_euler_maruyama_matches_scalar_gaussian_and_is_differentiable():
    kernel = _scalar_kernel()
    context = phx.stochastic.StateSpaceStepContext.empty()
    state = jnp.asarray([2.0])
    start = jnp.asarray(0.2)
    end = jnp.asarray(0.6)
    next_state = jnp.asarray([1.7])

    parameters = jax.jit(kernel.parameters)(state, start, end, context)
    expected_mean = state + (end - start) * (-0.5 * state)
    expected_variance = (end - start) * 0.3**2
    residual = next_state[0] - expected_mean[0]
    expected_log_density = -0.5 * (
        residual**2 / expected_variance + jnp.log(2.0 * jnp.pi * expected_variance)
    )

    assert jnp.allclose(kernel.mean(state, start, end, context), expected_mean)
    assert jnp.allclose(parameters.covariance, expected_variance.reshape((1, 1)))
    assert jnp.allclose(
        jax.jit(kernel.log_prob)(next_state, state, start, end, context),
        expected_log_density,
        atol=1e-12,
        rtol=1e-12,
    )
    gradient = eqx.filter_grad(
        lambda candidate: candidate.log_prob(next_state, state, start, end, context)
    )(kernel)
    assert jnp.isfinite(gradient.system.vector_field.model.scale)


def test_euler_maruyama_preserves_rectangular_singular_diffusion():
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, args: jnp.zeros_like(state),
        state_layout=phx.dynamics.StateLayout((2,)),
        system_id="rank-one-system",
    )
    noise = phx.solver.WienerTerm(
        "rank-one",
        lambda time, state, args: jnp.asarray([[1.0], [2.0]]),
        (1,),
    )
    kernel = phx.stochastic.EulerMaruyamaTransitionKernel(
        system,
        (noise,),
        state_shape=(2,),
        noise_shape=(1,),
        process_id="rank-one-diffusion",
    )
    context = phx.stochastic.StateSpaceStepContext.empty()
    state = jnp.asarray([0.1, -0.3])
    mean = kernel.mean(state, 0.0, 0.25, context)

    assert kernel.covariance(state, 0.0, 0.25, context).shape == (2, 2)
    assert jnp.isfinite(
        kernel.log_prob(mean + jnp.asarray([0.2, 0.4]), state, 0.0, 0.25, context)
    )
    assert jnp.isneginf(
        kernel.log_prob(mean + jnp.asarray([0.2, 0.41]), state, 0.0, 0.25, context)
    )


def test_euler_maruyama_handles_multiaxis_state_and_invalid_interval():
    state_layout = phx.dynamics.StateLayout((2, 2))
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, args: jnp.ones_like(state),
        state_layout=state_layout,
        system_id="matrix-state-system",
    )
    noise = phx.solver.WienerTerm(
        "matrix-noise",
        lambda time, state, args: jnp.arange(8.0).reshape((2, 2, 2)) / 10.0,
        (2,),
    )
    kernel = phx.stochastic.EulerMaruyamaTransitionKernel(
        system,
        (noise,),
        state_shape=(2, 2),
        noise_shape=(2,),
        process_id="matrix-state-diffusion",
    )
    context = phx.stochastic.StateSpaceStepContext.empty()
    state = jnp.zeros((2, 2))

    assert kernel.mean(state, 0.0, 0.5, context).shape == (2, 2)
    assert kernel.covariance(state, 0.0, 0.5, context).shape == (4, 4)
    invalid_sample = kernel.sample(jr.key(0), state, 1.0, 1.0, context)
    assert not bool(invalid_sample.valid)
    assert invalid_sample.process_id == "matrix-state-diffusion"
    assert invalid_sample.approximation_id == "euler-maruyama"
    assert jnp.isneginf(kernel.log_prob(state, state, 1.0, 1.0, context))


def test_trajectory_quasi_likelihood_respects_masks_weights_inputs_and_ids():
    state_layout = phx.dynamics.StateLayout((1,))
    input_layout = phx.dynamics.InputLayout((1,))
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, inputs, args: -state + inputs,
        state_layout=state_layout,
        input_layout=input_layout,
        system_id="controlled-linear-system",
    )
    noise = phx.solver.WienerTerm(
        "controlled-noise",
        lambda time, state, args: jnp.asarray([[0.2]]),
        (1,),
    )
    kernel = phx.stochastic.EulerMaruyamaTransitionKernel(
        system,
        (noise,),
        state_shape=(1,),
        noise_shape=(1,),
        process_id="controlled-linear-diffusion",
    )
    trajectory = phx.dynamics.TrajectoryData(
        jnp.asarray([0.0, 0.1, 0.4, 0.9]),
        jnp.asarray([[0.0], [0.05], [0.1], [0.2]]),
        state_layout=state_layout,
        transition_valid=jnp.asarray([True, False, True]),
        weights=jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        inputs=jnp.asarray([[0.5], [0.25], [-0.1]]),
        input_layout=input_layout,
        source_id="controlled-trajectory",
        dataset_id="controlled-trajectory-data",
    )
    transitions = trajectory.transitions()
    likelihood = phx.stochastic.EulerMaruyamaQuasiLikelihood(kernel)

    result = jax.jit(likelihood.evaluate)(transitions)

    assert result.log_density.shape == (3,)
    assert result.log_density[1] == 0.0
    assert jnp.array_equal(result.transition_valid, jnp.asarray([True, False, True]))
    assert jnp.isfinite(result.mean_negative_log_likelihood)
    assert result.effective_weight == transitions.weights[0] + transitions.weights[2]
    assert result.process_id == "controlled-linear-diffusion"
    assert result.approximation_id == "euler-maruyama"
    assert result.dataset_id == "controlled-trajectory-data"
