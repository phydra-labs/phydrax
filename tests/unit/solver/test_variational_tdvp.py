import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


_HAMILTONIAN = jnp.asarray(
    [
        [-1.0, -0.5, -0.5, 0.0],
        [-0.5, 1.0, 0.0, -0.5],
        [-0.5, 0.0, 1.0, -0.5],
        [0.0, -0.5, -0.5, -1.0],
    ]
)


class _TableModel(eqx.Module):
    parameters: jax.Array

    def __call__(self, configuration):
        bits = (configuration > 0).astype(jnp.int32)
        index = 2 * bits[0] + bits[1]
        value = self.parameters[index]
        if jnp.iscomplexobj(value):
            return phx.operators.LogAmplitude(
                jnp.real(value), jnp.exp(1j * jnp.imag(value))
            )
        return phx.operators.LogAmplitude(value, 1.0 + 0.0j)


def _operator():
    def diagonal(configurations):
        return -configurations[..., 0] * configurations[..., 1]

    def connections(configurations):
        first = configurations.at[..., 0].multiply(-1)
        second = configurations.at[..., 1].multiply(-1)
        connected = jnp.stack((first, second), axis=-2)
        shape = configurations.shape[:-1] + (2,)
        return phx.operators.ConnectedConfigurations(
            connected,
            -0.5 * jnp.ones(shape),
            jnp.ones(shape, dtype=bool),
            configuration_shape=(2,),
        )

    return phx.operators.CallableDiscreteQuantumOperator(
        diagonal,
        connections,
        configuration_shape=(2,),
        operator_id="tdvp-ising",
    )


def _kernel():
    def sample(key, current):
        index = jr.randint(key, (), 0, current.shape[0])
        return current.at[index].multiply(-1)

    def log_prob(_proposed, current):
        return -jnp.log(float(current.shape[0]))

    return phx.sampling.MetropolisHastings(
        phx.sampling.CallableProposal(
            sample,
            log_prob,
            proposal_id="tdvp-spin-flip",
        )
    )


def _problem(parameters):
    return phx.solver.VariationalMonteCarloProblem(
        _TableModel(parameters),
        _operator(),
        _kernel(),
        jnp.asarray([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=jnp.int32),
    )


def _exact_energy(model):
    state = jnp.exp(model.parameters)
    return jnp.real(jnp.vdot(state, _HAMILTONIAN @ state) / jnp.vdot(state, state))


def test_imaginary_time_tdvp_decreases_exact_energy():
    problem = _problem(jnp.asarray([0.2, -0.1, 0.1, -0.2]))
    policy = phx.solver.VariationalTDVPPolicy(
        "imaginary-time",
        num_steps=2,
        step_size=0.03,
        draws_per_step=24,
        transitions_per_draw=2,
        warmup_steps=4,
        final_evaluation_draws=16,
        damping=0.1,
        final_chain_diagnostics=False,
    )
    result = phx.solver.solve_variational_tdvp(problem, policy, key=jr.key(30))

    assert result.successful
    assert result.completed_steps == 2
    assert result.times.shape == (3,)
    assert result.parameter_trajectory.shape == (3, 4)
    assert _exact_energy(result.final_state.model) < _exact_energy(problem.model)


@pytest.mark.parametrize("mode", ["real-time", "imaginary-time"])
def test_tdvp_stationary_eigenstate_has_zero_velocity(mode):
    _eigenvalues, eigenvectors = jnp.linalg.eigh(_HAMILTONIAN)
    ground = eigenvectors[:, 0]
    phase = jnp.sign(ground)
    parameters = jnp.log(jnp.abs(ground)) + 1j * jnp.where(phase < 0.0, jnp.pi, 0.0)
    problem = phx.solver.VariationalMonteCarloProblem(
        _TableModel(parameters),
        _operator(),
        _kernel(),
        jnp.asarray([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=jnp.int32),
        complex_parameter_mode="holomorphic",
    )
    policy = phx.solver.VariationalTDVPPolicy(
        mode,
        num_steps=1,
        step_size=1e-3,
        draws_per_step=8,
        final_evaluation_draws=8,
        damping=0.1,
        final_chain_diagnostics=False,
    )
    result = phx.solver.solve_variational_tdvp(problem, policy, key=jr.key(31))

    assert result.successful
    assert jnp.allclose(
        result.final_state.parameter_coordinates,
        problem.initial_coordinates,
        atol=1e-8,
    )
    assert result.velocity_norm_history[0] < 1e-8
