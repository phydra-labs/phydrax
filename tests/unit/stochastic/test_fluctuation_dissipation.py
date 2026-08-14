import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._model import AbstractArrayModel


class _QuadraticEnergy1D(AbstractArrayModel):
    in_size: int = 1
    out_size: str = "scalar"

    def __call__(self, state, /, *, key=None):
        del key
        return 0.5 * state[0] ** 2


class _QuadraticEnergy2D(AbstractArrayModel):
    in_size: int = 2
    out_size: str = "scalar"

    def __call__(self, state, /, *, key=None):
        del key
        return 0.5 * jnp.vdot(state, state).real


class _VariableMobility(AbstractArrayModel):
    in_size: int = 1
    out_size: int = 1

    def __call__(self, state, /, *, key=None):
        del key
        return jnp.asarray([1.0 + state[0]])


class _StateSkew(AbstractArrayModel):
    in_size: int = 2
    out_size: int = 1

    def __call__(self, state, /, *, key=None):
        del key
        return state[:1]


class _RankOneMobility(AbstractArrayModel):
    in_size: int = 2
    out_size: int = 3

    def __call__(self, state, /, *, key=None):
        del state, key
        return jnp.asarray([1.0, 0.0, 0.0])


def _variable_mobility_dynamics(temperature=0.7):
    field = phx.nn.models.PortHamiltonianVectorField(
        state_size=1,
        energy=_QuadraticEnergy1D(),
        dissipation_model=_VariableMobility(),
        dissipation_structure="positive_semidefinite",
        key=jr.key(0),
    )
    return phx.stochastic.IsothermalPortHamiltonianDynamics(
        field,
        temperature=temperature,
        process_id="variable-mobility",
    )


def test_variable_mobility_includes_complete_ito_correction():
    temperature = 0.7
    dynamics = _variable_mobility_dynamics(temperature)
    state = jnp.asarray([0.2])
    mobility = (1.0 + state[0]) ** 2
    mobility_derivative = 2.0 * (1.0 + state[0])
    expected_correction = temperature * mobility_derivative
    expected_drift = -mobility * state[0] + expected_correction

    assert jnp.allclose(
        dynamics.ito_correction(state),
        jnp.asarray([expected_correction]),
        atol=1e-12,
        rtol=1e-12,
    )
    assert jnp.allclose(
        jax.jit(dynamics.drift)(0.0, state),
        jnp.asarray([expected_drift]),
        atol=1e-12,
        rtol=1e-12,
    )
    assert jnp.allclose(
        dynamics.diffusion_covariance(state),
        jnp.asarray([[2.0 * temperature * mobility]]),
        atol=1e-12,
        rtol=1e-12,
    )
    expected_generator = state[0] * expected_drift + temperature * mobility
    assert jnp.allclose(
        dynamics.energy_generator(state),
        expected_generator,
        atol=1e-12,
        rtol=1e-12,
    )
    assert jnp.allclose(
        dynamics.stationary_fokker_planck_residual(state),
        0.0,
        atol=1e-10,
        rtol=0.0,
    )


def test_state_dependent_skew_field_has_exact_divergence_correction():
    temperature = 0.4
    field = phx.nn.models.PortHamiltonianVectorField(
        state_size=2,
        energy=_QuadraticEnergy2D(),
        interconnection_model=_StateSkew(),
        dissipation_model=_RankOneMobility(),
        dissipation_structure="positive_semidefinite",
        key=jr.key(1),
    )
    dynamics = phx.stochastic.IsothermalPortHamiltonianDynamics(
        field,
        temperature=temperature,
        process_id="state-skew",
    )
    state = jnp.asarray([0.3, -0.2])

    assert jnp.allclose(
        dynamics.ito_correction(state),
        jnp.asarray([0.0, -temperature]),
        atol=1e-12,
        rtol=1e-12,
    )
    assert jnp.allclose(
        dynamics.diffusion_covariance(state),
        2.0 * temperature * field.dissipation_matrix(state),
        atol=1e-12,
        rtol=1e-12,
    )
    assert jnp.allclose(
        dynamics.stationary_fokker_planck_residual(state),
        0.0,
        atol=1e-10,
        rtol=0.0,
    )


def test_constant_structure_has_exact_zero_correction():
    field = phx.nn.models.PortHamiltonianVectorField(
        state_size=2,
        energy=_QuadraticEnergy2D(),
        initial_damping=0.2,
        key=jr.key(2),
    )
    dynamics = phx.stochastic.IsothermalPortHamiltonianDynamics(
        field,
        temperature=0.5,
        process_id="constant-structure",
    )

    assert jnp.array_equal(
        dynamics.ito_correction(jnp.asarray([0.4, -0.1])),
        jnp.zeros((2,)),
    )


def test_isothermal_dynamics_rejects_nonequilibrium_configuration():
    with pytest.raises(ValueError, match="strictly positive"):
        phx.stochastic.IsothermalPortHamiltonianDynamics(
            phx.nn.models.PortHamiltonianVectorField(
                state_size=1,
                key=jr.key(3),
            ),
            temperature=0.0,
            process_id="invalid-temperature",
        )
    with pytest.raises(ValueError, match="control"):
        phx.stochastic.IsothermalPortHamiltonianDynamics(
            phx.nn.models.PortHamiltonianVectorField(
                state_size=1,
                control_size=1,
                key=jr.key(4),
            ),
            temperature=0.5,
            process_id="controlled",
        )
    with pytest.raises(ValueError, match="dissipation"):
        phx.stochastic.IsothermalPortHamiltonianDynamics(
            phx.nn.models.PortHamiltonianVectorField(
                state_size=1,
                dissipative=False,
                key=jr.key(5),
            ),
            temperature=0.5,
            process_id="conservative",
        )


def test_thermodynamic_kernel_and_solver_preserve_process_contracts():
    dynamics = _variable_mobility_dynamics()
    kernel = dynamics.transition_kernel()
    context = phx.stochastic.StateSpaceStepContext.empty()
    state = jnp.asarray([0.1])
    sample = kernel.sample(jr.key(6), state, 0.0, 0.05, context)

    assert kernel.dynamics is dynamics
    assert sample.valid
    assert sample.process_id == "variable-mobility"
    assert sample.approximation_id == "euler-maruyama"
    assert jnp.isfinite(kernel.log_prob(sample.values, state, 0.0, 0.05, context))

    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
    )
    observation = phx.stochastic.GaussianObservationModel(
        lambda hidden, time, context: hidden,
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        kernel,
        observation,
        model_id="thermodynamic-state-space",
    )
    assert model.approximation_id == "euler-maruyama"
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.05, 0.1]),
        jnp.asarray([[0.1], [0.2]]),
        sequence_id="thermodynamic-observations",
    )
    state_space_problem = phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="thermodynamic-filtering-problem",
    )
    step_context = state_space_problem.step_context(0, 0)
    assert jnp.isfinite(
        model.transition.log_prob(
            jnp.asarray([0.2]),
            state,
            0.0,
            0.05,
            step_context,
        )
    )

    problem = phx.solver.DifferentialProblem(
        dynamics.drift,
        state,
        t0=0.0,
        t1=0.05,
        wiener_terms=(dynamics.wiener_term(basis_id="thermal-basis"),),
        interpretation="ito",
    )
    realization = phx.stochastic.WienerRealization(
        jr.key(7),
        (1,),
        support=(0.0, 0.05),
        tolerance=1e-3,
        noise_id="thermal-basis",
    )
    solution = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([0.0, 0.05]),
        realization=realization,
        dt0=0.005,
    )
    assert solution.states.shape == (2, 1)
    assert bool(solution.successful)
