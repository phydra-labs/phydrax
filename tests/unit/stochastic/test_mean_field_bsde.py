import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _tree_paths(*, num_steps=3, repeats=32):
    branches = 2**num_steps
    indices = jnp.repeat(jnp.arange(branches, dtype=jnp.int32), repeats)
    bits = (indices[:, None] >> jnp.arange(num_steps, dtype=jnp.int32)) & 1
    dt = 1.0 / num_steps
    increments = (2.0 * bits - 1.0) * jnp.sqrt(dt)
    states = jnp.concatenate(
        (jnp.zeros((indices.shape[0], 1)), jnp.cumsum(increments, axis=1)),
        axis=1,
    )
    return phx.stochastic.BSDEPathBatch(
        jnp.linspace(0.0, 1.0, num_steps + 1),
        states[..., None],
        increments[..., None],
        sample_shape=(indices.shape[0],),
        state_shape=(1,),
        noise_shape=(1,),
        path_id="mean-field-tree",
        process_id="wiener",
    )


def test_empirical_mean_field_interpolates_weighted_lagrangian_law():
    mean_field = phx.stochastic.EmpiricalMeanField(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[[0.0], [2.0]], [[2.0], [4.0]]]),
        sample_shape=(2,),
        state_shape=(1,),
        mean_field_id="weighted-flow",
        weights=jnp.asarray([[1.0, 1.0], [3.0, 3.0]]),
    )

    snapshot = mean_field.snapshot(0.5)

    assert snapshot.valid
    assert jnp.allclose(snapshot.particles[:, 0], jnp.asarray([1.0, 3.0]))
    assert jnp.allclose(snapshot.weights, jnp.asarray([0.25, 0.75]))
    assert jnp.allclose(snapshot.mean, jnp.asarray([2.5]))
    assert jnp.allclose(snapshot.covariance, jnp.asarray([[0.75]]))
    assert jnp.allclose(snapshot.effective_sample_size, 1.6)
    assert jnp.allclose(snapshot.expectation(lambda state: state**2), jnp.asarray([7.0]))
    assert not mean_field.snapshot(-0.1).valid


def test_mean_field_control_adapter_builds_hamiltonian_bsde_and_recovers_policy():
    paths = _tree_paths()
    mean_field = phx.stochastic.EmpiricalMeanField.from_paths(paths)
    adapter = phx.stochastic.MeanFieldBSDEControlAdapter(
        lambda time, state, law, value, z, args: -z.reshape((1,)),
        lambda time, state, law, action, args: 0.5 * action**2,
        lambda time, state, law, action, args: action,
        control_shape=(1,),
        output_shape=(1,),
        noise_shape=(1,),
        adapter_id="linear-quadratic",
    )
    problem = phx.stochastic.adapt_mean_field_control_bsde(
        lambda key: paths,
        mean_field,
        lambda time, state, law, args: -state + law.mean,
        lambda time, state, law, args: jnp.ones((1, 1)),
        lambda state, law, args: 0.5 * (state - law.mean) ** 2,
        adapter,
        state_shape=(1,),
        problem_id="mean-field-control",
        process_id="wiener",
    )
    bsde = problem.as_bsde_problem()

    z = jnp.asarray([[2.0]])
    generator = bsde.generator(
        jnp.asarray(0.5),
        jnp.asarray([1.0]),
        jnp.asarray([0.0]),
        z,
        None,
    )
    control = phx.stochastic.evaluate_mean_field_bsde_control(
        problem,
        0.5,
        jnp.asarray([1.0]),
        jnp.asarray([0.0]),
        z,
    )

    assert jnp.allclose(generator, jnp.asarray([-2.0]))
    assert jnp.allclose(control, jnp.asarray([-2.0]))
    assert jnp.allclose(bsde.terminal(jnp.asarray([1.0]), None), jnp.asarray([0.5]))
    assert bsde.sample(jr.key(0)).path_id == paths.path_id

    result = phx.solver.solve_bsde_least_squares(
        bsde,
        phx.solver.PolynomialBSDERegressionBasis((1,), 2),
        paths=paths,
        ridge=1e-8,
    )
    assert result.successful
    assert jnp.all(jnp.isfinite(result.values))
    assert jnp.all(jnp.isfinite(result.controls))


def test_mean_field_adapter_rejects_forward_support_mismatch():
    paths = _tree_paths()
    shifted_paths = phx.stochastic.BSDEPathBatch(
        paths.times + 1.0,
        paths.states,
        paths.wiener_increments,
        sample_shape=paths.sample_shape,
        state_shape=paths.state_shape,
        noise_shape=paths.noise_shape,
        path_id="shifted",
        process_id=paths.process_id,
    )
    mean_field = phx.stochastic.EmpiricalMeanField.from_paths(paths)
    problem = phx.stochastic.MeanFieldBSDEProblem(
        lambda key: shifted_paths,
        mean_field,
        lambda time, state, law, args: jnp.zeros((1,)),
        lambda time, state, law, args: jnp.ones((1, 1)),
        lambda time, state, law, value, control, args: jnp.zeros((1,)),
        lambda state, law, args: jnp.zeros((1,)),
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id="mismatched-support",
        process_id="wiener",
    )

    with pytest.raises(ValueError, match="share time support"):
        problem.as_bsde_problem().sample(jr.key(1))
