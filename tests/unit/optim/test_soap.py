import jax
import jax.numpy as jnp
import optax
import pytest

import phydrax as phx


def test_soap_preconditioner_respects_axis_resource_bounds():
    parameters = {
        "matrix": jnp.zeros((2, 3), dtype=jnp.float64),
        "scalar": jnp.asarray(1.0),
        "vector": jnp.zeros((4,), dtype=jnp.float64),
    }
    transformation = phx.optim.scale_by_soap(
        max_preconditioner_size=2,
        precondition_1d=False,
        preconditioner_dtype=jnp.float32,
    )
    state = transformation.init(parameters)

    matrix = state.covariance["matrix"].matrices
    assert matrix[0].shape == (2, 2)
    assert matrix[0].dtype == jnp.float32
    assert matrix[1] is None
    low_precision_state = phx.optim.scale_by_soap().init(
        jnp.zeros((2, 2), dtype=jnp.float16)
    )
    assert low_precision_state.covariance.matrices[0].dtype == jnp.float32
    assert state.covariance["scalar"].matrices == ()
    assert state.covariance["vector"].matrices == (None,)


def test_soap_vector_path_matches_sign_normalization_after_basis_warmup():
    parameters = jnp.asarray([1.0, -2.0])
    gradients = jnp.asarray([2.0, -4.0])
    optimizer = phx.optim.soap(
        0.1,
        b1=0.0,
        b2=0.0,
        eps=1e-12,
        precondition_1d=False,
    )
    state = optimizer.init(parameters)

    warmup, state = optimizer.update(gradients, state, parameters)
    updates, state = optimizer.update(gradients, state, parameters)

    assert state.count == 2
    assert jnp.array_equal(warmup, jnp.zeros_like(parameters))
    assert jnp.allclose(updates, jnp.asarray([-0.1, 0.1]), atol=1e-10)


def test_soap_weight_decay_starts_after_basis_warmup():
    parameters = jnp.asarray([2.0, -4.0])
    gradients = jnp.zeros_like(parameters)
    optimizer = phx.optim.soap(
        0.1,
        b1=0.0,
        b2=0.0,
        weight_decay=0.5,
    )
    state = optimizer.init(parameters)

    warmup, state = optimizer.update(gradients, state, parameters)
    updates, _ = optimizer.update(gradients, state, parameters)

    assert jnp.array_equal(warmup, jnp.zeros_like(parameters))
    assert jnp.allclose(updates, -0.05 * parameters)


def test_soap_schedule_counts_effective_parameter_updates():
    parameters = jnp.asarray([1.0])
    gradients = jnp.asarray([2.0])
    optimizer = phx.optim.soap(
        lambda count: 0.1 * (count + 1),
        b1=0.0,
        b2=0.0,
        eps=1e-12,
    )
    state = optimizer.init(parameters)

    _, state = optimizer.update(gradients, state, parameters)
    first, state = optimizer.update(gradients, state, parameters)
    second, _ = optimizer.update(gradients, state, parameters)

    assert jnp.allclose(first, -0.1)
    assert jnp.allclose(second, -0.2)


def test_soap_mixed_preconditioner_dtype_is_jittable_and_orthogonal():
    parameters = jnp.asarray(
        [[1.0, -2.0], [0.5, 3.0]],
        dtype=jnp.float64,
    )
    gradients = jnp.asarray(
        [[2.0, -1.0], [0.25, 4.0]],
        dtype=jnp.float64,
    )
    transformation = phx.optim.scale_by_soap(
        b1=0.9,
        b2=0.99,
        precondition_frequency=1,
        preconditioner_dtype=jnp.float32,
    )
    state = transformation.init(parameters)

    @jax.jit
    def step(current_state):
        return transformation.update(gradients, current_state, parameters)

    warmup, state = step(state)
    directions, state = step(state)

    assert jnp.array_equal(warmup, jnp.zeros_like(parameters))
    assert jnp.all(jnp.isfinite(directions))
    for basis in state.basis.matrices:
        assert basis.dtype == jnp.float32
        assert jnp.allclose(basis.T @ basis, jnp.eye(2), atol=2e-5)


def test_soap_decreases_dense_quadratic_under_jit_scan():
    initial = jnp.asarray([[3.0, -2.0], [1.0, 4.0]])
    target = jnp.asarray([[0.5, 1.0], [-1.0, 0.25]])
    optimizer = phx.optim.soap(
        0.05,
        b1=0.9,
        b2=0.99,
        precondition_frequency=2,
    )
    state = optimizer.init(initial)

    def body(carry, _):
        parameters, optimizer_state = carry
        gradients = 2.0 * (parameters - target)
        updates, optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            parameters,
        )
        return (optax.apply_updates(parameters, updates), optimizer_state), None

    (trained, _), _ = jax.jit(
        lambda carry: jax.lax.scan(body, carry, xs=None, length=80)
    )((initial, state))
    initial_loss = jnp.sum(jnp.square(initial - target))
    final_loss = jnp.sum(jnp.square(trained - target))

    assert final_loss < 0.1 * initial_loss


def test_soap_checkpoint_resume_matches_uninterrupted_steps(tmp_path):
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Parameter(2.0)
    component = domain.component()
    condition = phx.conditions.Residual("u", component, lambda value: value)
    batch = component.points({"x": jnp.asarray([[0.1], [0.4], [0.7], [0.9]])})
    term = phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(
            phx.integration.from_samples(
                phx.integration.mean_over(component),
                batch,
            )
        ),
    )


    solver = phx.solver.FunctionalSolver(functions={"u": field}, terms=(term,))
    plan = phx.solver.FunctionalTrainingPlan(
        checkpoint=phx.solver.FunctionalCheckpointPolicy(
            tmp_path / "soap",
            every=1,
        )
    )
    solver.solve(
        num_iter=1,
        optim=phx.optim.soap(0.05),
        keep_best=False,
        log_every=0,
        training=plan,
    )
    resumed = solver.solve(
        num_iter=3,
        optim=phx.optim.soap(0.05),
        keep_best=False,
        log_every=0,
        training=plan,
        resume=True,
    )
    uninterrupted = solver.solve(
        num_iter=3,
        optim=phx.optim.soap(0.05),
        keep_best=False,
        log_every=0,
        training=phx.solver.FunctionalTrainingPlan(),
    )

    assert isinstance(resumed.training_state.optimizer_state, phx.optim.SOAPState)
    assert jnp.allclose(
        resumed.training_state.current_functions["u"].func(),
        uninterrupted.training_state.current_functions["u"].func(),
    )
@pytest.mark.parametrize(
    "parameters",
    (
        jnp.asarray([1, 2], dtype=jnp.int32),
        jnp.asarray([1.0 + 1.0j], dtype=jnp.complex64),
    ),
)
def test_soap_rejects_nonreal_parameter_coordinates(parameters):
    with pytest.raises(TypeError, match="real floating-point"):
        phx.optim.soap().init(parameters)


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"b1": 1.0}, "b1"),
        ({"b2": -0.1}, "b2"),
        ({"preconditioner_decay": 1.0}, "preconditioner_decay"),
        ({"eps": 0.0}, "eps"),
        ({"precondition_frequency": 0}, "precondition_frequency"),
        ({"max_preconditioner_size": 0}, "max_preconditioner_size"),
        ({"preconditioner_dtype": jnp.float16}, "preconditioner_dtype"),
    ),
)
def test_soap_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        phx.optim.soap(**kwargs)
