import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _quaternion_coordinates(base_shape=(1,)):
    algebra = phx.metrix.algebra.QuaternionAlgebraSpec()
    coordinates = phx.linalg.AlgebraCoordinatePlan(
        algebra,
        public_storage="real_coordinates",
        public_dtype=jnp.float64,
    ).prepare(base_shape)
    return algebra, coordinates


def test_diffrax_algebra_policy_preserves_quaternion_public_layout():
    algebra, coordinates = _quaternion_coordinates()
    product = algebra.prepare_product()
    imaginary = jnp.asarray([0.0, 1.0, 0.0, 0.0])
    problem = phx.solver.DifferentialProblem(
        lambda time, state, args: product(imaginary, state),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
        t0=0.0,
        t1=0.2,
    )
    solution = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([0.2]),
        algebra_state_policy=phx.solver.DiffraxAlgebraStatePolicy(coordinates),
        rtol=1e-9,
        atol=1e-11,
    )
    evidence = solution.temporal_evidence.state_packing

    assert solution.states.shape == (1, 1, 4)
    assert jnp.allclose(
        solution.states[0, 0],
        jnp.asarray([jnp.cos(0.2), jnp.sin(0.2), 0.0, 0.0]),
        atol=2e-9,
    )
    assert isinstance(evidence, phx.solver.AlgebraStatePackingEvidence)
    assert evidence.algebra_id == algebra.algebra_id
    assert evidence.backend_shape == (4, 1)


def test_delay_and_segmented_delay_accept_real_algebra_coordinates():
    initial = jnp.asarray([[1.0, 0.0, 0.0, 0.0]])
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: jnp.zeros_like(state),
        lambda time, args: initial,
        (phx.solver.ConstantDelay("lag", 0.1),),
        t0=0.0,
        t1=0.2,
    )
    times = jnp.asarray([0.0, 0.1, 0.2])
    whole = phx.solver.solve_diffrax_delay(
        problem,
        save_times=times,
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.05,
    )
    segmented = phx.solver.solve_diffrax_delay_segmented(
        problem,
        save_times=times,
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.05,
        max_segments=8,
    )

    assert jnp.array_equal(whole.states, jnp.broadcast_to(initial, whole.states.shape))
    assert jnp.array_equal(segmented.states, whole.states)


def test_jump_differential_keeps_algebra_state_separate_from_real_hazards():
    initial = jnp.asarray([[1.0, 0.0, 0.0, 0.0]])
    differential = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.zeros_like(state),
        initial,
        t0=0.0,
        t1=0.2,
    )
    process = phx.stochastic.JumpProcess(
        lambda time, state, args: jnp.asarray([0.0]),
        lambda state, channel, mark, args: state,
        state_shape=initial.shape,
        num_channels=1,
        process_id="quaternion-no-jump",
    )
    realization = phx.stochastic.PoissonClockRealization(
        jr.key(4),
        1,
        support=(0.0, 0.2),
        max_events_per_channel=1,
        process_id=process.process_id,
    )
    solution = phx.solver.solve_jump_differential(
        phx.solver.JumpDifferentialProblem(differential, process),
        realization,
        save_times=jnp.asarray([0.0, 0.2]),
    )

    assert solution.states.shape == (2, 1, 4)
    assert jnp.array_equal(
        solution.states, jnp.broadcast_to(initial, solution.states.shape)
    )


def test_rough_dynamics_accept_explicit_real_octonion_coordinates():
    algebra = phx.metrix.algebra.OctonionAlgebraSpec()
    product = algebra.prepare_product(backend="sparse")
    direction = jnp.eye(8)[1]
    initial = jnp.eye(8)[0]
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: product(direction, state)[..., None],
        initial,
        driver_dimension=1,
    )
    partition = jnp.linspace(0.0, 0.1, 17)
    control = phx.stochastic.GeometricRoughPath.from_values(
        partition,
        partition[:, None],
    )
    solution = phx.solver.solve_rough_differential(
        problem,
        control,
        save_times=jnp.asarray([0.1]),
        solver=phx.solver.Davie(),
    )

    assert solution.states.shape == (1, 8)
    assert jnp.all(jnp.isfinite(solution.states))


def test_unit_complex_quaternion_and_nonassociative_geometry_boundaries():
    complex_geometry = phx.metrix.algebra.UnitComplexStateGeometry()
    quaternion_geometry = phx.metrix.algebra.UnitQuaternionStateGeometry()
    complex_point = jnp.asarray([1.0, 0.0])
    quaternion_point = jnp.asarray([1.0, 0.0, 0.0, 0.0])

    assert bool(complex_geometry.contains(complex_point))
    assert bool(quaternion_geometry.contains(quaternion_point))
    assert bool(
        quaternion_geometry.contains(
            quaternion_geometry.retract(
                quaternion_point,
                jnp.asarray([0.0, 0.2, 0.0, 0.0]),
            )
        )
    )
    with pytest.raises(ValueError, match="Nonassociative"):
        phx.metrix.algebra.unit_algebra_state_geometry(
            phx.metrix.algebra.OctonionAlgebraSpec()
        )
