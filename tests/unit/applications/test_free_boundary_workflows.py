#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import phydrax as phx


class _TrainableFront(eqx.Module):
    offset: jnp.ndarray

    def __call__(self, time):
        return time[0] + self.offset


def test_exact_stefan_all_representations_satisfy_shared_problem():
    result = phx.applications.free_boundary.ExactStefanBenchmark().run(
        points_per_block=32,
        key=jr.key(0),
    )

    assert result.explicit.total < 1.0e-20
    assert result.reference.total < 1.0e-20
    assert result.implicit.total < 1.0e-3


def test_stefan_fit_optimizes_a_trainable_representation_end_to_end():
    benchmark = phx.applications.free_boundary.ExactStefanBenchmark()
    batch = phx.applications.free_boundary.stefan_collocation_batch(
        benchmark.parameters,
        interior_points=16,
        ambient_points=16,
        boundary_points=16,
        interface_points=16,
        initial_points=16,
        key=jr.key(8),
    )
    model = phx.applications.free_boundary.ExplicitFrontStefanPINN(
        benchmark.fields.temperature,
        _TrainableFront(jnp.asarray(0.4)),
    )
    loss = lambda candidate: phx.applications.free_boundary.explicit_front_stefan_loss(
        candidate,
        batch,
        benchmark.parameters,
        benchmark.data,
    )
    initial = loss(model).total
    fitted = phx.applications.free_boundary.fit_stefan_pinn(
        model,
        loss,
        steps=8,
        optimizer=optax.adam(0.05),
        jit=True,
    )

    assert fitted.loss_history.shape == (8,)
    assert fitted.final_loss.total < initial
    assert abs(float(fitted.model.front.offset) - 0.5) < 0.1


def test_relaxed_first_passage_weights_form_one_stopping_law():
    distance = jnp.asarray(((1.0, 0.1, -0.1, -1.0),))
    result = phx.applications.free_boundary.relaxed_first_passage_weights(
        distance,
        width=0.2,
        particle_phase="outside",
    )

    assert jnp.all(result.stopping_weights >= 0.0)
    assert jnp.all(result.cumulative_stopping_probability <= 1.0 + 1.0e-14)
    np.testing.assert_allclose(
        result.cumulative_stopping_probability + result.survival_probability,
        1.0,
        atol=1.0e-14,
    )


def test_stationary_probabilistic_stefan_moments_have_zero_loss():
    times = jnp.asarray((0.0, 0.5, 1.0))
    domain_points = jnp.asarray(((-1.0,), (1.0,)))
    paths_outside = jnp.ones((4, 3, 1))
    paths_inside = -jnp.ones((4, 3, 1))
    batch = phx.applications.free_boundary.ProbabilisticStefanBatch(
        times=times,
        domain_points=domain_points,
        domain_weights=jnp.asarray((1.0, 1.0)),
        initial_solid_fraction=jnp.asarray((1.0, 0.0)),
        liquid_paths=paths_outside,
        solid_paths=paths_inside,
        test_centers=jnp.asarray(((0.0,),)),
        test_inverse_widths=jnp.asarray((1.0,)),
    )
    model = phx.applications.free_boundary.ProbabilisticLevelSetStefan(
        lambda point: point[0]
    )
    parameters = phx.applications.free_boundary.ProbabilisticStefanParameters(
        latent_heat=1.0,
        interface_width=0.1,
    )

    result = phx.applications.free_boundary.probabilistic_stefan_moment_loss(
        model,
        batch,
        parameters,
    )

    np.testing.assert_allclose(result.total, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.moment_residual, 0.0, atol=1.0e-14)


def test_benchmark_ladder_observables_are_exact_on_reference_data():
    times = jnp.asarray((0.0, 0.5, 1.0))
    modes = jnp.asarray((2, 3))
    rates = jnp.asarray((0.2, -0.1))
    initial = jnp.asarray((0.1, 0.05))
    amplitudes = initial[None, :] * jnp.exp(times[:, None] * rates[None, :])
    instability = phx.applications.free_boundary.mullins_sekerka_benchmark(
        amplitudes,
        times,
        modes,
        rates,
        initial,
    )
    topology = phx.applications.free_boundary.topology_event_benchmark(
        jnp.asarray((1, 1, 2)),
        jnp.asarray((1, 1, 2)),
        times,
    )
    obstacle = phx.applications.free_boundary.obstacle_complementarity_benchmark(
        jnp.asarray((1.0, 2.0)),
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((2.0, 0.0)),
    )

    assert instability.relative_l2_error == 0.0
    assert bool(topology.event_detected)
    assert topology.event_time_error == 0.0
    assert obstacle.gap_violation == 0.0
    assert obstacle.dual_violation == 0.0
    assert obstacle.complementarity_residual == 0.0


def test_hysing_fsi_and_fracture_benchmark_contracts():
    angle = jnp.linspace(0.0, 2.0 * jnp.pi, 65)[:-1]
    contour = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
    bubble = phx.applications.free_boundary.hysing_bubble_benchmark(
        jnp.ones((4,)),
        jnp.asarray(((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5))),
        jnp.full((4,), jnp.pi / 4.0),
        jnp.full((4,), 2.0),
        contour,
    )
    signal = jnp.sin(2.0 * jnp.pi * 2.0 * jnp.arange(64) * 0.01)
    fsi = phx.applications.free_boundary.turek_hron_fsi_benchmark(
        signal,
        signal,
        signal,
        signal,
        signal,
        signal,
        0.01,
    )
    crack = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    fracture = phx.applications.free_boundary.phase_field_fracture_benchmark(
        jnp.asarray(((0.0, 0.0), (0.5, 0.0), (1.0, 0.0))),
        jnp.asarray((0.0, 1.0, 2.0)),
        jnp.asarray((0.0, 1.0, 2.0)),
        jnp.asarray((0.0, 0.5, 1.0)),
        crack,
        crack,
    )

    assert abs(float(bubble.area) - jnp.pi) < 1.0e-12
    assert abs(float(bubble.circularity) - 1.0) < 2.0e-3
    assert bubble.mean_rise_velocity == 2.0
    assert fsi.tip_relative_l2 == 0.0
    assert fsi.dominant_frequency_error == 0.0
    assert fracture.irreversibility_violation == 0.0
    assert fracture.crack_path_hausdorff == 0.0


def test_trajectory_split_never_leaks_one_trajectory():
    result = phx.applications.free_boundary.trajectory_disjoint_ood_split(
        ("a", "a", "b", "b", "c", "c", "d", "d"),
        jnp.asarray((False, False, False, False, False, False, True, True)),
        validation_fraction=0.34,
        interpolation_test_fraction=0.33,
    )
    sets = [
        set(map(int, result.train_indices)),
        set(map(int, result.validation_indices)),
        set(map(int, result.interpolation_test_indices)),
        set(map(int, result.extrapolation_test_indices)),
    ]

    assert set.union(*sets) == set(range(8))
    assert all(
        left.isdisjoint(right) for i, left in enumerate(sets) for right in sets[i + 1 :]
    )
    assert set(map(int, result.extrapolation_test_indices)) == {6, 7}
