import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _spatiotemporal_trajectory():
    times = jnp.asarray([0.0, 0.2, 0.6, 1.0])
    spatial_axis = phx.domain.FourierAxisSpec(4).materialize(0.0, 1.0)
    path = jnp.arange(3.0)[:, None, None]
    time = times[None, :, None]
    space = spatial_axis.nodes[None, None, :]
    states = path + time + space**2
    valid = jnp.asarray(
        [
            [True, True, True, True],
            [True, False, False, False],
            [True, True, True, True],
        ]
    )
    realization = phx.stochastic.WienerRealization(
        jr.key(0),
        (1,),
        support=(0.0, 1.0),
        sample_shape=(3,),
    )
    trajectory = phx.stochastic.StochasticTrajectory(
        times,
        states,
        valid=valid,
        realization_axes=("path",),
        realization_shape=(3,),
        state_axes=("space",),
        realizations=(realization,),
    )
    discretization = phx.solver.TensorGridDiscretization((spatial_axis,))
    return trajectory, discretization


def test_staged_time_then_path_reduction_contains_failed_paths():
    trajectory, _ = _spatiotemporal_trajectory()
    path_target = phx.stochastic.trajectory_measure(trajectory, mode="path")
    time_target = phx.stochastic.time_measure(trajectory)
    values = path_target.samples

    time_estimate = phx.integration.integrate(values, time_target)
    path_estimate = phx.integration.integrate(time_estimate.value, path_target)

    expected_paths = jnp.sum(
        trajectory.states * time_target.weights.data[..., None],
        axis=1,
    )
    expected = jnp.mean(expected_paths[jnp.asarray([0, 2])], axis=0)
    assert time_estimate.status[1] == int(
        phx.integration.IntegrationStatus.NO_VALID_SAMPLES
    )
    assert jnp.isnan(time_estimate.value.data[1]).all()
    assert path_estimate.value.dims == ("space",)
    assert jnp.allclose(path_estimate.value.data, expected)
    assert path_estimate.successful


def test_staged_space_time_path_reduction_is_jittable_and_differentiable():
    trajectory, discretization = _spatiotemporal_trajectory()
    path_target = phx.stochastic.trajectory_measure(trajectory, mode="path")
    time_target = phx.stochastic.time_measure(trajectory)
    spatial_target = phx.solver.spatial_measure(
        discretization,
        spatial_dims="space",
    )
    dims = ("path", "time", "space")

    def staged(scale):
        values = cx.Field(scale * trajectory.states, dims=dims)
        spatial = phx.integration.integrate(values, spatial_target).value
        temporal = phx.integration.integrate(spatial, time_target).value
        return phx.integration.integrate(temporal, path_target).value.data

    spatial_values = jnp.sum(
        trajectory.states * discretization.quadrature_weights,
        axis=-1,
    )
    path_values = jnp.sum(spatial_values * time_target.weights.data, axis=-1)
    expected = jnp.mean(path_values[jnp.asarray([0, 2])])
    compiled = jax.jit(staged)(jnp.asarray(2.0))
    derivative = jax.grad(staged)(jnp.asarray(1.0))

    assert jnp.allclose(compiled, 2.0 * expected)
    assert jnp.allclose(derivative, expected)
