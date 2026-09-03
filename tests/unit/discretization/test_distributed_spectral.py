#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.spectral._distributed import (
    DistributedSpectralExecutionPlan,
    SpectralMeshTopology,
    SpectralResourceError,
    SpectralTranspose,
)


def _slab(shape=(8, 8, 6), *, state_shape=(), padded_shape=None, devices=None):
    selected = (jax.devices("cpu")[0],) if devices is None else tuple(devices)
    topology = SpectralMeshTopology(
        (len(selected),),
        devices=selected,
        axis_names=("spectral",),
    )
    return DistributedSpectralExecutionPlan(
        topology,
        shape,
        state_shape=state_shape,
        padded_shape=padded_shape,
        coefficient_dtype=jnp.complex64,
    )


def test_one_device_is_real_identity_realization_with_round_trip_and_derivative():
    plan = _slab()
    x = jnp.arange(8)[:, None, None] * (2.0 * jnp.pi / 8.0)
    y = jnp.arange(8)[None, :, None] * (2.0 * jnp.pi / 8.0)
    z = jnp.arange(6)[None, None, :] * (2.0 * jnp.pi / 6.0)
    values = (jnp.sin(2.0 * x) + 0.25j * jnp.cos(y - z)).astype(jnp.complex64)

    modal = plan.to_modal(values)
    restored = plan.to_physical(modal)
    derivative = plan.to_physical(plan.modal_derivative(modal, 0))

    np.testing.assert_allclose(restored, values, rtol=2e-5, atol=2e-5)
    expected_derivative = jnp.broadcast_to(2.0 * jnp.cos(2.0 * x), derivative.real.shape)
    np.testing.assert_allclose(derivative.real, expected_derivative, rtol=3e-5, atol=3e-5)
    assert restored.sharding == plan.physical_layout.sharding(plan.topology)
    assert modal.sharding == plan.modal_layout.sharding(plan.topology)
    assert plan.report.host_gather is False


def test_padding_round_trip_global_reductions_and_autodiff():
    plan = _slab((6, 6, 4), padded_shape=(10, 12, 8))
    key = jax.random.key(8)
    modal = (
        jax.random.normal(key, plan.spatial_shape)
        + 1j * jax.random.normal(jax.random.fold_in(key, 1), plan.spatial_shape)
    ).astype(jnp.complex64)
    padded = plan.pad_modal(modal)

    np.testing.assert_allclose(plan.unpad_modal(padded), modal, rtol=2e-6, atol=2e-6)
    assert padded.shape == plan.padded_shape
    assert padded.sharding == plan.padded_modal_layout.sharding(plan.topology)

    diagnostics = plan.diagnostics(modal)
    np.testing.assert_allclose(diagnostics.total, jnp.sum(modal), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        diagnostics.maximum_absolute, jnp.max(jnp.abs(modal)), rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        diagnostics.l2_norm, jnp.linalg.norm(modal), rtol=2e-6, atol=2e-6
    )
    assert bool(diagnostics.finite)

    direction = jnp.ones(plan.spatial_shape, dtype=jnp.complex64)
    _, tangent = jax.jvp(
        lambda value: plan.to_physical(plan.to_modal(value)), (modal,), (direction,)
    )
    np.testing.assert_allclose(tangent, direction, rtol=2e-5, atol=2e-5)
    _, pullback = jax.vjp(
        lambda value: jnp.real(plan.to_physical(plan.to_modal(value))), modal
    )
    gradient = pullback(jnp.ones(plan.spatial_shape, dtype=jnp.float32))[0]
    np.testing.assert_allclose(gradient, jnp.ones_like(gradient), rtol=2e-5, atol=2e-5)


def test_rotational_dealiasing_matches_full_complex_reference_and_projector_adapter():
    plan = _slab((4, 4, 4), state_shape=(3,), padded_shape=(6, 6, 6))
    key = jax.random.key(2)
    velocity = (
        jax.random.normal(key, (4, 4, 4, 3))
        + 1j * jax.random.normal(jax.random.fold_in(key, 1), (4, 4, 4, 3))
    ).astype(jnp.complex64)

    class IdentityProjector:
        @staticmethod
        def project(value):
            return value

    distributed = plan.rotational_nonlinear(velocity, projector=IdentityProjector())
    padded = plan.pad_modal(velocity)
    physical = jnp.fft.ifftn(padded, axes=(0, 1, 2), norm="ortho")
    derivatives = []
    for axis, size in enumerate(plan.padded_shape):
        wave = jnp.fft.fftfreq(size) * size
        multiplier_shape = [1, 1, 1, 1]
        multiplier_shape[axis] = size
        derivative_modal = padded * (1j * wave).reshape(multiplier_shape)
        derivatives.append(jnp.fft.ifftn(derivative_modal, axes=(0, 1, 2), norm="ortho"))
    curl = jnp.stack(
        (
            derivatives[1][..., 2] - derivatives[2][..., 1],
            derivatives[2][..., 0] - derivatives[0][..., 2],
            derivatives[0][..., 1] - derivatives[1][..., 0],
        ),
        axis=-1,
    )
    reference = plan.unpad_modal(
        jnp.fft.fftn(jnp.cross(physical, curl), axes=(0, 1, 2), norm="ortho")
    )
    np.testing.assert_allclose(distributed, reference, rtol=3e-5, atol=3e-5)


def test_channel_distribution_keeps_y_replicated_and_zero_mode_atomic():
    devices = tuple(jax.devices("cpu"))
    if len(devices) >= 4:
        topology = SpectralMeshTopology(
            (2, 2),
            devices=devices[:4],
            axis_names=("channel_x", "channel_z"),
        )
    elif len(devices) >= 2:
        topology = SpectralMeshTopology(
            (2,),
            devices=devices[:2],
            axis_names=("channel_x",),
        )
    else:
        topology = SpectralMeshTopology.one_device()
    plan = DistributedSpectralExecutionPlan(
        topology,
        (8, 7, 8),
        schedule="channel",
        state_shape=(3,),
        horizontal_axes=(0, 2),
    )
    assert plan.physical_layout.partition[1] is None
    assert plan.modal_layout.partition[1] is None
    assert plan.report.zero_mode_atomic
    state = jnp.arange(8 * 7 * 8 * 3, dtype=jnp.float32)
    state = state.reshape((8, 7, 8, 3)).astype(jnp.complex64)
    zero = plan.channel_zero_mode(state)
    np.testing.assert_array_equal(zero, state[0, :, 0, :])
    doubled = plan.execute_channel(lambda value: 2.0 * value, state)
    np.testing.assert_array_equal(doubled, 2.0 * state)


def test_resource_refusal_topology_mismatch_and_no_host_gather_guardrails(monkeypatch):
    topology = SpectralMeshTopology.one_device()
    with pytest.raises(SpectralResourceError) as caught:
        DistributedSpectralExecutionPlan(topology, (16, 16, 16), maximum_bytes=128)
    assert not caught.value.report.accepted
    assert caught.value.report.total_bytes > caught.value.report.maximum_bytes

    other = SpectralMeshTopology((1,), devices=(jax.devices()[0],), axis_names=("other",))
    with pytest.raises(ValueError, match="identity mismatch"):
        _slab().physical_layout.sharding(other)

    plan = _slab((4, 4, 4))
    values = jnp.ones((4, 4, 4), dtype=jnp.complex64)
    with monkeypatch.context() as guard:
        guard.setattr(
            jax,
            "device_get",
            lambda *_: pytest.fail("host gather is forbidden"),
        )
        restored = plan.to_physical(plan.to_modal(values))
    np.testing.assert_allclose(restored, values, rtol=1e-5, atol=1e-5)
    source = inspect.getsource(SpectralTranspose.execute)
    assert "device_get" not in source
    assert "process_allgather" not in source


def test_multi_device_slab_and_pencil_when_process_exposes_forced_cpu_devices():
    devices = tuple(jax.devices("cpu"))
    if len(devices) < 2:
        pytest.skip(
            "Run under --xla_force_host_platform_device_count to exercise collectives."
        )
    slab = _slab((8, 8, 6), devices=devices[:2])
    values = (
        jnp.arange(8 * 8 * 6, dtype=jnp.float32).reshape((8, 8, 6)).astype(jnp.complex64)
    )
    np.testing.assert_allclose(
        slab.to_physical(slab.to_modal(values)), values, rtol=2e-5, atol=2e-5
    )
    transpose = slab.physical_to_modal[0]
    transposed = transpose.execute(values, slab.topology)
    restored = slab.modal_to_physical[0].execute(transposed, slab.topology)
    np.testing.assert_array_equal(restored, values)

    if len(devices) < 4:
        pytest.skip("Four CPU devices are required for the two-dimensional pencil mesh.")
    topology = SpectralMeshTopology(
        (2, 2),
        devices=devices[:4],
        axis_names=("px", "py"),
    )
    pencil = DistributedSpectralExecutionPlan(
        topology,
        (8, 8, 8),
        schedule="pencil",
    )
    pencil_values = jnp.arange(8**3, dtype=jnp.float32).reshape((8, 8, 8))
    pencil_values = pencil_values.astype(jnp.complex64)
    restored = pencil.to_physical(pencil.to_modal(pencil_values))
    np.testing.assert_allclose(
        restored,
        pencil_values,
        rtol=3e-5,
        atol=3e-5,
    )
