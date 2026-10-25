#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax import config as pxconfig
from phydrax.discretization import (
    AxisDomain,
    ChebyshevBasisPlan,
    FourierBasisPlan,
    TensorSpectralDiscretization,
    TensorSpectralPlan,
)
from phydrax.discretization.spectral import SpectralPrecisionPolicy
from phydrax.discretization.spectral._distributed import (
    DistributedSpectralExecutionPlan,
    SpectralLayout,
    SpectralMeshTopology,
    SpectralResourceError,
    SpectralSchedule,
    SpectralTranspose,
)


def _single_precision() -> SpectralPrecisionPolicy:
    return SpectralPrecisionPolicy(jnp.float32)


def _slab(
    shape: Sequence[int] = (8, 8, 6),
    *,
    owner_id: str = "distributed-test-owner",
    state_shape: Sequence[int] = (),
    admitted_payload_shapes: Sequence[Sequence[int]] | None = None,
    padded_shape: Sequence[int] | None = None,
    devices: Sequence[jax.Device] | None = None,
    precision: SpectralPrecisionPolicy | None = None,
    maximum_bytes: int = 2 * 1024**3,
) -> DistributedSpectralExecutionPlan:
    selected = (jax.devices()[0],) if devices is None else tuple(devices)
    topology = SpectralMeshTopology(
        (len(selected),),
        devices=selected,
        axis_names=("spectral",),
    )
    payloads = (
        (tuple(state_shape),)
        if admitted_payload_shapes is None
        else admitted_payload_shapes
    )
    return DistributedSpectralExecutionPlan(
        topology,
        shape,
        owner_id=owner_id,
        precision=_single_precision() if precision is None else precision,
        admitted_payload_shapes=payloads,
        state_shape=state_shape,
        padded_shape=padded_shape,
        maximum_bytes=maximum_bytes,
    )


def _periodic_discretization(
    shape: Sequence[int], precision: SpectralPrecisionPolicy
) -> TensorSpectralDiscretization:
    return TensorSpectralPlan(
        tuple(FourierBasisPlan(count) for count in shape),
        axis_names=tuple(f"x{axis}" for axis in range(len(shape))),
        precision=precision,
    ).prepare(tuple(AxisDomain.periodic(0.0, 2.0 * np.pi) for _ in shape))


def test_forward_inverse_match_independent_full_complex_references() -> None:
    plan = _slab()
    x = np.arange(8)[:, None, None] * (2.0 * np.pi / 8.0)
    y = np.arange(8)[None, :, None] * (2.0 * np.pi / 8.0)
    z = np.arange(6)[None, None, :] * (2.0 * np.pi / 6.0)
    values = (np.sin(2.0 * x) + 0.25j * np.cos(y - z)).astype(np.complex64)

    modal = plan.to_modal(values)
    reference_modal = np.fft.fftn(values, axes=(0, 1, 2), norm="ortho").astype(
        np.complex64
    )
    restored = plan.to_physical(modal)
    reference_physical = np.fft.ifftn(
        reference_modal, axes=(0, 1, 2), norm="ortho"
    ).astype(np.complex64)
    independent_inverse = plan.to_physical(values)
    reference_inverse = np.fft.ifftn(values, axes=(0, 1, 2), norm="ortho").astype(
        np.complex64
    )

    np.testing.assert_allclose(modal, reference_modal, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(restored, reference_physical, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        independent_inverse, reference_inverse, rtol=2e-5, atol=2e-5
    )
    assert modal.dtype == jnp.dtype(plan.precision.coefficient_dtype)
    assert restored.dtype == jnp.dtype(plan.precision.coefficient_dtype)
    assert restored.sharding == plan.physical_layout.sharding(plan.topology)
    assert modal.sharding == plan.modal_layout.sharding(plan.topology)
    sequence_ids = (
        plan.report.forward_sequence_id,
        plan.report.inverse_sequence_id,
        plan.report.padded_forward_sequence_id,
        plan.report.padded_inverse_sequence_id,
    )
    compiled_modal = jax.jit(lambda state: plan.to_modal(state))(jnp.asarray(values))
    np.testing.assert_allclose(compiled_modal, reference_modal, rtol=2e-5, atol=2e-5)
    assert sequence_ids == (
        plan.report.forward_sequence_id,
        plan.report.inverse_sequence_id,
        plan.report.padded_forward_sequence_id,
        plan.report.padded_inverse_sequence_id,
    )


def test_explicit_cpu_topology_on_accelerator_default() -> None:
    if jax.default_backend() == "cpu":
        pytest.skip("Requires an accelerator default and an explicit CPU topology.")
    cpu = jax.devices("cpu")[0]
    plan = _slab((4, 4, 4), devices=(cpu,)).prepare()
    values = np.arange(64, dtype=np.float32).reshape(4, 4, 4).astype(np.complex64)
    actual = plan.to_modal(values)
    expected = np.fft.fftn(values, norm="ortho").astype(np.complex64)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert actual.devices() == {cpu}


def test_padding_derivative_and_fft_only_preparation_evidence() -> None:
    plan = _slab((6, 6, 4), padded_shape=(10, 12, 8))
    key = jax.random.key(8)
    modal = (
        jax.random.normal(key, plan.spatial_shape)
        + 1j * jax.random.normal(jax.random.fold_in(key, 1), plan.spatial_shape)
    ).astype(jnp.complex64)
    padded = plan.pad_modal(modal)
    padded_physical = (
        jax.random.normal(jax.random.fold_in(key, 2), plan.padded_shape)
        + 1j * jax.random.normal(jax.random.fold_in(key, 3), plan.padded_shape)
    ).astype(jnp.complex64)
    padded_modal = plan.to_modal(padded_physical, padded=True)
    padded_reference = np.fft.fftn(np.asarray(padded_physical), norm="ortho").astype(
        np.complex64
    )
    np.testing.assert_allclose(padded_modal, padded_reference, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(
        plan.to_physical(padded_modal, padded=True),
        padded_physical,
        rtol=3e-5,
        atol=3e-5,
    )
    derivative = plan.to_physical(plan.modal_derivative(plan.to_modal(modal), 0))

    np.testing.assert_allclose(plan.unpad_modal(padded), modal, rtol=2e-6, atol=2e-6)
    assert derivative.shape == plan.spatial_shape
    assert padded.shape == plan.padded_shape
    assert padded.sharding == plan.padded_modal_layout.sharding(plan.topology)
    report = plan.report
    assert report.owner_id == plan.owner_id
    assert report.numerical_id == plan.numerical_id
    assert report.execution_id == plan.execution_id
    assert report.precision_policy_id == plan.precision.policy_id
    assert report.admitted_payload_shapes == ((),)
    assert report.collective_count == 1
    assert report.collective_payload_bytes == report.resource.collective_payload_bytes
    assert report.forward_sequence_id != report.inverse_sequence_id
    assert report.padded_forward_sequence_id != report.padded_inverse_sequence_id
    assert report.host_gather is False
    assert report.differentiable
    assert report.resource.transform_workspace_bytes > 0
    assert report.resource.peak_live_bytes == (
        2 * report.resource.padded_storage_bytes
        + report.resource.transform_workspace_bytes
    )
    assert report.resource.collective_payload_bytes > 0
    assert report.resource.accepted


def test_exact_payload_admission_refuses_before_array_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _slab(
        (4, 4, 4),
        state_shape=(3,),
        admitted_payload_shapes=((), (3,), (3, 3)),
        padded_shape=(6, 6, 6),
    )
    scalar = jnp.ones((4, 4, 4), dtype=jnp.complex64)
    vector = jnp.ones((4, 4, 4, 3), dtype=jnp.complex64)
    tensor = jnp.ones((4, 4, 4, 3, 3), dtype=jnp.complex64)
    assert plan.place_batched(scalar, representation="modal").shape == scalar.shape
    assert plan.place_batched(vector, representation="modal").shape == vector.shape
    assert plan.place_batched(tensor, representation="modal").shape == tensor.shape
    padded_tensor = jnp.ones((6, 6, 6, 3, 3), dtype=jnp.complex64)
    assert (
        plan.place_batched(padded_tensor, representation="modal", padded=True).shape
        == padded_tensor.shape
    )
    assert plan.report.resource.canonical_storage_bytes == 4**3 * 3 * 3 * 8
    undeclared_host = np.ones((4, 4, 4, 2), dtype=np.complex64)
    wrong_leading_host = np.ones((5, 4, 4, 3), dtype=np.complex64)

    with monkeypatch.context() as guard:
        guard.setattr(
            jnp,
            "asarray",
            lambda *_args, **_kwargs: pytest.fail(
                "payload refusal must precede jnp.asarray"
            ),
        )
        guard.setattr(
            jax,
            "device_put",
            lambda *_args, **_kwargs: pytest.fail(
                "payload refusal must precede placement"
            ),
        )
        with pytest.raises(ValueError, match="not admitted"):
            plan.place_batched(undeclared_host, representation="modal")
        with pytest.raises(ValueError, match="must begin with shape"):
            plan.place_batched(wrong_leading_host, representation="modal")
    with pytest.raises(TypeError):
        plan.place_batched(
            np.full((4, 4, 4), "not-numeric", dtype=np.str_),
            representation="modal",
        )

    state_included = DistributedSpectralExecutionPlan(
        SpectralMeshTopology.one_device(),
        (4, 4, 4),
        owner_id="payload-owner",
        precision=_single_precision(),
        admitted_payload_shapes=((), (), (3,)),
        state_shape=(3,),
    )
    assert state_included.admitted_payload_shapes == ((), (3,))


def test_identity_partitions_numerics_execution_and_owner_binding() -> None:
    base = _slab((4, 4, 4), owner_id="owner-a")
    other_owner = _slab((4, 4, 4), owner_id="owner-b")
    wider_payload = _slab(
        (4, 4, 4),
        owner_id="owner-a",
        admitted_payload_shapes=((), (3,)),
    )
    larger_budget = _slab((4, 4, 4), owner_id="owner-a", maximum_bytes=3 * 1024**3)
    renamed_topology = SpectralMeshTopology(
        (1,), devices=(jax.devices()[0],), axis_names=("renamed",)
    )
    renamed = DistributedSpectralExecutionPlan(
        renamed_topology,
        (4, 4, 4),
        owner_id="owner-a",
        precision=_single_precision(),
        admitted_payload_shapes=((),),
    )
    scaled = DistributedSpectralExecutionPlan(
        base.topology,
        (4, 4, 4),
        owner_id="owner-a",
        precision=_single_precision(),
        admitted_payload_shapes=((),),
        transform_scale=2.0,
    )
    default_precision = DistributedSpectralExecutionPlan(
        base.topology,
        (4, 4, 4),
        owner_id="owner-a",
        admitted_payload_shapes=((),),
    )

    assert base.numerical_id == other_owner.numerical_id
    assert base.execution_id == other_owner.execution_id
    assert base.plan_id != other_owner.plan_id
    assert base.report.forward_sequence_id == other_owner.report.forward_sequence_id
    assert base.numerical_id == wider_payload.numerical_id
    assert base.execution_id != wider_payload.execution_id
    assert base.numerical_id == larger_budget.numerical_id
    assert base.execution_id != larger_budget.execution_id
    assert base.numerical_id == renamed.numerical_id
    assert base.execution_id != renamed.execution_id
    assert base.numerical_id != scaled.numerical_id
    assert default_precision.precision.coefficient_dtype == "complex64"

    double = SpectralPrecisionPolicy(jnp.float64)
    if pxconfig.x64_enabled():
        different_precision = _slab((4, 4, 4), precision=double)
        assert different_precision.numerical_id != base.numerical_id
        assert (
            different_precision.report.forward_sequence_id
            != base.report.forward_sequence_id
        )


def test_transpose_provider_and_scratch_policy_are_execution_identity() -> None:
    base = _slab((4, 4, 4))
    other_budget = DistributedSpectralExecutionPlan(
        base.topology,
        base.spatial_shape,
        owner_id=base.owner_id,
        precision=base.precision,
        admitted_payload_shapes=((),),
        maximum_external_scratch_bytes=32 * 1024**2,
    )
    assert base.transpose_provider == "native"
    assert base.report.transpose_provider == "native"
    assert base.report.transpose_evidence is None
    assert base.numerical_id == other_budget.numerical_id
    assert base.execution_id != other_budget.execution_id
    resource = base.report.resource
    assert resource.transpose_packing_bytes == 0
    assert resource.external_device_scratch_bytes == 0
    assert resource.external_resident_device_bytes == 0
    assert resource.external_host_bytes == 0
    with pytest.raises(TypeError, match="CollectiveBinding"):
        DistributedSpectralExecutionPlan(
            base.topology,
            base.spatial_shape,
            owner_id=base.owner_id,
            precision=base.precision,
            admitted_payload_shapes=((),),
            transpose_provider="cudecomp",
        )
    with pytest.raises(ValueError, match="maximum_external_scratch_bytes"):
        DistributedSpectralExecutionPlan(
            base.topology,
            base.spatial_shape,
            owner_id=base.owner_id,
            precision=base.precision,
            admitted_payload_shapes=((),),
            maximum_external_scratch_bytes=0,
        )


@pytest.mark.parametrize(
    ("schedule", "mesh_shape", "shape"),
    [
        ("pencil", (2, 4), (8, 16, 8)),
        ("pencil", (4, 2), (16, 8, 8)),
        ("slab", (1,), (8, 8, 8, 2)),
        ("channel", (1,), (8, 8, 8)),
    ],
    ids=["pencil2x4", "pencil4x2", "rank4", "channel"],
)
def test_cudecomp_refuses_unsupported_layouts_without_native_fallback(
    schedule: SpectralSchedule, mesh_shape: tuple[int, ...], shape: tuple[int, ...]
) -> None:
    count = prod(mesh_shape)
    devices = tuple(jax.devices())
    if len(devices) < count:
        pytest.skip(f"Requires {count} physical or explicitly configured devices.")
    topology = SpectralMeshTopology(mesh_shape, devices=devices[:count])
    with pytest.raises(ValueError, match="direct three-dimensional slabs only"):
        DistributedSpectralExecutionPlan(
            topology,
            shape,
            owner_id="unsupported-cudecomp-layout",
            precision=_single_precision(),
            admitted_payload_shapes=((),),
            schedule=schedule,
            transpose_provider="cudecomp",
        )


def test_transform_autodiff_and_hilbert_adjoint() -> None:
    plan = _slab((4, 4, 4))
    key = jax.random.key(4)
    value = (
        jax.random.normal(key, plan.spatial_shape)
        + 1j * jax.random.normal(jax.random.fold_in(key, 1), plan.spatial_shape)
    ).astype(jnp.complex64)
    direction = jnp.ones(plan.spatial_shape, dtype=jnp.complex64)

    _, tangent = jax.jvp(
        lambda state: plan.to_physical(plan.to_modal(state)),
        (value,),
        (direction,),
    )
    np.testing.assert_allclose(tangent, direction, rtol=2e-5, atol=2e-5)
    _, pullback = jax.vjp(
        lambda state: jnp.real(plan.to_physical(plan.to_modal(state))), value
    )
    gradient = pullback(jnp.ones(plan.spatial_shape, dtype=jnp.float32))[0]
    np.testing.assert_allclose(gradient, jnp.ones_like(gradient), rtol=2e-5, atol=2e-5)

    test_modal = (
        jax.random.normal(jax.random.fold_in(key, 2), plan.spatial_shape)
        + 1j * jax.random.normal(jax.random.fold_in(key, 3), plan.spatial_shape)
    ).astype(jnp.complex64)
    _, forward_tangent = jax.jvp(plan.to_modal, (value,), (direction,))
    np.testing.assert_allclose(
        forward_tangent, plan.to_modal(direction), rtol=2e-5, atol=2e-5
    )
    _, inverse_tangent = jax.jvp(plan.to_physical, (test_modal,), (direction,))
    np.testing.assert_allclose(
        inverse_tangent, plan.to_physical(direction), rtol=2e-5, atol=2e-5
    )
    lhs = jnp.vdot(plan.to_modal(value), test_modal)
    rhs = jnp.vdot(value, plan.to_physical(test_modal))
    np.testing.assert_allclose(lhs, rhs, rtol=3e-5, atol=3e-5)


def test_reductions_cast_before_overflow_sensitive_products() -> None:
    if not pxconfig.x64_enabled():
        pytest.skip("float64 reduction evidence requires JAX x64 mode")
    precision = SpectralPrecisionPolicy(
        jnp.float32,
        coefficient_dtype=jnp.complex64,
        transform_dtype=jnp.complex128,
        nonlinear_dtype=jnp.float32,
        reduction_dtype=jnp.float64,
        certification_dtype=jnp.float64,
    )
    plan = _slab((2, 2), precision=precision)
    values = jnp.full((2, 2), 2.0e19 + 1.0e19j, dtype=jnp.complex64)
    diagnostics = plan.diagnostics(values)
    reference = np.linalg.norm(np.asarray(values, dtype=np.complex128))

    assert diagnostics.l2_norm.dtype == jnp.float64
    assert bool(jnp.isfinite(diagnostics.l2_norm))
    np.testing.assert_allclose(diagnostics.l2_norm, reference, rtol=2e-7)
    inner = plan.global_inner_product(values, values)
    assert inner.dtype == jnp.complex128
    assert bool(jnp.isfinite(inner))
    cancellation = jnp.asarray(((1.0e8, 1.0), (-1.0e8, 1.0)), dtype=jnp.complex64)
    cancellation_diagnostics = plan.diagnostics(cancellation)
    np.testing.assert_allclose(cancellation_diagnostics.total, 2.0 + 0.0j, atol=0.0)
    transformed = plan.to_modal(cancellation)
    assert transformed.dtype == jnp.complex64
    np.testing.assert_allclose(
        transformed,
        np.fft.fftn(np.asarray(cancellation, dtype=np.complex128), norm="ortho").astype(
            np.complex64
        ),
        rtol=2e-6,
        atol=2e-6,
    )


def test_unavailable_requested_precision_is_refused() -> None:
    if pxconfig.x64_enabled():
        pytest.skip("active JAX policy honors float64/complex128")
    precision = SpectralPrecisionPolicy(jnp.float64)
    with pytest.raises(ValueError, match="cannot honor"):
        _slab((4, 4), precision=precision)


def test_channel_family_axes_and_zero_mode_contracts() -> None:
    topology = SpectralMeshTopology.one_device()
    plan = DistributedSpectralExecutionPlan(
        topology,
        (8, 7, 8),
        owner_id="channel-owner",
        precision=_single_precision(),
        admitted_payload_shapes=((3,),),
        schedule="channel",
        state_shape=(3,),
        horizontal_axes=(0, 2),
    )
    reversed_axes = DistributedSpectralExecutionPlan(
        topology,
        (8, 7, 8),
        owner_id="channel-owner",
        precision=_single_precision(),
        admitted_payload_shapes=((3,),),
        schedule="channel",
        state_shape=(3,),
        horizontal_axes=(2, 0),
    )
    assert plan.physical_layout.partition[1] is None
    assert plan.modal_layout.partition[1] is None
    assert plan.report.zero_mode_atomic
    assert plan.report.collective_count == 0
    assert plan.report.collective_payload_bytes == 0
    assert plan.report.resource.transform_workspace_bytes == 0
    assert plan.report.resource.collective_payload_bytes == 0
    assert (
        plan.report.resource.peak_live_bytes
        == 2 * plan.report.resource.padded_storage_bytes
    )
    assert plan.report.resource.accepted
    assert plan.numerical_id != reversed_axes.numerical_id
    state = jnp.arange(8 * 7 * 8 * 3, dtype=jnp.float32)
    state = state.reshape((8, 7, 8, 3)).astype(jnp.complex64)
    np.testing.assert_array_equal(plan.channel_zero_mode(state), state[0, :, 0, :])
    np.testing.assert_array_equal(
        plan.execute_channel(lambda value: 2.0 * value, state), 2.0 * state
    )
    with pytest.raises(ValueError, match="Chebyshev axis"):
        plan.to_modal(state)

    periodic = _periodic_discretization((4, 4, 4), _single_precision())
    with pytest.raises(ValueError, match="Fourier-Chebyshev-Fourier"):
        DistributedSpectralExecutionPlan.from_discretization(
            topology,
            periodic,
            admitted_payload_shapes=((),),
            schedule="channel",
        )
    wrong_order = TensorSpectralPlan(
        (ChebyshevBasisPlan(5), FourierBasisPlan(4), FourierBasisPlan(4)),
        axis_names=("y", "x", "z"),
        precision=_single_precision(),
    ).prepare(
        (
            AxisDomain.interval(-1.0, 1.0),
            AxisDomain.periodic(0.0, 2.0 * np.pi),
            AxisDomain.periodic(0.0, 2.0 * np.pi),
        )
    )
    with pytest.raises(ValueError, match="Fourier-Chebyshev-Fourier"):
        DistributedSpectralExecutionPlan.from_discretization(
            topology,
            wrong_order,
            admitted_payload_shapes=((),),
            schedule="channel",
        )

    channel = TensorSpectralPlan(
        (FourierBasisPlan(4), ChebyshevBasisPlan(5), FourierBasisPlan(4)),
        axis_names=("x", "y", "z"),
        precision=_single_precision(),
    ).prepare(
        (
            AxisDomain.periodic(0.0, 2.0 * np.pi),
            AxisDomain.interval(-1.0, 1.0),
            AxisDomain.periodic(0.0, 2.0 * np.pi),
        )
    )
    derived = DistributedSpectralExecutionPlan.from_discretization(
        topology, channel, admitted_payload_shapes=((),)
    )
    assert derived.schedule == "channel"
    assert derived.owner_id == channel.prepared_id
    assert derived.precision.policy_id == channel.plan.precision.policy_id


def test_rank_one_slab_pencil_and_process_qualified_topology_refusals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = SpectralMeshTopology.one_device()
    with pytest.raises(SpectralResourceError) as rank_one:
        DistributedSpectralExecutionPlan(
            topology,
            (8,),
            owner_id="rank-one-owner",
            precision=_single_precision(),
            admitted_payload_shapes=((),),
        )
    assert any(
        "spatial rank at least two" in reason for reason in rank_one.value.report.reasons
    )

    with pytest.raises(SpectralResourceError) as pencil:
        DistributedSpectralExecutionPlan(
            topology,
            (4, 4, 4),
            owner_id="pencil-owner",
            precision=_single_precision(),
            admitted_payload_shapes=((),),
            schedule="pencil",
        )
    assert "two-dimensional mesh" in pencil.value.report.reasons[0]
    device = jax.devices()[0]
    assert topology.device_keys == ((int(device.process_index), int(device.id)),)
    with monkeypatch.context() as guard:
        guard.setattr(jax, "devices", lambda *_args, **_kwargs: ())
        with pytest.raises(RuntimeError, match=str(topology.device_keys[0])):
            topology.require_available()


def test_resource_refusal_and_no_host_gather_guardrails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = SpectralMeshTopology.one_device()
    with pytest.raises(SpectralResourceError) as caught:
        DistributedSpectralExecutionPlan(
            topology,
            (16, 16, 16),
            owner_id="resource-owner",
            precision=_single_precision(),
            admitted_payload_shapes=((),),
            maximum_bytes=128,
        )
    assert not caught.value.report.accepted
    assert caught.value.report.peak_live_bytes > caught.value.report.maximum_bytes
    assert caught.value.report.collective_payload_bytes > 0
    assert caught.value.report.peak_live_bytes == (
        2 * caught.value.report.padded_storage_bytes
        + caught.value.report.transform_workspace_bytes
    )

    other = SpectralMeshTopology((1,), devices=(jax.devices()[0],), axis_names=("other",))
    with pytest.raises(ValueError, match="identity mismatch"):
        _slab().physical_layout.sharding(other)

    plan = _slab((4, 4, 4))
    values = jnp.ones((4, 4, 4), dtype=jnp.complex64)
    with monkeypatch.context() as guard:
        guard.setattr(
            jax,
            "device_get",
            lambda *_args, **_kwargs: pytest.fail("host gather is forbidden"),
        )
        restored = plan.to_physical(plan.to_modal(values))
    np.testing.assert_allclose(restored, values, rtol=1e-5, atol=1e-5)


def test_rotational_dealiasing_matches_independent_reference() -> None:
    plan = _slab(
        (4, 4, 4),
        state_shape=(3,),
        admitted_payload_shapes=((3,),),
        padded_shape=(6, 6, 6),
    )
    key = jax.random.key(2)
    velocity = (
        jax.random.normal(key, (4, 4, 4, 3))
        + 1j * jax.random.normal(jax.random.fold_in(key, 1), (4, 4, 4, 3))
    ).astype(jnp.complex64)

    distributed = plan.rotational_nonlinear(velocity)
    padded = plan.pad_modal(velocity)
    physical = jnp.fft.ifftn(padded, axes=(0, 1, 2), norm="ortho")
    derivatives: list[Array] = []
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


@pytest.mark.parametrize("device_count", [1, 2, 3, 4, 5, 8])
def test_public_transpose_and_multi_device_slab_execution(device_count: int) -> None:
    devices = tuple(jax.devices())
    if len(devices) < device_count:
        pytest.skip(f"Requires {device_count} physical or explicitly configured devices.")
    shape = (4 * device_count, 4 * device_count, 6)
    padded_shape = (6 * device_count, 6 * device_count, 10)
    slab = _slab(shape, padded_shape=padded_shape, devices=devices[:device_count])
    resource = slab.report.resource
    with pytest.raises(SpectralResourceError) as refused:
        _slab(
            shape,
            padded_shape=padded_shape,
            devices=devices[:device_count],
            maximum_bytes=resource.peak_live_bytes - 1,
        )
    assert refused.value.report.peak_live_bytes == resource.peak_live_bytes
    values = (
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        / np.float32(np.prod(shape))
    ).astype(np.complex64)
    for padded, transform_shape in ((False, shape), (True, padded_shape)):
        field = (
            np.arange(np.prod(transform_shape), dtype=np.float32).reshape(transform_shape)
            / np.float32(np.prod(transform_shape))
        ).astype(np.complex64)
        reference = np.fft.fftn(field, norm="ortho").astype(np.complex64)
        np.testing.assert_allclose(
            slab.to_modal(field, padded=padded), reference, rtol=3e-5, atol=8e-5
        )
        np.testing.assert_allclose(
            slab.to_physical(reference, padded=padded), field, rtol=3e-5, atol=8e-5
        )
        _, tangent = jax.jvp(
            lambda value: slab.to_modal(value, padded=padded),
            (jnp.asarray(field),),
            (jnp.asarray(field * np.complex64(0.3 + 0.2j)),),
        )
        np.testing.assert_allclose(
            tangent, reference * np.complex64(0.3 + 0.2j), rtol=3e-5, atol=8e-5
        )
        probe = field * np.complex64(0.3 + 0.2j)
        _, pullback = jax.vjp(
            lambda value: slab.to_modal(value, padded=padded), jnp.asarray(field)
        )
        adjoint = jnp.conj(pullback(jnp.conj(jnp.asarray(probe)))[0])
        np.testing.assert_allclose(
            adjoint, np.fft.ifftn(probe, norm="ortho"), rtol=3e-5, atol=8e-5
        )

    forward = SpectralTranspose(
        slab.physical_layout,
        slab.modal_layout,
        slab.topology,
    )
    inverse = SpectralTranspose(
        slab.modal_layout,
        slab.physical_layout,
        slab.topology,
    )
    transposed = forward.execute(values, slab.topology)
    restored = inverse.execute(transposed, slab.topology)
    np.testing.assert_array_equal(restored, values)
    modal_probe = (values * np.complex64(0.3 + 0.2j)).astype(np.complex64)
    lhs = slab.global_inner_product(transposed, modal_probe, representation="modal")
    reverse_probe = inverse.execute(modal_probe, slab.topology)
    rhs = slab.global_inner_product(values, reverse_probe, representation="physical")
    np.testing.assert_allclose(lhs, rhs, rtol=2e-6, atol=2e-6)


def _pencil_topology(shape: tuple[int, int]) -> SpectralMeshTopology:
    devices = tuple(jax.devices())
    count = prod(shape)
    if len(devices) < count:
        pytest.skip(f"Needs {count} physical or explicitly configured devices.")
    return SpectralMeshTopology(shape, devices=devices[:count], axis_names=("px", "py"))


@pytest.mark.parametrize(
    "mesh_shape",
    [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2)],
    ids=["2x2", "2x3", "3x2", "2x4", "4x2"],
)
@pytest.mark.parametrize("spatial_rank", [3, 4], ids=["rank3", "rank4"])
def test_pencil_matches_independent_reference_and_adjoint(
    mesh_shape: tuple[int, int], spatial_rank: int
) -> None:
    topology = _pencil_topology(mesh_shape)
    px, py = mesh_shape
    shape = (4 * px, 6 * py, 2 * prod(mesh_shape)) + ((3,) if spatial_rank == 4 else ())
    padded_shape = (6 * px, 8 * py, 3 * prod(mesh_shape)) + (
        (4,) if spatial_rank == 4 else ()
    )
    payload = (2, 3)
    pencil = DistributedSpectralExecutionPlan(
        topology,
        shape,
        owner_id="pencil-multi-device",
        precision=_single_precision(),
        admitted_payload_shapes=(payload,),
        state_shape=payload,
        padded_shape=padded_shape,
        schedule="pencil",
    )
    single = _slab(
        shape,
        owner_id=pencil.owner_id,
        state_shape=payload,
        padded_shape=padded_shape,
    )
    assert pencil.numerical_id == single.numerical_id
    assert pencil.execution_id != single.execution_id
    assert pencil.report.collective_count == 1
    assert pencil.report.forward_sequence_id != pencil.report.padded_forward_sequence_id
    resource = pencil.report.resource
    assert resource.accepted
    assert resource.collective_payload_bytes == resource.padded_storage_bytes
    assert resource.transform_workspace_bytes == 2 * resource.padded_storage_bytes
    with pytest.raises(SpectralResourceError) as refused:
        DistributedSpectralExecutionPlan(
            topology,
            shape,
            owner_id=pencil.owner_id,
            precision=pencil.precision,
            admitted_payload_shapes=(payload,),
            state_shape=payload,
            padded_shape=padded_shape,
            schedule="pencil",
            maximum_bytes=resource.peak_live_bytes - 1,
        )
    assert refused.value.report.peak_live_bytes == resource.peak_live_bytes
    axes = tuple(range(spatial_rank))
    rng = np.random.default_rng(17)
    for padded, counts in ((False, shape), (True, padded_shape)):
        values = (
            rng.normal(size=(*counts, *payload))
            + 1j * rng.normal(size=(*counts, *payload))
        ).astype(np.complex64)
        reference = np.fft.fftn(values, axes=axes, norm="ortho").astype(np.complex64)
        modal = pencil.to_modal(values, padded=padded)
        np.testing.assert_allclose(modal, reference, rtol=3e-5, atol=8e-5)
        np.testing.assert_allclose(
            pencil.to_physical(reference, padded=padded), values, rtol=3e-5, atol=8e-5
        )
        direction = values * np.complex64(0.3 + 0.2j)
        _, tangent = jax.jvp(
            lambda value: pencil.to_modal(value, padded=padded),
            (jnp.asarray(values),),
            (jnp.asarray(direction),),
        )
        np.testing.assert_allclose(
            tangent, np.fft.fftn(direction, axes=axes, norm="ortho"), rtol=3e-5, atol=8e-5
        )
        _, pullback = jax.vjp(
            lambda value: pencil.to_modal(value, padded=padded), jnp.asarray(values)
        )
        adjoint = jnp.conj(pullback(jnp.conj(jnp.asarray(direction)))[0])
        np.testing.assert_allclose(
            adjoint,
            np.fft.ifftn(direction, axes=axes, norm="ortho"),
            rtol=3e-5,
            atol=8e-5,
        )
        if not padded:
            np.testing.assert_allclose(
                pencil.unpad_modal(pencil.pad_modal(modal)), modal, rtol=2e-6, atol=2e-6
            )


@pytest.mark.parametrize(
    "mesh_shape",
    [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2)],
    ids=["2x2", "2x3", "3x2", "2x4", "4x2"],
)
def test_pencil_transpose_uses_declared_axis_names(
    mesh_shape: tuple[int, int],
) -> None:
    topology = _pencil_topology(mesh_shape)
    shape = (6 * mesh_shape[1], 6 * mesh_shape[0], 2 * prod(mesh_shape), 2)
    physical = SpectralLayout(shape, ("py", "px", None, None), "physical", topology)
    modal = SpectralLayout(shape, (None, None, ("px", "py"), None), "modal", topology)
    forward = SpectralTranspose(physical, modal, topology)
    inverse = SpectralTranspose(modal, physical, topology)
    values = np.arange(prod(shape), dtype=np.float32).reshape(shape).astype(np.complex64)
    transposed = forward.execute(values, topology)
    np.testing.assert_array_equal(transposed, values)
    np.testing.assert_array_equal(inverse.execute(transposed, topology), values)
    assert transposed.sharding == modal.sharding(topology)
    assert forward.mesh_shape == mesh_shape[::-1]


@pytest.mark.parametrize(
    "mesh_shape", [(2, 2), (2, 4), (4, 2)], ids=["2x2", "2x4", "4x2"]
)
def test_pencil_packing_budget_refuses_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
    mesh_shape: tuple[int, int],
) -> None:
    topology = _pencil_topology(mesh_shape)
    shape = (16, 16, prod(mesh_shape))
    precision = _single_precision()
    payload_bytes = prod(shape) * 11 * np.dtype(precision.coefficient_dtype).itemsize
    with monkeypatch.context() as guard:
        guard.setattr(
            jnp,
            "asarray",
            lambda *_args, **_kwargs: pytest.fail(
                "FFT budget admission must precede arrays"
            ),
        )
        guard.setattr(
            jax,
            "device_put",
            lambda *_args, **_kwargs: pytest.fail(
                "FFT budget admission must precede placement"
            ),
        )
        with pytest.raises(SpectralResourceError) as refused:
            DistributedSpectralExecutionPlan(
                topology,
                shape,
                owner_id="pencil-packing-budget",
                precision=precision,
                admitted_payload_shapes=((1, 11), (1, 6), (1, 2)),
                state_shape=(1, 3),
                schedule="pencil",
                maximum_bytes=3 * payload_bytes,
            )
        admitted = DistributedSpectralExecutionPlan(
            topology,
            shape,
            owner_id="pencil-packing-budget",
            precision=precision,
            admitted_payload_shapes=((1, 11), (1, 6), (1, 2)),
            state_shape=(1, 3),
            schedule="pencil",
            maximum_bytes=4 * payload_bytes,
        )
    assert not refused.value.report.accepted
    assert refused.value.report.transform_workspace_bytes == 2 * payload_bytes
    assert refused.value.report.collective_payload_bytes == payload_bytes
    assert refused.value.report.peak_live_bytes == 4 * payload_bytes
    assert admitted.report.resource.accepted
    assert admitted.report.resource.peak_live_bytes == 4 * payload_bytes


@eqx.filter_jit
def _field_roundtrip(plan: DistributedSpectralExecutionPlan, values: Array) -> Array:
    return jnp.real(plan.to_physical_batched(plan.to_modal_batched(values)[..., :6]))


@eqx.filter_jit
def _field_and_divergence_roundtrip(
    plan: DistributedSpectralExecutionPlan, values: Array
) -> Array:
    field = jnp.real(plan.to_physical_batched(plan.to_modal_batched(values)[..., :6]))
    return jnp.real(plan.to_physical_batched(plan.to_modal_batched(field)[..., :2]))


@pytest.mark.parametrize(
    "mesh_shape",
    [(4,), (8,), (2, 2), (2, 4), (4, 2)],
    ids=["slab4", "slab8", "pencil2x2", "pencil2x4", "pencil4x2"],
)
def test_composed_payloads_after_prior_executable(
    mesh_shape: tuple[int, ...],
) -> None:
    devices = tuple(jax.devices())
    count = prod(mesh_shape)
    if len(devices) < count:
        pytest.skip(f"Requires {count} physical or explicitly configured devices.")
    topology = SpectralMeshTopology(
        mesh_shape,
        devices=devices[:count],
        axis_names=("spectral",) if len(mesh_shape) == 1 else ("px", "py"),
    )
    shape = (16, 16, prod(mesh_shape))
    if not pxconfig.x64_enabled():
        pytest.skip("The PIC payload regression requires float64/complex128.")
    plan = DistributedSpectralExecutionPlan(
        topology,
        shape,
        owner_id="spectral-pic-payload-history",
        precision=SpectralPrecisionPolicy(jnp.float64),
        admitted_payload_shapes=((1, 11), (1, 6), (1, 2)),
        state_shape=(1, 3),
        schedule="slab" if len(mesh_shape) == 1 else "pencil",
    )
    values = np.random.default_rng(7).normal(size=(*shape, 1, 11))
    placed = plan.place_batched(values, representation="physical")
    # Equinox's Callable return annotation omits its runtime lowering API.
    first = _field_roundtrip.lower(plan, placed).compile()  # ty: ignore[unresolved-attribute]
    first_result = jax.block_until_ready(first(plan, placed))
    np.testing.assert_allclose(first_result, values[..., :6], rtol=1e-12, atol=1e-12)
    chain = _field_and_divergence_roundtrip.lower(plan, placed).compile()  # ty: ignore[unresolved-attribute]
    for offset in (0.0, 0.25):
        changed = plan.place_batched(values + offset, representation="physical")
        actual = jax.block_until_ready(chain(plan, changed))
        np.testing.assert_allclose(
            actual, values[..., :2] + offset, rtol=1e-12, atol=1e-12
        )


# --- owner-region pencil topologies ---------------------------------------------------


def _owner_blocks(values: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Owner-major pencil blocks: owner r holds coordinate (r % px, r // px)."""
    px, py = grid
    b0, b1 = values.shape[0] // px, values.shape[1] // py
    return np.stack(
        [
            values[
                (owner % px) * b0 : (owner % px + 1) * b0,
                (owner // px) * b1 : (owner // px + 1) * b1,
            ]
            for owner in range(px * py)
        ]
    )


@pytest.mark.parametrize("grid", [(2, 3), (3, 2)], ids=["2x3", "3x2"])
def test_owner_region_lane_pencil_matches_reference_and_exact_transposes(
    grid: tuple[int, int],
) -> None:
    """A non-square pencil grid factoring one lane owner axis (first axis fastest)."""
    px, py = grid
    owners = px * py
    topology = SpectralMeshTopology.owner_region(
        grid, owner_axis="owner", devices=(jax.devices()[0],), lanes=True
    )
    assert topology.lanes == owners and topology.mesh_shape == grid
    assert topology.owner_axis == "owner"
    lane = SpectralMeshTopology.lane_reference(owners, axis_name="owner")
    assert topology.topology_id != lane.topology_id
    shape = (2 * px, 2 * py, 2 * owners)
    plan = DistributedSpectralExecutionPlan(
        topology,
        shape,
        owner_id="owner-region-lane-pencil",
        admitted_payload_shapes=(),
        precision=_single_precision(),
        schedule="pencil",
    )
    names = topology.mesh_axis_names
    assert plan.physical_layout.partition == (names[0], names[1], None)
    assert plan.modal_layout.partition == (None, None, (names[1], names[0]))
    assert plan.physical_layout.local_shape(topology) == (2, 2, 2 * owners)
    assert plan.modal_layout.local_shape(topology) == (2 * px, 2 * py, 2)
    assert plan.report.collective_count == 1
    values = np.random.default_rng(3).normal(size=shape)
    with pytest.raises(ValueError, match="lane-reference|owner-region"):
        plan.to_modal(values)
    blocks = jnp.asarray(_owner_blocks(values, grid))
    modal = jax.vmap(plan.forward_local, axis_name="owner")(blocks)
    expected = np.fft.fftn(values, norm="ortho")
    for owner in range(owners):
        np.testing.assert_allclose(
            modal[owner],
            expected[:, :, 2 * owner : 2 * owner + 2],
            rtol=1e-5,
            atol=1e-5,
        )
    back = jax.vmap(plan.inverse_local, axis_name="owner")(modal)
    np.testing.assert_allclose(jnp.real(back), blocks, rtol=1e-5, atol=1e-5)

    # The pencil transposes run on the owner axis in the canonical flattened
    # (axis_1, axis_0) order; the reverse transpose is their exact transpose.
    (forward,) = (
        stage.transpose for stage in plan._forward_sequence if stage.transpose is not None
    )
    (reverse,) = (
        stage.transpose for stage in plan._inverse_sequence if stage.transpose is not None
    )
    assert forward.mesh_axis == (names[1], names[0])
    assert forward.collective_axis == "owner" and reverse.collective_axis == "owner"
    to_slab = jax.vmap(forward.apply_local, axis_name="owner")
    to_pencil = jax.vmap(reverse.apply_local, axis_name="owner")
    cotangent = jnp.asarray(
        np.random.default_rng(4).normal(size=(owners, 2 * px, 2 * py, 2))
    )
    (transposed,) = jax.linear_transpose(to_slab, blocks)(cotangent)
    np.testing.assert_array_equal(transposed, to_pencil(cotangent))
    np.testing.assert_array_equal(to_pencil(to_slab(blocks)), blocks)

    # A pencil layout whose flattened order is not the owner order is refused.
    square = (6, 6, 2 * owners)
    swapped = SpectralLayout(square, (names[1], names[0], None), "physical", topology)
    slab_order = SpectralLayout(square, (None, None, names), "modal", topology)
    with pytest.raises(ValueError, match="owner order"):
        SpectralTranspose(swapped, slab_order, topology)


def test_owner_region_device_pencil_is_the_canonical_pencil_collective() -> None:
    """Owner devices bound per coordinate; one owner-axis all-to-all = 2-D pencil."""
    devices = tuple(jax.devices())
    if len(devices) < 4:
        pytest.skip("Needs 4 physical or explicitly configured devices.")
    grid = (2, 2)
    owners_devices = devices[:4]
    topology = SpectralMeshTopology.owner_region(
        grid, owner_axis="owner", devices=owners_devices
    )
    for owner, device in enumerate(owners_devices):
        assert topology.mesh.devices[owner % 2, owner // 2] == device
    assert topology.device_ids == tuple(int(device.id) for device in owners_devices)
    shape = (4, 6, 8)
    plan = DistributedSpectralExecutionPlan(
        topology,
        shape,
        owner_id="owner-region-device-pencil",
        admitted_payload_shapes=(),
        precision=_single_precision(),
        schedule="pencil",
    )
    values = np.random.default_rng(5).normal(size=shape)
    with pytest.raises(ValueError, match="owner-region"):
        plan.to_modal(values)
    owner_mesh = jax.sharding.Mesh(np.asarray(owners_devices, dtype=object), ("owner",))
    blocks = _owner_blocks(values, grid).reshape((-1, 3, 8))
    mapped = jax.shard_map(
        plan.forward_local,
        mesh=owner_mesh,
        in_specs=jax.sharding.PartitionSpec("owner"),
        out_specs=jax.sharding.PartitionSpec("owner"),
        check_vma=False,
    )
    modal = np.asarray(mapped(jnp.asarray(blocks))).reshape((4, 4, 6, 2))
    canonical_topology = SpectralMeshTopology(
        jax.sharding.Mesh(topology.mesh.devices, ("px", "py"))
    )
    canonical = DistributedSpectralExecutionPlan(
        canonical_topology,
        shape,
        owner_id="owner-region-device-pencil",
        admitted_payload_shapes=(),
        precision=_single_precision(),
        schedule="pencil",
    )
    reference = np.asarray(canonical.to_modal(values))
    expected = np.fft.fftn(values, norm="ortho")
    for owner in range(4):
        np.testing.assert_allclose(
            modal[owner], reference[:, :, 2 * owner : 2 * owner + 2], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            modal[owner], expected[:, :, 2 * owner : 2 * owner + 2], rtol=1e-5, atol=1e-5
        )
