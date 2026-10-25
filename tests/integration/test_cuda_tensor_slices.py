"""Real CUDA residual slices, host checkpoints, and explicit device placement.

All runtime discovery is deferred until test execution; no CuPy or native-provider
mock is used. Multi-device cases skip only when their physical device count is
unavailable, never when a selected provider fails to execute.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Literal, TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray


if TYPE_CHECKING:
    from jax import Array

    from phydrax.tensor_network import PreparedSlicedContraction

type TensorProvider = Literal["cutensor", "cutensornet"]
type TensorScalar = np.float32 | np.float64 | np.complex64 | np.complex128
type TensorArray = NDArray[TensorScalar]


pytestmark = pytest.mark.cuda_native
_PROVIDERS = ("cutensor", "cutensornet")
_DTYPES = ("float32", "float64", "complex64", "complex128")


@contextmanager
def _cuda_case(provider: TensorProvider) -> Iterator[SimpleNamespace]:
    # Entered by test bodies only, not at collection time.
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from phydrax import config, tensor_network as tn
    from phydrax._cuda_native import CudaNativeUnavailableError, load_cuda_component
    from phydrax._execution_device import device_platform
    from phydrax._jax_config import default_device

    devices = tuple(
        device for device in jax.local_devices() if device_platform(device) == "cuda"
    )
    if not devices:
        pytest.skip("Real CUDA hardware is unavailable")
    try:
        load_cuda_component(provider)
    except CudaNativeUnavailableError as error:
        if error.issue.code == "companion-missing" and not error.issue.paths:
            pytest.skip(str(error))
        raise
    with (
        config.context(config.NumericsConfig(enable_x64=True)),
        default_device(devices[0]),
    ):
        yield SimpleNamespace(jax=jax, jnp=jnp, eqx=eqx, tn=tn, devices=devices)


def _values(shape: tuple[int, ...], dtype: str, offset: int = 0) -> TensorArray:
    real = (np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 1 + offset) / 9
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        real = real + 1j * (np.flip(real) + 0.2)
    return np.asarray(real, dtype=dtype)


def _binding(
    runtime: SimpleNamespace,
    provider: TensorProvider,
    dtype: str,
    *,
    extent: int = 5,
    labels: tuple[str, ...] = ("a",),
    batch_size: int = 4,
    nonfinite: bool = False,
) -> tuple[PreparedSlicedContraction, tuple[TensorArray, ...]]:
    tn = runtime.tn
    values = (
        _values((2, extent), dtype),
        _values((extent, 2), dtype, 1),
        _values((2, 2), dtype, 2),
    )
    if nonfinite:
        values[0][0, 0] = np.nan
    structure = tn.ContractionStructure(
        (
            tn.ContractionOperand(
                "left", (tn.ContractionLeg("i", 2), tn.ContractionLeg("a", extent))
            ),
            tn.ContractionOperand(
                "middle", (tn.ContractionLeg("a", extent), tn.ContractionLeg("b", 2))
            ),
            tn.ContractionOperand(
                "right", (tn.ContractionLeg("b", 2), tn.ContractionLeg("k", 2))
            ),
        ),
        ("i", "k"),
    )
    precision = tn.TensorNetworkPrecisionPolicy(
        storage_dtype=dtype,
        contraction_dtype=dtype,
        accumulation_dtype=dtype,
        output_dtype=dtype,
    )
    # Contract middle/right first, not the usual left-to-right sequence. Slicing
    # must retain this imported path even when a removed mode changes its cost.
    plan = tn.plan_contraction(
        structure, dtype=dtype, precision=precision, path=((1, 2), (0, 1))
    )
    policy = tn.ContractionExecutionPolicy(
        provider,
        maximum_workspace_bytes=2**23,
        maximum_cache_bytes=2**26,
        maximum_cache_entries=4,
        maximum_batch_size=8,
        maximum_derivative_bytes=2**27,
    )
    prepared = tn.prepare_contraction(
        plan,
        tuple(runtime.jax.device_put(value, runtime.devices[0]) for value in values),
        execution=policy,
    )
    sliced = tn.plan_sliced_contraction(
        plan,
        labels,
        batch_size=batch_size,
        resources=tn.SlicingResourcePolicy(
            maximum_slices=32,
            maximum_batch_elements=4096,
            maximum_checkpoint_bytes=4096,
        ),
    )
    return tn.prepare_sliced_contraction(prepared, sliced), values


def _assert_close(actual: Array | ArrayLike, expected: ArrayLike, dtype: str) -> None:
    rtol, atol = (1e-4, 2e-5) if dtype in ("float32", "complex64") else (5e-11, 5e-12)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol
    )
    assert np.asarray(actual).dtype == np.dtype(dtype)


def _replace(
    runtime: SimpleNamespace,
    binding: PreparedSlicedContraction,
    values: tuple[Array, ...],
) -> PreparedSlicedContraction:
    return runtime.eqx.tree_at(
        lambda item: item.original.operands, binding, tuple(values)
    )


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cuda_slice_residual_identity_pure_local_jit_jvp_vjp_and_mixed_second(
    provider: TensorProvider, dtype: str
) -> None:
    with _cuda_case(provider) as runtime:
        binding, values = _binding(runtime, provider, dtype, labels=("a", "b"))
        tn, jax = runtime.tn, runtime.jax
        assert binding.original.plan.path == ((1, 2), (0, 1))
        assert binding.plan.residual.path == binding.original.plan.path
        assert binding.executor.policy is binding.original.executor.policy
        assert binding.executor.policy.provider == provider
        assert binding.executor.build_identity == binding.original.executor.build_identity
        assert (
            binding.executor.external_workspace_bound_bytes
            == binding.original.executor.external_workspace_bound_bytes
        )
        assert (
            binding.executor.external_cache_bound_bytes
            == binding.original.executor.external_cache_bound_bytes
        )
        assert binding.executor.vendor_host_bound_bytes is None
        execution_id = binding.executor.lowering_id

        def local(assignment: Array, *operands: Array) -> Array:
            return tn.execute_slice_assignment(
                _replace(runtime, binding, operands), assignment
            )

        tangents = tuple(
            _values(value.shape, dtype, index + 4) for index, value in enumerate(values)
        )
        cotangent = _values((2, 2), dtype, 3)
        device_tangents = tuple(
            jax.device_put(value, runtime.devices[0]) for value in tangents
        )
        device_cotangent = jax.device_put(cotangent, runtime.devices[0])

        def transformed(
            assignment: Array, *operands: Array
        ) -> tuple[Array, Array, tuple[Array, ...], Array]:
            evaluate = lambda *xs: local(assignment, *xs)
            value, tangent = jax.jvp(evaluate, operands, device_tangents)
            bars = jax.vjp(evaluate, *operands)[1](device_cotangent)
            a, b, c = operands
            mixed = jax.jvp(
                lambda middle: jax.jvp(
                    lambda left: local(assignment, left, middle, c),
                    (a,),
                    (device_tangents[0],),
                )[1],
                (b,),
                (device_tangents[1],),
            )[1]
            return value, tangent, bars, mixed

        assignment = runtime.jnp.asarray((0, 0), dtype=runtime.jnp.int32)
        executable = (
            jax.jit(transformed).lower(assignment, *binding.original.operands).compile()
        )
        for p, q in ((0, 0), (2, 1), (4, 0)):
            current_assignment = runtime.jnp.asarray((p, q), dtype=runtime.jnp.int32)
            value, tangent, bars, mixed = executable(
                current_assignment, *binding.original.operands
            )
            a, b, c = values
            da, db, dc = tangents
            expected = np.outer(a[:, p], c[q]) * b[p, q]
            expected_tangent = (
                np.outer(da[:, p], c[q]) * b[p, q]
                + np.outer(a[:, p], c[q]) * db[p, q]
                + np.outer(a[:, p], dc[q]) * b[p, q]
            )
            a_bar, b_bar, c_bar = (np.zeros_like(item) for item in values)
            a_bar[:, p] = np.sum(cotangent * c[q][None, :], axis=1) * b[p, q]
            b_bar[p, q] = np.sum(cotangent * np.outer(a[:, p], c[q]))
            c_bar[q] = np.sum(cotangent * a[:, p, None], axis=0) * b[p, q]
            _assert_close(value, expected, dtype)
            _assert_close(tangent, expected_tangent, dtype)
            for actual, expected_bar in zip(bars, (a_bar, b_bar, c_bar), strict=True):
                _assert_close(actual, expected_bar, dtype)
            _assert_close(mixed, np.outer(da[:, p], c[q]) * db[p, q], dtype)
        assert binding.executor.lowering_id == execution_id

        assignments = tn.mixed_radix_assignments((5, 2), maximum_assignments=32)
        np.testing.assert_array_equal(
            np.asarray(assignments),
            np.asarray([(p, q) for p in range(5) for q in range(2)], dtype=np.int32),
        )
        # Bound each map to the execution policy; one map never includes all ten
        # assignments when maximum_batch_size is eight.
        first = jax.jit(
            jax.vmap(lambda assignment: local(assignment, *binding.original.operands))
        )(assignments[:5])
        second = jax.jit(
            jax.vmap(lambda assignment: local(assignment, *binding.original.operands))
        )(assignments[5:])
        _assert_close(
            runtime.jnp.sum(first, axis=0) + runtime.jnp.sum(second, axis=0),
            values[0] @ values[1] @ values[2],
            dtype,
        )


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("mode", ("serial", "batched"))
@pytest.mark.parametrize(
    "logarithmic_scaling", (False, True), ids=("ordinary", "log-scaled")
)
def test_cuda_slice_host_checkpoint_order_and_numeric_version(
    provider: TensorProvider,
    dtype: str,
    mode: str,
    logarithmic_scaling: bool,
) -> None:
    with _cuda_case(provider) as runtime:
        binding, values = _binding(runtime, provider, dtype, labels=("a", "b"))
        tn = runtime.tn
        ranges = (
            tn.SliceRange(0, 3, 10),
            tn.SliceRange(3, 7, 10),
            tn.SliceRange(7, 10, 10),
        )
        partials = tuple(
            tn.execute_sliced_contraction(
                binding,
                slice_range=part,
                mode=mode,
                logarithmic_scaling=logarithmic_scaling,
            )
            for part in ranges
        )
        for partial, part in zip(partials, ranges, strict=True):
            assert bool(partial.evidence.finite)
            assert not bool(partial.evidence.complete)
            assert not bool(partial.evidence.accepted)
            assert partial.evidence.processed_slices == part.stop - part.start
            assert partial.evidence.order == "lexicographic-last-label-fastest"
            assert partial.evidence.execution_id == binding.executor.lowering_id
            assert partial.checkpoint.execution_id == binding.executor.lowering_id
            expected = np.zeros((2, 2), dtype=dtype)
            for ordinal in range(part.start, part.stop):
                p, q = divmod(ordinal, 2)
                expected += np.outer(values[0][:, p], values[2][q]) * values[1][p, q]
            _assert_close(partial.value, expected, dtype)
        checkpoints = tuple(partial.checkpoint for partial in partials)
        merged = tn.merge_slice_checkpoints(binding.plan, checkpoints)
        full = tn.execute_sliced_contraction(
            binding, mode=mode, logarithmic_scaling=logarithmic_scaling
        )
        expected = np.einsum("ia,ab,bk->ik", *values, optimize=False)
        _assert_close(merged.value, expected, dtype)
        _assert_close(full.value, expected, dtype)
        assert bool(merged.evidence.accepted) and bool(full.evidence.accepted)
        assert merged.evidence.processed_slices == full.evidence.processed_slices == 10
        assert (
            merged.evidence.execution_id
            == full.evidence.execution_id
            == binding.executor.lowering_id
        )
        assert (
            merged.checkpoint.execution_id
            == full.checkpoint.execution_id
            == binding.executor.lowering_id
        )
        for invalid in (
            checkpoints[::-1],
            checkpoints[1:],
            (checkpoints[0], checkpoints[0], *checkpoints[1:]),
        ):
            with pytest.raises(ValueError, match="ordered|contiguous|cover"):
                tn.merge_slice_checkpoints(binding.plan, invalid)
        refreshed = tn.refresh_contraction(
            binding.original, tuple(value * 2 for value in binding.original.operands)
        )
        refreshed_binding = tn.prepare_sliced_contraction(refreshed, binding.plan)
        newer = tn.execute_sliced_contraction(
            refreshed_binding, slice_range=ranges[-1]
        ).checkpoint
        with pytest.raises(ValueError, match="numeric preparation|ordered"):
            tn.merge_slice_checkpoints(binding.plan, (*checkpoints[:-1], newer))


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cuda_slice_host_checkpoint_and_distributed_orchestration_refuse_transformations(
    provider: TensorProvider, dtype: str
) -> None:
    with _cuda_case(provider) as runtime:
        binding, _ = _binding(runtime, provider, dtype)
        tn, jax = runtime.tn, runtime.jax
        mesh = tn.create_tensor_network_mesh((runtime.devices[0],))
        placement = tn.plan_slice_placement(binding.plan, mesh)

        def host_checkpoint(left: Array) -> Array:
            current = _replace(runtime, binding, (left, *binding.original.operands[1:]))
            return tn.execute_sliced_contraction(current).value

        def host_placement(left: Array) -> Array | None:
            prepared = runtime.eqx.tree_at(
                lambda item: item.operands,
                binding.original,
                (left, *binding.original.operands[1:]),
            )
            return tn.execute_distributed_slices(prepared, placement).aggregate

        for operation in (host_checkpoint, host_placement):
            with pytest.raises(TypeError, match="host-only|cannot be transformed"):
                jax.jit(operation)(binding.original.operands[0])
            with pytest.raises(TypeError, match="host-only|cannot be transformed"):
                jax.grad(lambda left: runtime.jnp.real(runtime.jnp.sum(operation(left))))(
                    binding.original.operands[0]
                )
            with pytest.raises(TypeError, match="host-only|cannot be transformed"):
                jax.vmap(operation)(
                    runtime.jnp.stack(
                        (binding.original.operands[0], binding.original.operands[0])
                    )
                )


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("layout", ("single", "device-subset", "uneven", "idle-device"))
def test_cuda_slice_placement_device_subsets_uneven_counts_and_inactive_padding(
    provider: TensorProvider,
    dtype: str,
    layout: str,
) -> None:
    with _cuda_case(provider) as runtime:
        from phydrax._cuda_native import cuda_native_resources, release_cuda_component

        if layout != "single" and len(runtime.devices) < 2:
            pytest.skip("This placement case requires two physical CUDA devices")
        devices = {
            "single": runtime.devices[:1],
            "device-subset": runtime.devices[-1:],
            "uneven": runtime.devices[:2][::-1],
            "idle-device": runtime.devices[:2][::-1],
        }[layout]
        extent = 1 if layout == "idle-device" else 5
        binding, values = _binding(runtime, provider, dtype, extent=extent)
        tn = runtime.tn
        mesh = tn.create_tensor_network_mesh(devices)
        placement = tn.plan_slice_placement(
            binding.plan,
            mesh,
            resources=tn.PlacementResourcePolicy(
                maximum_devices=len(devices),
                maximum_local_bytes=2**28,
                maximum_transfer_bytes=2**20,
            ),
        )
        assert placement.slice_plan.residual.path == binding.original.plan.path
        release_cuda_component(provider)
        result = tn.execute_distributed_slices(binding.original, placement)
        assert result.aggregate is not None, result.evidence.failure
        _assert_close(result.aggregate, values[0] @ values[1] @ values[2], dtype)
        assert bool(result.evidence.accepted)
        assert bool(result.evidence.completed_all_slices)
        assert result.evidence.failure == ""
        assert result.evidence.deterministic_reduction
        assert result.evidence.device_count == len(devices)
        assert result.evidence.transfer.device_ids == mesh.device_ids
        active_steps = [step for step in placement.replay.steps if step.stop > step.start]
        active_devices = {devices[step.device_ordinal].id for step in active_steps}
        assert sum(step.stop - step.start for step in active_steps) == extent
        # Completed transfer counts exclude inactive lanes and idle-device operand
        # replication. This observes actual orchestration, not implementation text.
        operand_bytes = sum(value.nbytes for value in values) + 4
        output_bytes = 4 * np.dtype(dtype).itemsize
        expected_completed = len(active_steps) * operand_bytes + extent * (
            4 + output_bytes
        )
        assert result.evidence.transfer.completed_bytes == expected_completed
        assert (
            result.evidence.transfer.completed_bytes
            <= result.evidence.transfer.planned_bytes
        )
        if layout in ("uneven", "idle-device"):
            assert placement.replay.padded_slice_count > extent
            assert (
                result.evidence.transfer.completed_bytes
                < result.evidence.transfer.planned_bytes
            )
        if layout == "idle-device":
            assert len(active_steps) == 1 < len(devices)
        ids = {operation.partition for operation in binding.executor.operations}
        observed = [
            part
            for part in cuda_native_resources().partitions
            if part.component == provider and part.partition in ids and part.live
        ]
        assert {part.device for part in observed} == active_devices
        assert all(
            part.live <= part.capacity == binding.executor.policy.maximum_cache_entries
            for part in observed
        )


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cuda_slice_nonfinite_failure_returns_no_partial_aggregate(
    provider: TensorProvider, dtype: str
) -> None:
    with _cuda_case(provider) as runtime:
        binding, _ = _binding(runtime, provider, dtype, nonfinite=True)
        tn = runtime.tn
        mesh = tn.create_tensor_network_mesh(runtime.devices[:2])
        placement = tn.plan_slice_placement(binding.plan, mesh)
        result = tn.execute_distributed_slices(binding.original, placement)
        assert result.aggregate is None
        assert not bool(result.evidence.accepted)
        assert not bool(result.evidence.completed_all_slices)
        assert not bool(result.evidence.exact)
        assert not bool(result.evidence.finite)
        # A numerical failure is observed, not skipped or converted into success.
        assert result.evidence.failure == "non-finite-slice-result"
