"""Real CUDA tensor execution and native-owned transformation acceptance.

Discovery is deferred until a test runs. Only absent hardware or an absent optional
companion may skip; an installed provider's admission, compile, or runtime error
is a test failure. NumPy, not the native JAX executor, supplies numerical oracles.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from math import prod
from types import ModuleType, SimpleNamespace
from typing import Literal, TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray


if TYPE_CHECKING:
    from jax import Array

    from phydrax.tensor_network import ContractionExecutionPolicy, PreparedContraction

type TensorProvider = Literal["cutensor", "cutensornet"]
type TensorScalar = np.float32 | np.float64 | np.complex64 | np.complex128
type TensorArray = NDArray[TensorScalar]


pytestmark = pytest.mark.cuda_native
_PROVIDERS = ("cutensor", "cutensornet")
_DTYPES = ("float32", "float64", "complex64", "complex128")
_CASES = (
    ("scalar-product", ",->", ((), ())),
    ("scalar-reduction", "i,i->", ((3,), (3,))),
    ("outer", "i,j->ji", ((2,), (3,))),
    ("hyperedge", "ij,jk,jl->ilk", ((2, 3), (3, 2), (3, 2))),
    ("repeated-label", "iij,jk->ki", ((2, 2, 3), (3, 2))),
    ("permutation", "ijk,->kij", ((2, 3, 2), ())),
    ("singleton", "iaj,ajk->kia", ((2, 1, 3), (1, 3, 2))),
    ("exclusive-reduced-mode", "ij,k->i", ((2, 3), (2,))),
)


@contextmanager
def _cuda_case(provider: TensorProvider) -> Iterator[SimpleNamespace]:
    # Called only by test bodies, never during module collection.
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from phydrax import config, tensor_network as tn
    from phydrax._cuda_native import CudaNativeUnavailableError, load_cuda_component
    from phydrax._execution_device import device_platform
    from phydrax.execution import default_device

    devices = tuple(
        device for device in jax.local_devices() if device_platform(device) == "cuda"
    )
    if not devices:
        pytest.skip("Real CUDA hardware is unavailable")
    try:
        component = load_cuda_component(provider)
    except CudaNativeUnavailableError as error:
        if error.issue.code == "companion-missing" and not error.issue.paths:
            pytest.skip(str(error))
        raise
    with (
        config.context(config.NumericsConfig(enable_x64=True)),
        default_device(devices[0]),
    ):
        yield SimpleNamespace(
            jax=jax,
            jnp=jnp,
            eqx=eqx,
            tn=tn,
            devices=devices,
            device=devices[0],
            component=component,
        )


def _values(shape: tuple[int, ...], dtype: str, offset: int = 0) -> TensorArray:
    count = prod(shape)
    real = (np.arange(count, dtype=np.float64).reshape(shape) + 1 + offset) / 7
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        real = (
            real
            + 1j
            * (np.arange(count, dtype=np.float64).reshape(shape)[::-1] if shape else 2.0)
            / 11
        )
    return np.asarray(real, dtype=dtype)


def _policy(
    tn: ModuleType, provider: TensorProvider, **overrides: int
) -> ContractionExecutionPolicy:
    bounds = dict(
        maximum_workspace_bytes=2**23,
        maximum_cache_bytes=2**26,
        maximum_cache_entries=4,
        maximum_batch_size=8,
        maximum_derivative_bytes=2**27,
    )
    return tn.ContractionExecutionPolicy(provider, **(bounds | overrides))


def _prepare(
    runtime: SimpleNamespace,
    provider: TensorProvider,
    equation: str,
    operands: tuple[TensorArray, ...],
    *,
    execution: ContractionExecutionPolicy | None = None,
    path: tuple[tuple[int, ...], ...] | None = None,
) -> PreparedContraction:
    tn = runtime.tn
    inputs, output = equation.split("->")
    structure = tn.ContractionStructure(
        tuple(
            tn.ContractionOperand(
                f"operand-{index}",
                tuple(
                    tn.ContractionLeg(label, size)
                    for label, size in zip(labels, value.shape, strict=True)
                ),
            )
            for index, (labels, value) in enumerate(
                zip(inputs.split(","), operands, strict=True)
            )
        ),
        tuple(output),
    )
    if path is None:
        path = tuple((0, 1) for _ in range(len(operands) - 1))
    dtype = str(operands[0].dtype)
    precision = tn.TensorNetworkPrecisionPolicy(
        storage_dtype=dtype,
        contraction_dtype=dtype,
        accumulation_dtype=dtype,
        output_dtype=dtype,
    )
    plan = tn.plan_contraction(structure, dtype=dtype, precision=precision, path=path)
    arrays = tuple(runtime.jax.device_put(value, runtime.device) for value in operands)
    return tn.prepare_contraction(
        plan,
        arrays,
        execution=_policy(tn, provider) if execution is None else execution,
    )


def _evaluator(
    runtime: SimpleNamespace, prepared: PreparedContraction
) -> Callable[..., Array]:
    def evaluate(*values: Array) -> Array:
        current = runtime.eqx.tree_at(lambda item: item.operands, prepared, tuple(values))
        return runtime.tn.execute_contraction(current).value

    return evaluate


def _assert_close(actual: Array | ArrayLike, expected: ArrayLike, dtype: str) -> None:
    rtol, atol = (8e-5, 1e-5) if dtype in ("float32", "complex64") else (3e-11, 3e-12)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol
    )
    assert np.asarray(actual).dtype == np.dtype(dtype)


def _numpy_jvp(
    equation: str,
    values: tuple[TensorArray, ...],
    tangents: tuple[TensorArray, ...],
) -> TensorArray:
    terms = tuple(
        np.asarray(
            np.einsum(
                equation,
                *(values[:index] + (tangent,) + values[index + 1 :]),
                optimize=False,
            )
        )
        for index, tangent in enumerate(tangents)
    )
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def _numpy_vjp(
    equation: str,
    values: tuple[TensorArray, ...],
    cotangent: TensorArray,
) -> tuple[TensorArray, ...]:
    # Basis evaluation is deliberately independent of the provider's transpose
    # recipes, including its diagonal scatter and exclusive-mode broadcasts.
    results = []
    for index, value in enumerate(values):
        derivative = np.zeros_like(value)
        for entry in np.ndindex(value.shape):
            basis = np.zeros_like(value)
            basis[entry] = 1
            image = np.einsum(
                equation,
                *(values[:index] + (basis,) + values[index + 1 :]),
                optimize=False,
            )
            derivative[entry] = np.sum(cotangent * image)
        results.append(derivative)
    return tuple(results)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("name,equation,shapes", _CASES, ids=[case[0] for case in _CASES])
def test_cuda_tensor_numpy_primal_jvp_and_bilinear_vjp(
    provider: TensorProvider,
    dtype: str,
    name: str,
    equation: str,
    shapes: tuple[tuple[int, ...], ...],
) -> None:
    with _cuda_case(provider) as runtime:
        values = tuple(_values(shape, dtype, index) for index, shape in enumerate(shapes))
        prepared = _prepare(runtime, provider, equation, values)
        evaluate = _evaluator(runtime, prepared)
        expected = np.einsum(equation, *values, optimize=False)
        tangents = tuple(
            _values(shape, dtype, index + 3) for index, shape in enumerate(shapes)
        )
        device_tangents = tuple(
            runtime.jax.device_put(value, runtime.device) for value in tangents
        )
        cotangent = _values(expected.shape, dtype, 2)
        device_cotangent = runtime.jax.device_put(cotangent, runtime.device)

        actual, tangent = runtime.jax.jit(
            lambda *xs: runtime.jax.jvp(evaluate, xs, device_tangents)
        )(*prepared.operands)
        _assert_close(actual, expected, dtype)
        _assert_close(tangent, _numpy_jvp(equation, values, tangents), dtype)
        pullback = runtime.jax.jit(
            lambda *xs: runtime.jax.vjp(evaluate, *xs)[1](device_cotangent)
        )(*prepared.operands)
        for actual_bar, expected_bar in zip(
            pullback, _numpy_vjp(equation, values, cotangent), strict=True
        ):
            _assert_close(actual_bar, expected_bar, dtype)

        zero = runtime.jax.jit(
            lambda *xs: runtime.jax.jvp(
                evaluate, xs, tuple(runtime.jnp.zeros_like(x) for x in xs)
            )[1]
        )(*prepared.operands)
        _assert_close(zero, np.zeros_like(expected), dtype)
        zero_bars = runtime.jax.jit(
            lambda *xs: runtime.jax.vjp(evaluate, *xs)[1](
                runtime.jnp.zeros_like(device_cotangent)
            )
        )(*prepared.operands)
        for bar, value in zip(zero_bars, values, strict=True):
            _assert_close(bar, np.zeros_like(value), dtype)

        result = runtime.tn.execute_contraction(prepared)
        assert bool(result.evidence.accepted)
        assert result.evidence.provider == provider
        assert result.evidence.execution_id == prepared.executor.lowering_id
        assert result.value.devices() == {runtime.device}
        assert prepared.executor.vendor_host_bound_bytes is None
        if name == "repeated-label":
            assert prepared.executor.normalization_bytes > 0
            diagonal_bar = np.asarray(pullback[0])
            assert np.count_nonzero(diagonal_bar[0, 1]) == 0
            assert np.count_nonzero(diagonal_bar[1, 0]) == 0


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cuda_tensor_current_data_transforms_share_one_compiled_executable(
    provider: TensorProvider, dtype: str
) -> None:
    with _cuda_case(provider) as runtime:
        jax, jnp = runtime.jax, runtime.jnp
        values = (_values((2, 3), dtype), _values((3, 2), dtype, 2))
        prepared = _prepare(runtime, provider, "ij,jk->ik", values)
        evaluate = _evaluator(runtime, prepared)

        def transformed(
            a: Array,
            b: Array,
            da: Array,
            db: Array,
            cotangent: Array,
        ) -> tuple[Array, Array, tuple[Array, ...], Array, Array, Array, Array]:
            value, tangent = jax.jvp(evaluate, (a, b), (da, db))
            bars = jax.vjp(evaluate, a, b)[1](cotangent)
            # The closed-over b has an AD zero tangent, not a numerical buffer.
            one_slot = jax.jvp(lambda x: evaluate(x, b), (a,), (da,))[1]
            mixed = jax.jvp(
                lambda y: jax.jvp(lambda x: evaluate(x, y), (a,), (da,))[1],
                (b,),
                (db,),
            )[1]
            loss = lambda x: jnp.sum(jnp.abs(evaluate(x, b)) ** 2)
            gradient = jax.grad(loss)(a)
            hvp = jax.jvp(jax.grad(loss), (a,), (da,))[1]
            return value, tangent, bars, one_slot, mixed, gradient, hvp

        da, db = (_values(value.shape, dtype, 4) for value in values)
        cotangent = _values((2, 2), dtype, 1)
        arguments = tuple(
            jax.device_put(value, runtime.device)
            for value in (*values, da, db, cotangent)
        )
        executable = jax.jit(transformed).lower(*arguments).compile()
        for offset in (0, 3, -2):
            a = _values((2, 3), dtype, offset)
            b = _values((3, 2), dtype, offset + 2)
            current = tuple(
                jax.device_put(value, runtime.device)
                for value in (a, b, da, db, cotangent)
            )
            value, tangent, bars, one_slot, mixed, gradient, hvp = executable(*current)
            _assert_close(value, a @ b, dtype)
            _assert_close(tangent, da @ b + a @ db, dtype)
            _assert_close(bars[0], cotangent @ b.T, dtype)
            _assert_close(bars[1], a.T @ cotangent, dtype)
            _assert_close(one_slot, da @ b, dtype)
            _assert_close(mixed, da @ db, dtype)
            # JAX's real-loss complex gradient convention is bilinear pairing:
            # dL = Re(sum(gradient * da)), not a Hermitian VJP convention.
            _assert_close(gradient, 2 * np.conj(a @ b) @ b.T, dtype)
            _assert_close(hvp, 2 * np.conj(da @ b) @ b.T, dtype)

        batched_a = np.stack((values[0], values[0] * 2, values[0] * -1))
        mapped = jax.jit(jax.vmap(evaluate, in_axes=(0, None)))(
            jax.device_put(batched_a, runtime.device),
            prepared.operands[1],
        )
        _assert_close(mapped, np.stack(tuple(a @ values[1] for a in batched_a)), dtype)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cuda_tensor_repeated_operand_chain_rule_and_second_order(
    provider: TensorProvider, dtype: str
) -> None:
    with _cuda_case(provider) as runtime:
        x, dx = _values((3,), dtype), _values((3,), dtype, 2)
        prepared = _prepare(runtime, provider, "i,i->", (x, x))
        evaluate = _evaluator(runtime, prepared)
        repeated = lambda value: evaluate(value, value)
        real_loss = lambda value: runtime.jnp.real(repeated(value))
        compiled = runtime.jax.jit(
            lambda value, tangent: (
                runtime.jax.jvp(repeated, (value,), (tangent,)),
                runtime.jax.grad(real_loss)(value),
                runtime.jax.jvp(runtime.jax.grad(real_loss), (value,), (tangent,))[1],
            )
        )
        (value, tangent), gradient, hvp = compiled(
            prepared.operands[0],
            runtime.jax.device_put(dx, runtime.device),
        )
        _assert_close(value, np.sum(x * x), dtype)
        _assert_close(tangent, 2 * np.sum(x * dx), dtype)
        _assert_close(gradient, 2 * x, dtype)
        _assert_close(hvp, 2 * dx, dtype)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cuda_tensor_refresh_concurrency_and_policy_cache_isolation(
    provider: TensorProvider, dtype: str
) -> None:
    with _cuda_case(provider) as runtime:
        from phydrax._cuda_native import cuda_native_resources

        values = (_values((2, 3), dtype), _values((3, 2), dtype, 1))
        prepared = _prepare(
            runtime,
            provider,
            "ij,jk->ik",
            values,
            execution=_policy(runtime.tn, provider, maximum_cache_entries=1),
        )
        changed = tuple(value * np.asarray(2, dtype=dtype) for value in values)
        refreshed = runtime.tn.refresh_contraction(
            prepared,
            tuple(runtime.jax.device_put(value, runtime.device) for value in changed),
        )
        assert refreshed.executor is prepared.executor
        assert refreshed.prepared_id != prepared.prepared_id
        assert refreshed.operand_id != prepared.operand_id
        assert int(refreshed.numeric_version) == int(prepared.numeric_version) + 1
        _assert_close(
            runtime.tn.execute_contraction(refreshed).value,
            changed[0] @ changed[1],
            dtype,
        )
        isolated = _prepare(
            runtime,
            provider,
            "ij,jk->ik",
            values,
            execution=_policy(runtime.tn, provider, maximum_cache_entries=2),
        )
        assert isolated.executor.lowering_id != prepared.executor.lowering_id
        assert {op.partition for op in isolated.executor.operations}.isdisjoint(
            {op.partition for op in prepared.executor.operations}
        )
        executables = tuple(
            runtime.jax.jit(_evaluator(runtime, item)).lower(*item.operands).compile()
            for item in (prepared, isolated)
        )
        work = []
        for index in range(4):
            operands = tuple(
                value * np.asarray(index + 1, dtype=dtype) for value in values
            )
            arrays = tuple(
                runtime.jax.device_put(value, runtime.device) for value in operands
            )
            work.append((executables[index % 2], arrays, operands[0] @ operands[1]))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(executable, *arrays) for executable, arrays, _ in work]
            for future, (_, _, expected) in zip(futures, work, strict=True):
                _assert_close(future.result(), expected, dtype)
        partitions = cuda_native_resources().partitions
        for item in (prepared, isolated):
            ids = {op.partition for op in item.executor.operations}
            observed = [
                part
                for part in partitions
                if part.component == provider and part.partition in ids
            ]
            assert observed
            assert all(
                0
                < part.live
                <= part.capacity
                == item.executor.policy.maximum_cache_entries
                for part in observed
            )


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cuda_tensor_registry_release_rebuild_and_resource_admission(
    provider: TensorProvider, dtype: str
) -> None:
    with _cuda_case(provider) as runtime:
        from phydrax._cuda_native import cuda_native_resources, release_cuda_component

        values = (_values((2, 3), dtype), _values((3, 2), dtype, 1))
        policy = _policy(runtime.tn, provider, maximum_cache_entries=2)
        prepared = _prepare(runtime, provider, "ij,jk->ik", values, execution=policy)
        evaluate = _evaluator(runtime, prepared)
        executable = runtime.jax.jit(evaluate).lower(*prepared.operands).compile()
        _assert_close(executable(*prepared.operands), values[0] @ values[1], dtype)
        executor = prepared.executor
        assert executor.build_identity and executor.lowering_id
        assert executor.external_workspace_bound_bytes == policy.maximum_workspace_bytes
        assert executor.external_cache_bound_bytes == policy.maximum_cache_bytes
        assert executor.vendor_host_bound_bytes is None
        assert executor.derivative_contract
        ids = {operation.partition for operation in executor.operations}
        before = [
            part
            for part in cuda_native_resources().partitions
            if part.component == provider and part.partition in ids
        ]
        assert before and all(0 < part.live <= 2 for part in before)
        assert release_cuda_component(provider) > 0
        assert not any(
            part.live
            for part in cuda_native_resources().partitions
            if part.component == provider
        )
        # The already-compiled callable owns a reconstructible recipe, not a stale
        # vendor pointer. A different numerical allocation must rebuild and rebind.
        changed = tuple(value * np.asarray(3, dtype=dtype) for value in values)
        arrays = tuple(runtime.jax.device_put(value, runtime.device) for value in changed)
        _assert_close(executable(*arrays), changed[0] @ changed[1], dtype)
        rebuilt = [
            part
            for part in cuda_native_resources().partitions
            if part.component == provider and part.partition in ids
        ]
        assert rebuilt and all(
            0 < part.live <= part.capacity == 2 and part.rebuilds > 0 for part in rebuilt
        )
        cotangent = runtime.jax.device_put(_values((2, 2), dtype), runtime.device)
        bars = runtime.jax.jit(lambda *xs: runtime.jax.vjp(evaluate, *xs)[1](cotangent))(
            *arrays
        )
        _assert_close(bars[0], np.asarray(cotangent) @ changed[1].T, dtype)
        _assert_close(bars[1], changed[0].T @ np.asarray(cotangent), dtype)
        # Primal and both transpose descriptors exceed capacity two. Replaying
        # the retained primal after this pressure must reconstruct any eviction.
        _assert_close(executable(*prepared.operands), values[0] @ values[1], dtype)
        pressured = [
            part
            for part in cuda_native_resources().partitions
            if part.component == provider and part.partition in ids
        ]
        assert pressured and all(
            0 < part.live <= part.capacity == 2 for part in pressured
        )
        resources = runtime.tn.ContractionResourcePolicy(
            maximum_workspace_bytes=prepared.plan.cost.workspace_bytes
            + policy.maximum_cache_bytes
            - 1,
        )
        constrained_plan = runtime.tn.plan_contraction(
            prepared.plan.structure,
            dtype=dtype,
            precision=prepared.plan.precision,
            path=prepared.plan.path,
            resources=resources,
        )
        with pytest.raises(MemoryError, match="maximum_workspace_bytes"):
            runtime.tn.prepare_contraction(
                constrained_plan, prepared.operands, execution=policy
            )
        with pytest.raises(MemoryError, match="maximum_derivative_bytes"):
            _prepare(
                runtime,
                provider,
                "ij,jk->ik",
                values,
                execution=_policy(runtime.tn, provider, maximum_derivative_bytes=1),
            )
        bounded = _prepare(
            runtime,
            provider,
            "ij,jk->ik",
            values,
            execution=_policy(runtime.tn, provider, maximum_batch_size=1),
        )
        oversized = runtime.jnp.stack((bounded.operands[0], bounded.operands[0]))
        with pytest.raises(MemoryError, match="maximum_batch_size"):
            runtime.jax.jit(
                runtime.jax.vmap(_evaluator(runtime, bounded), in_axes=(0, None))
            )(
                oversized,
                bounded.operands[1],
            )


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cuda_tensor_unary_and_imported_nary_path_contract(
    provider: TensorProvider, dtype: str
) -> None:
    with _cuda_case(provider) as runtime:
        # An empty binary path is a complete one-input plan; cuTENSOR still
        # executes a unary permutation/reduction and owns its true transpose.
        value = _values((2, 3), dtype)
        unary = _prepare(runtime, provider, "ij->ji", (value,))
        evaluate = _evaluator(runtime, unary)
        actual = runtime.jax.jit(evaluate)(*unary.operands)
        _assert_close(actual, value.T, dtype)
        cotangent = runtime.jax.device_put(_values((3, 2), dtype, 2), runtime.device)
        bar = runtime.jax.jit(lambda x: runtime.jax.vjp(evaluate, x)[1](cotangent)[0])(
            unary.operands[0]
        )
        _assert_close(bar, np.asarray(cotangent).T, dtype)
        inputs = tuple(_values((2,), dtype, index) for index in range(3))
        with pytest.raises(ValueError, match="unary/binary|binary"):
            _prepare(runtime, provider, "i,i,i->", inputs, path=((0, 1, 2),))
