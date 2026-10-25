# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded installed-wheel CUDA campaign; scientific formulas remain with their owners.

The launcher copies the harness, then invokes the selected interpreter with -I.
The worker is Windows/single-GPU only. This is not peer or collective evidence.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING


if TYPE_CHECKING:
    from jax import Array

    from phydrax._cuda_native import CudaNativeComponentName
    from phydrax.linalg import PreparedLinearSolve


@dataclass(frozen=True)
class Case:
    name: str
    capacity: int
    function: Callable[..., Any]
    arguments: tuple[Any, ...]
    check: Callable[[Any], None]
    retained: Any = None
    provider: CudaNativeComponentName | None = None
    derivative: Callable[..., Any] | None = None
    derivative_check: Callable[[Any], None] | None = None
    derivative_reason: str | None = None


def _agree(
    actual: Any,
    expected: Any,
    tolerance: float,
    *,
    absolute_tolerance: float | None = None,
) -> None:
    import jax
    import numpy as np

    actual_leaves, actual_tree = jax.tree.flatten(actual)
    expected_leaves, expected_tree = jax.tree.flatten(expected)
    if actual_tree != expected_tree:
        raise ValueError("Scientific result structure differs from its reference")
    for value, reference in zip(actual_leaves, expected_leaves, strict=True):
        left, right = np.asarray(value), np.asarray(reference)
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError("Nonfinite scientific result/reference")
        np.testing.assert_allclose(
            left,
            right,
            rtol=tolerance,
            atol=tolerance if absolute_tolerance is None else absolute_tolerance,
        )


def _payload(value: Any) -> Any:
    """Serialize observed diagnostic dataclasses without losing native statuses."""
    from dataclasses import fields, is_dataclass

    import jax
    import numpy as np

    if is_dataclass(value):
        return {
            field.name: _payload(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, dict):
        return {key: _payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_payload(item) for item in value]
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        return np.asarray(value).tolist()
    return value


def _measure(case: Case, preparation: float | None, repeats: int) -> dict[str, Any]:
    import equinox as eqx
    import jax

    from benchmarks._runtime import (
        compiler_evidence,
        logical_array_bytes,
        measure_host,
        measure_lower_and_compile,
        measure_repeated,
        measure_synchronized,
    )
    from phydrax._cuda_native import cuda_native_resources, release_cuda_component
    from phydrax._execution_sampling import sample_device_memory, sample_host_memory

    # Fixtures may prepare on-device metadata. This explicit round trip measures
    # transfer separately; fixture preparation remains independently recorded.
    host, read_inputs = measure_host(lambda: jax.device_get(case.arguments))
    arguments, upload = measure_synchronized(lambda: jax.device_put(host))
    compiled, timing = measure_lower_and_compile(
        # Equinox preserves the callable annotation but returns its lowering wrapper.
        lambda: eqx.filter_jit(case.function).lower(*arguments),  # ty: ignore[unresolved-attribute]
        lambda lowered: lowered.compile(),
    )
    first, first_seconds = measure_synchronized(lambda: compiled(*arguments))
    case.check(first)  # correctness gates precede every warm timing series
    for value in jax.tree.leaves(first):
        if isinstance(value, jax.Array) and value.devices() != {jax.local_devices()[0]}:
            raise ValueError("Scientific result escaped the admitted CUDA device")
    result, warm = measure_repeated(
        lambda: compiled(*arguments), warmup=1, repeats=repeats
    )
    case.check(result)
    _, download = measure_host(lambda: jax.device_get(result))
    record = {
        "name": case.name,
        "capacity": case.capacity,
        "provider": case.provider,
        "passed": True,
        "preparation_seconds": preparation,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "first_seconds": first_seconds,
        "warm": warm.to_dict(),
        "transfers": {
            "fixture_readback_seconds": read_inputs,
            "host_to_device_seconds": upload,
            "device_to_host_seconds": download,
        },
        "compiler": asdict(
            compiler_evidence(
                compiled.compiled.cost_analysis(),
                compiled.compiled.memory_analysis(),
                source="jax-compiled",
                unavailable_reason="Compiler counter unavailable",
                opaque_custom_calls=case.provider is not None,
            )
        ),
        "logical_retained_bytes": logical_array_bytes(
            (case.retained, arguments, first, result)
        ),
        "logical_retained_bytes_scope": "prepared scientific state, transferred inputs, first and latest result; excludes oracle host storage, executable and opaque vendor allocations",
        "host_memory": sample_host_memory().to_payload(),
        "device_memory": [
            sample.to_payload() for sample in sample_device_memory(jax.local_devices())
        ],
        "native_retained_bytes": {
            "value": None,
            "unavailable_reason": "Native registry exposes entry counts, not vendor allocation byte counters",
        }
        if case.provider
        else None,
        "solver_evidence": {
            name: _payload(getattr(result, name))
            for name in ("status", "successful", "success", "diagnostics", "provenance")
            if hasattr(result, name)
        },
        "derivatives": {"status": "not-applicable", "reason": case.derivative_reason},
    }
    derivative_function = case.derivative
    if derivative_function is not None:
        derivative, derivative_timing = measure_lower_and_compile(
            lambda: eqx.filter_jit(derivative_function).lower(*arguments),  # ty: ignore[unresolved-attribute]
            lambda lowered: lowered.compile(),
        )
        value, elapsed = measure_synchronized(lambda: derivative(*arguments))
        if case.derivative_check is None:
            raise ValueError("Derivative timing requires an agreement oracle")
        case.derivative_check(value)
        _, samples = measure_repeated(
            lambda: derivative(*arguments), warmup=1, repeats=repeats
        )
        record["derivatives"] = {
            "status": "measured",
            "lowering_seconds": derivative_timing.lowering_seconds,
            "compilation_seconds": derivative_timing.compilation_seconds,
            "first_seconds": elapsed,
            "warm": samples.to_dict(),
            "compiler": asdict(
                compiler_evidence(
                    derivative.compiled.cost_analysis(),
                    derivative.compiled.memory_analysis(),
                    source="jax-compiled",
                    unavailable_reason="Compiler counter unavailable",
                )
            ),
        }
    if case.provider is not None:
        resources = cuda_native_resources()
        record["native_resources"] = asdict(resources)
        # A released registry forces real native state reconstruction in the same
        # compiled program; it does not pretend that executable compilation is cold.
        release_cuda_component(case.provider)
        rebuilt, elapsed = measure_synchronized(lambda: compiled(*arguments))
        case.check(rebuilt)
        record["cold_native_rebuild_seconds"] = elapsed
        record["cold_scope"] = "native registry released; compiled executable retained"
    return record


def _ordinary_cases() -> list[Case]:
    from benchmarks.metal_execution_scaling import _cases

    return [
        Case(
            item.name,
            item.capacity,
            item.function,
            item.inputs,
            lambda value, item=item: _agree(
                value,
                item.reference,
                item.relative_tolerance,
                absolute_tolerance=item.absolute_tolerance,
            ),
            derivative_reason="Index/reduction/contraction throughput row; AD measured in compensated and MPM rows",
        )
        for item in _cases()
    ]


def _array_jvp(
    function: Callable[[Array], Array], direction: Array, value: Array
) -> Array:
    import jax

    return jax.jvp(function, (value,), (direction,))[1]


def _solve_jvp(prepared: PreparedLinearSolve, direction: Array, value: Array) -> Any:
    import jax

    from phydrax.linalg import solve

    # The primal row checks status; the tangent belongs to the numerical value.
    return jax.jvp(lambda rhs: solve(prepared, rhs).value, (value,), (direction,))[1]


def _dense_or_fft(size: int, dtype: Any, route: str) -> Case:
    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np

    tolerance = (
        3e-4
        if np.dtype(dtype).itemsize
        <= (8 if np.issubdtype(dtype, np.complexfloating) else 4)
        else 1e-10
    )

    rng = np.random.default_rng(408)
    values = rng.normal(size=size).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values = values + 1j * rng.normal(size=size).astype(dtype)
    retained = None
    prepared: PreparedLinearSolve | None = None
    if route == "fft":
        function = jnp.fft.fft
        expected = np.fft.fft(values)
        direction = np.ones_like(values)
        expected_tangent = np.fft.fft(direction)
    else:
        matrix = (2 * np.eye(size) - np.eye(size, k=1) - np.eye(size, k=-1)).astype(dtype)
        retained = matrix
        direction = np.ones_like(values)
        if route == "sparse-apply":
            import phydrax as phx

            rows, columns = np.nonzero(matrix)
            relation = phx.sparse.EdgeRelation(
                columns.astype(np.int64),
                rows.astype(np.int64),
                source_size=size,
                target_size=size,
            )
            operator = phx.sparse.SparseLinearMap(relation, matrix[rows, columns])
            retained = operator
            function = eqx.Partial(phx.sparse.SparseLinearMap.mv, operator)
            expected = matrix @ values
            expected_tangent = matrix @ direction
        else:
            import phydrax as phx

            # This prepared-direct profile reuses its factors for derivatives.
            # The native residual request matches the predeclared oracle accuracy;
            # product defaults and the independent Krylov route remain unchanged.
            prepared = phx.linalg.prepare(
                phx.linalg.LinearSystem(
                    phx.linalg.DenseLinearOperator(jnp.asarray(matrix))
                ),
                phx.linalg.LinearSolvePolicy(
                    derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(
                        route="primal-factors",
                        relative_tolerance=tolerance,
                        absolute_tolerance=tolerance,
                    )
                ),
            )
            retained = prepared
            function = eqx.Partial(phx.linalg.solve, prepared)
            expected = np.linalg.solve(matrix, values)
            expected_tangent = np.linalg.solve(matrix, direction)

    def check(value: Any) -> None:
        if prepared is not None:
            if not bool(np.all(np.asarray(value.successful))):
                raise ValueError(f"Native dense solve failed: {value.status}")
            value = value.value
        _agree(value, expected, tolerance)

    derivative = (
        eqx.Partial(_solve_jvp, prepared, jnp.asarray(direction))
        if prepared is not None
        else eqx.Partial(_array_jvp, function, jnp.asarray(direction))
    )
    return Case(
        f"ordinary-{route}-{np.dtype(dtype).name}",
        size,
        function,
        (values,),
        check,
        retained=retained,
        derivative=derivative,
        derivative_check=lambda value: _agree(value, expected_tangent, tolerance),
    )


def _compensated(size: int, dtype: Any) -> Case:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from phydrax._numerics._compensated import compensated_sum

    # Same nonlinear recurrence and derivative contract as the owning unit case.
    values = np.arange(size, dtype=dtype) / size
    direction = np.full_like(values, 0.25)

    def loss(value: jax.Array) -> jax.Array:
        return compensated_sum(value * value)

    def derivatives(value: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        gradient = jax.grad(loss)(value)
        _, tangent = jax.jvp(loss, (value,), (jnp.asarray(direction),))
        _, hvp = jax.jvp(jax.grad(loss), (value,), (jnp.asarray(direction),))
        return gradient, tangent, hvp

    expected = (
        2 * values,
        np.sum(2 * values * direction, dtype=np.float64),
        2 * direction,
    )
    tolerance = 2e-5 if dtype == np.float32 else 1e-11
    return Case(
        f"compensated-{np.dtype(dtype).name}",
        size,
        loss,
        (values,),
        lambda result: _agree(result, np.sum(values.astype(np.float64) ** 2), tolerance),
        derivative=derivatives,
        derivative_check=lambda result: _agree(result, expected, tolerance),
    )


def _mpm(size: int, dtype: Any) -> Case:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from tests.unit.discretization.test_mpm_apic_streaming_moments import (
        _assert_moments,
        _derivatives,
        _operands,
        _probes,
        _reference_all,
        _streaming_all,
    )

    operands = _operands(23, size, 27, 97, 3, 3, 3, dtype)
    probes = _probes(operands, 29)
    directions = (
        jnp.full_like(operands[2], 0.25),
        jnp.full_like(operands[3], 0.25),
        jnp.full_like(operands[4], 0.25),
        jnp.full_like(operands[5], 0.25),
    )
    expected = _derivatives(_reference_all, probes, operands, directions)
    jax.block_until_ready(expected)
    tolerance = 3e-4 if dtype == np.float32 else 1e-9
    return Case(
        f"mpm-apic-{np.dtype(dtype).name}",
        size,
        _streaming_all,
        operands,
        lambda result: _assert_moments(result, operands, dtype),
        retained=(probes, directions),
        derivative=lambda *args: _derivatives(_streaming_all, probes, args, directions),
        derivative_check=lambda result: _agree(result, expected, tolerance),
    )


def _solver(provider: Literal["cudss", "amgx"], size: int, dtype: Any) -> Case:
    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np

    import phydrax as phx
    from phydrax.backends import AmgXPolicy, plan_amgx, prepare_amgx, solve_amgx

    # Existing Poisson/provider fixture, at two controlling matrix capacities.
    matrix = (2 * np.eye(size) - np.eye(size, k=1) - np.eye(size, k=-1)).astype(dtype)
    rhs = np.ones(size, dtype=dtype)
    rows, columns = np.nonzero(matrix)
    relation = phx.sparse.EdgeRelation(
        columns.astype(np.int32),
        rows.astype(np.int32),
        source_size=size,
        target_size=size,
    )
    operator = phx.sparse.SparseLinearMap(
        relation,
        jnp.asarray(matrix[rows, columns]),
        operator_id=f"windows-{provider}-{size}",
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "asserted"},
        ),
    )
    system = phx.linalg.LinearSystem(operator)
    reference = np.linalg.solve(matrix.astype(np.float64), rhs.astype(np.float64))
    if provider == "cudss":
        prepared_cudss = phx.linalg.prepare(
            system, phx.linalg.LinearSolvePolicy(phx.linalg.SparseLDLT(provider="cudss"))
        )
        solve = eqx.Partial(phx.linalg.solve, prepared_cudss)
        prepared = prepared_cudss
    else:
        prepared_amgx = prepare_amgx(system, plan_amgx(system, AmgXPolicy()))
        solve = eqx.Partial(solve_amgx, prepared_amgx)
        prepared = prepared_amgx

    def check(result: Any) -> None:
        successful = result.successful if provider == "cudss" else result.success
        if not bool(np.all(np.asarray(successful))):
            raise ValueError(f"{provider} reported unsuccessful solve: {result}")
        _agree(result.value, reference, 3e-4 if dtype == np.float32 else 1e-7)

    return Case(
        f"{provider}-{np.dtype(dtype).name}",
        size,
        solve,
        (rhs,),
        check,
        retained=prepared,
        provider=provider,
        derivative_reason="Native solve row records native solver status; AD is measured on canonical numerical owners",
    )


def _quda(extent: int, dtype: Any, operation: str) -> Case:
    import jax.numpy as jnp
    import numpy as np

    from phydrax.backends import create_lattice_provider
    from tests.unit.backends.test_lattice_providers import _wilson_case

    operator, spinor, links = _wilson_case((extent,) * 4, dtype)
    provider = create_lattice_provider("quda")
    reference = create_lattice_provider("jax-native-lattice")
    if operation == "wilson_dslash":
        function = lambda value: provider.wilson_dslash(operator, value)
        expected = reference.wilson_dslash(operator, spinor)
        arguments = (spinor,)
    else:
        mask = jnp.ones(
            (extent**4, 6 if operation == "gauge_action" else 4), dtype=jnp.bool_
        )
        kwargs = {"lattice_shape": (extent,) * 4, "periodic": (True,) * 4}
        function = lambda value: getattr(provider, operation)(value, 5.7, mask, **kwargs)
        expected = getattr(reference, operation)(links, 5.7, mask, **kwargs)
        arguments = (links,)
    tolerance = 8e-5 if dtype == np.complex64 else 4e-11
    return Case(
        f"quda-{operation}-{np.dtype(dtype).name}",
        extent**4,
        function,
        arguments,
        lambda value: _agree(value, expected, tolerance),
        retained=operator,
        provider="quda",
        derivative_reason="QUDA owner admits jit only, not JVP/VJP; no substituted derivative",
    )


def _cutensor(size: int, repeats: int) -> dict[str, Any]:
    import jax
    import numpy as np

    from benchmarks._runtime import measure_host, measure_synchronized
    from benchmarks.cuda_tensor_contraction import _case, _derivative_reference
    from phydrax.tensor_network import (
        ContractionLeg,
        ContractionOperand,
        ContractionStructure,
        plan_contraction,
    )

    labels = ("a", "b", "c", "d", "e")
    structure = ContractionStructure(
        tuple(
            ContractionOperand(
                f"matrix-{index}",
                (
                    ContractionLeg(labels[index], size),
                    ContractionLeg(labels[index + 1], size),
                ),
            )
            for index in range(4)
        ),
        ("e", "a"),
    )
    generator = np.random.default_rng(408)
    host = tuple(
        generator.normal(size=(size, size)).astype(np.float64) / np.sqrt(size)
        for _ in range(4)
    )
    operands, transfer = measure_synchronized(lambda: jax.device_put(host))
    plan, planning = measure_host(lambda: plan_contraction(structure, dtype="float64"))
    record = _case(
        "native-path-cutensor-executor",
        plan,
        operands,
        (host[0] @ host[1] @ host[2] @ host[3]).T,
        _derivative_reference(host),
        "cutensor",
        repeats,
        tuple(jax.local_devices()),
    )
    record.update(
        capacity=size,
        input_transfer_seconds=transfer,
        planning_seconds=planning,
        passed=True,
    )
    return record


def campaign(repeats: int, cutensor: bool, record: dict[str, Any]) -> None:
    import jax
    import numpy as np

    from benchmarks._runtime import (
        capture_environment,
        installed_build_fingerprint,
        measure_synchronized,
    )
    from phydrax import config as pxconfig
    from phydrax._cuda_native import release_cuda_component
    from tests._support.cuda_execution_worker import cuda_runtime, inventory

    if sys.platform != "win32":
        raise ValueError("This campaign qualifies native Windows only")
    if jax.process_count() != 1 or len(jax.devices()) != 1:
        raise ValueError("Windows campaign requires exactly one process and one GPU")
    if jax.local_devices()[0].platform != "gpu" or not pxconfig.x64_enabled():
        raise ValueError("CUDA GPU with enabled x64 is required")
    if pxconfig.compilation_cache_enabled():
        raise ValueError("Compilation timing requires the persistent JAX cache disabled")
    record["persistent_jax_compilation_cache"] = False
    record["installed_build_fingerprint"] = installed_build_fingerprint()
    record["inventory"] = inventory(cuda_runtime())
    record["environment"] = capture_environment().to_dict()
    from tests.unit import test_compensated_reductions as compensated_contract
    from tests.unit.discretization import test_mpm_apic_streaming_moments as mpm_contract

    compensated_contract.test_compensated_contracts()
    for dtype in (np.float32, np.float64):
        compensated_contract.test_long_reduction_nonfinite_lanes_remain_explicit(dtype)
        mpm_contract.test_masks_nonfinite_metadata_and_zero_weights(dtype)
    for dtype in (np.float32, np.float64, np.complex64, np.complex128):
        compensated_contract.test_seeded_recurrence_preserves_reference_components(
            4097, dtype
        )
    record["semantic_gates"] = {
        "compensated_cancellation_seeded_nonfinite": "passed",
        "mpm_masks_nonfinite_zero_weights": "passed",
    }
    ordinary, preparation = measure_synchronized(_ordinary_cases)
    record["ordinary_fixture_preparation_seconds"] = preparation
    for case in ordinary:
        row = _measure(case, None, repeats)
        row["preparation_scope"] = "shared ordinary_fixture_preparation_seconds"
        record["cases"].append(row)
    factories: list[Callable[[], Case]] = [
        lambda n=n, dtype=dtype, route=route: _dense_or_fft(n, dtype, route)
        for n in (32, 128)
        for dtype in (np.float32, np.float64, np.complex64, np.complex128)
        for route in ("dense-solve", "fft", "sparse-apply")
    ]
    factories += [
        lambda n=n, dtype=dtype: _compensated(n, dtype)
        for n in (65, 4097)
        for dtype in (np.float32, np.float64)
    ]
    factories += [
        lambda n=n, dtype=dtype: _mpm(n, dtype)
        for n in (7, 130)
        for dtype in (np.float32, np.float64)
    ]
    factories += [
        lambda provider=provider, n=n, dtype=dtype: _solver(provider, n, dtype)
        for provider in ("cudss", "amgx")
        for n in (64, 128)
        for dtype in (np.float32, np.float64)
    ]
    factories += [
        lambda n=n, dtype=dtype, operation=operation: _quda(n, dtype, operation)
        for n in (4, 8)
        for dtype in (np.complex64, np.complex128)
        for operation in ("gauge_action", "gauge_force", "wilson_dslash")
    ]
    for factory in factories:
        case = None
        try:
            case, preparation = measure_synchronized(factory)
            record["cases"].append(_measure(case, preparation, repeats))
        finally:
            if case is not None and case.provider is not None:
                release_cuda_component(case.provider)
    if cutensor:
        for size in (8, 32):
            try:
                record["cases"].append(_cutensor(size, repeats))
            finally:
                release_cuda_component("cutensor")
    record["passed"] = True


def main(argv: Sequence[str] | None = None) -> int:
    from tools._accelerator_process import read_json_evidence, run, write_json_evidence
    from tools.qualify_cuda_wheels import (
        clean_environment,
        copy_harness,
        installed_command,
        validate_work_path,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--cutensor", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        not 1 <= args.repeats <= 20
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("repeats must be 1..20 and timeout finite and positive")
    root = args.work_root.absolute()
    if sys.platform == "win32":
        root = validate_work_path(root, Path.home())
    args.python = validate_work_path(args.python, root)
    args.output = validate_work_path(args.output, root)
    record: dict[str, Any] = {
        "passed": False,
        "cases": [],
        "errors": [],
        "scope": "Windows single-GPU scientific timing, not peer/collective qualification",
    }
    record["requested_environment"] = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("JAX_", "XLA_", "CUDA_", "PHYDRAX_"))
    }
    try:
        if args.worker:
            if (
                not sys.flags.isolated
                or Path(sys.executable).resolve() != args.python.resolve()
            ):
                raise ValueError(
                    "Worker requires the exact isolated installed interpreter"
                )
            campaign(args.repeats, args.cutensor, record)
        else:
            root.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="windows-cuda-", dir=root))
            harness = directory / "harness"
            record["harness_sha256"] = copy_harness(harness)
            output = directory / "worker.json"
            environment = clean_environment(library_path=None, work_root=root)
            environment.update(
                JAX_PLATFORMS="cuda",
                JAX_ENABLE_X64="true",
                JAX_ENABLE_COMPILATION_CACHE="false",
            )
            command = installed_command(
                args.python,
                harness,
                "benchmarks.windows_cuda",
                [
                    "--worker",
                    "--python",
                    str(args.python),
                    "--work-root",
                    str(root),
                    "--output",
                    str(output),
                    "--repeats",
                    str(args.repeats),
                    *(["--cutensor"] if args.cutensor else []),
                ],
            )
            record["command"] = run(
                command, cwd=harness, environment=environment, timeout=args.timeout
            )
            result = read_json_evidence(output)
            record["result"] = result
            record["passed"] = (
                record["command"]["returncode"] == 0
                and not record["command"]["timed_out"]
                and result.get("passed") is True
                and not result.get("errors")
            )
            if not record["passed"]:
                record["errors"].append(
                    "Installed campaign failed, timed out, or lacked authoritative success"
                )
    except Exception as error:
        record["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if args.worker:
            from phydrax._cuda_native import release_cuda_component

            for provider in (
                "cudss",
                "amgx",
                "quda",
                *(("cutensor",) if args.cutensor else ()),
            ):
                try:
                    release_cuda_component(provider)
                except Exception as error:
                    record["passed"] = False
                    record["errors"].append(
                        f"{provider} teardown: {type(error).__name__}: {error}"
                    )
        write_json_evidence(args.output, record)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
