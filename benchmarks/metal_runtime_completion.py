"""Phase-separated XOR capacity and optimized-program inspection evidence.

Run unchanged through an installed wheel. Every case checks its executed output
against an independent NumPy oracle on the identical host inputs: integer cases
must agree bit for bit, and the CHLO ``sinh`` case must stay within a declared
float32 bound of a host float64 evaluation. Inspection queries are host-only and
are timed separately before and after execution. This is correctness and phase
evidence for fixed capacities, not a speedup claim.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from numpy.typing import NDArray

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    DurationDistribution,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax import config


_ROOT = Path(__file__).resolve().parents[1]
_XOR_DTYPES = ("bool", "int8", "uint64")
_XOR_LAYOUTS = {
    "long-axis-1024": (1024,),
    "long-axis-65536": (65_536,),
    "long-axis-1048577": (1_048_577,),
    "many-outputs-17x4096": (17, 4096),
    "many-outputs-17x65536": (17, 65_536),
}
_CHAIN_LENGTHS = (16, 256, 2048)
_SINH_RELATIVE_TOLERANCE = 1e-5
_SINH_ABSOLUTE_TOLERANCE = 1e-7


@dataclass(frozen=True)
class Case:
    name: str
    capacity: int
    function: Callable[..., Array]
    inputs: tuple[NDArray[Any], ...]
    reference: NDArray[Any]
    exact: bool


def _xor_reduce(values: Array, initial: Array) -> Array:
    return lax.reduce(values, initial, lax.bitwise_xor, (0,))


def _random_words(shape: tuple[int, ...], dtype: str, seed: int) -> NDArray[Any]:
    rng = np.random.default_rng(seed)
    if dtype == "bool":
        return rng.integers(0, 2, size=shape, dtype=np.uint8).astype(np.bool_)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)


def _chain_constants(length: int) -> NDArray[np.uint64]:
    rng = np.random.default_rng(length)
    return rng.integers(0, np.iinfo(np.uint64).max, size=(length, 2), dtype=np.uint64)


def _chain_program(constants: NDArray[np.uint64]) -> Callable[[Array], Array]:
    def program(values: Array) -> Array:
        for mask, offset in constants:
            values = (values ^ jnp.uint64(mask)) + jnp.uint64(offset)
        return values

    return program


def _chain_reference(
    values: NDArray[np.uint64], constants: NDArray[np.uint64]
) -> NDArray[np.uint64]:
    # Fixed-width NumPy uint64 arithmetic wraps modulo 2**64, as StableHLO does.
    result = values.copy()
    for mask, offset in constants:
        result = (result ^ mask) + offset
    return result


def _sinh(values: Array) -> Array:
    return jnp.sinh(values)


def _cases() -> list[Case]:
    cases = []
    for seed, (layout, shape) in enumerate(_XOR_LAYOUTS.items()):
        for dtype in _XOR_DTYPES:
            values = _random_words(shape, dtype, seed)
            initial = np.zeros((), dtype)
            cases.append(
                Case(
                    f"xor-reduce-{layout}-{dtype}",
                    int(np.prod(shape)),
                    _xor_reduce,
                    (values, initial),
                    np.bitwise_xor.reduce(values, axis=0, dtype=values.dtype),
                    True,
                )
            )
    values = _random_words((256,), "uint64", 7)
    for length in _CHAIN_LENGTHS:
        constants = _chain_constants(length)
        cases.append(
            Case(
                f"inspection-uint64-chain-{length}",
                length,
                _chain_program(constants),
                (values,),
                _chain_reference(values, constants),
                True,
            )
        )
    arguments = np.linspace(-4.0, 4.0, 257, dtype=np.float32)
    cases.append(
        Case(
            "inspection-chlo-sinh",
            arguments.size,
            _sinh,
            (arguments,),
            np.sinh(arguments.astype(np.float64)),
            False,
        )
    )
    return cases


def _host_samples(
    operation: Callable[[], Any], repeats: int
) -> tuple[Any, DurationDistribution]:
    result = None
    samples = []
    for _ in range(repeats):
        result, elapsed = measure_host(operation)
        samples.append(elapsed)
    return result, DurationDistribution(tuple(samples))


def _accuracy(case: Case, host: NDArray[Any]) -> dict[str, Any]:
    if case.exact:
        mismatches = int(np.count_nonzero(host != case.reference))
        return {
            "passed": host.dtype == case.reference.dtype
            and host.shape == case.reference.shape
            and mismatches == 0,
            "mismatched_elements": mismatches,
            "reference": "independent fixed-width NumPy integer oracle on identical inputs",
        }
    error = np.abs(host.astype(np.float64) - case.reference)
    bound = _SINH_ABSOLUTE_TOLERANCE + _SINH_RELATIVE_TOLERANCE * np.abs(case.reference)
    return {
        "passed": bool(np.all(np.isfinite(host)) and np.all(error <= bound)),
        "maximum_absolute_error": float(np.max(error, initial=0.0)),
        "maximum_error_over_bound": float(np.max(error / bound, initial=0.0)),
        "relative_tolerance": _SINH_RELATIVE_TOLERANCE,
        "absolute_tolerance": _SINH_ABSOLUTE_TOLERANCE,
        "reference": "host float64 evaluation of the identical float32 inputs",
    }


def _measure(case: Case, repeats: int) -> dict[str, Any]:
    inputs, transfer = measure_synchronized(
        lambda: tuple(jax.device_put(value) for value in case.inputs)
    )
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(case.function).lower(*inputs), lambda lowered: lowered.compile()
    )
    # Inspection happens before any execution, then is repeated and checked
    # again after execution; it must never change or execute the program.
    text_before, first_text = measure_host(compiled.as_text)
    if text_before is None:
        raise RuntimeError("Selected backend does not expose optimized-program text.")
    _, repeated_text = _host_samples(compiled.as_text, repeats)
    executable = compiled.runtime_executable()
    if executable is None:
        raise RuntimeError("Selected backend does not expose a runtime executable.")
    modules, first_modules = measure_host(executable.hlo_modules)
    _, repeated_modules = _host_samples(executable.hlo_modules, repeats)
    first, first_seconds = measure_synchronized(lambda: compiled(*inputs))
    output, warm = measure_repeated(lambda: compiled(*inputs), warmup=1, repeats=repeats)
    host, readback = measure_host(lambda: np.asarray(output))
    text_after = compiled.as_text()
    devices = output.devices()
    if len(devices) != 1 or next(iter(devices)).platform != jax.default_backend():
        raise RuntimeError("Benchmark output escaped the selected backend.")
    accuracy = _accuracy(case, host)
    digest = hashlib.sha256()
    for value in case.inputs:
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes(order="C"))
    text_bytes = text_before.encode()
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="native-compiled-executable",
        unavailable_reason="Backend did not expose compiler cost or memory analysis.",
    )
    inspection_stable = text_after == text_before and len(modules) == 1
    return {
        "case": case.name,
        "capacity": case.capacity,
        "input_sha256": digest.hexdigest(),
        "dtype": str(output.dtype),
        "shape": list(output.shape),
        "devices": sorted(str(device) for device in devices),
        "passed": bool(accuracy["passed"]) and inspection_stable,
        "accuracy": accuracy,
        "inspection": {
            "text_bytes": len(text_bytes),
            "text_sha256": hashlib.sha256(text_bytes).hexdigest(),
            "identical_before_and_after_execution": text_after == text_before,
            "hlo_module_count": len(modules),
            "note": (
                "Compilation already queries the optimized program for layouts; "
                "these timings are the frontend text/HLO conversions afterwards."
            ),
        },
        "timing": {
            "input_transfer_and_sync_seconds": transfer,
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
            "first_as_text_seconds": first_text,
            "repeated_as_text": repeated_text.to_seconds_dict(),
            "first_hlo_modules_seconds": first_modules,
            "repeated_hlo_modules": repeated_modules.to_seconds_dict(),
            "first_execution_and_sync_seconds": first_seconds,
            "warm_synchronized": warm.to_seconds_dict(),
            "output_readback_seconds": readback,
        },
        "compiler": asdict(evidence),
        "logical_bytes": {
            "inputs": logical_array_bytes(inputs),
            "outputs": logical_array_bytes(output),
            "retained": logical_array_bytes(
                (inputs, first, output, case.inputs, case.reference, host)
            ),
        },
    }


def run(repeats: int) -> dict[str, Any]:
    if repeats < 1 or repeats > 100:
        raise ValueError("repeats must be between 1 and 100.")
    if not config.x64_enabled():
        raise ValueError("Exact 64-bit integer cases require x64 enabled.")
    cases, preparation = measure_host(_cases)
    records = [_measure(case, repeats) for case in cases]
    return {
        "identity": capture_benchmark_identity(
            _ROOT, Path(__file__), records[0].keys()
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "host_input_and_reference_preparation_seconds": preparation,
        "scope": (
            "Fixed bounded capacities on the selected backend; no CPU numerical "
            "fallback and no speedup claim. Unknown compiler memory remains unknown."
        ),
        "records": records,
        "passed": all(record["passed"] for record in records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.repeats)
    write_json_atomic(arguments.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
