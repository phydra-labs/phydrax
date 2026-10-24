"""Optimized-program inspection on the real, explicitly selected MPS device.

``PJRT_Executable_OptimizedProgram`` returns the finalized module the plugin
executes as MLIR text; XLA's ``PjRtCApiExecutable::GetHloModule`` converts it to
the ``HloModule`` behind ``hlo_modules()`` and ``Compiled.as_text()``. JAX runs
that conversion while compiling (parameter/output layouts), so every program
the frontend actually emits must convert.

The frontend input is read as an MLIR module and the report as HLO
instructions: the report is the inlined module the device walks, CHLO
composites and single-device Shardy constraints convert, admitted ``mhlo.*``
custom calls are published as ``mps.mhlo.*`` (a spelling the plugin admits and
executes identically, while unknown targets in either namespace stay refused),
a query is identical before and after execution, never runs a host callback,
and is safe under concurrent queries and after JAX drops its stage objects.
The C-level two-call size protocol is exercised by XLA's own client on every
query.
Run this device file with METAL_STARTUP_ENVIRONMENT (mps only, x64 enabled).
"""

from __future__ import annotations

import gc
import math
import re
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax.experimental import io_callback
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from phydrax._jax_config import default_device
from tests._support.metal_qualification import metal_device, record_metal_execution


pytestmark = pytest.mark.metal_native

type RecordProperty = Callable[[str, object], None]

_HOST_CALLBACK_TARGET = "phydrax_metal.host_callback"
_CUSTOM_CALL_TARGET = re.compile(r'custom_call_target="([^"]*)"')

_UNARY_INPUT = np.linspace(-0.9, 0.9, 16, dtype=np.float32)
# Distinct values, so the top three and their indices are unambiguous.
_TOPK_INPUT = np.asarray([3.0, -1.0, 7.5, 0.0, 2.0, 9.0, -4.0, 5.0], np.float32)
_ADMITTED_MHLO_TARGETS = ("mhlo.sinh", "mhlo.erf_inv", "mhlo.topk")
_UNKNOWN_TARGETS = ("mhlo.not_a_target", "mps.mhlo.not_a_target", "mps.mhlo.eigh")


@pytest.fixture
def device() -> Iterator[jax.Device]:
    selected = metal_device()
    with default_device(selected):
        yield selected


def _put(value: np.ndarray, device: jax.Device) -> jax.Array:
    result = jax.device_put(value, device)
    assert result.dtype == value.dtype
    return result


def _input_operations(lowered: jax.stages.Lowered) -> list[tuple[str, str | None]]:
    """Name and custom-call target of every operation JAX lowered, regions included."""
    module = lowered.compiler_ir("stablehlo")
    assert module is not None
    found: list[tuple[str, str | None]] = []

    def visit(operations: Any) -> None:
        for view in operations:
            operation = view.operation
            target = None
            if operation.name == "stablehlo.custom_call":
                target = ir.StringAttr(operation.attributes["call_target_name"]).value
            found.append((operation.name, target))
            for region in operation.regions:
                for block in region.blocks:
                    visit(block.operations)

    with module.context:
        visit(module.body.operations)
    return found


@dataclass(frozen=True)
class _Instruction:
    opcode: str
    target: str | None
    composite: str | None
    replicated: bool


@dataclass(frozen=True)
class _Report:
    text: str
    computations: int
    instructions: tuple[_Instruction, ...]

    def opcodes(self) -> Counter[str]:
        return Counter(instruction.opcode for instruction in self.instructions)

    def custom_call_targets(self) -> Counter[str | None]:
        return Counter(
            instruction.target
            for instruction in self.instructions
            if instruction.opcode == "kCustomCall"
        )

    def composites(self) -> Counter[str | None]:
        return Counter(
            instruction.composite
            for instruction in self.instructions
            if instruction.composite is not None
        )


def _report(module: Any) -> _Report:
    computations = module.computations()
    instructions = []
    for computation in computations:
        for instruction in computation.instructions():
            opcode = instruction.opcode.name
            text = instruction.to_string()
            target = None
            if opcode == "kCustomCall":
                match = _CUSTOM_CALL_TARGET.search(text)
                assert match is not None, text
                target = match.group(1)
            composite = (
                instruction.get_frontend_attribute("composite.name")
                if opcode == "kCall"
                else None
            )
            instructions.append(
                _Instruction(opcode, target, composite, "sharding={replicated}" in text)
            )
    return _Report(module.to_string(), len(computations), tuple(instructions))


def _optimized_program(executable: Any) -> _Report:
    """The report behind ``hlo_modules()``: one module, converted by XLA."""
    modules = executable.hlo_modules()
    assert len(modules) == 1, modules
    return _report(modules[0])


def test_optimized_program_is_the_inlined_executed_module(
    device: jax.Device, record_property: RecordProperty
) -> None:
    inner = jax.jit(lambda value: jnp.tanh(value) * np.float32(2.0))

    def program(x: jax.Array) -> jax.Array:
        return inner(x) + inner(x + np.float32(1.0))

    host = np.linspace(-2.0, 2.0, 16, dtype=np.float32)
    x = _put(host, device)
    lowered = jax.jit(program).lower(x)
    submitted = Counter(name for name, _ in _input_operations(lowered))
    assert submitted["func.call"] == 2, submitted
    assert submitted["stablehlo.tanh"] == 1, submitted

    compiled = lowered.compile()
    reported = _optimized_program(compiled.runtime_executable())
    opcodes = reported.opcodes()
    # Both calls were inlined and the dead callee removed: one computation.
    assert reported.computations == 1, reported.text
    assert opcodes["kCall"] == 0, reported.text
    assert opcodes["kTanh"] == 2, reported.text
    assert opcodes["kParameter"] == 1, reported.text
    assert opcodes["kCustomCall"] == 0, reported.text

    # The public text route parses to the same program.
    public = _report(xla_client.hlo.hlo_module_from_text(compiled.as_text()))
    assert public.computations == reported.computations
    assert public.opcodes() == opcodes

    result = compiled(x)
    two = np.float32(2.0)
    expected = np.tanh(host) * two + np.tanh(host + np.float32(1.0)) * two
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5, atol=1e-6)
    record_metal_execution(record_property, result)


def test_optimized_program_converts_active_chlo_and_shardy_modules(
    device: jax.Device, record_property: RecordProperty
) -> None:
    mesh = Mesh(np.asarray([device]), ("x",))
    constrained = NamedSharding(mesh, PartitionSpec("x"))

    def program(x: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        y = jax.lax.with_sharding_constraint(jnp.sinh(x), constrained)
        return y, jax.random.normal(key, (16,), jnp.float32)

    host = np.linspace(-2.0, 2.0, 16, dtype=np.float32)
    x = _put(host, device)
    key = jax.random.key(7)
    lowered = jax.jit(program).lower(x, key)
    operations = _input_operations(lowered)
    submitted = Counter(name for name, _ in operations)
    # The active frontend emits CHLO and Shardy, never mhlo.* custom calls.
    assert submitted["chlo.sinh"] == 1, submitted
    assert submitted["chlo.erf_inv"] >= 1, submitted
    constraints = submitted["sdy.sharding_constraint"]
    assert constraints >= 1, submitted
    assert submitted["sdy.mesh"] >= 1, submitted
    assert not [target for _, target in operations if target is not None], operations

    compiled = lowered.compile()
    reported = _optimized_program(compiled.runtime_executable())
    # chlo.sinh executes as its composite; erf_inv reaches the plugin as
    # StableHLO arithmetic. Each single-device constraint is the identity.
    assert reported.composites() == Counter({"chlo.sinh": 1}), reported.text
    targets = reported.custom_call_targets()
    assert targets == Counter({"Sharding": constraints}), reported.text
    sharding = [i for i in reported.instructions if i.target == "Sharding"]
    assert all(instruction.replicated for instruction in sharding), reported.text

    y, samples = compiled(x, key)
    np.testing.assert_allclose(np.asarray(y), np.sinh(host), rtol=1e-5, atol=1e-6)
    drawn = np.asarray(samples)
    assert drawn.dtype == np.float32 and drawn.shape == (16,)
    assert np.isfinite(drawn).all()
    np.testing.assert_array_equal(np.asarray(compiled(x, key)[1]), drawn)
    record_metal_execution(record_property, y, samples)


def test_optimized_program_is_identical_before_and_after_execution(
    device: jax.Device, record_property: RecordProperty
) -> None:
    host = np.linspace(-3.0, 3.0, 32, dtype=np.float32)
    x = _put(host, device)
    program = jax.jit(lambda v: jnp.tanh(v) * np.float32(3.0) + np.float32(1.0))
    compiled = program.lower(x).compile()
    executable = compiled.runtime_executable()
    assert executable is not None
    fingerprint = executable.fingerprint
    # The executable fingerprint stays the SHA-256 of the submitted artifact.
    assert re.fullmatch(rb"[0-9a-f]{64}", fingerprint), fingerprint

    before = _optimized_program(executable)
    first = jax.block_until_ready(compiled(x))
    between = _optimized_program(executable)
    second = jax.block_until_ready(compiled(x))
    after = _optimized_program(executable)
    assert before == between == after
    assert executable.fingerprint == fingerprint

    expected = np.tanh(host) * np.float32(3.0) + np.float32(1.0)
    np.testing.assert_allclose(np.asarray(first), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(second), np.asarray(first))
    record_metal_execution(record_property, first, second)


def test_optimized_program_query_never_runs_host_callbacks(
    device: jax.Device, record_property: RecordProperty
) -> None:
    calls: list[str] = []
    signature = jax.ShapeDtypeStruct((8,), np.float32)

    def record(value: np.ndarray) -> np.ndarray:
        calls.append("io")
        return np.asarray(value, np.float32) + np.float32(1.0)

    def double(value: np.ndarray) -> np.ndarray:
        calls.append("pure")
        return np.asarray(value, np.float32) * np.float32(2.0)

    def program(x: jax.Array) -> jax.Array:
        echoed = io_callback(record, signature, x, ordered=True)
        return jax.pure_callback(double, signature, echoed)

    host = np.arange(8, dtype=np.float32)
    x = _put(host, device)
    lowered = jax.jit(program).lower(x)
    targets = Counter(target for _, target in _input_operations(lowered) if target)
    assert targets == Counter({_HOST_CALLBACK_TARGET: 2}), targets

    # Compilation itself already converted the program (JAX reads layouts).
    compiled = lowered.compile()
    executable = compiled.runtime_executable()
    reports = [_optimized_program(executable) for _ in range(3)]
    assert calls == []
    assert reports[1:] == reports[:1] * 2
    assert reports[0].custom_call_targets() == Counter({_HOST_CALLBACK_TARGET: 2})

    result = jax.block_until_ready(compiled(x))
    assert calls == ["io", "pure"]
    assert _optimized_program(executable) == reports[0]
    assert calls == ["io", "pure"]
    expected = (host + np.float32(1.0)) * np.float32(2.0)
    np.testing.assert_array_equal(np.asarray(result), expected)
    record_metal_execution(record_property, result)


def test_optimized_program_concurrent_queries_and_executable_lifetime(
    device: jax.Device, record_property: RecordProperty
) -> None:
    host = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    x = _put(host, device)
    program = jax.jit(lambda v: jnp.sin(v) * np.float32(2.0) - v)
    compiled = program.lower(x).compile()
    executable = compiled.runtime_executable()
    reference = _optimized_program(executable)

    with ThreadPoolExecutor(max_workers=8) as pool:
        queries = [pool.submit(_optimized_program, executable) for _ in range(32)]
        results = [compiled(x) for _ in range(16)]
        reports = [query.result(timeout=600) for query in queries]
    assert all(report == reference for report in reports)
    expected = np.sin(host) * np.float32(2.0) - host
    for result in results:
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5, atol=1e-6)

    # The cached program lives with the PJRT executable, not with JAX's stage
    # objects or caches.
    del compiled, program
    jax.clear_caches()
    gc.collect()
    assert _optimized_program(executable) == reference
    record_metal_execution(record_property, results[0], results[-1])


def _custom_call(target: str, x: jax.Array) -> tuple[jax.Array, ...]:
    """Call ``target`` explicitly, as a jax.ffi consumer of the plugin would."""
    if target.endswith("mhlo.topk"):
        results = (
            jax.ShapeDtypeStruct((3,), np.float32),
            jax.ShapeDtypeStruct((3,), np.int32),
        )
        return tuple(jax.ffi.ffi_call(target, results)(x))
    return (jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(x.shape, x.dtype))(x),)


def _check_admitted(
    target: str, host: np.ndarray, results: tuple[jax.Array, ...]
) -> None:
    if target == "mhlo.topk":
        values, indices = (np.asarray(result) for result in results)
        assert indices.dtype == np.int32
        np.testing.assert_array_equal(values, np.asarray([9.0, 7.5, 5.0], np.float32))
        np.testing.assert_array_equal(indices, np.asarray([5, 2, 7], np.int32))
        return
    (result,) = (np.asarray(value) for value in results)
    assert result.dtype == np.float32
    if target == "mhlo.sinh":
        np.testing.assert_allclose(result, np.sinh(host), rtol=1e-5, atol=1e-6)
        return
    # Host reference for the inverse: erf of the device result recovers x.
    recovered = np.asarray([math.erf(float(value)) for value in result])
    np.testing.assert_allclose(recovered, host, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("target", _ADMITTED_MHLO_TARGETS)
def test_optimized_program_publishes_admitted_mhlo_targets(
    device: jax.Device, record_property: RecordProperty, target: str
) -> None:
    published = "mps." + target
    host = _TOPK_INPUT if target == "mhlo.topk" else _UNARY_INPUT
    x = _put(host, device)
    lowered = jax.jit(lambda value: _custom_call(target, value)).lower(x)
    submitted = Counter(call for _, call in _input_operations(lowered) if call)
    assert submitted == Counter({target: 1}), submitted

    # The reserved mhlo.* namespace is published as mps.mhlo.*, so XLA's HLO
    # conversion (run by JAX while compiling and by hlo_modules()) succeeds.
    compiled = lowered.compile()
    reported = _optimized_program(compiled.runtime_executable())
    assert reported.custom_call_targets() == Counter({published: 1}), reported.text
    results = tuple(compiled(x))
    _check_admitted(target, host, results)

    # The published spelling is admitted, dispatched like mhlo.X, and publishes
    # the same program.
    republished = jax.jit(lambda value: _custom_call(published, value)).lower(x).compile()
    again_reported = _optimized_program(republished.runtime_executable())
    assert again_reported.computations == reported.computations
    assert again_reported.instructions == reported.instructions
    again = tuple(republished(x))
    for first, second in zip(results, again, strict=True):
        np.testing.assert_array_equal(np.asarray(second), np.asarray(first))
    record_metal_execution(record_property, *results, *again)


@pytest.mark.parametrize("target", _UNKNOWN_TARGETS)
def test_optimized_program_namespace_never_admits_unknown_targets(
    device: jax.Device, record_property: RecordProperty, target: str
) -> None:
    x = _put(_UNARY_INPUT, device)
    refusal = (
        rf"(?s)(?=.*phydrax-metal)"
        rf"(?=.*custom_call target '{re.escape(target)}')(?=.*not supported)"
    )
    with pytest.raises(jax.errors.JaxRuntimeError, match=refusal):
        jax.jit(lambda value: _custom_call(target, value)).lower(x).compile()
    # An admitted target in the published namespace still executes.
    (result,) = jax.jit(lambda value: _custom_call("mps.mhlo.sinh", value))(x)
    _check_admitted("mhlo.sinh", _UNARY_INPUT, (result,))
    record_metal_execution(record_property, result)
