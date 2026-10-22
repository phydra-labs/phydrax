#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compiler/effect tests; CPU observers do not qualify a vendor or MPI runtime."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, ShapeDtypeStruct
from jax._src import dispatch as jax_dispatch
from jax._src.interpreters import mlir as _mlir
from jax.core import AbstractValue, ShapedArray
from jax.extend import core
from jax.interpreters import ad, mlir
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import phydrax._execution_collective_ffi as collective
import phydrax._execution_collectives as collectives
from phydrax import config as pxconfig
from phydrax._execution_collectives import (
    CollectiveBinding,
    CollectiveLease,
    CollectiveLeaseSpec,
    CollectiveMPIProvenance,
)
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup


@pytest.fixture(autouse=True)
def _cpu_default() -> Iterator[None]:
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        yield


@pytest.fixture
def admitted_spec(monkeypatch: pytest.MonkeyPatch) -> CollectiveLeaseSpec:
    # This fixture isolates compilation from communicator setup. Actual live,
    # closed, failed and host-thread admission belongs to the lease tests.
    device = jax.local_devices(backend="cpu")[0]
    spec = CollectiveLeaseSpec(
        group_id="collective-effect-test",
        provider_id="effect-observer",
        configuration_id="configuration-a",
        preparation_id="preparation-a",
        process_indices=(device.process_index,),
        device_keys=((device.process_index, device.id),),
        mpi_build_identity="compiler-test-no-mpi",
    )

    def admit(lease_key: CollectiveLeaseSpec) -> None:
        if lease_key.group_id not in (spec.group_id, "other-effect-group"):
            raise ValueError("unexpected test group")

    monkeypatch.setattr(collective, "require_collective_lease", admit)
    return spec


@pytest.fixture
def cpu_effect_observer(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Execute the real ordered callback custom call, without simulating MPI."""
    observed: list[str] = []

    def observer_lowering(
        ctx: mlir.LoweringRuleContext,
        *operands: mlir.ir.Value,
        target_name: str,
        lease_key: CollectiveLeaseSpec,
        **params: Any,
    ) -> Sequence[mlir.ir.Value]:
        del params
        effect = collective._CollectiveEffect(lease_key.group_id)

        def shaped_arrays(avals: Sequence[AbstractValue]) -> tuple[ShapedArray, ...]:
            arrays: list[ShapedArray] = []
            for aval in avals:
                if not isinstance(aval, ShapedArray):
                    raise TypeError(
                        "the CPU effect observer accepts only array abstract values"
                    )
                arrays.append(aval)
            return tuple(arrays)

        input_avals = shaped_arrays(ctx.avals_in)
        output_avals = shaped_arrays(ctx.avals_out)

        def observe(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
            observed.append(target_name)
            # Numeric-result tests transfer the input unchanged; they qualify
            # token execution and result liveness, not a collective operation.
            return tuple(arrays[-1] for _ in output_avals)

        outputs, token, _ = mlir.emit_python_callback(
            ctx,
            observe,
            ctx.tokens_in.get(effect),
            list(operands),
            input_avals,
            output_avals,
            has_side_effect=True,
            returns_token=True,
        )
        if not isinstance(token, mlir.ir.Value):
            raise TypeError("the CPU effect observer did not return an IR token")
        flat_outputs: list[mlir.ir.Value] = []
        for output in outputs:
            if not isinstance(output, mlir.ir.Value):
                raise TypeError("the CPU effect observer returned a non-array IR result")
            flat_outputs.append(output)
        ctx.set_tokens_out(mlir.TokenSet({effect: token}))
        return flat_outputs

    monkeypatch.setitem(
        mlir._platform_specific_lowerings["cpu"],
        collective._collective_p,
        _mlir.LoweringRuleEntry(observer_lowering, inline=True),
    )
    return observed


def _call(
    spec: CollectiveLeaseSpec,
    lease_id: Array,
    ticket: Array,
    data: Array,
    *,
    target: str = "collective_effect_observer",
) -> Array:
    return collective.collective_ffi_call(
        target,
        (ShapeDtypeStruct(data.shape, data.dtype),),
        (lease_id, ticket, data),
        lease_key=spec,
        attributes={"operation_id": "observe", "operation": np.int64(0)},
    )[0]


def _arguments() -> tuple[ShapeDtypeStruct, ShapeDtypeStruct, ShapeDtypeStruct]:
    return (
        ShapeDtypeStruct((), np.int64),
        ShapeDtypeStruct((), np.int64),
        ShapeDtypeStruct((), np.float64),
    )


def test_group_identity_orders_different_provider_configurations(
    admitted_spec: CollectiveLeaseSpec,
) -> None:
    other_provider = replace(
        admitted_spec, provider_id="another-provider", configuration_id="configuration-b"
    )
    other_group = replace(admitted_spec, group_id="other-effect-group")

    def shared_group(lease_id: Array, ticket: Array, data: Array) -> Array:
        _call(admitted_spec, lease_id, ticket, data)
        return _call(other_provider, lease_id, ticket, data)

    def independent_groups(lease_id: Array, ticket: Array, data: Array) -> Array:
        _call(admitted_spec, lease_id, ticket, data)
        return _call(other_group, lease_id, ticket, data)

    shared = jax.make_jaxpr(shared_group)(*_arguments())
    independent = jax.make_jaxpr(independent_groups)(*_arguments())
    assert len(shared.effects) == 1
    assert len(independent.effects) == 2
    assert shared.effects < independent.effects


def test_unused_collective_outputs_still_execute_in_group_order(
    admitted_spec: CollectiveLeaseSpec, cpu_effect_observer: list[str]
) -> None:
    other_config = replace(admitted_spec, configuration_id="configuration-b")

    def invoke(lease_id: Array, ticket: Array, data: Array) -> Array:
        _call(admitted_spec, lease_id, ticket, data, target="collective_first")
        _call(other_config, lease_id, ticket, data, target="collective_second")
        return data

    device = jax.local_devices(backend="cpu")[0]
    arguments = (
        jax.device_put(np.int64(1), device),
        jax.device_put(np.int64(1), device),
        jax.device_put(np.float64(7), device),
    )
    executable = (
        jax.jit(invoke, in_shardings=tuple(value.sharding for value in arguments))
        .lower(*arguments)
        .compile()
    )
    result = executable(*arguments)
    jax.block_until_ready(result)
    jax.effects_barrier()
    np.testing.assert_array_equal(np.asarray(result), np.asarray(7, dtype=np.float64))
    assert cpu_effect_observer == ["collective_first", "collective_second"]


@pytest.mark.parametrize("metadata_index", (0, 1))
@pytest.mark.parametrize(
    "invalid_spec",
    (ShapeDtypeStruct((1,), np.int64), ShapeDtypeStruct((), np.int32)),
)
def test_invocation_metadata_requires_dynamic_scalar_int64(
    admitted_spec: CollectiveLeaseSpec,
    metadata_index: int,
    invalid_spec: ShapeDtypeStruct,
) -> None:
    arguments = list(_arguments())
    arguments[metadata_index] = invalid_spec

    def invoke(lease_id: Array, ticket: Array, data: Array) -> Array:
        return _call(admitted_spec, lease_id, ticket, data)

    with pytest.raises(ValueError, match="scalar int64"):
        jax.make_jaxpr(invoke)(*arguments)


def test_caller_cannot_override_collective_admission_attributes(
    admitted_spec: CollectiveLeaseSpec,
) -> None:
    def invoke(lease_id: Array, ticket: Array, data: Array) -> tuple[Array, ...]:
        return collective.collective_ffi_call(
            "collective_effect_observer",
            (),
            (lease_id, ticket, data),
            lease_key=admitted_spec,
            attributes={"collective_call_id": "forged"},
        )

    with pytest.raises(ValueError, match="reserved"):
        jax.make_jaxpr(invoke)(*_arguments())


@pytest.mark.parametrize("mapped_data", (False, True))
def test_vmap_refuses_even_unmapped_collectives(
    admitted_spec: CollectiveLeaseSpec, mapped_data: bool
) -> None:
    def invoke(lease_id: Array, ticket: Array, data: Array) -> Array:
        def lane(value: Array) -> Array:
            return _call(admitted_spec, lease_id, ticket, value if mapped_data else data)

        return jax.vmap(lane)(jnp.ones((2,), dtype=np.float64))

    with pytest.raises(ValueError, match="does not support vmap"):
        jax.make_jaxpr(invoke)(*_arguments())


@pytest.mark.parametrize("transform", (jax.grad, jax.jacfwd, jax.jacrev))
def test_raw_collective_autodiff_is_refused(
    admitted_spec: CollectiveLeaseSpec, transform: Callable[..., Any]
) -> None:
    def invoke(lease_id: Array, ticket: Array, data: Array) -> Array:
        return transform(lambda value: _call(admitted_spec, lease_id, ticket, value))(
            data
        )

    with pytest.raises(ValueError, match="cannot be differentiated"):
        jax.make_jaxpr(invoke)(*_arguments())


def test_raw_collective_transpose_is_refused(
    admitted_spec: CollectiveLeaseSpec,
) -> None:
    def invoke(lease_id: Array, ticket: Array, data: Array) -> tuple[Array, ...]:
        return jax.linear_transpose(
            lambda value: _call(admitted_spec, lease_id, ticket, value), data
        )(data)

    with pytest.raises(ValueError, match="cannot be differentiated"):
        jax.make_jaxpr(invoke)(*_arguments())


def test_no_cpu_numerical_fallback_is_registered(
    admitted_spec: CollectiveLeaseSpec,
) -> None:
    def invoke(lease_id: Array, ticket: Array, data: Array) -> Array:
        return _call(admitted_spec, lease_id, ticket, data)

    with pytest.raises(NotImplementedError, match="phydrax_collective_ffi"):
        jax.jit(invoke).lower(*_arguments())


def _scientific_derivative(spec: CollectiveLeaseSpec) -> Callable[..., Array]:
    # An identity derivative is a compiler-only mathematical owner harness; it
    # does not implement or qualify any numerical collective operation.
    @jax.custom_jvp
    def operation(lease_id: Array, ticket: Array, data: Array) -> Array:
        return _call(spec, lease_id, ticket, data)

    @operation.defjvp
    def derivative(
        primals: tuple[Array, ...], tangents: tuple[Array, ...]
    ) -> tuple[Array, Array]:
        return operation(*primals), tangents[2]

    return operation


def test_scientific_owner_can_supply_derivative_without_raw_ffi_ad(
    admitted_spec: CollectiveLeaseSpec,
) -> None:
    operation = _scientific_derivative(admitted_spec)

    def invoke(lease_id: Array, ticket: Array, data: Array) -> Array:
        return jax.grad(lambda value: operation(lease_id, ticket, value))(data)

    differentiated = jax.make_jaxpr(invoke)(*_arguments())
    assert len(differentiated.effects) == 1


def test_remat_cannot_duplicate_an_owned_collective_derivative(
    admitted_spec: CollectiveLeaseSpec,
) -> None:
    operation = jax.checkpoint(_scientific_derivative(admitted_spec))

    def invoke(lease_id: Array, ticket: Array, data: Array) -> Array:
        return jax.grad(lambda value: operation(lease_id, ticket, value))(data)

    with pytest.raises(NotImplementedError, match="Effects not supported.*checkpoint"):
        jax.make_jaxpr(invoke)(*_arguments())


def test_separate_async_executables_preserve_effects_without_numeric_results(
    admitted_spec: CollectiveLeaseSpec, cpu_effect_observer: list[str]
) -> None:
    observed = cpu_effect_observer

    def first(lease_id: Array, ticket: Array, data: Array) -> tuple[Array, ...]:
        return collective.collective_ffi_call(
            "first",
            (),
            (lease_id, ticket, data),
            lease_key=admitted_spec,
            attributes={"operation_id": "observe"},
        )

    def second(lease_id: Array, ticket: Array, data: Array) -> tuple[Array, ...]:
        return collective.collective_ffi_call(
            "second",
            (),
            (lease_id, ticket, data),
            lease_key=admitted_spec,
            attributes={"operation_id": "observe"},
        )

    device = jax.local_devices(backend="cpu")[0]
    arguments = (
        jax.device_put(np.int64(1), device),
        jax.device_put(np.int64(1), device),
        jax.device_put(np.float64(0), device),
    )
    shardings = tuple(value.sharding for value in arguments)
    executable_first = jax.jit(first, in_shardings=shardings).lower(*arguments).compile()
    executable_second = (
        jax.jit(second, in_shardings=shardings).lower(*arguments).compile()
    )
    # Deliberately no output synchronization between asynchronous dispatches.
    executable_first(*arguments)
    executable_second(*arguments)
    executable_first(*arguments)
    executable_second(*arguments)
    jax.effects_barrier()
    assert observed == ["first", "second", "first", "second"]


def test_shared_call_identity_allows_unequal_local_partitions(
    admitted_spec: CollectiveLeaseSpec,
) -> None:
    def invoke(lease_id: Array, ticket: Array, data: Array) -> Array:
        for local_data, partition, operation_id in (
            (data[:1], 0, "csr-action"),
            (data, 1, "csr-action"),
            (data, 1, "csr-transpose"),
        ):
            collective.collective_ffi_call(
                "collective_csr",
                (ShapeDtypeStruct(local_data.shape, local_data.dtype),),
                (lease_id, ticket, local_data),
                lease_key=admitted_spec,
                attributes={
                    "operation_id": operation_id,
                    "partition_index": np.int64(partition),
                },
            )
        return data

    metadata = ShapeDtypeStruct((), np.int64)
    traced = jax.make_jaxpr(invoke)(
        metadata, metadata, ShapeDtypeStruct((3,), np.float64)
    )
    schedule = collective.collective_schedule(traced, lease_key=admitted_spec)
    assert len(schedule) == 3
    assert schedule[0] == schedule[1]
    assert schedule[1] != schedule[2]


def test_shared_operation_identity_is_required(
    admitted_spec: CollectiveLeaseSpec,
) -> None:
    def invoke(lease_id: Array, ticket: Array, data: Array) -> tuple[Array, ...]:
        return collective.collective_ffi_call(
            "collective_effect_observer",
            (),
            (lease_id, ticket, data),
            lease_key=admitted_spec,
            attributes={},
        )

    with pytest.raises(ValueError, match="requires a shared operation_id"):
        jax.make_jaxpr(invoke)(*_arguments())


def test_nested_jit_schedule_uses_semantics_not_local_shapes(
    admitted_spec: CollectiveLeaseSpec,
) -> None:
    @jax.jit
    def inner(lease_id: Array, ticket: Array, data: Array) -> Array:
        return _call(admitted_spec, lease_id, ticket, data)

    traced = jax.make_jaxpr(inner)(*_arguments())
    expected = collective.collective_call_identity(
        "collective_effect_observer", admitted_spec, "observe"
    )
    assert collective.collective_schedule(traced, lease_key=admitted_spec) == (expected,)


def test_schedule_rejects_captured_foreign_lease(
    admitted_spec: CollectiveLeaseSpec,
) -> None:
    foreign_spec = replace(admitted_spec, preparation_id="different-preparation")

    def invoke(lease_id: Array, ticket: Array, data: Array) -> Array:
        return _call(foreign_spec, lease_id, ticket, data)

    traced = jax.make_jaxpr(invoke)(*_arguments())
    with pytest.raises(ValueError, match="different lease specification"):
        collective.collective_schedule(traced, lease_key=admitted_spec)


def test_transformed_symbolic_zero_activity_changes_preflight_schedule(
    admitted_spec: CollectiveLeaseSpec,
) -> None:
    @jax.custom_jvp
    def scientific(lease_id: Array, ticket: Array, data: Array) -> Array:
        return _call(admitted_spec, lease_id, ticket, data)

    @scientific.defjvp
    def derivative(
        primals: tuple[Array, ...], tangents: tuple[Array, ...]
    ) -> tuple[Array, Array]:
        return scientific(*primals), _call(
            admitted_spec, primals[0], primals[1], tangents[2], target="tangent_observer"
        )

    def active(lease_id: Array, ticket: Array, data: Array) -> tuple[Array, Array]:
        return jax.jvp(
            lambda value: scientific(lease_id, ticket, value),
            (data,),
            (jnp.ones_like(data),),
        )

    def inactive(lease_id: Array, ticket: Array, data: Array) -> tuple[Array, Array]:
        return jax.jvp(
            lambda value: scientific(lease_id, ticket, jax.lax.stop_gradient(value)),
            (data,),
            (jnp.ones_like(data),),
        )

    active_schedule = collective.collective_schedule(
        jax.make_jaxpr(active)(*_arguments()), lease_key=admitted_spec
    )
    inactive_schedule = collective.collective_schedule(
        jax.make_jaxpr(inactive)(*_arguments()), lease_key=admitted_spec
    )
    assert len(active_schedule) == 2
    assert len(inactive_schedule) == 1
    assert active_schedule != inactive_schedule


@pytest.mark.parametrize("control_flow", ("cond", "while", "remat"))
def test_schedule_refuses_unproved_control_flow(
    admitted_spec: CollectiveLeaseSpec, control_flow: str
) -> None:
    def invoke(lease_id: Array, ticket: Array, data: Array) -> Array:
        def operation(value: Array) -> Array:
            return _call(admitted_spec, lease_id, ticket, value)

        if control_flow == "cond":
            return jax.lax.cond(data > 0, operation, lambda value: value, data)
        if control_flow == "while":
            return jax.lax.while_loop(
                lambda value: value < 2, lambda value: operation(value) + 1, data
            )
        return jax.checkpoint(operation)(data)

    traced = jax.make_jaxpr(invoke)(*_arguments())
    with pytest.raises(ValueError, match="fixed collective schedule"):
        collective.collective_schedule(traced, lease_key=admitted_spec)


def test_fixed_scan_expands_its_actual_body_schedule(
    admitted_spec: CollectiveLeaseSpec,
) -> None:
    def invoke(lease_id: Array, ticket: Array, data: Array) -> Array:
        def body(value: Array, unused: None) -> tuple[Array, None]:
            del unused
            return _call(admitted_spec, lease_id, ticket, value), None

        result, _ = jax.lax.scan(body, data, None, length=3)
        return result

    traced = jax.make_jaxpr(invoke)(*_arguments())
    expected = collective.collective_call_identity(
        "collective_effect_observer", admitted_spec, "observe"
    )
    assert (
        collective.collective_schedule(traced, lease_key=admitted_spec) == (expected,) * 3
    )


def test_unregistered_mathematical_collective_requires_owner_schedule(
    admitted_spec: CollectiveLeaseSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    primitive = core.Primitive("collective_schedule_owner_harness")
    primitive.multiple_results = True

    def abstract(*avals: Any, lease_key: CollectiveLeaseSpec) -> Any:
        return collective.collective_effectful_abstract_eval(
            *avals, result_avals=(avals[-1],), lease_key=lease_key
        )

    primitive.def_effectful_abstract_eval(abstract)

    def invoke(lease_id: Array, ticket: Array, data: Array) -> Array:
        return primitive.bind(lease_id, ticket, data, lease_key=admitted_spec)[0]

    traced = jax.make_jaxpr(invoke)(*_arguments())
    with pytest.raises(ValueError, match="collective_schedule_owner_harness"):
        collective.collective_schedule(traced, lease_key=admitted_spec)

    def owner_schedule(
        equation: core.JaxprEqn,
    ) -> tuple[tuple[CollectiveLeaseSpec, str], ...]:
        spec = equation.params["lease_key"]
        return (
            (spec, collective.collective_call_identity("first", spec, "first-phase")),
            (spec, collective.collective_call_identity("second", spec, "second-phase")),
        )

    monkeypatch.setattr(collective, "_schedule_rules", dict(collective._schedule_rules))
    collective.register_collective_schedule_rule(primitive, owner_schedule)
    assert collective.collective_schedule(traced, lease_key=admitted_spec) == (
        collective.collective_call_identity("first", admitted_spec, "first-phase"),
        collective.collective_call_identity("second", admitted_spec, "second-phase"),
    )


class _ProgramBridge(collectives._NativeBridge):
    """Host authorization boundary double; no MPI/numeric operation is provided."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.ticket = 0

    def validate(self, lease_id: int) -> None:
        if lease_id != 71:
            raise ValueError("unexpected security-test lease ID")

    def agree_program(self, lease_id: int, schedule: tuple[str, ...]) -> None:
        self.validate(lease_id)
        self.events.append("agree-program")

    def begin(self, lease_id: int, schedule: tuple[str, ...]) -> int:
        self.validate(lease_id)
        self.events.append("begin")
        self.ticket += 1
        return self.ticket

    def end(self, lease_id: int, ticket: int) -> None:
        self.validate(lease_id)
        if ticket != self.ticket:
            raise ValueError("unexpected security-test invocation ticket")
        self.events.append("end")

    def fail(self, lease_id: int, reason: str) -> None:
        self.validate(lease_id)
        self.events.append(f"fail:{reason}")


class _ProgramSupervisor:
    supervisor_id = "program-security-test"

    def __init__(self) -> None:
        self.failures: list[str] = []

    def require_active(
        self, process_indices: tuple[int, ...], timeout_seconds: float
    ) -> None:
        if process_indices != (jax.process_index(),) or timeout_seconds <= 0:
            raise ValueError("unexpected security-test supervisor membership")

    def fail(self, reason: str) -> None:
        self.failures.append(reason)


@pytest.fixture
def program_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[CollectiveLease, _ProgramBridge]]:
    device = jax.local_devices(backend="cpu")[0]
    processes = (device.process_index,)
    devices = ((device.process_index, device.id),)
    spec = CollectiveLeaseSpec(
        group_id="program-security-group",
        provider_id="program-security-observer",
        configuration_id="program-security-configuration",
        preparation_id="program-security-preparation",
        process_indices=processes,
        device_keys=devices,
        mpi_build_identity="program-security-no-mpi",
    )
    group = ExecutionGroup(
        ExecutionGroupSpec(spec.group_id, processes, devices), (device,)
    )
    bridge = _ProgramBridge()
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        lease = CollectiveLease(
            spec,
            71,
            bridge,
            group,
            object(),
            _ProgramSupervisor(),
            10.0,
            CollectiveMPIProvenance(
                "program-security-no-mpi",
                "/not-a-runtime-library",
                "no-library",
                "no-header",
                "no-MPI-execution",
            ),
        )
        monkeypatch.setattr(collectives, "_LIVE", {spec: lease})
        yield lease, bridge
        jax.effects_barrier()


def test_prepared_program_does_not_reexecute_python_at_invocation(
    program_lease: tuple[CollectiveLease, _ProgramBridge],
) -> None:
    lease, bridge = program_lease
    staged_calls: list[str] = []

    def numerical_program(binding: CollectiveBinding, value: Array) -> Array:
        del binding
        staged_calls.append("staged")
        return value + 1

    value = jnp.asarray(3, dtype=np.float64)
    program = collectives.prepare_collective_program(
        numerical_program, lease.binding, value, lease_key=lease.spec
    )
    assert staged_calls == ["staged"]
    assert bridge.events == ["agree-program"]
    first = lease.invoke(program, lease.binding, value)
    second = lease.invoke(program, lease.binding, value)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(4, dtype=np.float64))
    np.testing.assert_array_equal(np.asarray(second), np.asarray(4, dtype=np.float64))
    assert staged_calls == ["staged"]
    assert bridge.events == ["agree-program", "begin", "end", "begin", "end"]


def test_invoke_rejects_opaque_compiled_wrapper_before_executing_it(
    program_lease: tuple[CollectiveLease, _ProgramBridge],
) -> None:
    lease, bridge = program_lease
    wrapper_calls: list[str] = []
    opaque_calls: list[str] = []

    def observe(value: np.ndarray) -> None:
        del value
        opaque_calls.append("opaque")

    def opaque_program(value: Array) -> Array:
        jax.debug.callback(observe, value, ordered=True)
        return value + 1

    opaque = jax.jit(opaque_program).lower(ShapeDtypeStruct((), np.float64)).compile()

    def wrapper(binding: CollectiveBinding, value: Array) -> Array:
        del binding
        wrapper_calls.append("executed")
        opaque(np.float64(7)).block_until_ready()
        jax.effects_barrier()
        return value

    with pytest.raises(TypeError):
        lease.invoke(wrapper, lease.binding, jnp.asarray(3, dtype=np.float64))  # ty: ignore[invalid-argument-type]
    assert wrapper_calls == []
    assert opaque_calls == []
    assert bridge.events == []


def test_invoke_rejects_raw_compiled_without_authorizing_a_ticket(
    program_lease: tuple[CollectiveLease, _ProgramBridge],
) -> None:
    lease, bridge = program_lease

    def numerical_program(binding: CollectiveBinding, value: Array) -> Array:
        del binding
        return value + 1

    value = jnp.asarray(3, dtype=np.float64)
    opaque = jax.jit(numerical_program).lower(lease.binding, value).compile()
    with pytest.raises(TypeError):
        lease.invoke(opaque, lease.binding, value)  # ty: ignore[invalid-argument-type]
    assert bridge.events == []


def test_explicit_program_staging_refuses_opaque_compiled_array_capture(
    program_lease: tuple[CollectiveLease, _ProgramBridge],
) -> None:
    lease, bridge = program_lease
    staged_calls: list[str] = []
    opaque_calls: list[str] = []

    def observe(value: np.ndarray) -> None:
        del value
        opaque_calls.append("opaque")

    def opaque_program(value: Array) -> Array:
        jax.debug.callback(observe, value, ordered=True)
        return value + 1

    opaque = jax.jit(opaque_program).lower(ShapeDtypeStruct((2,), np.float64)).compile()

    def wrapper(binding: CollectiveBinding, value: Array) -> Array:
        del binding, value
        staged_calls.append("staged")
        result = opaque(np.asarray([7, 8], dtype=np.float64))
        result.block_until_ready()
        jax.effects_barrier()
        return result

    # User code may execute at the explicit staging boundary, but the captured
    # result cannot become a supposedly dynamic collective program.
    with pytest.raises(ValueError, match="numerical constants"):
        collectives.prepare_collective_program(
            wrapper,
            lease.binding,
            jnp.asarray([3, 4], dtype=np.float64),
            lease_key=lease.spec,
        )
    assert staged_calls == ["staged"]
    assert opaque_calls == ["opaque"]
    assert "begin" not in bridge.events
    assert any(event.startswith("fail:") for event in bridge.events)


@pytest.fixture(scope="module")
def native_cpu_token_observer(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[str, ctypes.CDLL]:
    """Build a typed-FFI token observer, not an MPI or numerical provider."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.fail("the native ordered-token regression requires a C++ compiler")
    directory = tmp_path_factory.mktemp("collective_cpu_token")
    source = directory / "observer.cpp"
    library_path = directory / "observer.so"
    source.write_text(
        r"""
#include <array>
#include <cstdint>
#include <mutex>
#include <string_view>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace {
std::mutex mutex;
std::array<double, 64> payloads{};
std::array<int64_t, 64> tickets{};
int64_t count = 0;

ffi::Error observe(ffi::Token, ffi::BufferR0<ffi::S64> lease,
                   ffi::BufferR0<ffi::S64> ticket,
                   ffi::BufferR0<ffi::F64> payload, ffi::Result<ffi::Token>,
                   ffi::ResultBufferR0<ffi::F64> output,
                   std::string_view configuration, std::string_view call) {
  std::lock_guard<std::mutex> lock(mutex);
  if (*lease.typed_data() <= 0 || *ticket.typed_data() <= 0 ||
      configuration.empty() || call.empty())
    return ffi::Error::InvalidArgument("missing token-observer invocation metadata");
  if (count == static_cast<int64_t>(payloads.size()))
    return ffi::Error::InvalidArgument("token-observer record capacity exceeded");
  payloads[count] = *payload.typed_data();
  tickets[count] = *ticket.typed_data();
  ++count;
  *output->typed_data() = *payload.typed_data() + static_cast<double>(count);
  return ffi::Error::Success();
}
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(phx_test_ordered_cpu, observe,
  ffi::Ffi::Bind()
    .Arg<ffi::Token>()
    .Arg<ffi::BufferR0<ffi::S64>>().Arg<ffi::BufferR0<ffi::S64>>()
    .Arg<ffi::BufferR0<ffi::F64>>()
    .Ret<ffi::Token>().Ret<ffi::BufferR0<ffi::F64>>()
    .Attr<std::string_view>("collective_configuration_id")
    .Attr<std::string_view>("collective_call_id"));

extern "C" int64_t phx_test_ordered_count() {
  std::lock_guard<std::mutex> lock(mutex);
  return count;
}

extern "C" int phx_test_ordered_record(int64_t index, double* payload,
                                      int64_t* ticket) {
  std::lock_guard<std::mutex> lock(mutex);
  if (index < 0 || index >= count) return -1;
  *payload = payloads[index];
  *ticket = tickets[index];
  return 0;
}
""",
        encoding="utf-8",
    )
    build = subprocess.run(
        (
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            "-pthread",
            f"-I{jax.ffi.include_dir()}",
            str(source),
            "-o",
            str(library_path),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if build.returncode:
        pytest.fail(
            f"native token observer compilation failed:\n{build.stdout}\n{build.stderr}"
        )
    library = ctypes.CDLL(str(library_path))
    library.phx_test_ordered_count.argtypes = []
    library.phx_test_ordered_count.restype = ctypes.c_int64
    library.phx_test_ordered_record.argtypes = (
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
    )
    library.phx_test_ordered_record.restype = ctypes.c_int
    target = f"phydrax_test_ordered_cpu_{directory.name}"
    jax.ffi.register_ffi_target(
        target, jax.ffi.pycapsule(library.phx_test_ordered_cpu), platform="cpu"
    )
    return target, library


def test_ordered_math_wrapper_repeated_program_preserves_hidden_runtime_token(
    program_lease: tuple[CollectiveLease, _ProgramBridge],
    monkeypatch: pytest.MonkeyPatch,
    native_cpu_token_observer: tuple[str, ctypes.CDLL],
) -> None:
    lease, bridge = program_lease
    target, library = native_cpu_token_observer
    wrapper = core.Primitive("collective_no_callback_cpu_wrapper")

    def abstract(
        *avals: AbstractValue, lease_key: CollectiveLeaseSpec
    ) -> tuple[tuple[ShapedArray, ...], set[core.Effect]]:
        value = avals[-1]
        if not isinstance(value, ShapedArray):
            raise TypeError("the AOT token regression requires ordinary array values")
        return collective.collective_effectful_abstract_eval(
            *avals, result_avals=(value,), lease_key=lease_key
        )

    def mathematical_wrapper(
        lease_id: Array,
        ticket: Array,
        value: Array,
        *,
        lease_key: CollectiveLeaseSpec,
    ) -> tuple[Array, ...]:
        collective.collective_ffi_call(
            target,
            (ShapeDtypeStruct(value.shape, value.dtype),),
            (lease_id, ticket, value),
            lease_key=lease_key,
            attributes={"operation_id": "first"},
        )
        return collective.collective_ffi_call(
            target,
            (ShapeDtypeStruct(value.shape, value.dtype),),
            (lease_id, ticket, value + 10),
            lease_key=lease_key,
            attributes={"operation_id": "second"},
        )

    def schedule(
        equation: core.JaxprEqn,
    ) -> tuple[tuple[CollectiveLeaseSpec, str], ...]:
        spec = equation.params["lease_key"]
        return tuple(
            (spec, collective.collective_call_identity(target, spec, operation_id))
            for operation_id in ("first", "second")
        )

    monkeypatch.setattr(collective, "_schedule_rules", dict(collective._schedule_rules))
    monkeypatch.setattr(
        jax_dispatch,
        "prim_requires_devices_during_lowering",
        set(jax_dispatch.prim_requires_devices_during_lowering),
    )
    wrapper.multiple_results = True
    jax_dispatch.simple_impl(wrapper)
    wrapper.def_effectful_abstract_eval(abstract)
    collective.register_collective_schedule_rule(wrapper, schedule)
    monkeypatch.setitem(
        mlir._platform_specific_lowerings["cpu"],
        wrapper,
        _mlir.LoweringRuleEntry(
            mlir.lower_fun(mathematical_wrapper, multiple_results=True), inline=True
        ),
    )
    # Exercise the production token ABI lowering with a real CPU FFI handler,
    # not emit_python_callback (whose host-callback flag masks the AOT bug).
    monkeypatch.setitem(
        mlir._platform_specific_lowerings["cpu"],
        collective._collective_p,
        mlir._platform_specific_lowerings["cuda"][collective._collective_p],
    )

    @jax.jit
    def operation(binding: CollectiveBinding, value: Array) -> Array:
        return wrapper.bind(
            binding.lease_id,
            binding.invocation_ticket,
            value,
            lease_key=binding.spec,
        )[0]

    # No host callback is registered. The first native result is discarded;
    # its record and the second result prove token execution and ordering.
    # Reusing this exact artifact exercises the cached-AOT token-loss boundary.
    example = jnp.asarray(1, dtype=np.float64)
    program = collectives.prepare_collective_program(
        operation, lease.binding, example, lease_key=lease.spec
    )
    for index, value in enumerate((1, 2, 3), start=1):
        result = lease.invoke(
            program, lease.binding, jnp.asarray(value, dtype=np.float64)
        )
        np.testing.assert_array_equal(
            np.asarray(result), np.asarray(value + 10 + 2 * index, dtype=np.float64)
        )
    assert bridge.events == [
        "agree-program",
        "begin",
        "end",
        "begin",
        "end",
        "begin",
        "end",
    ]
    assert library.phx_test_ordered_count() == 6
    payloads: list[float] = []
    tickets: list[int] = []
    for index in range(6):
        payload = ctypes.c_double()
        ticket = ctypes.c_int64()
        status = library.phx_test_ordered_record(
            index, ctypes.byref(payload), ctypes.byref(ticket)
        )
        assert status == 0
        payloads.append(payload.value)
        tickets.append(ticket.value)
    assert payloads == [1, 11, 2, 12, 3, 13]
    assert tickets == [1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("reduction", (False, True))
def test_mathematical_owner_preserves_manual_variance_through_shard_map_ad(
    admitted_spec: CollectiveLeaseSpec,
    monkeypatch: pytest.MonkeyPatch,
    reduction: bool,
) -> None:
    devices = jax.local_devices(backend="cpu")
    if len(devices) < 2:
        pytest.skip("manual-variance runtime qualification requires two real CPU devices")
    mesh = Mesh(np.asarray(devices[:2]), ("collective_rank",))
    axis = mesh.axis_names[0]
    primitive = core.Primitive(f"collective_owner_variance_{reduction}")

    def implementation(
        lease_id: Array, ticket: Array, value: Array, *, inverse: bool
    ) -> Array:
        del lease_id, ticket
        if reduction:
            return jax.lax.psum(value, axis)
        return jax.lax.all_to_all(
            value,
            axis,
            split_axis=0 if inverse else 1,
            concat_axis=1 if inverse else 0,
            tiled=True,
        )

    def abstract(
        lease_id: AbstractValue,
        ticket: AbstractValue,
        value: AbstractValue,
        *,
        inverse: bool,
    ) -> tuple[ShapedArray, set[core.Effect]]:
        if not isinstance(value, ShapedArray):
            raise TypeError("the manual-variance owner requires an ordinary array")
        if reduction:
            # This is the actual psum replication proof, not a generic inference
            # from the input's shape, values, or other owners' variance.
            result = value.update(
                manual_axis_type=value.mat.update(varying=value.mat.varying - {axis})
            )
        else:
            shape = (
                (value.shape[0] // mesh.size, value.shape[1] * mesh.size)
                if inverse
                else (value.shape[0] * mesh.size, value.shape[1] // mesh.size)
            )
            result = value.update(shape=shape)
        outputs, effects = collective.collective_effectful_abstract_eval(
            lease_id, ticket, value, result_avals=(result,), lease_key=admitted_spec
        )
        return outputs[0], effects

    def jvp(
        primals: tuple[Array, ...],
        tangents: tuple[Array | ad.Zero, ...],
        *,
        inverse: bool,
    ) -> tuple[Array, Array]:
        lease_id, ticket, value = primals
        tangent = ad.instantiate_zeros(tangents[2])
        expected = jax.typeof(value)
        actual = jax.typeof(tangent)
        if not isinstance(expected, ShapedArray) or not isinstance(actual, ShapedArray):
            raise TypeError("the manual-variance JVP requires ordinary array values")
        missing = expected.mat.varying - actual.mat.varying
        if missing:
            tangent = jax.lax.pcast(tangent, tuple(missing), to="varying")
        return (
            primitive.bind(lease_id, ticket, value, inverse=inverse),
            primitive.bind(lease_id, ticket, tangent, inverse=inverse),
        )

    def transpose(
        cotangent: Array | ad.Zero,
        lease_id: Array,
        ticket: Array,
        value: Array | ad.UndefinedPrimal,
        *,
        inverse: bool,
    ) -> tuple[None, None, Array]:
        original = (
            value.aval if isinstance(value, ad.UndefinedPrimal) else jax.typeof(value)
        )
        cotangent = ad.instantiate_zeros(cotangent)
        actual = jax.typeof(cotangent)
        if not isinstance(original, ShapedArray) or not isinstance(actual, ShapedArray):
            raise TypeError("the manual-variance VJP requires ordinary array values")
        missing = original.mat.varying - actual.mat.varying
        if missing:
            cotangent = jax.lax.pcast(cotangent, tuple(missing), to="varying")
        if reduction:
            return None, None, cotangent
        return (
            None,
            None,
            primitive.bind(lease_id, ticket, cotangent, inverse=not inverse),
        )

    jax_dispatch.simple_impl(primitive)
    primitive.def_effectful_abstract_eval(abstract)
    monkeypatch.setitem(ad.primitive_jvps, primitive, jvp)
    monkeypatch.setitem(ad.primitive_transposes, primitive, transpose)
    monkeypatch.setitem(
        mlir._platform_specific_lowerings["cpu"],
        primitive,
        _mlir.LoweringRuleEntry(
            mlir.lower_fun(implementation, multiple_results=False), inline=True
        ),
    )

    def selected_body(lease_id: Array, ticket: Array, value: Array) -> Array:
        return primitive.bind(lease_id, ticket, value, inverse=False)

    def reference_body(lease_id: Array, ticket: Array, value: Array) -> Array:
        return implementation(lease_id, ticket, value, inverse=False)

    output_spec = PartitionSpec() if reduction else PartitionSpec(None, axis)
    mapped = jax.shard_map(
        selected_body,
        mesh=mesh,
        in_specs=(PartitionSpec(), PartitionSpec(), PartitionSpec(axis, None)),
        out_specs=output_spec,
        check_vma=True,
    )
    reference = jax.shard_map(
        reference_body,
        mesh=mesh,
        in_specs=(PartitionSpec(), PartitionSpec(), PartitionSpec(axis, None)),
        out_specs=output_spec,
        check_vma=True,
    )
    metadata = jax.device_put(
        np.asarray(1, dtype=np.int64), NamedSharding(mesh, PartitionSpec())
    )

    def selected(value: Array) -> Array:
        return mapped(metadata, metadata, value)

    def native_reference(value: Array) -> Array:
        return reference(metadata, metadata, value)

    host = np.arange(16, dtype=np.float64).reshape(4, 4).astype(np.complex128)
    host *= np.complex128(1 + 0.25j)
    value = jax.device_put(host, NamedSharding(mesh, PartitionSpec(axis, None)))
    actual, actual_pullback = jax.vjp(selected, value)
    expected, expected_pullback = jax.vjp(native_reference, value)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=1e-12
    )
    cotangent = jnp.ones_like(expected) * np.complex128(2 - 0.5j)
    np.testing.assert_allclose(
        np.asarray(actual_pullback(cotangent)[0]),
        np.asarray(expected_pullback(cotangent)[0]),
        rtol=1e-12,
        atol=1e-12,
    )
    tangent = jnp.ones_like(value)
    _, actual_tangent = jax.jvp(selected, (value,), (tangent,))
    _, expected_tangent = jax.jvp(native_reference, (value,), (tangent,))
    np.testing.assert_allclose(
        np.asarray(actual_tangent), np.asarray(expected_tangent), rtol=1e-12, atol=1e-12
    )
