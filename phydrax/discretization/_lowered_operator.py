#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class LoweredBufferSpec(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    placement: str = eqx.field(static=True)
    buffer_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        shape: Sequence[int],
        dtype: Any,
        /,
        *,
        placement: str = "device",
    ):
        identifier = str(name)
        shape_ = tuple(int(value) for value in shape)
        dtype_ = str(np.dtype(dtype))
        placement_ = str(placement)
        if not identifier or any(value <= 0 for value in shape_):
            raise ValueError("Lowered buffer name/shape are invalid.")
        if placement_ not in ("host", "device", "distributed"):
            raise ValueError("Unknown lowered buffer placement.")
        self.name = identifier
        self.shape = shape_
        self.dtype = dtype_
        self.placement = placement_
        self.buffer_id = canonical_fingerprint(
            {
                "kind": "lowered-buffer",
                "name": identifier,
                "shape": list(shape_),
                "dtype": dtype_,
                "placement": placement_,
            }
        )


class LoweredKernel(StrictModule):
    name: str = eqx.field(static=True)
    reads: tuple[str, ...] = eqx.field(static=True)
    writes: tuple[str, ...] = eqx.field(static=True)
    jax_action: Callable[[Mapping[str, Any]], Mapping[str, Any]] = eqx.field(static=True)
    numpy_action: Callable[[Mapping[str, Any]], Mapping[str, Any]] = eqx.field(
        static=True
    )
    halo_widths: tuple[int, ...] = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        reads: Sequence[str],
        writes: Sequence[str],
        jax_action: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        numpy_action: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        /,
        *,
        halo_widths: Sequence[int] = (),
    ):
        identifier = str(name)
        reads_ = tuple(str(value) for value in reads)
        writes_ = tuple(str(value) for value in writes)
        halos = tuple(int(value) for value in halo_widths)
        if not identifier or not writes_ or len(set(writes_)) != len(writes_):
            raise ValueError("Lowered kernel name/writes are invalid.")
        if any(not value for value in reads_ + writes_) or any(
            value < 0 for value in halos
        ):
            raise ValueError("Lowered kernel reads/halos are invalid.")
        if not callable(jax_action) or not callable(numpy_action):
            raise TypeError("Lowered kernel backends must be callable.")
        self.name = identifier
        self.reads = reads_
        self.writes = writes_
        self.jax_action = jax_action
        self.numpy_action = numpy_action
        self.halo_widths = halos
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "lowered-kernel",
                "name": identifier,
                "reads": list(reads_),
                "writes": list(writes_),
                "halos": list(halos),
            }
        )


class LoweredOperatorProgram(StrictModule, NonTrainableState):
    buffers: tuple[LoweredBufferSpec, ...]
    kernels: tuple[LoweredKernel, ...]
    program_id: str = eqx.field(static=True)

    def __init__(
        self,
        buffers: Sequence[LoweredBufferSpec],
        kernels: Sequence[LoweredKernel],
        /,
    ):
        buffers_ = tuple(buffers)
        kernels_ = tuple(kernels)
        if not buffers_ or any(
            not isinstance(value, LoweredBufferSpec) for value in buffers_
        ):
            raise TypeError("Lowered program requires buffer specifications.")
        if not kernels_ or any(
            not isinstance(value, LoweredKernel) for value in kernels_
        ):
            raise TypeError("Lowered program requires kernels.")
        names = tuple(value.name for value in buffers_)
        if len(set(names)) != len(names):
            raise ValueError("Lowered buffer names must be unique.")
        known = set(names)
        initialized = set(names)
        for kernel in kernels_:
            if not set(kernel.reads).issubset(initialized) or not set(
                kernel.writes
            ).issubset(known):
                raise ValueError("Lowered kernel reads/writes reference unknown buffers.")
            initialized.update(kernel.writes)
        self.buffers = buffers_
        self.kernels = kernels_
        self.program_id = canonical_fingerprint(
            {
                "kind": "lowered-operator-program",
                "buffers": [value.buffer_id for value in buffers_],
                "kernels": [value.kernel_id for value in kernels_],
            }
        )

    def validate_state(self, state: Mapping[str, Any], /) -> dict[str, Any]:
        values = dict(state)
        if set(values) != {value.name for value in self.buffers}:
            raise ValueError("Lowered state keys must exactly match program buffers.")
        for spec in self.buffers:
            array = np.asarray(values[spec.name])
            if array.shape != spec.shape or str(array.dtype) != spec.dtype:
                raise ValueError(f"Lowered buffer {spec.name!r} has wrong shape/dtype.")
        return values


class LoweredJAXBackend(StrictModule, NonTrainableState):
    program: LoweredOperatorProgram

    def __call__(self, state: Mapping[str, Any], /) -> dict[str, Array]:
        values = {name: jnp.asarray(value) for name, value in state.items()}
        for kernel in self.program.kernels:
            updates = dict(kernel.jax_action(values))
            if set(updates) != set(kernel.writes):
                raise ValueError("JAX lowered kernel returned wrong write set.")
            values.update(updates)
        return values

    def compile(self, /):
        return jax.jit(self)


class LoweredNumPyBackend(StrictModule, NonTrainableState):
    program: LoweredOperatorProgram

    def __call__(self, state: Mapping[str, Any], /) -> dict[str, np.ndarray]:
        values = {name: np.asarray(value) for name, value in state.items()}
        for kernel in self.program.kernels:
            updates = dict(kernel.numpy_action(values))
            if set(updates) != set(kernel.writes):
                raise ValueError("NumPy lowered kernel returned wrong write set.")
            values.update({name: np.asarray(value) for name, value in updates.items()})
        return values


class LoweredBackendParityReport(StrictModule):
    maximum_residual: Array
    passed: Array
    program_id: str = eqx.field(static=True)


def compare_lowered_backends(
    program: LoweredOperatorProgram,
    state: Mapping[str, Any],
    /,
    *,
    tolerance: float = 1e-10,
) -> LoweredBackendParityReport:
    validated = program.validate_state(state)
    jax_result = LoweredJAXBackend(program)(validated)
    numpy_result = LoweredNumPyBackend(program)(validated)
    residual = jnp.asarray(0.0)
    for name in jax_result:
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(jax_result[name])))
        residual = jnp.maximum(
            residual,
            jnp.max(jnp.abs(jax_result[name] - numpy_result[name])) / scale,
        )
    return LoweredBackendParityReport(
        residual,
        residual <= float(tolerance),
        program.program_id,
    )


__all__ = [
    "LoweredBackendParityReport",
    "LoweredBufferSpec",
    "LoweredJAXBackend",
    "LoweredKernel",
    "LoweredNumPyBackend",
    "LoweredOperatorProgram",
    "compare_lowered_backends",
]
