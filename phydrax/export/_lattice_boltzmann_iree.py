#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from ..backends._types import BackendAvailability
from ..backends.iree import iree_availability
from ..discretization.lattice_boltzmann._execution import (
    ReferenceLatticeBoltzmannExecutionPlan,
)
from ..discretization.lattice_boltzmann._fused import (
    FusedLatticeBoltzmannExecutionPlan,
)
from ._iree import IREEExportPolicy, IREEExportResult, save_iree


LatticeBoltzmannIREEExportMode: TypeAlias = Literal["forward", "forward-vjp"]
_LBMExportPlan = (
    ReferenceLatticeBoltzmannExecutionPlan | FusedLatticeBoltzmannExecutionPlan
)


@dataclass(frozen=True, slots=True)
class LatticeBoltzmannIREEContract:
    """Static forward and optional transpose ABI for one fixed realization."""

    execution_mode: LatticeBoltzmannIREEExportMode
    execution_plan_id: str
    lattice_id: str
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    step_count: int
    step_size: float
    initial_time: float
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    differentiable_input_names: tuple[str, ...]
    vjp_input_names: tuple[str, ...]
    vjp_output_names: tuple[str, ...]
    supports_reverse_mode: bool
    contract_id: str

    def pack_inputs(self, *values: Array) -> tuple[Array, ...]:
        if len(values) != len(self.input_names):
            raise ValueError("LBM IREE input count does not match contract.")
        packed: list[Array] = []
        for value, shape, dtype in zip(
            values, self.input_shapes, self.input_dtypes, strict=True
        ):
            if tuple(value.shape) != shape:
                raise ValueError("LBM IREE input shape does not match contract.")
            if np.dtype(value.dtype).str != dtype:
                raise TypeError("LBM IREE input dtype does not match contract.")
            packed.append(value)
        return tuple(packed)

    def pack_vjp_inputs(
        self, primal_inputs: tuple[Array, ...], cotangent_final_populations: Array, /
    ) -> tuple[Array, ...]:
        primal = self.pack_inputs(*primal_inputs)
        if (
            tuple(cotangent_final_populations.shape) != self.input_shapes[0]
            or np.dtype(cotangent_final_populations.dtype).str != self.input_dtypes[0]
        ):
            raise ValueError("LBM IREE output cotangent does not match populations.")
        return (*primal, cotangent_final_populations)


@dataclass(frozen=True, slots=True)
class LatticeBoltzmannIREEExportBundle:
    """Atomically published forward artifact and optional exact VJP artifact."""

    forward: IREEExportResult
    vjp: IREEExportResult | None
    contract: LatticeBoltzmannIREEContract

    @property
    def path(self) -> Path:
        return self.forward.path


def lattice_boltzmann_iree_availability() -> BackendAvailability:
    """Return the matched IREE compiler/runtime availability evidence."""

    return iree_availability()


def prepare_lattice_boltzmann_iree_contract(
    plan: _LBMExportPlan,
    initial_populations: Array,
    /,
    *,
    step_count: int,
    step_size: float,
    runtime_arrays: tuple[Array, ...] = (),
    runtime_array_names: tuple[str, ...] = (),
    differentiable_inputs: tuple[str, ...] = ("populations",),
    t0: float = 0.0,
    mode: LatticeBoltzmannIREEExportMode = "forward",
) -> LatticeBoltzmannIREEContract:
    """Prepare a fixed-shape forward/VJP ABI without invoking the compiler."""

    if not isinstance(
        plan,
        (ReferenceLatticeBoltzmannExecutionPlan, FusedLatticeBoltzmannExecutionPlan),
    ):
        raise TypeError("IREE LBM export requires a reference or fused execution plan.")
    if mode not in ("forward", "forward-vjp"):
        raise ValueError("Unknown LBM IREE export mode.")
    arrays = (initial_populations, *tuple(runtime_arrays))
    if any(not eqx.is_array(value) for value in arrays):
        raise TypeError("IREE LBM export accepts explicit JAX array inputs only.")
    velocity_set = plan.velocity_set
    if (
        initial_populations.ndim != velocity_set.dimension + 1
        or initial_populations.shape[-1] != velocity_set.population_count
    ):
        raise ValueError("IREE LBM populations must use the lattice trailing-Q shape.")
    names = ("populations", *tuple(str(name) for name in runtime_array_names))
    if len(names) != len(arrays) or len(set(names)) != len(names):
        raise ValueError("Runtime array names must be unique and match runtime arrays.")
    selected = tuple(str(name) for name in differentiable_inputs)
    if len(set(selected)) != len(selected) or any(name not in names for name in selected):
        raise ValueError("Differentiable inputs must name declared array inputs.")
    if mode == "forward-vjp" and not selected:
        raise ValueError("Forward-VJP export requires differentiable array inputs.")
    count = int(step_count)
    dt = float(step_size)
    initial_time = float(t0)
    if count <= 0 or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("IREE LBM export requires positive step_count and step_size.")
    if not math.isfinite(initial_time):
        raise ValueError("IREE LBM initial time must be finite.")
    shapes = tuple(tuple(int(size) for size in value.shape) for value in arrays)
    dtypes = tuple(np.dtype(value.dtype).str for value in arrays)
    vjp_inputs = (*names, "cotangent_final_populations")
    vjp_outputs = tuple(f"cotangent_{name}" for name in selected)
    metadata = {
        "kind": "lattice-boltzmann-iree-contract",
        "execution_mode": mode,
        "execution_plan": plan.plan_id,
        "lattice": velocity_set.lattice_id,
        "input_shapes": shapes,
        "input_dtypes": dtypes,
        "step_count": count,
        "step_size": dt,
        "initial_time": initial_time,
        "input_names": names,
        "differentiable_inputs": selected,
    }
    return LatticeBoltzmannIREEContract(
        mode,
        plan.plan_id,
        velocity_set.lattice_id,
        shapes,
        dtypes,
        count,
        dt,
        initial_time,
        names,
        ("final_populations",),
        selected,
        vjp_inputs,
        vjp_outputs,
        mode == "forward-vjp",
        canonical_fingerprint(metadata),
    )


def save_lattice_boltzmann_iree(
    plan: _LBMExportPlan,
    path: str | Path,
    /,
    *,
    initial_populations: Array,
    step_count: int,
    step_size: float,
    runtime_arrays: tuple[Array, ...] = (),
    runtime_array_names: tuple[str, ...] = (),
    differentiable_inputs: tuple[str, ...] = ("populations",),
    static_args: Any = None,
    t0: float = 0.0,
    mode: LatticeBoltzmannIREEExportMode = "forward",
    policy: IREEExportPolicy | None = None,
    validate: bool = True,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
) -> LatticeBoltzmannIREEExportBundle:
    """Publish separate fixed forward and VJP artifacts through the native path."""

    runtime = tuple(runtime_arrays)
    contract = prepare_lattice_boltzmann_iree_contract(
        plan,
        initial_populations,
        step_count=step_count,
        step_size=step_size,
        runtime_arrays=runtime,
        runtime_array_names=runtime_array_names,
        differentiable_inputs=differentiable_inputs,
        t0=t0,
        mode=mode,
    )
    lattice_boltzmann_iree_availability().require("compiled-inference")

    def forward(*inputs, key=None):
        if key is not None:
            raise ValueError("IREE LBM export requires key=None.")
        populations, *runtime_values = inputs
        execution_args = (
            static_args if not runtime_values else (static_args, tuple(runtime_values))
        )
        return plan.realize(
            populations,
            step_count=contract.step_count,
            step_size=contract.step_size,
            args=execution_args,
            t0=contract.initial_time,
        ).final_populations

    destination = Path(path)
    forward_path = (
        destination
        if mode == "forward"
        else destination.with_name(f"{destination.stem}-forward{destination.suffix}")
    )
    primal_inputs = contract.pack_inputs(initial_populations, *runtime)
    forward_artifact = save_iree(
        forward,
        forward_path,
        inputs=primal_inputs,
        input_names=contract.input_names,
        policy=policy,
        key=None,
        validate=validate,
        rtol=rtol,
        atol=atol,
    )
    if mode == "forward":
        return LatticeBoltzmannIREEExportBundle(forward_artifact, None, contract)

    name_to_index = {name: index for index, name in enumerate(contract.input_names)}
    selected_indices = tuple(
        name_to_index[name] for name in contract.differentiable_input_names
    )

    def transpose(*inputs, key=None):
        if key is not None:
            raise ValueError("IREE LBM VJP export requires key=None.")
        *primals, cotangent = inputs
        _, pullback = jax.vjp(forward, *primals)
        cotangents = pullback(cotangent)
        return tuple(cotangents[index] for index in selected_indices)

    sample_final = forward(*primal_inputs)
    sample_cotangent = jax.numpy.zeros_like(sample_final)
    vjp_path = destination.with_name(f"{destination.stem}-vjp{destination.suffix}")
    vjp_artifact = save_iree(
        transpose,
        vjp_path,
        inputs=contract.pack_vjp_inputs(primal_inputs, sample_cotangent),
        input_names=contract.vjp_input_names,
        policy=policy,
        key=None,
        validate=validate,
        rtol=rtol,
        atol=atol,
    )
    return LatticeBoltzmannIREEExportBundle(forward_artifact, vjp_artifact, contract)


__all__ = [
    "LatticeBoltzmannIREEContract",
    "LatticeBoltzmannIREEExportBundle",
    "LatticeBoltzmannIREEExportMode",
    "lattice_boltzmann_iree_availability",
    "prepare_lattice_boltzmann_iree_contract",
    "save_lattice_boltzmann_iree",
]
