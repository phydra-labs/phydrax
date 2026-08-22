#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
    PreparedStencilProgram,
)


class StencilStateLayout(StrictModule):
    """Stable trailing-field packing for one tensor stencil program."""

    field_names: tuple[str, ...] = eqx.field(static=True)
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_names: Sequence[str],
        spatial_shape: Sequence[int],
        /,
    ):
        fields = tuple(str(name) for name in field_names)
        shape = tuple(int(size) for size in spatial_shape)
        if (
            not fields
            or any(not name for name in fields)
            or len(set(fields)) != len(fields)
            or not shape
            or any(size <= 0 for size in shape)
        ):
            raise ValueError("Stencil state fields and spatial dimensions must be valid.")
        self.field_names = fields
        self.spatial_shape = shape
        self.state_shape = shape + (len(fields),)
        self.layout_id = canonical_fingerprint(
            {
                "kind": "stencil-state-layout",
                "fields": list(fields),
                "spatial_shape": list(shape),
            }
        )

    def pack(self, fields: Mapping[str, ArrayLike], /) -> Array:
        if set(fields) != set(self.field_names):
            raise ValueError("Stencil fields must match the layout exactly.")
        values = tuple(jnp.asarray(fields[name]) for name in self.field_names)
        if any(value.shape != self.spatial_shape for value in values):
            raise ValueError("Every stencil field must match the spatial shape.")
        return jnp.stack(values, axis=-1)

    def unpack(self, state: ArrayLike, /) -> dict[str, Array]:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(f"Stencil state must have shape {self.state_shape}.")
        return {name: value[..., index] for index, name in enumerate(self.field_names)}


class CompiledStencilDynamics(StrictModule):
    """Executable stencil program, packed state layout, and full provenance."""

    program: PreparedStencilProgram
    layout: StencilStateLayout
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(self, program: PreparedStencilProgram, /):
        if not isinstance(program, PreparedStencilProgram):
            raise TypeError("program must be a PreparedStencilProgram.")
        layout = StencilStateLayout(
            program.plan.field_names,
            program.plan.discretization.grid.shape,
        )
        compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-stencil-dynamics",
                "program": program.prepared_id,
                "layout": layout.layout_id,
            }
        )
        residual_key = DiscretizationKey(
            "stencil_program",
            DiscretizationRole.RESIDUAL,
            domain_labels=program.plan.discretization.grid.axis_names,
        )
        discretization = program.plan.discretization
        self.program = program
        self.layout = layout
        self.discretization_bundle = DiscretizationBundle(
            (
                DiscretizationRecord(
                    discretization.key,
                    type(discretization).__name__,
                    discretization.prepared_id,
                    numeric_version=discretization.numeric_version,
                ),
                DiscretizationRecord(
                    residual_key,
                    "prepared-stencil-program",
                    program.prepared_id,
                    dependency_key_ids=(discretization.key.key_id,),
                ),
            )
        )
        self.compilation_id = compilation_id

    def drift(self, time: Array, state: Array, args: Any) -> Array:
        del time, args
        fields = self.layout.unpack(state)
        derivatives = self.program(fields)
        complete = {
            name: derivatives[name]
            if name in derivatives
            else jnp.zeros(self.layout.spatial_shape, dtype=state.dtype)
            for name in self.layout.field_names
        }
        return self.layout.pack(complete)


def compile_stencil_dynamics(
    program: PreparedStencilProgram,
    /,
) -> CompiledStencilDynamics:
    return CompiledStencilDynamics(program)


__all__ = [
    "CompiledStencilDynamics",
    "StencilStateLayout",
    "compile_stencil_dynamics",
]
