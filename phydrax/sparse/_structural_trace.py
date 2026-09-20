#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._pattern import SparsePattern
from ._structural_interpret import _prop_jaxpr


DerivativeKind = Literal["jacobian", "hessian"]


def _coo_from_dependencies(
    dependencies: list[set[int]],
    /,
) -> tuple[list[int], list[int]]:
    rows: list[int] = []
    columns: list[int] = []
    for row, values in enumerate(dependencies):
        for column in sorted(values):
            rows.append(row)
            columns.append(column)
    return rows, columns


def trace_sparse_pattern(
    function: Callable[[Array, Any], Array],
    coordinates: Array,
    sample_args: Any,
    /,
    *,
    source_size: int,
    target_size: int,
    derivative_kind: DerivativeKind,
) -> SparsePattern:
    """Trace one global structural derivative pattern from a JAXPR."""

    if derivative_kind == "jacobian":
        traced_function = lambda value: function(value, sample_args)
        expected_shape = (int(target_size), int(source_size))
        symmetric = False
    elif derivative_kind == "hessian":

        def scalar_function(value):
            output = jnp.asarray(function(value, sample_args))
            if output.size != 1:
                raise ValueError("Sparse Hessian tracing requires scalar output.")
            return jnp.reshape(output, ())

        traced_function = jax.grad(scalar_function)
        expected_shape = (int(source_size), int(source_size))
        symmetric = True
    else:
        raise ValueError(f"Unknown sparse derivative kind {derivative_kind!r}.")

    closed = jax.make_jaxpr(traced_function)(coordinates)
    jaxpr = closed.jaxpr
    if len(jaxpr.invars) != 1:
        raise ValueError("Structural sparsity tracing requires one coordinate input.")
    input_size = int(np.prod(jaxpr.invars[0].aval.shape, dtype=np.int64))
    if input_size != source_size:
        raise ValueError(
            f"Coordinate JAXPR has size {input_size}; expected {source_size}."
        )
    input_dependencies = [[{index} for index in range(source_size)]]
    constants = {
        variable: np.asarray(value)
        for variable, value in zip(jaxpr.constvars, closed.consts, strict=True)
    }
    output_dependencies = _prop_jaxpr(jaxpr, input_dependencies, constants)
    flattened: list[set[int]] = []
    for output in output_dependencies:
        flattened.extend(output)
    if len(flattened) != expected_shape[0]:
        raise ValueError(
            "Structural sparsity output size does not match the declared target space."
        )
    rows, columns = _coo_from_dependencies(flattened)
    if symmetric:
        entries = set(zip(rows, columns, strict=True))
        entries.update((column, row) for row, column in tuple(entries))
        ordered = sorted(entries)
        rows = [row for row, _ in ordered]
        columns = [column for _, column in ordered]
    return SparsePattern.from_coo(
        rows,
        columns,
        expected_shape,
        symmetric=symmetric,
        origin="structural",
    )


__all__ = ["trace_sparse_pattern"]
