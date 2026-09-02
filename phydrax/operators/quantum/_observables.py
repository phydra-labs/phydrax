#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._propagation import apply_local_operator_to_state
from ._register import _target_wire_ids, HilbertRegisterLayout


class LocalObservable(StrictModule):
    """One ordered local observable matrix on a finite Hilbert register."""

    matrix: Array
    target_wire_ids: tuple[str, ...] = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    finite: Array
    hermiticity_residual: Array
    hermitian: Array
    valid: Array
    schema_id: str = eqx.field(static=True)
    observable_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        target_wire_ids: Sequence[str],
        /,
        *,
        hermiticity_tolerance: float = 1e-8,
    ):
        value = jnp.asarray(matrix)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("Observable matrix must have exact square shape (dT, dT).")
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError(
                "Observable matrix must use complex floating-point coordinates."
            )
        tolerance = float(hermiticity_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("hermiticity_tolerance must be finite and nonnegative.")
        targets = _target_wire_ids(target_wire_ids)
        adjoint = jnp.swapaxes(jnp.conj(value), -1, -2)
        finite = jnp.all(jnp.isfinite(value))
        residual = jnp.max(jnp.abs(value - adjoint))
        hermitian = residual <= tolerance
        schema_id = canonical_fingerprint(
            {
                "kind": "local-observable-schema",
                "targets": targets,
                "shape": value.shape,
                "dtype": str(value.dtype),
                "hermiticity_tolerance": tolerance,
            }
        )
        self.matrix = value
        self.target_wire_ids = targets
        self.hermiticity_tolerance = tolerance
        self.finite = finite
        self.hermiticity_residual = residual
        self.hermitian = hermitian
        self.valid = finite & hermitian
        self.schema_id = schema_id
        self.observable_id = canonical_fingerprint(
            {
                "kind": "local-observable",
                "schema": schema_id,
                "matrix": array_tree_fingerprint(value)["sha256"],
            }
        )


def _observable_inputs(
    layout: HilbertRegisterLayout,
    observable: LocalObservable,
    /,
) -> None:
    if not isinstance(layout, HilbertRegisterLayout):
        raise TypeError("layout must be a HilbertRegisterLayout.")
    if not isinstance(observable, LocalObservable):
        raise TypeError("observable must be a LocalObservable.")
    if observable.matrix.shape[0] != layout.target_dimension(observable.target_wire_ids):
        raise ValueError(
            "Observable matrix dimension does not match its ordered targets."
        )


def local_state_expectation(
    layout: HilbertRegisterLayout,
    observable: LocalObservable,
    state: ArrayLike,
    /,
) -> Array:
    """Evaluate one local observable on state vectors without a global matrix."""
    _observable_inputs(layout, observable)
    vector = jnp.asarray(state)
    transformed = apply_local_operator_to_state(
        layout,
        observable.matrix,
        observable.target_wire_ids,
        vector,
    )
    return oe.contract("...i,...i->...", jnp.conj(vector), transformed)


def local_density_expectation(
    layout: HilbertRegisterLayout,
    observable: LocalObservable,
    density: ArrayLike,
    /,
) -> Array:
    """Evaluate one local observable on density matrices without a global matrix."""
    _observable_inputs(layout, observable)
    value = jnp.asarray(density)
    if (
        value.ndim < 2
        or value.shape[-2:] != (layout.dimension, layout.dimension)
        or not jnp.issubdtype(value.dtype, jnp.complexfloating)
    ):
        raise ValueError(
            "Density must have complex shape (..., layout.dimension, layout.dimension)."
        )
    if value.dtype != observable.matrix.dtype:
        raise TypeError("Observable and density dtypes must match exactly.")
    right_action = apply_local_operator_to_state(
        layout,
        jnp.swapaxes(observable.matrix, -1, -2),
        observable.target_wire_ids,
        value,
    )
    return jnp.trace(right_action, axis1=-2, axis2=-1)


__all__ = [
    "LocalObservable",
    "local_density_expectation",
    "local_state_expectation",
]
