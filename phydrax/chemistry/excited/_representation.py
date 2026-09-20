#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Representation-correct excited-state amplitudes and normalization evidence."""

from __future__ import annotations

from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract


class TDAStateRepresentation(StrictModule, NonTrainableState):
    amplitudes: Array
    orthonormality_residual: Array
    representation_id: str = eqx.field(static=True)

    def __init__(self, amplitudes: ArrayLike, /):
        values = jnp.asarray(amplitudes)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("TDA amplitudes must have shape (basis, roots).")
        residual = jnp.max(
            jnp.abs(jnp.conj(values.T) @ values - jnp.eye(values.shape[1])),
            initial=0.0,
        )
        self.amplitudes = values
        self.orthonormality_residual = residual
        self.representation_id = canonical_fingerprint(
            {
                "kind": "tda-state-representation",
                "arrays": array_tree_fingerprint(np.asarray(values)),
            }
        )


class RPAStateRepresentation(StrictModule, NonTrainableState):
    x_amplitudes: Array
    y_amplitudes: Array
    left_amplitudes: Array
    symplectic_norms: Array
    biorthogonality_residual: Array
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        x_amplitudes: ArrayLike,
        y_amplitudes: ArrayLike,
        left_amplitudes: ArrayLike,
        /,
    ):
        x = jnp.asarray(x_amplitudes)
        y = jnp.asarray(y_amplitudes, dtype=x.dtype)
        left = jnp.asarray(left_amplitudes, dtype=x.dtype)
        if (
            x.ndim != 2
            or y.shape != x.shape
            or left.shape != (2 * x.shape[0], x.shape[1])
        ):
            raise ValueError("RPA X/Y and left amplitudes have incompatible shapes.")
        right = jnp.concatenate((x, y), axis=0)
        metric_right = jnp.concatenate((x, -y), axis=0)
        norms = jnp.real(contract("ir,ir->r", jnp.conj(right), metric_right))
        overlap = jnp.conj(left.T) @ right
        residual = jnp.max(jnp.abs(overlap - jnp.eye(x.shape[1])), initial=0.0)
        self.x_amplitudes = x
        self.y_amplitudes = y
        self.left_amplitudes = left
        self.symplectic_norms = norms
        self.biorthogonality_residual = residual
        self.representation_id = canonical_fingerprint(
            {
                "kind": "rpa-state-representation",
                "arrays": array_tree_fingerprint(
                    {
                        "x": np.asarray(x),
                        "y": np.asarray(y),
                        "left": np.asarray(left),
                        "symplectic_norms": np.asarray(norms),
                    }
                ),
            }
        )


class BiorthogonalStateRepresentation(StrictModule, NonTrainableState):
    right_amplitudes: Array
    left_amplitudes: Array
    biorthogonality_residual: Array
    method: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        right_amplitudes: ArrayLike,
        left_amplitudes: ArrayLike,
        method: str,
        /,
    ):
        right = jnp.asarray(right_amplitudes)
        left = jnp.asarray(left_amplitudes, dtype=right.dtype)
        method_ = str(method).strip()
        if right.ndim != 2 or left.shape != right.shape or not method_:
            raise ValueError("Biorthogonal amplitudes and method identity are invalid.")
        residual = jnp.max(
            jnp.abs(jnp.conj(left.T) @ right - jnp.eye(right.shape[1])),
            initial=0.0,
        )
        self.right_amplitudes = right
        self.left_amplitudes = left
        self.biorthogonality_residual = residual
        self.method = method_
        self.representation_id = canonical_fingerprint(
            {
                "kind": "biorthogonal-state-representation",
                "method": method_,
                "arrays": array_tree_fingerprint(
                    {"right": np.asarray(right), "left": np.asarray(left)}
                ),
            }
        )


class CIStateRepresentation(StrictModule, NonTrainableState):
    coefficients: Array
    determinant_ids: tuple[int, ...] = eqx.field(static=True)
    orthonormality_residual: Array
    representation_id: str = eqx.field(static=True)

    def __init__(self, coefficients: ArrayLike, determinant_ids: tuple[int, ...], /):
        values = jnp.asarray(coefficients)
        determinants = tuple(determinant_ids)
        if values.ndim != 2 or values.shape[0] != len(determinants):
            raise ValueError("CI coefficients must align with determinant identities.")
        residual = jnp.max(
            jnp.abs(jnp.conj(values.T) @ values - jnp.eye(values.shape[1])),
            initial=0.0,
        )
        self.coefficients = values
        self.determinant_ids = determinants
        self.orthonormality_residual = residual
        self.representation_id = canonical_fingerprint(
            {
                "kind": "ci-state-representation",
                "determinants": list(determinants),
                "arrays": array_tree_fingerprint(np.asarray(values)),
            }
        )


ExcitedStateRepresentation: TypeAlias = (
    TDAStateRepresentation
    | RPAStateRepresentation
    | BiorthogonalStateRepresentation
    | CIStateRepresentation
)


__all__ = [
    "BiorthogonalStateRepresentation",
    "CIStateRepresentation",
    "ExcitedStateRepresentation",
    "RPAStateRepresentation",
    "TDAStateRepresentation",
]
