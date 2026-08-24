#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..finite_difference._coefficients import fornberg_weights


MortarSide: TypeAlias = Literal["left", "right"]


class NormCompatibleInterpolationPlan(StrictModule, NonTrainableState):
    """Local polynomial prolongation and norm-adjoint mortar restriction."""

    left_coordinates: Array
    right_coordinates: Array
    left_weights: Array
    right_weights: Array
    interpolation_order: int = eqx.field(static=True)
    mortar_side: MortarSide = eqx.field(static=True)
    left_to_mortar_matrix: Array
    right_to_mortar_matrix: Array
    mortar_to_left_matrix: Array
    mortar_to_right_matrix: Array
    compatibility_residual: float = eqx.field(static=True)
    constant_residual: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_coordinates: ArrayLike,
        right_coordinates: ArrayLike,
        left_weights: ArrayLike,
        right_weights: ArrayLike,
        /,
        *,
        interpolation_order: int = 4,
    ):
        left_x = np.asarray(left_coordinates, dtype=float).reshape((-1,))
        right_x = np.asarray(right_coordinates, dtype=float).reshape((-1,))
        left_h = np.asarray(left_weights, dtype=float).reshape((-1,))
        right_h = np.asarray(right_weights, dtype=float).reshape((-1,))
        order = int(interpolation_order)
        if (
            left_x.size < 2
            or right_x.size < 2
            or left_h.shape != left_x.shape
            or right_h.shape != right_x.shape
            or np.any(~np.isfinite(left_x))
            or np.any(~np.isfinite(right_x))
            or np.any(~np.isfinite(left_h))
            or np.any(~np.isfinite(right_h))
            or np.any(left_h <= 0.0)
            or np.any(right_h <= 0.0)
            or order <= 0
        ):
            raise ValueError("Interpolation coordinates, norms, or order are invalid.")
        if np.any(np.diff(left_x) <= 0.0) or np.any(np.diff(right_x) <= 0.0):
            raise ValueError(
                "Interface interpolation coordinates must increase strictly."
            )
        if not np.allclose(
            [left_x[0], left_x[-1]],
            [right_x[0], right_x[-1]],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("Interface interpolation endpoints must coincide.")
        if left_x.size == right_x.size:
            if not np.allclose(left_x, right_x, rtol=1e-10, atol=1e-12):
                raise ValueError("Equal-size interface coordinates must be conforming.")
            mortar_side: MortarSide = "left"
            left_to_mortar = np.eye(left_x.size)
            right_to_mortar = np.eye(right_x.size)
        elif left_x.size > right_x.size:
            mortar_side = "left"
            left_to_mortar = np.eye(left_x.size)
            right_to_mortar = _local_interpolation(right_x, left_x, order)
        else:
            mortar_side = "right"
            left_to_mortar = _local_interpolation(left_x, right_x, order)
            right_to_mortar = np.eye(right_x.size)
        mortar_weights = left_h if mortar_side == "left" else right_h
        left_restriction = (
            np.diag(1.0 / left_h) @ left_to_mortar.T @ np.diag(mortar_weights)
        )
        right_restriction = (
            np.diag(1.0 / right_h) @ right_to_mortar.T @ np.diag(mortar_weights)
        )
        left_compatibility = np.diag(mortar_weights) @ left_to_mortar - (
            left_restriction.T @ np.diag(left_h)
        )
        right_compatibility = np.diag(mortar_weights) @ right_to_mortar - (
            right_restriction.T @ np.diag(right_h)
        )
        compatibility = float(
            max(
                np.max(np.abs(left_compatibility)),
                np.max(np.abs(right_compatibility)),
            )
        )
        constant = float(
            max(
                np.max(np.abs(left_to_mortar @ np.ones(left_x.size) - 1.0)),
                np.max(np.abs(right_to_mortar @ np.ones(right_x.size) - 1.0)),
            )
        )
        if compatibility > 1e-12 or constant > 1e-10:
            raise RuntimeError("Norm-compatible interpolation construction failed.")
        self.left_coordinates = jnp.asarray(left_x)
        self.right_coordinates = jnp.asarray(right_x)
        self.left_weights = jnp.asarray(left_h)
        self.right_weights = jnp.asarray(right_h)
        self.interpolation_order = order
        self.mortar_side = mortar_side
        self.left_to_mortar_matrix = jnp.asarray(left_to_mortar)
        self.right_to_mortar_matrix = jnp.asarray(right_to_mortar)
        self.mortar_to_left_matrix = jnp.asarray(left_restriction)
        self.mortar_to_right_matrix = jnp.asarray(right_restriction)
        self.compatibility_residual = compatibility
        self.constant_residual = constant
        self.plan_id = canonical_fingerprint(
            {
                "kind": "norm-compatible-interface-interpolation",
                "left_coordinates": array_tree_fingerprint(left_x),
                "right_coordinates": array_tree_fingerprint(right_x),
                "left_weights": array_tree_fingerprint(left_h),
                "right_weights": array_tree_fingerprint(right_h),
                "interpolation_order": order,
                "mortar_side": mortar_side,
            }
        )

    def left_to_mortar(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.shape[0] != self.left_coordinates.size:
            raise ValueError(
                "Left trace leading size is incompatible with interpolation."
            )
        return oe.contract("ij,j...->i...", self.left_to_mortar_matrix, value)

    def right_to_mortar(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.shape[0] != self.right_coordinates.size:
            raise ValueError(
                "Right trace leading size is incompatible with interpolation."
            )
        return oe.contract("ij,j...->i...", self.right_to_mortar_matrix, value)

    def mortar_to_left(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.shape[0] != self.left_to_mortar_matrix.shape[0]:
            raise ValueError(
                "Mortar trace leading size is incompatible with interpolation."
            )
        return oe.contract("ij,j...->i...", self.mortar_to_left_matrix, value)

    def mortar_to_right(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.shape[0] != self.right_to_mortar_matrix.shape[0]:
            raise ValueError(
                "Mortar trace leading size is incompatible with interpolation."
            )
        return oe.contract("ij,j...->i...", self.mortar_to_right_matrix, value)


def _local_interpolation(
    source: np.ndarray,
    target: np.ndarray,
    order: int,
    /,
) -> np.ndarray:
    width = min(int(order) + 1, source.size)
    matrix = np.zeros((target.size, source.size), dtype=float)
    for row, coordinate in enumerate(target):
        insertion = int(np.searchsorted(source, coordinate))
        start = int(np.clip(insertion - width // 2, 0, source.size - width))
        indices = np.arange(start, start + width)
        matrix[row, indices] = fornberg_weights(source[indices], coordinate, 0)
    return matrix


__all__ = ["MortarSide", "NormCompatibleInterpolationPlan"]
