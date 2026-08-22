#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import factorial
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


HighResolutionMethod: TypeAlias = Literal["weno_z", "teno", "mp5"]
ReconstructionBoundary: TypeAlias = Literal["periodic", "outflow"]


def _uniform_windows(
    values: Array,
    offsets: tuple[int, ...],
    boundary: ReconstructionBoundary,
    /,
) -> Array:
    count = values.shape[0]
    indices = jnp.arange(count)[:, None] + jnp.asarray(offsets)[None, :]
    indices = (
        jnp.mod(indices, count)
        if boundary == "periodic"
        else jnp.clip(indices, 0, count - 1)
    )
    return values[indices]


def _weno_candidates(window: Array, /) -> tuple[Array, Array]:
    um2, um1, center, up1, up2 = (window[:, index] for index in range(5))
    candidates = jnp.stack(
        (
            (2.0 * um2 - 7.0 * um1 + 11.0 * center) / 6.0,
            (-um1 + 5.0 * center + 2.0 * up1) / 6.0,
            (2.0 * center + 5.0 * up1 - up2) / 6.0,
        ),
        axis=1,
    )
    smoothness = jnp.stack(
        (
            13.0 / 12.0 * (um2 - 2.0 * um1 + center) ** 2
            + 0.25 * (um2 - 4.0 * um1 + 3.0 * center) ** 2,
            13.0 / 12.0 * (um1 - 2.0 * center + up1) ** 2 + 0.25 * (um1 - up1) ** 2,
            13.0 / 12.0 * (center - 2.0 * up1 + up2) ** 2
            + 0.25 * (3.0 * center - 4.0 * up1 + up2) ** 2,
        ),
        axis=1,
    )
    return candidates, smoothness


def _nonlinear_combination(
    candidates: Array,
    smoothness: Array,
    method: HighResolutionMethod,
    epsilon: float,
    power: int,
    cutoff: float,
    /,
    *,
    optimal: Array | None = None,
) -> Array:
    optimal_ = (
        jnp.asarray((0.1, 0.6, 0.3), dtype=candidates.dtype)
        if optimal is None
        else jnp.asarray(optimal, dtype=candidates.dtype)
    )
    shape = (1, 3) + (1,) * (smoothness.ndim - 2)
    optimal_ = optimal_.reshape(shape) if optimal_.ndim == 1 else optimal_
    tau = jnp.abs(smoothness[:, :1] - smoothness[:, 2:3])
    if method == "weno_z":
        alpha = optimal_ * (1.0 + (tau / (smoothness + epsilon)) ** power)
    else:
        gamma = (1.0 + tau / (smoothness + epsilon)) ** 6
        normalized = gamma / jnp.sum(gamma, axis=1, keepdims=True)
        active = normalized >= cutoff
        alpha = optimal_ * active
        alpha_sum = jnp.sum(alpha, axis=1, keepdims=True)
        alpha = jnp.where(alpha_sum > 0.0, alpha, optimal_)
    weights = alpha / jnp.sum(alpha, axis=1, keepdims=True)
    return jnp.sum(weights * candidates, axis=1)


def _minmod(*values: Array) -> Array:
    stacked = jnp.stack(values)
    same_positive = jnp.all(stacked > 0.0, axis=0)
    same_negative = jnp.all(stacked < 0.0, axis=0)
    magnitude = jnp.min(jnp.abs(stacked), axis=0)
    return jnp.where(same_positive, magnitude, jnp.where(same_negative, -magnitude, 0.0))


def _median(left: Array, center: Array, right: Array, /) -> Array:
    return left + _minmod(center - left, right - left)


def _mp5_left(window: Array, /, *, alpha: float = 4.0) -> Array:
    um2, um1, center, up1, up2 = (window[:, index] for index in range(5))
    unlimited = (2.0 * um2 - 13.0 * um1 + 47.0 * center + 27.0 * up1 - 3.0 * up2) / 60.0
    monotone = center + _minmod(up1 - center, alpha * (center - um1))
    accepted = (unlimited - center) * (unlimited - monotone) <= 1e-14
    second_minus = um2 - 2.0 * um1 + center
    second_center = um1 - 2.0 * center + up1
    second_plus = center - 2.0 * up1 + up2
    second_face = _minmod(
        4.0 * second_center - second_plus,
        4.0 * second_plus - second_center,
        second_center,
        second_plus,
    )
    upper_linear = center + alpha * (center - um1)
    average = 0.5 * (center + up1)
    median_derivative = average - 0.5 * second_face
    limited_center = (
        center
        + 0.5 * (center - um1)
        + 4.0
        / 3.0
        * _minmod(
            second_minus,
            second_center,
        )
    )
    minimum = jnp.maximum(
        jnp.minimum(jnp.minimum(center, up1), median_derivative),
        jnp.minimum(jnp.minimum(center, upper_linear), limited_center),
    )
    maximum = jnp.minimum(
        jnp.maximum(jnp.maximum(center, up1), median_derivative),
        jnp.maximum(jnp.maximum(center, upper_linear), limited_center),
    )
    limited = _median(minimum, unlimited, maximum)
    return jnp.where(accepted, unlimited, limited)


class HighResolutionReconstructionPlan(StrictModule, NonTrainableState):
    """Uniform periodic/outflow WENO-Z, TENO, or MP5 face reconstruction."""

    method: HighResolutionMethod = eqx.field(static=True)
    boundary: ReconstructionBoundary = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    power: int = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: HighResolutionMethod = "weno_z",
        /,
        *,
        boundary: ReconstructionBoundary = "periodic",
        epsilon: float = 1e-12,
        power: int = 2,
        cutoff: float = 1e-6,
    ):
        epsilon_ = float(epsilon)
        cutoff_ = float(cutoff)
        power_ = int(power)
        if method not in ("weno_z", "teno", "mp5") or boundary not in (
            "periodic",
            "outflow",
        ):
            raise ValueError("Unknown high-resolution method or boundary policy.")
        if (
            not np.isfinite(epsilon_)
            or epsilon_ <= 0.0
            or power_ <= 0
            or not np.isfinite(cutoff_)
            or cutoff_ <= 0.0
            or cutoff_ >= 1.0
        ):
            raise ValueError("High-resolution epsilon, power, or cutoff is invalid.")
        self.method = method
        self.boundary = boundary
        self.epsilon = epsilon_
        self.power = power_
        self.cutoff = cutoff_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "high-resolution-reconstruction",
                "method": method,
                "boundary": boundary,
                "epsilon": epsilon_,
                "power": power_,
                "cutoff": cutoff_,
            }
        )

    @property
    def radius(self) -> int:
        return 3

    def _left(self, windows: Array, /) -> Array:
        if self.method == "mp5":
            return _mp5_left(windows)
        candidates, smoothness = _weno_candidates(windows)
        return _nonlinear_combination(
            candidates,
            smoothness,
            self.method,
            self.epsilon,
            self.power,
            self.cutoff,
        )

    def reconstruct(self, values: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(values)
        if value.ndim < 1 or value.shape[0] < 6:
            raise ValueError(
                "High-resolution reconstruction requires at least six cells."
            )
        left_windows = _uniform_windows(
            value,
            (-2, -1, 0, 1, 2),
            self.boundary,
        )
        right_windows = _uniform_windows(
            value,
            (3, 2, 1, 0, -1),
            self.boundary,
        )
        return self._left(left_windows), self._left(right_windows)


class CharacteristicSystem(StrictModule):
    """Face-local left/right eigenvectors and wave speeds for one hyperbolic system."""

    eigensystem: Callable[[Array, Array, Any], tuple[Array, Array, Array]] = eqx.field(
        static=True
    )
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        eigensystem: Callable[[Array, Array, Any], tuple[Array, Array, Array]],
        /,
        *,
        system_id: str | None = None,
    ):
        if not callable(eigensystem):
            raise TypeError("Characteristic eigensystem must be callable.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "characteristic-system",
                    "eigensystem": repr(eigensystem),
                }
            )
            if system_id is None
            else str(system_id)
        )
        if not identifier:
            raise ValueError("system_id must be non-empty.")
        self.eigensystem = eigensystem
        self.system_id = identifier


class CharacteristicReconstructionPlan(StrictModule):
    """Face-local characteristic projection before nonlinear reconstruction."""

    reconstruction: HighResolutionReconstructionPlan
    system: CharacteristicSystem
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: HighResolutionReconstructionPlan,
        system: CharacteristicSystem,
        /,
    ):
        if not isinstance(
            reconstruction, HighResolutionReconstructionPlan
        ) or not isinstance(system, CharacteristicSystem):
            raise TypeError(
                "Characteristic reconstruction requires reconstruction/system plans."
            )
        self.reconstruction = reconstruction
        self.system = system
        self.plan_id = canonical_fingerprint(
            {
                "kind": "characteristic-reconstruction",
                "reconstruction": reconstruction.plan_id,
                "system": system.system_id,
            }
        )

    def reconstruct(
        self,
        values: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array]:
        value = jnp.asarray(values)
        if value.ndim != 2 or value.shape[0] < 6:
            raise ValueError("Characteristic state must have shape (cells, components).")
        adjacent = (
            jnp.roll(value, -1, axis=0)
            if self.reconstruction.boundary == "periodic"
            else jnp.concatenate((value[1:], value[-1:]), axis=0)
        )
        left_matrix, right_matrix, wave_speeds = self.system.eigensystem(
            value,
            adjacent,
            args,
        )
        expected = (value.shape[0], value.shape[1], value.shape[1])
        if left_matrix.shape != expected or right_matrix.shape != expected:
            raise ValueError("Characteristic eigenvector matrices have wrong shape.")
        if wave_speeds.shape != value.shape:
            raise ValueError("Characteristic wave speeds must match state components.")
        left_windows = _uniform_windows(
            value,
            (-2, -1, 0, 1, 2),
            self.reconstruction.boundary,
        )
        right_windows = _uniform_windows(
            value,
            (3, 2, 1, 0, -1),
            self.reconstruction.boundary,
        )
        left_characteristic = jnp.einsum("nij,nsj->nsi", left_matrix, left_windows)
        right_characteristic = jnp.einsum("nij,nsj->nsi", left_matrix, right_windows)
        left_reconstructed = self.reconstruction._left(left_characteristic)
        right_reconstructed = self.reconstruction._left(right_characteristic)
        return (
            jnp.einsum("nij,nj->ni", right_matrix, left_reconstructed),
            jnp.einsum("nij,nj->ni", right_matrix, right_reconstructed),
            wave_speeds,
        )


class NonuniformWENOReconstructionPlan(StrictModule, NonTrainableState):
    """Periodic fifth-order WENO-Z/TENO prepared from nonuniform cell edges."""

    cell_edges: Array
    method: Literal["weno_z", "teno"] = eqx.field(static=True)
    left_indices: Array
    right_indices: Array
    left_coefficients: Array
    right_coefficients: Array
    left_smoothness: Array
    right_smoothness: Array
    left_optimal: Array
    right_optimal: Array
    epsilon: float = eqx.field(static=True)
    power: int = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_edges: ArrayLike,
        /,
        *,
        method: Literal["weno_z", "teno"] = "weno_z",
        epsilon: float = 1e-12,
        power: int = 2,
        cutoff: float = 1e-6,
    ):
        edges = np.asarray(cell_edges, dtype=float).reshape((-1,))
        if (
            edges.size < 7
            or np.any(~np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
            or method not in ("weno_z", "teno")
        ):
            raise ValueError("Nonuniform WENO requires at least six positive cells.")
        left = _prepare_nonuniform_side(edges, "left")
        right = _prepare_nonuniform_side(edges, "right")
        self.cell_edges = jnp.asarray(edges)
        self.method = method
        self.left_indices = jnp.asarray(left[0])
        self.right_indices = jnp.asarray(right[0])
        self.left_coefficients = jnp.asarray(left[1])
        self.right_coefficients = jnp.asarray(right[1])
        self.left_smoothness = jnp.asarray(left[2])
        self.right_smoothness = jnp.asarray(right[2])
        self.left_optimal = jnp.asarray(left[3])
        self.right_optimal = jnp.asarray(right[3])
        self.epsilon = float(epsilon)
        self.power = int(power)
        self.cutoff = float(cutoff)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nonuniform-weno-reconstruction",
                "edges": array_tree_fingerprint(edges),
                "method": method,
                "epsilon": float(epsilon),
                "power": int(power),
                "cutoff": float(cutoff),
            }
        )

    def _side(
        self,
        values: Array,
        indices: Array,
        coefficients: Array,
        smoothness_matrices: Array,
        optimal: Array,
        /,
    ) -> Array:
        windows = values[indices]
        candidates = jnp.einsum("nkj,nkj...->nk...", coefficients, windows)
        smoothness = jnp.einsum(
            "nkij,nki...,nkj...->nk...",
            smoothness_matrices,
            windows,
            windows,
        )
        optimal_shape = optimal.shape + (1,) * (smoothness.ndim - 2)
        return _nonlinear_combination(
            candidates,
            smoothness,
            self.method,
            self.epsilon,
            self.power,
            self.cutoff,
            optimal=optimal.reshape(optimal_shape),
        )

    def reconstruct(self, values: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(values)
        if value.shape[0] != self.cell_edges.size - 1:
            raise ValueError("Nonuniform WENO values must match prepared cell count.")
        return (
            self._side(
                value,
                self.left_indices,
                self.left_coefficients,
                self.left_smoothness,
                self.left_optimal,
            ),
            self._side(
                value,
                self.right_indices,
                self.right_coefficients,
                self.right_smoothness,
                self.right_optimal,
            ),
        )


def _logical_cell_bounds(edges: np.ndarray, index: int, /) -> tuple[float, float]:
    count = edges.size - 1
    period = edges[-1] - edges[0]
    quotient, remainder = divmod(index, count)
    return edges[remainder] + quotient * period, edges[remainder + 1] + quotient * period


def _polynomial_data(
    edges: np.ndarray,
    logical_indices: tuple[int, ...],
    target_cell: int,
    face_coordinate: float,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    degree = len(logical_indices) - 1
    average_matrix = np.zeros((degree + 1, degree + 1))
    for row, index in enumerate(logical_indices):
        lower, upper = _logical_cell_bounds(edges, index)
        width = upper - lower
        average_matrix[row] = [
            (upper ** (power + 1) - lower ** (power + 1)) / ((power + 1) * width)
            for power in range(degree + 1)
        ]
    inverse = np.linalg.inv(average_matrix)
    evaluation = (
        np.asarray([face_coordinate**power for power in range(degree + 1)]) @ inverse
    )
    target_lower, target_upper = _logical_cell_bounds(edges, target_cell)
    target_width = target_upper - target_lower
    smoothness_coeff = np.zeros((degree + 1, degree + 1))
    for derivative in range(1, degree + 1):
        derivative_matrix = np.zeros((degree + 1, degree + 1))
        for power in range(derivative, degree + 1):
            factor = factorial(power) / factorial(power - derivative)
            derivative_matrix[power - derivative, power] = factor
        integral = np.zeros((degree + 1, degree + 1))
        for left_power in range(degree + 1):
            for right_power in range(degree + 1):
                exponent = left_power + right_power
                integral[left_power, right_power] = (
                    target_upper ** (exponent + 1) - target_lower ** (exponent + 1)
                ) / (exponent + 1)
        smoothness_coeff += (
            target_width ** (2 * derivative - 1)
            * inverse.T
            @ derivative_matrix.T
            @ integral
            @ derivative_matrix
            @ inverse
        )
    return evaluation, smoothness_coeff


def _prepare_nonuniform_side(edges: np.ndarray, side: str, /):
    count = edges.size - 1
    all_indices = []
    coefficients = []
    smoothness = []
    optimal = []
    for face_index in range(count):
        face_coordinate = edges[face_index + 1]
        if side == "left":
            stencils = (
                (face_index - 2, face_index - 1, face_index),
                (face_index - 1, face_index, face_index + 1),
                (face_index, face_index + 1, face_index + 2),
            )
            full = tuple(range(face_index - 2, face_index + 3))
            target_cell = face_index
        else:
            stencils = (
                (face_index + 3, face_index + 2, face_index + 1),
                (face_index + 2, face_index + 1, face_index),
                (face_index + 1, face_index, face_index - 1),
            )
            full = tuple(range(face_index + 3, face_index - 2, -1))
            target_cell = face_index + 1
        candidate_coefficients = []
        candidate_smoothness = []
        embedded = np.zeros((3, 5))
        for stencil_index, stencil in enumerate(stencils):
            coefficient, beta = _polynomial_data(
                edges,
                stencil,
                target_cell,
                face_coordinate,
            )
            candidate_coefficients.append(coefficient)
            candidate_smoothness.append(beta)
            for local, logical in enumerate(stencil):
                embedded[stencil_index, full.index(logical)] = coefficient[local]
        full_coefficient, _ = _polynomial_data(
            edges,
            full,
            target_cell,
            face_coordinate,
        )
        weights, _, _, _ = np.linalg.lstsq(embedded.T, full_coefficient, rcond=None)
        if np.any(weights < -1e-10):
            raise ValueError("Nonuniform WENO geometry has non-positive optimal weights.")
        weights = np.maximum(weights, 0.0)
        weights /= np.sum(weights)
        all_indices.append([[value % count for value in stencil] for stencil in stencils])
        coefficients.append(candidate_coefficients)
        smoothness.append(candidate_smoothness)
        optimal.append(weights)
    return (
        np.asarray(all_indices, dtype=np.int32),
        np.asarray(coefficients),
        np.asarray(smoothness),
        np.asarray(optimal),
    )


__all__ = [
    "CharacteristicReconstructionPlan",
    "CharacteristicSystem",
    "HighResolutionMethod",
    "HighResolutionReconstructionPlan",
    "NonuniformWENOReconstructionPlan",
    "ReconstructionBoundary",
]
