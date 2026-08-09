#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import sqrt

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._multiresolution import MultiresolutionCoefficients
from ._wavelet import WaveletBoundary


def _canonical_columns(matrix: np.ndarray, /) -> np.ndarray:
    result = np.asarray(matrix, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _discrete_legendre_analysis(order: int, /) -> np.ndarray:
    nodes = (np.arange(order, dtype=float) + 0.5) / float(order)
    vandermonde = np.polynomial.legendre.legvander(2.0 * nodes - 1.0, order - 1)
    orthogonal, _ = np.linalg.qr(vandermonde)
    return _canonical_columns(orthogonal).T


def _alpert_analysis(order: int, /) -> np.ndarray:
    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(
        max(16, 4 * order)
    )
    low_rows = np.zeros((order, 2 * order), dtype=float)
    for branch in range(2):
        lower = 0.5 * branch
        points = lower + 0.25 * (quadrature_nodes + 1.0)
        weights = 0.25 * quadrature_weights
        coarse_values = np.stack(
            [
                sqrt(2 * degree + 1)
                * np.polynomial.legendre.legval(
                    2.0 * points - 1.0,
                    [0.0] * degree + [1.0],
                )
                for degree in range(order)
            ]
        )
        local_coordinate = 4.0 * points - (1.0 if branch == 0 else 3.0)
        fine_values = np.stack(
            [
                sqrt(2.0)
                * sqrt(2 * degree + 1)
                * np.polynomial.legendre.legval(
                    local_coordinate,
                    [0.0] * degree + [1.0],
                )
                for degree in range(order)
            ]
        )
        low_rows[:, branch * order : (branch + 1) * order] = (
            coarse_values * weights[None, :]
        ) @ fine_values.T
    left, _, right = np.linalg.svd(low_rows, full_matrices=True)
    low_rows = left @ right[:order]
    high_rows = right[order:]
    analysis = np.concatenate((low_rows, high_rows), axis=0)
    for row in range(analysis.shape[0]):
        pivot = int(np.argmax(np.abs(analysis[row])))
        if analysis[row, pivot] < 0.0:
            analysis[row] *= -1.0
    return analysis


class AlpertMultiwaveletTransform(StrictModule, NonTrainableState):
    """Shape-independent one-dimensional Alpert polynomial multiwavelet plan."""

    base_analysis: Array
    base_synthesis: Array
    level_analysis: Array
    level_synthesis: Array
    order: int = eqx.field(static=True)
    levels: int = eqx.field(static=True)
    boundary: WaveletBoundary = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        order: int = 3,
        levels: int = 3,
        boundary: WaveletBoundary = "periodization",
    ):
        order_value = int(order)
        level_count = int(levels)
        if min(order_value, level_count) <= 0:
            raise ValueError("Multiwavelet order and levels must be positive.")
        if boundary not in ("periodization", "symmetric", "zero"):
            raise ValueError(
                "Multiwavelet boundary must be 'periodization', 'symmetric', or 'zero'."
            )
        base = jnp.asarray(_discrete_legendre_analysis(order_value))
        level = jnp.asarray(_alpert_analysis(order_value))
        digest = array_tree_fingerprint((base, level))["sha256"]
        self.base_analysis = base
        self.base_synthesis = base.T
        self.level_analysis = level
        self.level_synthesis = level.T
        self.order = order_value
        self.levels = level_count
        self.boundary = boundary
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "alpert-multiwavelet-transform-v1",
                "order": order_value,
                "levels": level_count,
                "boundary": boundary,
                "matrices": digest,
            }
        )

    def _pad(self, values: Array, padded_points: int, /) -> Array:
        amount = padded_points - int(values.shape[-2])
        if amount == 0:
            return values
        pads = [(0, 0)] * values.ndim
        pads[-2] = (0, amount)
        mode = (
            "wrap"
            if self.boundary == "periodization"
            else "symmetric"
            if self.boundary == "symmetric"
            else "constant"
        )
        return jnp.pad(values, pads, mode=mode)

    def analysis(self, values: ArrayLike, /) -> MultiresolutionCoefficients:
        """Decompose channels-last samples into polynomial scale/detail arrays."""
        array = jnp.asarray(values)
        if array.ndim < 2:
            raise ValueError("Multiwavelet values require point and channel axes.")
        num_points = int(array.shape[-2])
        if num_points <= 1:
            raise ValueError("Multiwavelet point axes must contain at least two samples.")
        multiple = self.order * 2**self.levels
        padded_points = ((num_points + multiple - 1) // multiple) * multiple
        padded = self._pad(array, padded_points)
        cells = padded_points // self.order
        samples = padded.reshape(
            padded.shape[:-2] + (cells, self.order, padded.shape[-1])
        )
        approximation = oe.contract(
            "mp,...cpi->...cmi", self.base_analysis, samples
        )
        details: list[tuple[Array, ...]] = []
        shapes: list[tuple[int, ...]] = []
        for _ in range(self.levels):
            cells = int(approximation.shape[-3])
            if cells % 2:
                raise ValueError("Multiwavelet cell count must be divisible at every level.")
            shapes.append((cells, num_points, padded_points))
            paired = approximation.reshape(
                approximation.shape[:-3]
                + (cells // 2, 2 * self.order, approximation.shape[-1])
            )
            transformed = oe.contract(
                "mn,...pni->...pmi", self.level_analysis, paired
            )
            approximation = transformed[..., : self.order, :]
            details.append((transformed[..., self.order :, :],))
        return MultiresolutionCoefficients(
            approximation,
            tuple(reversed(details)),
            reconstruction_shapes=tuple(reversed(shapes)),
            transform_fingerprint=self.fingerprint,
        )

    def synthesis(self, coefficients: MultiresolutionCoefficients, /) -> Array:
        """Reconstruct channels-last samples and remove transform padding."""
        if not isinstance(coefficients, MultiresolutionCoefficients):
            raise TypeError("coefficients must be MultiresolutionCoefficients.")
        if coefficients.transform_fingerprint != self.fingerprint:
            raise ValueError("Multiwavelet coefficients belong to a different transform.")
        if coefficients.levels != self.levels:
            raise ValueError("Multiwavelet coefficient depth does not match this transform.")
        approximation = jnp.asarray(coefficients.scaling)
        num_points = coefficients.reconstruction_shapes[-1][1]
        padded_points = coefficients.reconstruction_shapes[-1][2]
        for detail_level, shape in zip(
            coefficients.details,
            coefficients.reconstruction_shapes,
            strict=True,
        ):
            if len(detail_level) != 1:
                raise ValueError("Alpert levels require exactly one polynomial detail bank.")
            detail = detail_level[0]
            merged = jnp.concatenate((approximation, detail), axis=-2)
            fine = oe.contract("nm,...pmi->...pni", self.level_synthesis, merged)
            approximation = fine.reshape(
                fine.shape[:-3] + (shape[0], self.order, fine.shape[-1])
            )
        samples = oe.contract(
            "pm,...cmi->...cpi", self.base_synthesis, approximation
        )
        output = samples.reshape(
            samples.shape[:-3] + (padded_points, samples.shape[-1])
        )
        return output[..., :num_points, :]

    def __call__(self, values: ArrayLike, /) -> MultiresolutionCoefficients:
        return self.analysis(values)


__all__ = ["AlpertMultiwaveletTransform"]
