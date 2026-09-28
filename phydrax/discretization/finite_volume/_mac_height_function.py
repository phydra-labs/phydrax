#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Height-function curvature and interface position on structured grids.

For an interface cell the column axis is ranked by the magnitude of the PLIC
normal.  Heights are sums of the volume fraction over ``2 w + 1`` cells along
the axis for the ``3^(D-1)`` neighboring columns (Cummins, Francois & Kothe
2005; Popinet 2009). A column is complete when its liquid end is full, its
opposite end is empty, and the samples between them form one monotone
full-to-empty interface band. Complete columns give a second-order curvature
from centered height derivatives and the interface crossing of the central
column. Other axes are tried when the ranked axis is incomplete.

Mixed cells without complete columns use a fixed-capacity, radius-bounded
weighted least-squares quadratic graph through primary valid PLIC facet
centroids (``FALLBACK``).  Samples are expressed in the tangent frame of the
target PLIC normal and screened by normal alignment.  Rank, condition,
support, residual, and bounded-stencil exhaustion remain explicit evidence;
fallback curvatures never become fit samples.  Every remaining interface-band
cell, and any primary estimate whose curvature exceeds the declared
resolution bound, is ``UNDERRESOLVED`` and is refused by status-driven
capillary actions.

Sign convention: curvature is ``div(n)`` for the normal out of the alpha
phase, i.e. positive for a convex alpha region (see ``_capillarity``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    HermitianSpectrum,
    prepare_local_block_factorization,
    solve_local_blocks_detailed,
)
from ._capillarity import CurvatureEvidence, CurvatureStatus
from ._structured_plic import StructuredPLICPlan, StructuredPLICReconstruction


class HeightFunctionCurvatureResult(StrictModule):
    """Height-function curvature evidence and interface positions.

    ``interface_position`` is the central-column interface crossing on
    height-function cells, the exact PLIC facet centroid on other mixed cells,
    and the fitted graph crossing on a pure interface-band fallback;
    ``position_valid`` marks where it is defined.  ``column_axis`` is the
    height axis of valid height-function cells and ``-1`` elsewhere. Per-cell
    fallback fields retain the attempted fit's support, numerical rank,
    condition, residual, candidate curvature, and bounded-resource refusal
    independently of the usable curvature field.

    """

    evidence: CurvatureEvidence
    interface_position: Array
    position_valid: Array
    column_axis: Array
    valid_count: Array
    fallback_count: Array
    underresolved_count: Array
    fallback_support_count: Array
    fallback_rank: Array
    fallback_condition: Array
    fallback_residual: Array
    fallback_candidate_curvature: Array
    fallback_attempted: Array
    fallback_resource_limited: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


class HeightFunctionCurvaturePlan(StrictModule, NonTrainableState):
    """Structured height-function curvature with explicit fallback evidence.

    Uniform spacing on every axis and at least ``2 * half_width + 1`` cells
    per axis are required.  The quadratic-fit radius is bounded by
    ``half_width``.  Its fixed capacity, normal-alignment threshold, residual
    threshold, rank cutoff, and condition limit are part of ``plan_id``.
    """

    plic: StructuredPLICPlan
    spacing: tuple[float, ...] = eqx.field(static=True)
    half_width: int = eqx.field(static=True)
    curvature_bound: float = eqx.field(static=True)
    fallback_radius: int = eqx.field(static=True)
    fallback_condition_limit: float = eqx.field(static=True)
    fallback_normal_alignment: float = eqx.field(static=True)
    fallback_residual_limit: float = eqx.field(static=True)
    fallback_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        plic: StructuredPLICPlan,
        /,
        *,
        half_width: int = 3,
        curvature_bound: float = 1.0,
        fallback_radius: int = 3,
        fallback_condition_limit: float = 1.0e6,
        fallback_normal_alignment: float = 0.5,
        fallback_residual_limit: float = 0.35,
    ) -> None:
        if not isinstance(plic, StructuredPLICPlan):
            raise TypeError("plic must be StructuredPLICPlan.")
        width = int(half_width)
        bound = float(curvature_bound)
        radius = int(fallback_radius)
        condition_limit = float(fallback_condition_limit)
        normal_alignment = float(fallback_normal_alignment)
        residual_limit = float(fallback_residual_limit)
        if width < 1:
            raise ValueError("half_width must be at least one.")
        if not np.isfinite(bound) or bound <= 0.0:
            raise ValueError("curvature_bound must be finite and positive.")
        if not 1 <= radius <= width:
            raise ValueError("fallback_radius must lie in [1, half_width].")
        if not np.isfinite(condition_limit) or condition_limit <= 1.0:
            raise ValueError(
                "fallback_condition_limit must be finite and greater than one."
            )
        if not np.isfinite(normal_alignment) or not 0.0 <= normal_alignment < 1.0:
            raise ValueError(
                "fallback_normal_alignment must be finite and lie in [0, 1)."
            )
        if not np.isfinite(residual_limit) or residual_limit <= 0.0:
            raise ValueError("fallback_residual_limit must be finite and positive.")
        discretization = plic.discretization
        dtype_epsilon = float(np.finfo(np.dtype(discretization.cell_volumes.dtype)).eps)
        effective_condition_limit = min(
            condition_limit,
            float(np.sqrt(0.1 / dtype_epsilon)),
        )
        spacing = []
        for axis, grid_axis in enumerate(discretization.grid.structured_axes):
            widths = np.asarray(grid_axis.interval_widths, dtype=np.float64)
            if not np.allclose(widths, widths[0], rtol=1.0e-10, atol=0.0):
                raise ValueError("Height functions require uniform spacing per axis.")
            if discretization.cell_shape[axis] < 2 * width + 1:
                raise ValueError(
                    "Height-function columns need at least 2 * half_width + 1 cells "
                    f"on axis {axis}; the stencil support is unavailable."
                )
            spacing.append(float(widths[0]))
        capacity = (2 * radius + 1) ** plic.dimension
        self.plic = plic
        self.spacing = tuple(spacing)
        self.half_width = width
        self.curvature_bound = bound
        self.fallback_radius = radius
        self.fallback_condition_limit = effective_condition_limit
        self.fallback_normal_alignment = normal_alignment
        self.fallback_residual_limit = residual_limit
        self.fallback_capacity = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "height-function-curvature-plan",
                "plic": plic.plan_id,
                "half_width": width,
                "curvature_bound": bound,
                "interface_delta": "plic-facet-measure-per-volume-then-youngs-norm",
                "fallback": {
                    "method": "primary-plic-centroid-weighted-local-quadratic",
                    "radius": radius,
                    "capacity": capacity,
                    "normal_alignment": normal_alignment,
                    "distance_weight_scale_cells": radius,
                    "residual_limit": residual_limit,
                    "gram_rank_relative_cutoff": (1.0 / effective_condition_limit**2),
                    "requested_condition_limit": condition_limit,
                    "effective_condition_limit": effective_condition_limit,
                    "dtype_condition_safety": "sqrt(0.1 / eps)",
                },
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.spacing)

    def _pad_constant(self, value: Array, width: int, fill: Array, /) -> Array:
        padded = value
        for axis, grid_axis in enumerate(self.plic.discretization.grid.structured_axes):
            pad_width = [(0, 0)] * padded.ndim
            pad_width[axis] = (width, width)
            if grid_axis.periodic:
                padded = jnp.pad(padded, pad_width, mode="wrap")
            else:
                padded = jnp.pad(padded, pad_width, mode="constant", constant_values=fill)
        return padded

    def _view(self, padded: Array, width: int, offset: tuple[int, ...], /) -> Array:
        shape = self.plic.discretization.cell_shape
        return padded[
            tuple(
                slice(width + delta, width + delta + size)
                for delta, size in zip(offset, shape, strict=True)
            )
        ]

    def interface_band(self, alpha: ArrayLike, /) -> Array:
        """Return mixed cells and cells adjacent to a represented phase jump."""

        value = jnp.asarray(alpha)
        band = self.plic.mixed_mask(value)
        padded = self.plic.pad(value, 1)
        for axis in range(self.dimension):
            for step in (-1, 1):
                offset = tuple(
                    step if other == axis else 0 for other in range(self.dimension)
                )
                band = band | (
                    jnp.abs(self._view(padded, 1, offset) - value)
                    > self.plic.mixed_tolerance
                )
        return band

    def _column_heights(
        self, padded: Array, axis: int, orientation: Array, /
    ) -> tuple[dict[tuple[int, ...], Array], Array]:
        """Interface offsets (physical) of the neighboring columns of ``axis``.

        ``orientation`` is ``+1`` when the alpha phase lies below the interface
        along ``axis`` and ``-1`` otherwise. A complete column contains exactly
        one monotone transition from full to empty in that orientation.
        """

        width = self.half_width
        tolerance = self.plic.mixed_tolerance
        tangential = [other for other in range(self.dimension) if other != axis]
        heights: dict[tuple[int, ...], Array] = {}
        complete = jnp.ones(self.plic.discretization.cell_shape, dtype=jnp.bool_)
        for shift in np.ndindex(*((3,) * len(tangential))):
            offset = [0] * self.dimension
            for other, delta in zip(tangential, shift, strict=True):
                offset[other] = delta - 1
            column = []
            for level in range(-width, width + 1):
                offset[axis] = level
                column.append(self._view(padded, width, tuple(offset)))
            total = sum(column[1:], start=column[0])
            lower_end = column[0]
            upper_end = column[-1]
            full_below = (lower_end >= 1.0 - tolerance) & (upper_end <= tolerance)
            full_above = (upper_end >= 1.0 - tolerance) & (lower_end <= tolerance)
            single_transition = (
                orientation * (column[1] - column[0]) <= tolerance
            )
            for level in range(1, 2 * width):
                phase_step = orientation * (column[level + 1] - column[level])
                single_transition = single_transition & (phase_step <= tolerance)
            complete = complete & single_transition & jnp.where(
                orientation > 0.0, full_below, full_above
            )
            key = tuple(delta - 1 for delta in shift)
            heights[key] = orientation * (total - (width + 0.5)) * self.spacing[axis]
        return heights, complete

    def _height_curvature(
        self,
        heights: dict[tuple[int, ...], Array],
        axis: int,
        orientation: Array,
        /,
    ) -> Array:
        tangential = [other for other in range(self.dimension) if other != axis]
        if self.dimension == 2:
            spacing = self.spacing[tangential[0]]
            slope = (heights[(1,)] - heights[(-1,)]) / (2.0 * spacing)
            second = (heights[(1,)] - 2.0 * heights[(0,)] + heights[(-1,)]) / spacing**2
            return -orientation * second / (1.0 + slope**2) ** 1.5
        first_spacing = self.spacing[tangential[0]]
        second_spacing = self.spacing[tangential[1]]
        hx = (heights[(1, 0)] - heights[(-1, 0)]) / (2.0 * first_spacing)
        hy = (heights[(0, 1)] - heights[(0, -1)]) / (2.0 * second_spacing)
        hxx = (
            heights[(1, 0)] - 2.0 * heights[(0, 0)] + heights[(-1, 0)]
        ) / first_spacing**2
        hyy = (
            heights[(0, 1)] - 2.0 * heights[(0, 0)] + heights[(0, -1)]
        ) / second_spacing**2
        hxy = (
            heights[(1, 1)] - heights[(1, -1)] - heights[(-1, 1)] + heights[(-1, -1)]
        ) / (4.0 * first_spacing * second_spacing)
        numerator = hxx * (1.0 + hy**2) + hyy * (1.0 + hx**2) - 2.0 * hxy * hx * hy
        return -orientation * numerator / (1.0 + hx**2 + hy**2) ** 1.5

    def _height_function(
        self, alpha: Array, normal: Array, /
    ) -> tuple[Array, Array, Array, Array]:
        """Return ranked height-function curvature, validity, axis, and offset."""

        padded = self.plic.pad(alpha, self.half_width)
        curvatures = []
        valids = []
        offsets = []
        for axis in range(self.dimension):
            orientation = jnp.where(normal[..., axis] >= 0.0, 1.0, -1.0).astype(
                alpha.dtype
            )
            heights, complete = self._column_heights(padded, axis, orientation)
            kappa = self._height_curvature(heights, axis, orientation)
            center = (0,) * (self.dimension - 1)
            curvatures.append(kappa)
            valids.append(complete & jnp.isfinite(kappa))
            offsets.append(heights[center])
        curvature_stack = jnp.stack(curvatures, axis=-1)
        valid_stack = jnp.stack(valids, axis=-1)
        offset_stack = jnp.stack(offsets, axis=-1)
        order = jnp.argsort(-jnp.abs(normal), axis=-1)
        ranked_valid = jnp.take_along_axis(valid_stack, order, axis=-1)
        first = jnp.argmax(ranked_valid, axis=-1)
        chosen_axis = jnp.take_along_axis(order, first[..., None], axis=-1)[..., 0]
        valid = jnp.any(valid_stack, axis=-1)
        curvature = jnp.take_along_axis(curvature_stack, chosen_axis[..., None], axis=-1)[
            ..., 0
        ]
        offset = jnp.take_along_axis(offset_stack, chosen_axis[..., None], axis=-1)[
            ..., 0
        ]
        return curvature, valid, chosen_axis, offset

    def _tangent_frame(self, normal: Array, /) -> tuple[Array, ...]:
        """Return a deterministic orthonormal tangent frame for unit normals."""

        helper_axis = jnp.argmin(jnp.abs(normal), axis=-1)
        helper = jax.nn.one_hot(helper_axis, self.dimension, dtype=normal.dtype)
        first = helper - jnp.sum(helper * normal, axis=-1, keepdims=True) * normal
        first = first / jnp.sqrt(jnp.sum(first**2, axis=-1, keepdims=True))
        if self.dimension == 2:
            return (first,)
        return first, jnp.cross(normal, first)

    def _quadratic_basis(self, coordinates: tuple[Array, ...], /) -> Array:
        """Return the dimension-specific complete quadratic graph basis."""

        if self.dimension == 2:
            (u,) = coordinates
            return jnp.stack((jnp.ones_like(u), u, u * u), axis=-1)
        u, v = coordinates
        return jnp.stack(
            (jnp.ones_like(u), u, v, u * u, u * v, v * v),
            axis=-1,
        )

    def _paraboloid_curvature(
        self,
        reconstruction: StructuredPLICReconstruction,
        attempted: Array,
        /,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        """Fit local PLIC facet centroids with rank and condition evidence."""

        discretization = self.plic.discretization
        shape = discretization.cell_shape
        dtype = discretization.cell_volumes.dtype
        scale = jnp.asarray(min(self.spacing), dtype=dtype)
        radius = self.fallback_radius
        coordinate_scale = jnp.asarray(float(radius), dtype=dtype)
        displacement = jnp.where(
            reconstruction.facet_valid[..., None],
            reconstruction.interface_point - discretization.cell_centers,
            jnp.zeros((), dtype=dtype),
        )
        padded_displacement = self._pad_constant(
            displacement, radius, jnp.zeros((), dtype=dtype)
        )
        padded_normal = self._pad_constant(
            reconstruction.normal, radius, jnp.zeros((), dtype=dtype)
        )
        padded_measure = self._pad_constant(
            reconstruction.facet_measure, radius, jnp.zeros((), dtype=dtype)
        )
        padded_valid = self._pad_constant(
            reconstruction.facet_valid,
            radius,
            jnp.zeros((), dtype=jnp.bool_),
        )
        target_normal = reconstruction.normal
        tangents = self._tangent_frame(target_normal)
        parameter_count = 3 if self.dimension == 2 else 6
        gram = jnp.zeros((*shape, parameter_count, parameter_count), dtype=dtype)
        moment = jnp.zeros((*shape, parameter_count), dtype=dtype)
        support_count = jnp.zeros(shape, dtype=jnp.int32)
        weight_sum = jnp.zeros(shape, dtype=dtype)
        samples = []
        for index in np.ndindex(*((2 * radius + 1,) * self.dimension)):
            offset = tuple(entry - radius for entry in index)
            center_offset = jnp.asarray(
                [
                    delta * width
                    for delta, width in zip(offset, self.spacing, strict=True)
                ],
                dtype=dtype,
            )
            relative = (
                self._view(padded_displacement, radius, offset) + center_offset
            ) / scale
            sample_normal = self._view(padded_normal, radius, offset)
            alignment = jnp.sum(sample_normal * target_normal, axis=-1)
            sample_valid = self._view(padded_valid, radius, offset)
            used = (
                sample_valid
                & jnp.isfinite(alignment)
                & (alignment >= self.fallback_normal_alignment)
            )
            coordinates = tuple(
                jnp.sum(relative * tangent, axis=-1) / coordinate_scale
                for tangent in tangents
            )
            height = jnp.sum(relative * target_normal, axis=-1)
            basis = self._quadratic_basis(coordinates)
            distance_square = jnp.sum(relative**2, axis=-1)
            facet_weight = self._view(padded_measure, radius, offset) / (
                scale ** (self.dimension - 1)
            )
            weight = jnp.where(
                used,
                facet_weight
                * jnp.exp(-0.5 * distance_square / coordinate_scale**2)
                * alignment**2,
                jnp.zeros((), dtype=dtype),
            )
            weighted_basis = weight[..., None] * basis
            gram = gram + weighted_basis[..., :, None] * basis[..., None, :]
            moment = moment + weighted_basis * height[..., None]
            support_count = support_count + used.astype(jnp.int32)
            weight_sum = weight_sum + weight
            samples.append((basis, height, weight))
        supported = support_count >= parameter_count
        solve_required = attempted & supported
        identity = jnp.eye(parameter_count, dtype=dtype)
        screened_gram = jnp.where(
            solve_required[..., None, None],
            gram,
            identity,
        )
        spectrum = HermitianSpectrum(
            screened_gram,
            tolerance=1.0 / self.fallback_condition_limit**2,
        )
        rank = jnp.where(
            solve_required,
            spectrum.numerical_rank,
            jnp.zeros((), dtype=jnp.int32),
        )
        condition = jnp.where(
            solve_required,
            jnp.sqrt(spectrum.condition_number),
            jnp.inf,
        )
        system_valid = (
            solve_required
            & spectrum.valid
            & (rank == parameter_count)
            & jnp.isfinite(condition)
            & (condition <= self.fallback_condition_limit)
        )
        factored_gram = jnp.where(
            system_valid[..., None, None],
            gram,
            identity,
        )
        factorization = prepare_local_block_factorization(
            factored_gram.reshape((-1, parameter_count, parameter_count)),
            positive_definite=True,
        )
        solved = solve_local_blocks_detailed(
            factorization,
            jnp.where(
                system_valid[..., None],
                moment,
                jnp.zeros((), dtype=dtype),
            ).reshape((-1, parameter_count)),
        )
        coefficients = solved.value.reshape((*shape, parameter_count))
        solve_failed = solved.failed_blocks.reshape(shape)
        residual_square = jnp.zeros(shape, dtype=dtype)
        for basis, height, weight in samples:
            error = jnp.sum(basis * coefficients, axis=-1) - height
            residual_square = residual_square + weight * error**2
        residual_dimensionless = jnp.sqrt(
            residual_square / jnp.maximum(weight_sum, jnp.finfo(dtype).tiny)
        )
        residual = jnp.where(
            attempted & supported,
            scale * residual_dimensionless,
            jnp.where(attempted, jnp.inf, jnp.zeros((), dtype=dtype)),
        )
        if self.dimension == 2:
            slope = coefficients[..., 1] / coordinate_scale
            second = 2.0 * coefficients[..., 2] / (coordinate_scale**2 * scale)
            curvature = -second / (1.0 + slope**2) ** 1.5
        else:
            hx = coefficients[..., 1] / coordinate_scale
            hy = coefficients[..., 2] / coordinate_scale
            hxx = 2.0 * coefficients[..., 3] / (coordinate_scale**2 * scale)
            hxy = coefficients[..., 4] / (coordinate_scale**2 * scale)
            hyy = 2.0 * coefficients[..., 5] / (coordinate_scale**2 * scale)
            numerator = hxx * (1.0 + hy**2) + hyy * (1.0 + hx**2) - 2.0 * hxy * hx * hy
            curvature = -numerator / (1.0 + hx**2 + hy**2) ** 1.5
        valid = (
            system_valid
            & ~solve_failed
            & jnp.isfinite(residual_dimensionless)
            & (residual_dimensionless <= self.fallback_residual_limit)
            & jnp.isfinite(curvature)
        )
        fit_offset = scale * coefficients[..., 0]
        return (
            curvature,
            valid,
            residual,
            support_count,
            rank,
            condition,
            fit_offset,
        )

    def evaluate(
        self,
        alpha: ArrayLike,
        reconstruction: StructuredPLICReconstruction,
        /,
    ) -> HeightFunctionCurvatureResult:
        """Return curvature evidence and interface positions for ``alpha``."""

        value = jnp.asarray(alpha)
        discretization = self.plic.discretization
        if value.shape != discretization.cell_shape:
            raise ValueError("Height-function alpha must match the cell shape.")
        if not isinstance(reconstruction, StructuredPLICReconstruction):
            raise TypeError("reconstruction must be StructuredPLICReconstruction.")
        if reconstruction.plan_id != self.plic.plan_id:
            raise ValueError("PLIC reconstruction belongs to another PLIC plan.")
        band = self.interface_band(value)
        height_curvature, height_valid, column_axis, column_offset = (
            self._height_function(value, reconstruction.normal)
        )
        bound = self.curvature_bound / min(self.spacing)
        height_valid = band & height_valid & (jnp.abs(height_curvature) <= bound)
        fallback_attempted = band & ~height_valid
        (
            fit_curvature,
            fit_valid,
            fit_residual,
            fit_support,
            fit_rank,
            fit_condition,
            fit_offset,
        ) = self._paraboloid_curvature(reconstruction, fallback_attempted)
        fit_valid = fit_valid & (jnp.abs(fit_curvature) <= bound)
        fallback_resource_limited = fallback_attempted & ~fit_valid
        curvature = jnp.where(
            height_valid,
            height_curvature,
            jnp.where(fit_valid, fit_curvature, 0.0),
        )
        status = jnp.where(
            ~band,
            int(CurvatureStatus.MISSING_INTERFACE),
            jnp.where(
                height_valid,
                int(CurvatureStatus.VALID),
                jnp.where(
                    fit_valid,
                    int(CurvatureStatus.FALLBACK),
                    int(CurvatureStatus.UNDERRESOLVED),
                ),
            ),
        ).astype(jnp.int8)
        curvature = jnp.where(band & (height_valid | fit_valid), curvature, 0.0)
        residual = jnp.where(
            fallback_attempted,
            fit_residual,
            jnp.zeros((), dtype=value.dtype),
        )
        unit_axis = jax.nn.one_hot(column_axis, self.dimension, dtype=value.dtype)
        height_position = (
            discretization.cell_centers + column_offset[..., None] * unit_axis
        )
        fit_position = (
            discretization.cell_centers + fit_offset[..., None] * reconstruction.normal
        )
        position_valid = height_valid | reconstruction.facet_valid | fit_valid
        position = jnp.where(
            height_valid[..., None],
            height_position,
            jnp.where(
                reconstruction.facet_valid[..., None],
                reconstruction.interface_point,
                jnp.where(
                    fit_valid[..., None],
                    fit_position,
                    discretization.cell_centers,
                ),
            ),
        )
        interface_delta, interface_delta_supported = self.plic.interface_delta(
            value, reconstruction
        )
        evidence = CurvatureEvidence(
            curvature,
            residual,
            status,
            interface_active=band,
            interface_delta=interface_delta,
            interface_delta_supported=interface_delta_supported,
            geometry_id=discretization.prepared_id,
            reconstruction_id=self.plic.plan_id,
            evidence_id=canonical_fingerprint(
                {"kind": "height-function-curvature-evidence", "plan": self.plan_id}
            ),
            tolerance=float(min(self.spacing)),
        )
        finite = jnp.all(jnp.isfinite(curvature)) & jnp.all(jnp.isfinite(position))
        return HeightFunctionCurvatureResult(
            evidence=evidence,
            interface_position=position,
            position_valid=position_valid,
            column_axis=jnp.where(height_valid, column_axis, -1).astype(jnp.int8),
            valid_count=jnp.sum(height_valid, dtype=jnp.int32),
            fallback_count=jnp.sum(fit_valid, dtype=jnp.int32),
            underresolved_count=jnp.sum(
                status == int(CurvatureStatus.UNDERRESOLVED), dtype=jnp.int32
            ),
            fallback_support_count=jnp.where(
                fallback_attempted,
                fit_support,
                jnp.zeros((), dtype=jnp.int32),
            ),
            fallback_rank=fit_rank,
            fallback_condition=jnp.where(
                fallback_attempted,
                fit_condition,
                jnp.zeros((), dtype=value.dtype),
            ),
            fallback_residual=residual,
            fallback_candidate_curvature=jnp.where(
                fallback_attempted,
                fit_curvature,
                jnp.zeros((), dtype=value.dtype),
            ),
            fallback_attempted=fallback_attempted,
            fallback_resource_limited=fallback_resource_limited,
            finite=finite,
            plan_id=self.plan_id,
        )


__all__ = ["HeightFunctionCurvaturePlan", "HeightFunctionCurvatureResult"]
