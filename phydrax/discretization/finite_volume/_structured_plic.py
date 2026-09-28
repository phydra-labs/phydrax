#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact piecewise-linear interface reconstruction in rectangular cells.

Each cell uses scaled coordinates ``xi = (x - x_lower) / h`` in ``[0, 1]^D``.
The PLIC facet is the plane ``m . xi = beta`` with ``m_i = n_i h_i`` for the
physical unit normal ``n`` pointing out of the alpha phase, and the alpha
phase occupies ``{m . xi <= beta}``.  The volume fraction of that region is the
closed-form piecewise polynomial of Scardovelli & Zaleski (2000, J. Comput.
Phys. 164:228-237), evaluated here in a cancellation-free form so that nearly
axis-aligned planes keep full relative accuracy.  Two-dimensional cells are the
exact ``m_3 = 0`` reduction of the three-dimensional formula.

The inverse (offset from volume fraction) is analytic on the quadratic,
cubic-root and linear branches.  On the two genuinely cubic branches the
volume is convex on ``[0, 1/2]`` (the cross-section area of a cube is
nondecreasing up to the middle plane), so Newton iteration started at the
upper end of the branch converges monotonically; it is executed by the native
:class:`~phydrax.nonlinear.LocalRootPlan` with implicit-function derivatives,
and its residual is reported as reconstruction evidence.
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
from ...nonlinear import LocalRootPlan
from ._structured import FiniteVolumeDiscretization


def _sorted_unit_normal(scaled_normal: Array, /) -> tuple[Array, Array]:
    """Return ascending ``|m| / sum|m|`` padded to three entries and ``sum|m|``."""

    magnitude = jnp.abs(scaled_normal)
    total = jnp.sum(magnitude, axis=-1)
    safe = jnp.where(total > 0.0, total, 1.0)
    normalized = jnp.sort(magnitude / safe[..., None], axis=-1)
    if normalized.shape[-1] == 2:
        normalized = jnp.concatenate(
            (jnp.zeros_like(normalized[..., :1]), normalized), axis=-1
        )
    return normalized, total


def _half_cube_volume(sorted_normal: Array, offset: Array, /) -> Array:
    """Volume of ``{a . xi <= b}`` in the unit cube for ``0 <= b <= 1/2``.

    ``sorted_normal`` holds ``a1 <= a2 <= a3`` with unit sum.  Every branch is
    written so that each division by a small ``a1`` or ``a2`` multiplies a
    numerator bounded by the same small quantity.
    """

    a1 = sorted_normal[..., 0]
    a2 = sorted_normal[..., 1]
    a3 = sorted_normal[..., 2]
    b = offset
    a12 = a1 + a2
    safe1 = jnp.where(a1 > 0.0, a1, 1.0)
    safe2 = jnp.where(a2 > 0.0, a2, 1.0)
    first = (b / safe1) * (b / safe2) * b / (6.0 * a3)
    second = (3.0 * (b / safe2) * (b - a1) + a1 * (a1 / safe2)) / (6.0 * a3)
    lower_excess = b - a2
    third = second - (lower_excess / safe1) * (lower_excess**2 / safe2) / (6.0 * a3)
    upper_excess = b - a3
    fourth = third - (upper_excess / safe1) * (upper_excess**2 / safe2) / (6.0 * a3)
    linear = (2.0 * b - a12) / (2.0 * a3)
    return jnp.where(
        b < a1,
        first,
        jnp.where(
            b < a2,
            second,
            jnp.where(
                b < jnp.minimum(a12, a3),
                third,
                jnp.where(a12 <= a3, linear, fourth),
            ),
        ),
    )


def _normalized_fraction(sorted_normal: Array, offset: Array, /) -> Array:
    """Volume fraction for a normalized offset ``b`` in ``[0, 1]``."""

    upper = offset > 0.5
    half = _half_cube_volume(sorted_normal, jnp.where(upper, 1.0 - offset, offset))
    return jnp.where(upper, 1.0 - half, half)


def plane_volume_fraction(scaled_normal: ArrayLike, offset: ArrayLike, /) -> Array:
    """Return the exact volume fraction of ``{m . xi <= beta}`` in ``[0, 1]^D``.

    ``scaled_normal`` has a trailing axis of length two or three and may have
    either sign per component; ``offset`` broadcasts against its leading axes.
    A zero normal yields ``1`` for nonnegative offset and ``0`` otherwise.
    """

    m = jnp.asarray(scaled_normal)
    if m.ndim < 1 or m.shape[-1] not in (2, 3):
        raise ValueError("PLIC normals must have a trailing dimension of 2 or 3.")
    beta = jnp.asarray(offset, dtype=m.dtype)
    sorted_normal, total = _sorted_unit_normal(m)
    positive_offset = beta - jnp.sum(jnp.minimum(m, 0.0), axis=-1)
    safe = jnp.where(total > 0.0, total, 1.0)
    # The volume is identically 0 below and 1 above the corner range; the
    # clamp is the exact extension of the polynomial, not a state repair.
    normalized = jnp.clip(positive_offset / safe, 0.0, 1.0)
    fraction = _normalized_fraction(sorted_normal, normalized)
    degenerate = jnp.where(beta >= 0.0, 1.0, 0.0).astype(m.dtype)
    return jnp.where(total > 0.0, fraction, degenerate)


def _offset_guess(sorted_normal: Array, target: Array, /) -> Array:
    """Analytic normalized offset for ``target <= 1/2`` (cubic-branch upper end)."""

    a1 = sorted_normal[..., 0]
    a2 = sorted_normal[..., 1]
    a3 = sorted_normal[..., 2]
    a12 = a1 + a2
    safe2 = jnp.where(a2 > 0.0, a2, 1.0)
    at_a1 = a1 * (a1 / safe2) / (6.0 * a3)
    at_a2 = (3.0 * a2 * (a2 - a1) + a1 * (a1 / safe2)) / (6.0 * a3)
    cubic_end = jnp.minimum(a12, a3)
    at_cubic_end = _half_cube_volume(sorted_normal, cubic_end)
    first = jnp.cbrt(6.0 * a1 * a2 * a3 * target)
    second = 0.5 * a1 + jnp.sqrt(
        jnp.maximum(2.0 * a2 * a3 * target - a1 * a1 / 12.0, 0.0)
    )
    linear = a3 * target + 0.5 * a12
    return jnp.where(
        target < at_a1,
        first,
        jnp.where(
            target < at_a2,
            second,
            jnp.where(
                target < at_cubic_end,
                cubic_end,
                jnp.where(a12 <= a3, linear, 0.5),
            ),
        ),
    )


class StructuredPLICReconstruction(StrictModule):
    """Exact per-cell PLIC planes of one volume-fraction field.

    ``normal`` is the physical unit normal out of the alpha phase and
    ``offset`` the scaled plane constant.  ``facet_measure`` and
    ``interface_point`` are the exact facet length/area and centroid.
    ``facet_valid`` marks mixed cells whose primary PLIC reconstruction is
    finite, converged, and within the volume residual tolerance.
    """

    normal: Array
    scaled_normal: Array
    offset: Array
    mixed: Array
    facet_measure: Array
    facet_valid: Array
    interface_point: Array
    residual: Array
    converged: Array
    finite: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


class StructuredPLICPlan(StrictModule, NonTrainableState):
    """Exact 2-D/3-D PLIC reconstruction and swept-volume geometry on a grid.

    Normals default to :meth:`interface_normal` (centered columns selected
    by Youngs' gradient); callers may pass their own normals (e.g. wall contact-angle
    overrides).  Cells with ``mixed_tolerance < alpha < 1 - mixed_tolerance``
    carry a facet; purer cells transport their own volume fraction exactly.
    """

    discretization: FiniteVolumeDiscretization
    offset_root: LocalRootPlan
    mixed_tolerance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteVolumeDiscretization,
        /,
        *,
        mixed_tolerance: float | None = None,
    ) -> None:
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError("discretization must be FiniteVolumeDiscretization.")
        if len(discretization.cell_shape) not in (2, 3):
            raise ValueError("Structured PLIC supports two or three dimensions.")
        epsilon = float(np.finfo(np.dtype(discretization.cell_volumes.dtype)).eps)
        tolerance = (
            float(np.sqrt(epsilon)) if mixed_tolerance is None else float(mixed_tolerance)
        )
        if not np.isfinite(tolerance) or not 0.0 < tolerance < 0.5:
            raise ValueError("mixed_tolerance must lie in (0, 1/2).")
        residual_tolerance = 256.0 * epsilon
        self.discretization = discretization
        self.mixed_tolerance = tolerance
        self.residual_tolerance = residual_tolerance
        self.offset_root = LocalRootPlan(
            maximum_steps=16,
            tolerance=residual_tolerance,
            minimum_derivative=float(np.finfo(np.float64).tiny),
            plan_id="structured-plic-offset",
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "structured-plic-plan",
                "discretization": discretization.prepared_id,
                "mixed_tolerance": tolerance,
                "volume": "scardovelli-zaleski-2000",
                "normal": "centered-column-youngs-selected",
                "offset_root": self.offset_root.plan_id,
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.discretization.cell_shape)

    def cell_widths(self) -> Array:
        """Return per-cell axis widths with a trailing dimension axis."""

        axes = self.discretization.grid.structured_axes
        dtype = self.discretization.cell_volumes.dtype
        widths = []
        for axis, grid_axis in enumerate(axes):
            shape = [1] * self.dimension
            shape[axis] = self.discretization.cell_shape[axis]
            widths.append(
                jnp.broadcast_to(
                    jnp.asarray(grid_axis.interval_widths, dtype=dtype).reshape(shape),
                    self.discretization.cell_shape,
                )
            )
        return jnp.stack(widths, axis=-1)

    def pad(self, value: ArrayLike, width: int, /) -> Array:
        """Pad cell data: periodic wrap or zero-gradient copy per axis."""

        padded = jnp.asarray(value)
        for axis, grid_axis in enumerate(self.discretization.grid.structured_axes):
            pad_width = [(0, 0)] * padded.ndim
            pad_width[axis] = (width, width)
            padded = jnp.pad(
                padded, pad_width, mode="wrap" if grid_axis.periodic else "edge"
            )
        return padded

    def mixed_mask(self, alpha: ArrayLike, /) -> Array:
        value = jnp.asarray(alpha)
        return (value > self.mixed_tolerance) & (value < 1.0 - self.mixed_tolerance)

    def _shifted_view(
        self, padded: Array, width: int, offset: tuple[int, ...], /
    ) -> Array:
        return padded[
            tuple(
                slice(width + delta, width + delta + size)
                for delta, size in zip(
                    offset, self.discretization.cell_shape, strict=True
                )
            )
        ]

    def _youngs_gradient(self, padded: Array, widths: Array, /) -> Array:
        """Youngs' ``(1, 2, 1)``-weighted centered alpha gradient."""

        dimension = self.dimension
        weights = {-1: 1.0, 0: 2.0, 1: 1.0}
        gradient = []
        for axis in range(dimension):
            total = jnp.zeros(self.discretization.cell_shape, dtype=padded.dtype)
            for index in np.ndindex(*((3,) * dimension)):
                offset = tuple(entry - 1 for entry in index)
                if offset[axis] == 0:
                    continue
                weight = offset[axis] * float(
                    np.prod(
                        [
                            weights[delta]
                            for other, delta in enumerate(offset)
                            if other != axis
                        ]
                    )
                )
                total = total + weight * self._shifted_view(padded, 1, offset)
            gradient.append(total / (2.0 * 4.0 ** (dimension - 1) * widths[..., axis]))
        return jnp.stack(gradient, axis=-1)

    def _centered_column_normal(
        self, padded: Array, widths: Array, axis: int, orientation: Array, /
    ) -> Array:
        """Unit normal from three-cell column heights along ``axis``.

        ``orientation`` is ``+1`` where the alpha phase lies below along
        ``axis``.  With heights ``H`` in cell units the normal out of the alpha
        phase is proportional to ``(-grad_t H, orientation / h_axis)``.
        """

        dimension = self.dimension
        tangential = [other for other in range(dimension) if other != axis]
        sums: dict[tuple[int, ...], Array] = {}
        for shift in np.ndindex(*((3,) * len(tangential))):
            offset = [0] * dimension
            for other, delta in zip(tangential, shift, strict=True):
                offset[other] = delta - 1
            total = jnp.zeros(self.discretization.cell_shape, dtype=padded.dtype)
            for level in (-1, 0, 1):
                offset[axis] = level
                total = total + self._shifted_view(padded, 1, tuple(offset))
            sums[tuple(delta - 1 for delta in shift)] = total
        components = []
        center = (0,) * len(tangential)
        for other in range(dimension):
            if other == axis:
                components.append(orientation / widths[..., axis])
                continue
            position = tangential.index(other)
            forward = list(center)
            backward = list(center)
            forward[position] = 1
            backward[position] = -1
            components.append(
                -(sums[tuple(forward)] - sums[tuple(backward)])
                / (2.0 * widths[..., other])
            )
        normal = jnp.stack(components, axis=-1)
        return normal / jnp.sqrt(jnp.sum(normal**2, axis=-1))[..., None]

    def interface_normal(self, alpha: ArrayLike, /) -> Array:
        """Return the unit normal out of the alpha phase (zero where flat).

        Centered-column candidates are formed from three-cell column sums
        along every axis (oriented by Youngs' ``(1, 2, 1)`` gradient); the
        candidate most aligned with its own column axis is used, in the spirit
        of the mixed Youngs-centered method of Aulisa et al. (2007).  Column
        heights are exact for planar interfaces spanned by the columns and
        second-order for curved ones.  Cells without an alpha gradient return
        a zero normal.
        """

        value = jnp.asarray(alpha)
        if value.shape != self.discretization.cell_shape:
            raise ValueError("PLIC alpha must match the cell shape.")
        padded = self.pad(value, 1)
        widths = self.cell_widths()
        youngs = -self._youngs_gradient(padded, widths)
        candidates = []
        alignment = []
        for axis in range(self.dimension):
            orientation = jnp.where(youngs[..., axis] >= 0.0, 1.0, -1.0).astype(
                value.dtype
            )
            candidate = self._centered_column_normal(padded, widths, axis, orientation)
            candidates.append(candidate)
            alignment.append(jnp.abs(candidate[..., axis]))
        best = jnp.argmax(jnp.stack(alignment, axis=-1), axis=-1)
        normal = jnp.take_along_axis(
            jnp.stack(candidates, axis=-2), best[..., None, None], axis=-2
        )[..., 0, :]
        flat = jnp.sum(youngs**2, axis=-1) == 0.0
        return jnp.where(flat[..., None], 0.0, normal)

    def offset(
        self, scaled_normal: ArrayLike, fraction: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Return the exact plane constant for ``fraction`` and its convergence."""

        m = jnp.asarray(scaled_normal)
        alpha = jnp.asarray(fraction, dtype=m.dtype)
        sorted_normal, total = _sorted_unit_normal(m)
        upper = alpha > 0.5
        # Rounding-level excursions of a stored fraction outside [0, 1] map to
        # the corner plane; the state itself is never modified.
        target = jnp.clip(jnp.where(upper, 1.0 - alpha, alpha), 0.0, 0.5)
        guess = _offset_guess(sorted_normal, target)
        flat_normal = sorted_normal.reshape((-1, 3))
        flat_target = target.reshape((-1,))
        flat_guess = guess.reshape((-1,))

        def polish(normal: Array, goal: Array, start: Array) -> tuple[Array, Array]:
            def residual(value: Array) -> Array:
                return _half_cube_volume(normal, value) - goal

            root, diagnostics = self.offset_root.solve_with_diagnostics(residual, start)
            return root, jnp.abs(diagnostics.residual) <= self.residual_tolerance

        root, converged = jax.vmap(polish)(flat_normal, flat_target, flat_guess)
        normalized = root.reshape(target.shape)
        normalized = jnp.where(upper, 1.0 - normalized, normalized)
        beta = normalized * total + jnp.sum(jnp.minimum(m, 0.0), axis=-1)
        reconstructed, derivative = jax.jvp(
            lambda offset: plane_volume_fraction(m, offset),
            (beta,),
            (jnp.ones_like(beta),),
        )
        error = reconstructed - alpha
        safe_derivative = jnp.where(
            jnp.isfinite(derivative) & (jnp.abs(derivative) > jnp.finfo(beta.dtype).tiny),
            derivative,
            jnp.ones_like(derivative),
        )
        lower = jnp.sum(jnp.minimum(m, 0.0), axis=-1)
        upper_bound = jnp.sum(jnp.maximum(m, 0.0), axis=-1)
        candidate = jnp.clip(beta - error / safe_derivative, lower, upper_bound)
        candidate_error = plane_volume_fraction(m, candidate) - alpha
        improved = jnp.abs(candidate_error) < jnp.abs(error)
        beta = jnp.where(improved, candidate, beta)
        final_error = jnp.where(improved, candidate_error, error)
        converged = converged.reshape(target.shape)
        return beta, converged & (jnp.abs(final_error) <= self.residual_tolerance)

    def reconstruct(
        self, alpha: ArrayLike, normal: ArrayLike | None = None, /
    ) -> StructuredPLICReconstruction:
        """Reconstruct exact PLIC planes with optional caller normals."""

        value = jnp.asarray(alpha)
        if value.shape != self.discretization.cell_shape:
            raise ValueError("PLIC alpha must match the cell shape.")
        unit = self.interface_normal(value) if normal is None else jnp.asarray(normal)
        if unit.shape != (*self.discretization.cell_shape, self.dimension):
            raise ValueError("PLIC normals must have shape cell_shape + (dimension,).")
        mixed = self.mixed_mask(value)
        # A mixed cell inside a locally uniform field has no gradient; any
        # plane reproduces its volume, so the first axis is used explicitly.
        fallback = jnp.zeros((self.dimension,), dtype=value.dtype).at[0].set(1.0)
        unit = jnp.where((jnp.sum(unit**2, axis=-1) > 0.0)[..., None], unit, fallback)
        widths = self.cell_widths()
        scaled = unit * widths
        beta, converged = self.offset(scaled, value)
        reconstructed = plane_volume_fraction(scaled, beta)
        residual = jnp.where(mixed, reconstructed - value, 0.0)
        area_rate = jax.jvp(
            lambda offset: plane_volume_fraction(scaled, offset),
            (beta,),
            (jnp.ones_like(beta),),
        )[1]
        facet = jnp.where(mixed, self.discretization.cell_volumes * area_rate, 0.0)
        coordinate_moments = []
        for axis in range(self.dimension):
            direction = jnp.zeros_like(scaled).at[..., axis].set(1.0)
            coordinate_moments.append(
                jax.jvp(
                    lambda normal_: plane_volume_fraction(normal_, beta),
                    (scaled,),
                    (direction,),
                )[1]
            )
        safe_area_rate = jnp.where(
            mixed & jnp.isfinite(area_rate) & (area_rate > 0.0),
            area_rate,
            jnp.ones_like(area_rate),
        )
        unit_centroid = (
            -jnp.stack(coordinate_moments, axis=-1) / safe_area_rate[..., None]
        )
        point = self.discretization.cell_centers + (unit_centroid - 0.5) * widths
        point = jnp.where(mixed[..., None], point, self.discretization.cell_centers)
        converged = jnp.where(mixed, converged, True)
        cell_finite = (
            jnp.isfinite(beta)
            & jnp.isfinite(facet)
            & jnp.all(jnp.isfinite(point), axis=-1)
            & jnp.all(jnp.isfinite(unit), axis=-1)
        )
        facet_valid = (
            mixed
            & cell_finite
            & converged
            & (facet > 0.0)
            & (jnp.abs(residual) <= self.residual_tolerance)
        )
        finite = jnp.all(cell_finite)
        valid = finite & jnp.all(~mixed | facet_valid)
        return StructuredPLICReconstruction(
            normal=unit,
            scaled_normal=scaled,
            offset=beta,
            mixed=mixed,
            facet_measure=facet,
            facet_valid=facet_valid,
            interface_point=point,
            residual=residual,
            converged=converged,
            finite=finite,
            valid=valid,
            plan_id=self.plan_id,
        )

    def interface_delta(
        self,
        alpha: ArrayLike,
        reconstruction: StructuredPLICReconstruction,
        /,
    ) -> tuple[Array, Array]:
        """Return a cell surface-delta estimate and its geometry support.

        A valid PLIC facet supplies its canonical measure divided by cell
        volume. If that geometry is unavailable in a mixed cell, the norm of
        the same Youngs gradient used to orient reconstruction supplies the
        bounded local estimate. Pure cells remain unsupported.
        """

        value = jnp.asarray(alpha)
        if value.shape != self.discretization.cell_shape:
            raise ValueError("PLIC alpha must match the cell shape.")
        if not isinstance(reconstruction, StructuredPLICReconstruction):
            raise TypeError("reconstruction must be StructuredPLICReconstruction.")
        if reconstruction.plan_id != self.plan_id:
            raise ValueError("PLIC reconstruction belongs to another PLIC plan.")
        gradient = self._youngs_gradient(self.pad(value, 1), self.cell_widths())
        gradient_delta = jnp.sqrt(jnp.sum(gradient * gradient, axis=-1))
        facet_delta = (
            reconstruction.facet_measure
            / self.discretization.cell_volumes.astype(value.dtype)
        )
        facet_supported = (
            reconstruction.facet_valid
            & jnp.isfinite(facet_delta)
            & (facet_delta > 0.0)
        )
        gradient_supported = (
            reconstruction.mixed
            & jnp.isfinite(gradient_delta)
            & (gradient_delta > 0.0)
        )
        supported = facet_supported | gradient_supported
        delta = jnp.where(
            facet_supported,
            facet_delta,
            jnp.where(gradient_supported, gradient_delta, 0.0),
        )
        return delta, supported

    def swept_fraction(
        self,
        alpha: ArrayLike,
        reconstruction: StructuredPLICReconstruction,
        axis: int,
        displacement: ArrayLike,
        /,
    ) -> Array:
        """Alpha fraction of the donor strip swept through each face of ``axis``.

        ``displacement`` is the signed face velocity times the step.  The
        donor is the upwind cell; its strip of normalized thickness
        ``|displacement| / h`` adjacent to the face is intersected exactly with
        the donor's PLIC plane.  Pure donors sweep their own volume fraction.
        Exterior donors beyond nonperiodic boundaries copy the boundary cell.
        """

        value = jnp.asarray(alpha)
        grid_axis = self.discretization.grid.structured_axes[axis]
        shift = jnp.asarray(displacement, dtype=value.dtype)
        widths = self.cell_widths()[..., axis]

        def neighbors(cell_value: Array) -> tuple[Array, Array]:
            moved = jnp.moveaxis(cell_value, axis, 0)
            if grid_axis.periodic:
                lower = jnp.roll(moved, 1, axis=0)
                upper = moved
            else:
                lower = jnp.concatenate((moved[:1], moved), axis=0)
                upper = jnp.concatenate((moved, moved[-1:]), axis=0)
            return jnp.moveaxis(lower, 0, axis), jnp.moveaxis(upper, 0, axis)

        def vector_neighbors(cell_value: Array) -> tuple[Array, Array]:
            lower, upper = zip(
                *(neighbors(cell_value[..., index]) for index in range(self.dimension)),
                strict=True,
            )
            return jnp.stack(lower, axis=-1), jnp.stack(upper, axis=-1)

        lower_alpha, upper_alpha = neighbors(value)
        lower_mixed, upper_mixed = neighbors(reconstruction.mixed)
        lower_offset, upper_offset = neighbors(reconstruction.offset)
        lower_width, upper_width = neighbors(widths)
        lower_normal, upper_normal = vector_neighbors(reconstruction.scaled_normal)
        forward = shift >= 0.0
        donor_alpha = jnp.where(forward, lower_alpha, upper_alpha)
        donor_mixed = jnp.where(forward, lower_mixed, upper_mixed)
        donor_offset = jnp.where(forward, lower_offset, upper_offset)
        donor_width = jnp.where(forward, lower_width, upper_width)
        donor_normal = jnp.where(forward[..., None], lower_normal, upper_normal)
        courant = jnp.abs(shift) / donor_width
        normal_component = donor_normal[..., axis]
        # Forward flux leaves through the donor's upper face, strip [1-c, 1];
        # backward flux leaves through its lower face, strip [0, c].
        strip_offset = jnp.where(
            forward, donor_offset - normal_component * (1.0 - courant), donor_offset
        )
        strip_normal = donor_normal.at[..., axis].set(normal_component * courant)
        geometric = plane_volume_fraction(strip_normal, strip_offset)
        return jnp.where(donor_mixed & (courant > 0.0), geometric, donor_alpha)


__all__ = [
    "StructuredPLICPlan",
    "StructuredPLICReconstruction",
    "plane_volume_fraction",
]
