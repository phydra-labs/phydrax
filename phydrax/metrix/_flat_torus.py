#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization._periodic_cell import PeriodicCell
from ..ein import contract
from ..linalg import SmallLinearSolvePlan, solve_small_linear
from ._manifold import _array_with_trailing_shape, AbstractGeodesicManifold
from ._state_geometry import AbstractStateGeometry


def _same_shape(value: Array, reference: Array, role: str, /) -> None:
    if value.shape != reference.shape:
        raise ValueError(f"{role} must have shape {reference.shape}; got {value.shape}.")


class FlatTorusStateGeometry(AbstractStateGeometry):
    """Principal-coordinate geometry for a flat torus with one common period."""

    period: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(self, period: float, /, *, geometry_id: str | None = None) -> None:
        value = float(period)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("period must be finite and positive.")
        self.period = value
        self.geometry_id = (
            canonical_fingerprint({"kind": "flat-torus-geometry", "period": value})
            if geometry_id is None
            else str(geometry_id)
        )
        if not self.geometry_id:
            raise ValueError("geometry_id must be non-empty.")
        self.retraction_method = "wrapped-addition"
        self.trivial = False
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = True
        self.supports_commutator_free = True

    def wrap(self, value: ArrayLike, /) -> Array:
        """Map real coordinates into ``[-period / 2, period / 2)``."""
        array = jnp.asarray(value)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError("Flat-torus coordinates must be real floating arrays.")
        return jnp.mod(array + 0.5 * self.period, self.period) - 0.5 * self.period

    def contains(self, state: ArrayLike, /) -> Array:
        array = jnp.asarray(state)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            return jnp.asarray(False)
        half = 0.5 * self.period
        return (
            jnp.all(jnp.isfinite(array)) & jnp.all(array >= -half) & jnp.all(array < half)
        )

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        state_array = jnp.asarray(state)
        vector_array = jnp.asarray(vector)
        _same_shape(vector_array, state_array, "Flat-torus tangent")
        return vector_array

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        state_array = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        _same_shape(local, state_array, "Flat-torus local tangent")
        return self.wrap(state_array + local)

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        _same_shape(point_array, state_array, "Flat-torus retraction point")
        return self.wrap(point_array - state_array)

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        velocity = jnp.asarray(local_velocity)
        _same_shape(local, state_array, "Flat-torus local tangent")
        _same_shape(velocity, local, "Flat-torus local velocity")
        return velocity

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        vector = jnp.asarray(tangent)
        _same_shape(point_array, state_array, "Flat-torus retraction point")
        _same_shape(vector, state_array, "Flat-torus tangent")
        return vector

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        covector = jnp.asarray(cotangent)
        _same_shape(local, state_array, "Flat-torus local tangent")
        _same_shape(covector, state_array, "Flat-torus cotangent")
        return covector

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        vector = jnp.asarray(tangent)
        _same_shape(point_array, state_array, "Flat-torus transport point")
        _same_shape(vector, state_array, "Flat-torus tangent")
        return vector

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        covector = jnp.asarray(cotangent)
        _same_shape(point_array, state_array, "Flat-torus transport point")
        _same_shape(covector, state_array, "Flat-torus cotangent")
        return covector

    def cut_locus_margin(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        _same_shape(point_array, state_array, "Flat-torus chart point")
        displacement = self.inverse_retract(state_array, point_array)
        return jnp.asarray(0.5 * self.period, dtype=state_array.dtype) - jnp.max(
            jnp.abs(displacement)
        )


def _unit_wrap(value: Array, /) -> Array:
    """Map real coordinates into ``[0, 1)`` with unit derivative.

    ``x - floor(x)`` can round to exactly ``1`` for tiny negative ``x``; that
    representative is mapped to ``0``.  The integer image is a branch choice,
    so it carries no derivative.
    """
    wrapped = value - jax.lax.stop_gradient(jnp.floor(value))
    return jnp.where(wrapped >= 1.0, wrapped - 1.0, wrapped)


class FlatTorusLogEvidence(StrictModule):
    """Logarithm branch evidence of one product of flat-torus point pairs.

    ``tangent`` is the selected shortest fractional displacement.
    ``distance`` and ``second_distance`` are the metric lengths of the best
    and second-best lattice images of each site pair; ``margin`` is their
    difference.  ``unique`` is ``margin > branch_tolerance``: a pair on (or
    within the tolerance of) the cut locus has no unique logarithm.
    """

    tangent: Array
    distance: Array
    second_distance: Array
    margin: Array
    unique: Array


class FlatTorusManifold(AbstractGeodesicManifold):
    """Unit-period flat torus of fractional points with an explicit constant metric.

    Points are fractional coordinates in ``[0, 1)`` with trailing dimension
    ``d``; leading entries of ``point_shape`` (for example ``(N, 3)`` for
    ``N`` sites) and any further leading axes form a product of independent
    sites.  The metric is constant and symmetric positive definite:

    * ``cell=None`` declares the **fractional Euclidean** metric ``G = I``.
      Its logarithm is the componentwise minimum image
      ``delta - round(delta)``.  This is *not* the shortest physical
      displacement of a general triclinic cell; it is the geodesic of the
      declared fractional representation only.
    * ``cell=PeriodicCell(H)`` with a square fully periodic row lattice
      ``H`` declares the **physical** metric ``G = H H^T``.  Its logarithm is
      the shortest lattice image selected by the native
      `PeriodicCell.minimum_image` owner, pulled back to fractional
      coordinates.

    ``exp``/``retract`` wrap ``x + v`` into ``[0, 1)``; transport is the
    identity (the torus is flat).  `log` refuses a pair whose best and
    second-best images are not separated by more than ``branch_tolerance``
    (the cut locus) instead of silently choosing a branch;
    `log_with_evidence` returns the branch margin without refusing.  For the
    physical metric the second-best image is searched over the cell's
    certified nearest-image stencil around the selected image.
    """

    point_shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    cell: PeriodicCell | None
    metric: Array
    inverse_metric: Array
    neighbor_shifts: Array
    branch_tolerance: float = eqx.field(static=True)
    metric_kind: str = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(
        self,
        point_shape: Sequence[int],
        /,
        *,
        cell: PeriodicCell | None = None,
        branch_tolerance: float = 0.0,
    ) -> None:
        shape = tuple(int(size) for size in point_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError(
                "FlatTorusManifold point_shape must be non-empty and positive."
            )
        dimension = shape[-1]
        tolerance = float(branch_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("branch_tolerance must be finite and nonnegative.")
        if cell is None:
            metric = np.eye(dimension, dtype=np.float64)
            inverse = metric
            # The componentwise branch structure needs no image stencil.
            neighbors = np.zeros((0, dimension), dtype=np.int32)
            kind = "fractional-euclidean"
            metric_identity: str | dict[str, str] = kind
        else:
            if not isinstance(cell, PeriodicCell):
                raise TypeError("cell must be a PeriodicCell or None.")
            if (
                cell.rank != dimension
                or cell.ambient_dimension != dimension
                or not cell.fully_periodic
            ):
                raise ValueError(
                    "A physical flat-torus metric needs a square, fully periodic "
                    "cell matching the trailing point dimension."
                )
            if dimension > 4:
                raise ValueError(
                    "Physical flat-torus metrics use the native small solve (d <= 4)."
                )
            vectors = np.asarray(cell.vectors, dtype=np.float64)
            metric = vectors @ vectors.T
            solved = solve_small_linear(
                SmallLinearSolvePlan(dimension),
                jnp.asarray(metric),
                jnp.eye(dimension, dtype=metric.dtype),
            )
            if not bool(solved.successful):
                raise ValueError("The cell metric H H^T is not admissibly invertible.")
            inverse = np.asarray(solved.value)
            inverse = 0.5 * (inverse + inverse.T)
            shifts = np.asarray(cell.image_shifts)
            neighbors = shifts[np.any(shifts != 0, axis=-1)]
            kind = "physical-cell"
            metric_identity = {"physical-cell": cell.cell_id}
        self.point_shape = shape
        self.dimension = dimension
        self.cell = cell
        self.metric = jnp.asarray(metric)
        self.inverse_metric = jnp.asarray(inverse)
        self.neighbor_shifts = jnp.asarray(neighbors, dtype=jnp.int32)
        self.branch_tolerance = tolerance
        self.metric_kind = kind
        self.manifold_id = "manifold:flat-torus:" + canonical_fingerprint(
            {
                "kind": "flat-torus-manifold",
                "point_shape": list(shape),
                "metric": metric_identity,
                "branch_tolerance": tolerance,
            }
        )
        self.retraction_method = "wrapped-addition"
        self.transport_method = "identity"
        self.transport_is_isometric = True
        self.transport_is_parallel = True

    def _point(self, value: ArrayLike, role: str, /) -> Array:
        array = _array_with_trailing_shape(value, self.point_shape, role)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError(f"{role} must be a real floating array.")
        return array

    def _pair(
        self, point: ArrayLike, other: ArrayLike, role: str, /
    ) -> tuple[Array, Array]:
        value = self._point(point, "Flat-torus point")
        partner = self._point(other, role)
        _same_shape(partner, value, role)
        return value, partner.astype(value.dtype)

    def wrap(self, value: ArrayLike, /) -> Array:
        """Map real fractional coordinates onto the unit torus ``[0, 1)``."""
        return _unit_wrap(self._point(value, "Flat-torus coordinates"))

    def contains(self, point: ArrayLike, /) -> Array:
        value = jnp.asarray(point)
        rank = len(self.point_shape)
        if (
            not jnp.issubdtype(value.dtype, jnp.floating)
            or value.ndim < rank
            or value.shape[value.ndim - rank :] != self.point_shape
        ):
            return jnp.asarray(False)
        return jnp.all(jnp.isfinite(value) & (value >= 0.0) & (value < 1.0))

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        """Largest distance of a coordinate outside ``[0, 1]`` (``inf`` if nonfinite)."""
        value = self._point(point, "Flat-torus point")
        defect = jnp.max(jnp.maximum(jnp.maximum(-value, value - 1.0), 0.0))
        return jnp.where(
            jnp.all(jnp.isfinite(value)),
            defect,
            jnp.asarray(jnp.inf, dtype=value.dtype),
        )

    def project_tangent(self, point: ArrayLike, ambient_vector: ArrayLike, /) -> Array:
        _, vector = self._pair(point, ambient_vector, "Flat-torus tangent")
        return vector

    def egrad_to_rgrad(self, point: ArrayLike, ambient_cotangent: ArrayLike, /) -> Array:
        """Metric gradient ``G^{-1} g``; the inverse comes from the native small solve."""
        cotangent = self.project_tangent(point, ambient_cotangent)
        if self.cell is None:
            return cotangent
        return contract(
            "...j,ij->...i",
            cotangent,
            self.inverse_metric.astype(cotangent.dtype),
            backend="jax",
        )

    def _lowered(self, tangent: Array, /) -> Array:
        """Covector ``G v`` of every site tangent."""
        if self.cell is None:
            return tangent
        return contract(
            "...i,ij->...j", tangent, self.metric.astype(tangent.dtype), backend="jax"
        )

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        left = self.project_tangent(point, left_tangent)
        right = self.project_tangent(point, right_tangent)
        return jnp.sum(self._lowered(left) * right)

    def exp(self, point: ArrayLike, tangent: ArrayLike, /) -> Array:
        value, step = self._pair(point, tangent, "Flat-torus tangent")
        return _unit_wrap(value + step)

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        return self.exp(point, tangent_step)

    def _branches(
        self, point: Array, destination: Array, /
    ) -> tuple[Array, Array, Array]:
        """Best fractional displacement and best/second-best squared image lengths."""
        raw = destination - point
        if self.cell is None:
            tangent = _unit_wrap(raw + 0.5) - 0.5
            best = jnp.sum(tangent * tangent, axis=-1)
            # The second-best image moves the coordinate nearest to its
            # half-period cut onto the other branch.
            second = best + jnp.min(1.0 - 2.0 * jnp.abs(tangent), axis=-1)
            return tangent, best, second
        vectors = self.cell.vectors.astype(raw.dtype)
        cartesian = contract("...i,ij->...j", raw, vectors, backend="jax")
        shortest = self.cell.minimum_image(cartesian)
        tangent = contract(
            "...i,ij->...j",
            shortest,
            self.cell.inverse_vectors.astype(raw.dtype),
            backend="jax",
        )
        best = jnp.sum(shortest * shortest, axis=-1)
        translations = contract(
            "si,ij->sj",
            self.neighbor_shifts.astype(raw.dtype),
            vectors,
            backend="jax",
        )
        candidates = shortest[..., None, :] - translations
        second = jnp.min(jnp.sum(candidates * candidates, axis=-1), axis=-1)
        return tangent, best, second

    def log_with_evidence(
        self, point: ArrayLike, destination: ArrayLike, /
    ) -> FlatTorusLogEvidence:
        """Shortest-image logarithm with its uniqueness margin; never refuses."""
        value, target = self._pair(point, destination, "Flat-torus destination")
        tangent, best, second = self._branches(value, target)
        distance = jnp.sqrt(jnp.maximum(best, 0.0))
        second_distance = jnp.sqrt(jnp.maximum(second, 0.0))
        margin = jax.lax.stop_gradient(second_distance - distance)
        return FlatTorusLogEvidence(
            tangent,
            distance,
            second_distance,
            margin,
            margin > self.branch_tolerance,
        )

    def log(self, point: ArrayLike, destination: ArrayLike, /) -> Array:
        """Shortest-image logarithm; refuses ambiguous (cut-locus) pairs."""
        evidence = self.log_with_evidence(point, destination)
        return eqx.error_if(
            evidence.tangent,
            ~jnp.all(evidence.unique),
            "Flat-torus logarithm is ambiguous: a site pair lies on the cut locus.",
        )

    def cut_locus_margin(self, point: ArrayLike, destination: ArrayLike, /) -> Array:
        """Smallest branch margin over every site pair."""
        return jnp.min(self.log_with_evidence(point, destination).margin)

    def squared_distance(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Sum of shortest-image squared metric lengths (branch independent)."""
        value, target = self._pair(left, right, "Flat-torus destination")
        _, best, _ = self._branches(value, target)
        return jnp.sum(best)

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        self.project_tangent(point, tangent_step)
        self._pair(point, destination, "Flat-torus destination")
        return self.project_tangent(destination, tangent)


__all__ = ["FlatTorusLogEvidence", "FlatTorusManifold", "FlatTorusStateGeometry"]
