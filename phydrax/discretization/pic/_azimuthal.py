#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Azimuthal-mode particle transfers for quasi-cylindrical PIC.

A quasi-cylindrical quantity is ``F(r, θ, z) = Re Σ_{m=0}^{M} F_m(r, z) e^{imθ}``
sampled on cell-centered radii ``r_j = (j + ½)Δr`` (``Δr = R/N_r``, the grid of
`phydrax.discretization.SharedGridHankelPlan`) and periodic axial nodes
``z_k = z₀ + kΔz``. A channel of *order offset* ``o`` carries, in mode ``m``,
the angular order ``p = m + o``: scalars and ``F_z`` have ``o = 0``, the
circular components ``F_± = (F_r ∓ iF_θ)/2`` have ``o = ∓1`` (so that
``F_x ∓ iF_y`` of mode ``m`` varies as ``e^{i(m∓1)θ}``).

Deposition assigns a particle of amplitude ``a`` at ``(r, θ, z)`` to mode ``m``
with the weight ``c_m a e^{−ipθ}`` (``c₀ = 1``, ``c_m = 2`` for ``m > 0``) and the
tensor cardinal B-spline of the shape order in ``(r, z)``; nodes below the axis
are folded onto their mirror nodes with the parity ``(−1)^p`` of an angular-order
``p`` quantity. The near-axis correction divides node content by the effective
volume ``V_j^{(q)} = 2πΔz ∫ G_j(r) (r/r_j)^q r dr`` of the folded shape ``G_j``
for ``q = |p|``, so the leading regular behavior ``r^{|p|}`` of every angular
order deposits exactly at every node (for ``p = 0``, a uniform density deposits
uniformly including the axis cell, which the plain ``2π r_j Δr Δz`` volume
overestimates by 13/12 for the linear shape — Ruyten, J. Comput. Phys. 105,
224, 1993; Verboncoeur, J. Comput. Phys. 174, 421, 2001). Gathering uses the same
folded shape and returns the azimuthal synthesis ``Σ_m F_{m,c} e^{ipθ}`` of each
channel at the particle angle.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from ..._admissibility import guard_derivative_validity
from ..._fingerprint import canonical_fingerprint
from ..._numerics._quadrature_rules import gauss_legendre_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import (
    canonical_row_route_ids,
    gather_routes,
    RelationExecutionPlan,
    RowRelation,
)
from ...typing import checked, parse
from .._cubical_whitney import PICShapeOrder
from ..particle._core import ParticleDiscretization
from ..spectral import SharedGridHankelOffset
from ..splatting._assignment import _basis_and_derivative


class QuasiCylindricalGrid(StrictModule, NonTrainableState):
    """Radial Hankel grid of radius ``R`` and a periodic uniform axial grid.

    Radial nodes ``r_j = (j + ½)R/N_r``; axial nodes ``z_k = lower + kΔz`` with
    ``Δz = (upper − lower)/N_z`` and period ``upper − lower``; modes
    ``m = 0, …, mode_count − 1``.
    """

    radius: float = eqx.field(static=True)
    radial_count: int = eqx.field(static=True)
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    axial_count: int = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)

    def __init__(
        self,
        radius: float,
        radial_count: int,
        lower: float,
        upper: float,
        axial_count: int,
        mode_count: int,
        /,
    ) -> None:
        values = (float(radius), float(lower), float(upper))
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Quasi-cylindrical grid bounds must be finite.")
        if values[0] <= 0.0 or values[2] <= values[1]:
            raise ValueError("radius must be positive and upper must exceed lower.")
        counts = (int(radial_count), int(axial_count), int(mode_count))
        if counts[0] < 2 or counts[1] < 2 or counts[2] < 1:
            raise ValueError(
                "radial_count and axial_count must be at least two and mode_count "
                "at least one."
            )
        self.radius, self.lower, self.upper = values
        self.radial_count, self.axial_count, self.mode_count = counts
        self.grid_id = canonical_fingerprint(
            {
                "kind": "quasi-cylindrical-grid",
                "radius": values[0],
                "radial_count": counts[0],
                "lower": values[1],
                "upper": values[2],
                "axial_count": counts[1],
                "mode_count": counts[2],
            }
        )

    @property
    def radial_spacing(self) -> float:
        return self.radius / self.radial_count

    @property
    def axial_spacing(self) -> float:
        return (self.upper - self.lower) / self.axial_count

    @property
    def length(self) -> float:
        return self.upper - self.lower

    @property
    def radial_coordinates(self) -> np.ndarray:
        return self.radial_spacing * (np.arange(self.radial_count) + 0.5)

    @property
    def axial_coordinates(self) -> np.ndarray:
        return self.lower + self.axial_spacing * np.arange(self.axial_count)


def _effective_volumes(grid: QuasiCylindricalGrid, order: int, /) -> np.ndarray:
    """Host table ``V[q, j]`` of folded-shape volumes for ``q = 0, …, M + 1``.

    Integrated in the radial index coordinate ``u = r/Δr − ½`` with a
    Gauss–Legendre rule on every half-cell, exact for the piecewise-polynomial
    integrand (spline degree plus ``q + 1``).
    """
    count = grid.radial_count
    rule = gauss_legendre_data(12)
    nodes = np.asarray(rule.nodes, dtype=np.float64)
    weights = np.asarray(rule.weights, dtype=np.float64)
    reach = 0.5 * (order + 1)
    edges = np.arange(-0.5, count - 0.5 + reach + 0.5, 0.5)
    lower, upper = edges[:-1], edges[1:]
    u = (0.5 * (upper - lower)[:, None] * nodes[None, :]) + (
        0.5 * (upper + lower)[:, None]
    )
    w = (0.5 * (upper - lower)[:, None] * weights[None, :]).reshape(-1)
    u = u.reshape(-1)
    index = np.arange(count, dtype=np.float64)[:, None]
    direct = np.asarray(_basis_and_derivative(order, jnp.asarray(u[None, :] - index))[0])
    mirror = np.asarray(
        _basis_and_derivative(order, jnp.asarray(u[None, :] + 1.0 + index))[0]
    )
    scale = 2.0 * math.pi * grid.axial_spacing * grid.radial_spacing**2
    tables = []
    for power in range(grid.mode_count + 1):
        folded = direct + (-1.0) ** power * mirror
        moment = folded * (u[None, :] + 0.5) ** (power + 1)
        tables.append(scale * (moment @ w) / (index[:, 0] + 0.5) ** power)
    return np.stack(tables)


class AzimuthalTransferPlan(StrictModule, NonTrainableState):
    """Azimuthal-mode deposition and gathering on one quasi-cylindrical grid."""

    grid: QuasiCylindricalGrid
    shape_order: PICShapeOrder = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, grid: QuasiCylindricalGrid, /, *, shape_order: PICShapeOrder = 1
    ) -> None:
        order = parse(shape_order, PICShapeOrder, "shape_order")
        if order + 1 > grid.radial_count or order + 1 > grid.axial_count:
            raise ValueError("The shape stencil is wider than the grid.")
        self.grid = grid
        self.shape_order = order
        self.plan_id = canonical_fingerprint(
            {
                "kind": "azimuthal-transfer-plan",
                "grid": grid.grid_id,
                "shape_order": order,
            }
        )

    def prepare(self, particles: ParticleDiscretization, /) -> PreparedAzimuthalTransfer:
        return PreparedAzimuthalTransfer(self, particles)


class _Routes(NamedTuple):
    relation: RowRelation
    weights: Array
    axis_slopes: Array
    mirrored: Array
    angle: Array
    on_axis: Array
    axis_tangent: Array
    support: Array


_AXIS_DERIVATIVE = (
    "An exactly on-axis particle has first position derivatives only for angular "
    "orders |p| <= 1."
)


@jax.custom_jvp
def _first_order_on_axis(position: Array, on_axis: Array, /) -> Array:
    """Identity whose first derivative is exact and whose nested derivatives
    are invalid on the rows of on-axis particles."""
    del on_axis
    return position


@_first_order_on_axis.defjvp
def _first_order_on_axis_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[Array, Array]:
    position, on_axis = primals
    # The axis route below is exact to first order only (the surrogate radius
    # and the transverse carrier have no curvature), so the tangent coefficient
    # is a unit whose own derivative is poisoned on the axis rows: every nested
    # derivative through an on-axis particle is NaN rather than a silent zero.
    coefficient = guard_derivative_validity(
        jnp.ones_like(position),
        ~on_axis[:, None],
        dependencies=position,
        message=_AXIS_DERIVATIVE,
    )
    return _first_order_on_axis(position, on_axis), coefficient * tangents[0]


class PreparedAzimuthalTransfer(StrictModule, NonTrainableState):
    """Prepared azimuthal transfer of one particle support.

    ``volumes[q, j]`` are the near-axis-corrected node volumes of angular order
    ``|p| = q``.
    """

    plan: AzimuthalTransferPlan
    volumes: Array
    stable_order: Array
    capacity: int = eqx.field(static=True)
    particles_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, plan: AzimuthalTransferPlan, particles: ParticleDiscretization, /
    ) -> None:
        if particles.ambient_dimension != 3:
            raise ValueError("Quasi-cylindrical particles carry Cartesian 3-D positions.")
        volumes = _effective_volumes(plan.grid, plan.shape_order)
        if not np.all(np.isfinite(volumes)) or not np.all(volumes > 0.0):
            raise ValueError("Near-axis volumes must be finite and positive.")
        ids = np.asarray(particles.particle_ids, dtype=np.int64)
        self.plan = plan
        self.volumes = jnp.asarray(volumes)
        self.stable_order = jnp.asarray(np.argsort(ids, kind="stable").astype(np.int32))
        self.capacity = particles.capacity
        self.particles_id = particles.prepared_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-azimuthal-transfer",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
            }
        )

    @property
    def route_width(self) -> int:
        return (self.plan.shape_order + 1) ** 2

    def _routes(self, position: Array, active: Array, /) -> _Routes:
        grid = self.plan.grid
        order = self.plan.shape_order
        width = order + 1
        # θ is undefined on the axis and the radius is not differentiable there;
        # the on-axis first derivative is carried by `axis_tangent` instead.
        on_axis = jnp.hypot(position[:, 0], position[:, 1]) == 0.0
        position = _first_order_on_axis(position, on_axis)
        x, y, z = position[:, 0], position[:, 1], position[:, 2]
        safe_x = jnp.where(on_axis, 1.0, x)
        safe_y = jnp.where(on_axis, 0.0, y)
        safe = jnp.hypot(safe_x, safe_y)
        radius = jnp.where(on_axis, 0.0, safe)
        # e^{-iθ}; an on-axis particle takes θ = 0.
        angle = jnp.where(on_axis, 1.0 + 0.0j, (safe_x - 1j * safe_y) / safe)
        u = radius / grid.radial_spacing - 0.5
        v = (z - grid.lower) / grid.axial_spacing
        first_u = jnp.floor(u - 0.5 * (order - 1)).astype(jnp.int32)
        first_v = jnp.floor(v - 0.5 * (order - 1)).astype(jnp.int32)
        offsets = jnp.arange(width, dtype=jnp.int32)
        radial_nodes = first_u[:, None] + offsets[None, :]
        axial_nodes = first_v[:, None] + offsets[None, :]
        radial_weight, radial_slope = _basis_and_derivative(
            order, u[:, None] - radial_nodes.astype(u.dtype)
        )
        axial_weight = _basis_and_derivative(
            order, v[:, None] - axial_nodes.astype(v.dtype)
        )[0]
        mirrored = radial_nodes < 0
        folded = jnp.where(mirrored, -1 - radial_nodes, radial_nodes)
        support = (
            active
            & jnp.all(jnp.isfinite(position), axis=-1)
            & (radial_nodes[:, -1] < grid.radial_count)
        )
        axial_index = jnp.mod(axial_nodes, grid.axial_count)
        radial_index = jnp.clip(folded, 0, grid.radial_count - 1)
        indices = (
            radial_index[:, :, None] * grid.axial_count + axial_index[:, None, :]
        ).reshape((position.shape[0], width * width))
        weights = (radial_weight[:, :, None] * axial_weight[:, None, :]).reshape(
            (position.shape[0], width * width)
        )
        axis_slopes = (radial_slope[:, :, None] * axial_weight[:, None, :]).reshape(
            (position.shape[0], width * width)
        ) / grid.radial_spacing
        # Zero-valued carrier of the on-axis transverse displacement (dx, dy).
        planar = position[:, :2]
        axis_tangent = jnp.where(
            on_axis[:, None], planar - lax.stop_gradient(planar), 0.0
        )
        mirrored_routes = jnp.broadcast_to(
            mirrored[:, :, None], (position.shape[0], width, width)
        ).reshape((position.shape[0], width * width))
        relation = RowRelation(
            indices,
            source_size=grid.radial_count * grid.axial_count,
            valid=jnp.broadcast_to(support[:, None], indices.shape),
        )
        return _Routes(
            relation,
            weights,
            axis_slopes,
            mirrored_routes,
            angle,
            on_axis,
            axis_tangent,
            support,
        )

    def _route_factors(self, routes: _Routes, offsets: tuple[int, ...], /) -> Array:
        """Route factors ``w (−1)^{p·mirrored} e^{−ipθ}`` ``[N, W, M+1, C]``.

        Folded on the axis, the factor of order ``p`` is ``A(r) e^{−ipθ}`` with
        ``A`` of parity ``(−1)^p`` in the signed radius. Orders ``|p| ≤ 1`` are
        C¹ in Cartesian ``(x, y)``: their exact first derivative is none for
        ``p = 0`` (``A'(0) = 0``) and ``A'(0)(dx ∓ i dy)`` for ``p = ±1``
        (``A(r)e^{∓iθ} → A'(0)(x ∓ iy)``). Higher orders are not differentiable
        on the axis (even ``p`` is discontinuous, odd ``p`` has the
        direction-dependent derivative ``A'(0)e^{−ipφ}``), so their transverse
        derivative there is invalidated rather than assigned.
        """
        modes = self.plan.grid.mode_count
        orders = np.arange(modes)[:, None] + np.asarray(offsets)[None, :]
        powers = jnp.stack(
            tuple(routes.angle**power for power in range(modes + 1)), axis=-1
        )
        # p = −1 only occurs for m = 0, o = −1: e^{+iθ} = conj(e^{−iθ}).
        phase = jnp.where(
            jnp.asarray(orders < 0),
            jnp.conj(routes.angle)[:, None, None],
            powers[:, np.abs(orders)],
        )
        dx, dy = routes.axis_tangent[:, 0], routes.axis_tangent[:, 1]
        axis_phase = jnp.where(
            jnp.asarray(np.abs(orders) == 1),
            dx[:, None, None] - 1j * np.sign(orders) * dy[:, None, None],
            0.0,
        )
        parity = jnp.where(
            routes.mirrored[:, :, None, None],
            jnp.asarray((-1.0) ** orders),
            1.0,
        )
        factors = parity * (
            routes.weights[:, :, None, None] * phase[:, None]
            + routes.axis_slopes[:, :, None, None] * axis_phase[:, None]
        )
        if np.all(np.abs(orders) <= 1):
            return factors
        undefined = guard_derivative_validity(
            dx + dy, ~routes.on_axis, message=_AXIS_DERIVATIVE
        )
        return (
            factors
            + jnp.where(jnp.asarray(np.abs(orders) > 1), undefined[:, None, None], 0.0)[
                :, None
            ]
        )

    def _require(self, position: Array, active: Array, /) -> None:
        if position.shape != (self.capacity, 3) or active.shape != (self.capacity,):
            raise ValueError("Positions must be capacity-by-3 with a capacity mask.")

    def deposit(
        self,
        position: Array,
        active: Array,
        amplitude: Array,
        offsets: tuple[int, ...],
        /,
    ) -> tuple[Array, Array]:
        """Mode densities ``[M+1, N_r, N_z, C]`` of per-particle amplitudes ``[N, C]``.

        Channel ``c`` has order offset ``offsets[c]``; returns the densities and
        success (every active particle's stencil lies inside the radial grid).
        """
        self._require(position, active)
        for value in offsets:
            parse(value, SharedGridHankelOffset, "offsets")
        if amplitude.shape != (self.capacity, len(offsets)):
            raise ValueError("amplitude must be capacity-by-channel.")
        grid = self.plan.grid
        routes = self._routes(position, active)
        factors = self._route_factors(routes, offsets)
        weight = np.where(np.arange(grid.mode_count) == 0, 1.0, 2.0)[None, :, None]
        content = jnp.where(active[:, None, None], amplitude[:, None, :], 0.0) * weight
        payload = (factors * content[:, None, :, :]).astype(jnp.complex128)
        # Deposition is the transpose of the particle-row gather: routes reduce
        # onto grid nodes in stable particle-identity order.
        execution = RelationExecutionPlan().prepare(
            routes.relation.as_edge_relation().transpose(),
            stable_route_ids=canonical_row_route_ids(self.stable_order, self.route_width),
        )
        reduced, evidence = execution.reduce(
            payload.reshape((-1, grid.mode_count, len(offsets))),
            accumulation="fast",
            output="dense",
        )
        nodes = jnp.moveaxis(
            reduced.reshape(
                (grid.radial_count, grid.axial_count, grid.mode_count, len(offsets))
            ),
            2,
            0,
        )
        power = np.abs(np.arange(grid.mode_count)[:, None] + np.asarray(offsets)[None, :])
        volume = jnp.moveaxis(self.volumes[power], -1, 1)[:, :, None, :]
        successful = evidence.successful & jnp.all(routes.support | ~active)
        return nodes / volume, successful

    def gather(
        self,
        position: Array,
        active: Array,
        values: Array,
        offsets: tuple[int, ...],
        /,
    ) -> tuple[Array, Array]:
        """Azimuthal syntheses ``Σ_m F_{m,c}(r, z) e^{ipθ}`` ``[N, C]`` and support."""
        self._require(position, active)
        grid = self.plan.grid
        expected = (grid.mode_count, grid.radial_count, grid.axial_count, len(offsets))
        if values.shape != expected:
            raise ValueError(f"Gathered values must have shape {expected}.")
        routes = self._routes(position, active)
        factors = self._route_factors(routes, offsets)
        flat = jnp.moveaxis(values, 0, 2).reshape(
            (grid.radial_count * grid.axial_count, grid.mode_count, len(offsets))
        )
        patches = gather_routes(routes.relation, flat)
        valid = routes.relation.valid[:, :, None, None]
        synthesis = jnp.sum(
            jnp.where(valid, jnp.conj(factors) * patches, 0.0), axis=(1, 2)
        )
        return jnp.where(routes.support[:, None], synthesis, 0.0), routes.support


__all__ = [
    "AzimuthalTransferPlan",
    "PreparedAzimuthalTransfer",
    "QuasiCylindricalGrid",
]
