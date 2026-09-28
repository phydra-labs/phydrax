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
import jax.numpy as jnp
import numpy as np
from jax import Array

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
from ...typing import parse
from ..particle._core import ParticleDiscretization
from ..spectral import SharedGridHankelOffset
from ..splatting._assignment import _basis_and_derivative
from ._transfer import PICShapeOrder


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

    def __init__(
        self, grid: QuasiCylindricalGrid, /, *, shape_order: PICShapeOrder = 1
    ) -> None:
        if not isinstance(grid, QuasiCylindricalGrid):
            raise TypeError("grid must be a QuasiCylindricalGrid.")
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
    mirrored: Array
    angle: Array
    support: Array


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

    def __init__(
        self, plan: AzimuthalTransferPlan, particles: ParticleDiscretization, /
    ) -> None:
        if not isinstance(plan, AzimuthalTransferPlan):
            raise TypeError("plan must be an AzimuthalTransferPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
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
        x, y, z = position[:, 0], position[:, 1], position[:, 2]
        radius = jnp.hypot(x, y)
        on_axis = radius == 0.0
        safe = jnp.where(on_axis, 1.0, radius)
        # e^{-iθ}; an on-axis particle takes θ = 0 (every p ≠ 0 weight vanishes there).
        angle = jnp.where(on_axis, 1.0 + 0.0j, (x - 1j * y) / safe)
        u = radius / grid.radial_spacing - 0.5
        v = (z - grid.lower) / grid.axial_spacing
        first_u = jnp.floor(u - 0.5 * (order - 1)).astype(jnp.int32)
        first_v = jnp.floor(v - 0.5 * (order - 1)).astype(jnp.int32)
        offsets = jnp.arange(width, dtype=jnp.int32)
        radial_nodes = first_u[:, None] + offsets[None, :]
        axial_nodes = first_v[:, None] + offsets[None, :]
        radial_weight = _basis_and_derivative(
            order, u[:, None] - radial_nodes.astype(u.dtype)
        )[0]
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
        mirrored_routes = jnp.broadcast_to(
            mirrored[:, :, None], (position.shape[0], width, width)
        ).reshape((position.shape[0], width * width))
        relation = RowRelation(
            indices,
            source_size=grid.radial_count * grid.axial_count,
            valid=jnp.broadcast_to(support[:, None], indices.shape),
        )
        return _Routes(relation, weights, mirrored_routes, angle, support)

    def _order_factors(
        self, routes: _Routes, offsets: tuple[int, ...], /
    ) -> tuple[Array, Array]:
        """``e^{−ipθ}`` per particle ``[N, M+1, C]`` and route parity ``[N, W, M+1, C]``."""
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
        parity = jnp.where(
            routes.mirrored[:, :, None, None],
            jnp.asarray((-1.0) ** orders),
            1.0,
        )
        return phase, parity

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
        phase, parity = self._order_factors(routes, offsets)
        weight = np.where(np.arange(grid.mode_count) == 0, 1.0, 2.0)[None, :, None]
        content = jnp.where(active[:, None, None], amplitude[:, None, :], 0.0) * (
            weight * phase
        )
        payload = (
            routes.weights[:, :, None, None] * parity * content[:, None, :, :]
        ).astype(jnp.complex128)
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
        phase, parity = self._order_factors(routes, offsets)
        flat = jnp.moveaxis(values, 0, 2).reshape(
            (grid.radial_count * grid.axial_count, grid.mode_count, len(offsets))
        )
        patches = gather_routes(routes.relation, flat)
        valid = routes.relation.valid[:, :, None, None]
        interpolated = jnp.sum(
            jnp.where(valid, routes.weights[:, :, None, None] * parity * patches, 0.0),
            axis=1,
        )
        synthesis = jnp.sum(interpolated * jnp.conj(phase), axis=1)
        return jnp.where(routes.support[:, None], synthesis, 0.0), routes.support


__all__ = [
    "AzimuthalTransferPlan",
    "PreparedAzimuthalTransfer",
    "QuasiCylindricalGrid",
]
