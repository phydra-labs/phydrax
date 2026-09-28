#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pairwise antisymmetric near-contact repulsion between color-gradient interfaces.

Montessori, Lauricella, Tirelli and Succi (J. Fluid Mech. 872, 327-347, 2019) add a
short-range repulsive force between opposing interfaces of a color-gradient model to
represent the disjoining pressure that keeps thin films intact in dense emulsions.
This owner realizes that interaction as an explicit sum over lattice node pairs
``(x, x + v)`` so that Newton's third law holds pair by pair:

    m(x, v) = A_ab g(|v|) [w_a(x) s-_a(x) w_b(x + v) s+_b(x + v) + [a != b] (a <-> b)],
    F(x)    = sum_v [m(x - v, v) - m(x, v)] e_v,

where ``c_k`` is the concentration of component ``k``, ``w_k = 4 c_k (1 - c_k)`` its
diffuse-interface indicator, ``n_k = grad c_k / |grad c_k|`` points into ``k``,
``s-_k(x) = max(0, -n_k(x) . e_v)`` and ``s+_k(y) = max(0, n_k(y) . e_v)`` select
interfaces that face each other across the film, ``e_v = v / |v|`` and
``g(d) = 1 - d / (R + 1)`` is the declared kernel of range ``R``. Pairs are enumerated
once over the canonical half-space of integer offsets ``0 < |v| <= R``; pairs that
touch a solid cell or wrap across a non-periodic face are excluded on both sides.

Because every pair contributes equal and opposite forces, the net force vanishes to
roundoff and the power ``sum_x F . u = sum_pairs m e_v . (u(x + v) - u(x))`` is
Galilean invariant: the repulsion only exchanges momentum between approaching
interfaces and does work only through their relative normal motion.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._discretization import LatticeBoltzmannDiscretization
from ._interfacial import isotropic_gradient
from ._lattice import LatticeBoltzmannVelocitySet
from ._scaling import LatticeBoltzmannScaling


# Films thicker than a few interface widths are resolved hydrodynamically; the
# interaction models the unresolved near-contact regime only.
_MAXIMUM_INTERACTION_RANGE = 8
# A node pair is reported active when its normalized facing weight
# w_a s-_a w_b s+_b exceeds this value, i.e. when both nodes lie inside diffuse
# interfaces; exponentially small tanh tails (for example a drop facing its own
# periodic image across the whole box) still act but are not counted.
_ACTIVE_PAIR_WEIGHT = 1.0e-3


def _half_space_offsets(dimension: int, radius: int, /) -> tuple[tuple[int, ...], ...]:
    """Integer offsets ``0 < |v| <= radius`` whose first nonzero component is positive."""

    offsets = [
        offset
        for offset in itertools.product(range(-radius, radius + 1), repeat=dimension)
        if 0 < sum(component * component for component in offset) <= radius * radius
        and next(component for component in offset if component != 0) > 0
    ]
    return tuple(
        sorted(
            offsets,
            key=lambda offset: (
                sum(component * component for component in offset),
                offset,
            ),
        )
    )


class NearContactRepulsionPlan(StrictModule, NonTrainableState):
    """Static structure of one pairwise antisymmetric near-contact repulsion.

    ``repelling_pairs`` names the component pairs whose facing interfaces repel; a
    pair may repeat one component (droplets of one color repelling each other
    across a film of another). The runtime strength vector of the color-gradient
    parameters follows the declared pair order. ``interaction_range`` is the
    largest node-pair distance in lattice cells.
    """

    repelling_pairs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    interaction_range: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        repelling_pairs: Sequence[Sequence[str]],
        /,
        *,
        interaction_range: int = 4,
    ) -> None:
        if isinstance(repelling_pairs, str):
            raise TypeError("repelling_pairs must be a sequence of component-id pairs.")
        pairs: list[tuple[str, str]] = []
        for pair in repelling_pairs:
            if isinstance(pair, str) or len(pair) != 2:
                raise ValueError("Every repelling pair must name exactly two components.")
            first, second = (str(value) for value in pair)
            if not first or not second:
                raise ValueError("Repelling-pair component ids must be non-empty.")
            pairs.append((min(first, second), max(first, second)))
        if not pairs:
            raise ValueError("A near-contact plan requires at least one repelling pair.")
        if len(set(pairs)) != len(pairs):
            raise ValueError("Repelling pairs must be unique.")
        if isinstance(interaction_range, bool) or not isinstance(interaction_range, int):
            raise TypeError("interaction_range must be an integer.")
        if not 1 <= interaction_range <= _MAXIMUM_INTERACTION_RANGE:
            raise ValueError(
                "interaction_range must lie in "
                f"[1, {_MAXIMUM_INTERACTION_RANGE}] lattice cells."
            )
        self.repelling_pairs = tuple(pairs)
        self.interaction_range = interaction_range
        self.plan_id = canonical_fingerprint(
            {
                "kind": "near-contact-repulsion-plan",
                "repelling_pairs": [list(pair) for pair in pairs],
                "interaction_range": interaction_range,
                "kernel": "linear-decay",
                "indicator": "four-c-one-minus-c",
            }
        )

    @property
    def pair_count(self) -> int:
        return len(self.repelling_pairs)

    def prepare(
        self,
        component_ids: Sequence[str],
        discretization: LatticeBoltzmannDiscretization,
        /,
    ) -> "PreparedNearContactRepulsion":
        return PreparedNearContactRepulsion(self, component_ids, discretization)


class NearContactForce(StrictModule):
    """Lattice-unit repulsion force density and its pair statistics."""

    force_density: Array
    active_pair_count: Array
    maximum_pair_force: Array


class NearContactRepulsionEvidence(StrictModule):
    """Physical-unit momentum and work evidence of one repulsion evaluation.

    ``net_force`` is the summed force (zero up to roundoff by construction);
    ``net_force_residual`` is its norm relative to the summed force magnitude.
    ``power`` is ``sum F . u`` over fluid cells and ``work`` is ``power * dt``, the
    midpoint-velocity work increment committed to the state ledger on acceptance.
    ``active_pair_count`` counts (node pair, repelling pair) combinations whose
    normalized facing weight exceeds ``1e-3`` (both nodes inside diffuse
    interfaces); ``maximum_pair_force`` is the largest pair force density.
    """

    net_force: Array
    net_force_residual: Array
    power: Array
    work: Array
    active_pair_count: Array
    maximum_pair_force: Array
    strength_valid: Array


class PreparedNearContactRepulsion(StrictModule, NonTrainableState):
    """Near-contact repulsion bound to component identity and one lattice grid."""

    plan: NearContactRepulsionPlan
    velocity_set: LatticeBoltzmannVelocitySet
    components: tuple[int, ...] = eqx.field(static=True)
    pair_components: tuple[tuple[int, int], ...] = eqx.field(static=True)
    offsets: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    grid_shape: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: NearContactRepulsionPlan,
        component_ids: Sequence[str],
        discretization: LatticeBoltzmannDiscretization,
        /,
    ) -> None:
        if not isinstance(plan, NearContactRepulsionPlan):
            raise TypeError("plan must be NearContactRepulsionPlan.")
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("discretization must be an LBM discretization.")
        identifiers = tuple(str(value) for value in component_ids)
        unknown = sorted(
            {value for pair in plan.repelling_pairs for value in pair} - set(identifiers)
        )
        if unknown:
            raise ValueError(f"Repelling pairs name unknown components {unknown}.")
        indexed = tuple(
            tuple(sorted(identifiers.index(value) for value in pair))
            for pair in plan.repelling_pairs
        )
        components = tuple(sorted({index for pair in indexed for index in pair}))
        local = tuple(
            (components.index(first), components.index(second))
            for first, second in indexed
        )
        velocity_set = discretization.velocity_set
        offsets = _half_space_offsets(velocity_set.dimension, plan.interaction_range)
        self.plan = plan
        self.velocity_set = velocity_set
        self.components = components
        self.pair_components = local
        self.offsets = offsets
        self.periodic = discretization.periodic
        self.grid_shape = discretization.grid.shape
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-near-contact-repulsion",
                "plan": plan.plan_id,
                "component_ids": list(identifiers),
                "discretization": discretization.prepared_id,
                "offset_count": len(offsets),
            }
        )

    @property
    def offset_count(self) -> int:
        return len(self.offsets)

    def _pair_mask(self, offset: Array, fluid: Array, /) -> Array:
        axes = tuple(range(len(self.grid_shape)))
        partner_fluid = jnp.roll(fluid, -offset, axis=axes)
        valid = fluid & partner_fluid
        indices = jnp.indices(self.grid_shape, dtype=jnp.int32)
        for axis, periodic in enumerate(self.periodic):
            if periodic:
                continue
            target = indices[axis] + offset[axis]
            valid = valid & (target >= 0) & (target < self.grid_shape[axis])
        return valid

    def force(
        self,
        concentrations: Array,
        fluid: Array,
        strengths: Array,
        gradient_floor: Array,
        /,
    ) -> NearContactForce:
        """Return the lattice force density for lattice-unit pair strengths."""

        dimension = self.velocity_set.dimension
        dtype = concentrations.dtype
        pair_strengths = jnp.asarray(strengths, dtype=dtype)
        floor = jnp.asarray(gradient_floor, dtype=dtype)
        values = concentrations[np.asarray(self.components)]
        indicator = 4.0 * values * (1.0 - values)
        gradients = jax.vmap(lambda field: isotropic_gradient(field, self.velocity_set))(
            values
        )
        magnitude = jnp.sqrt(ein.contract("m...d,m...d->m...", gradients, gradients))
        normals = jnp.where(
            (magnitude > floor)[..., None],
            gradients / jnp.maximum(magnitude, floor)[..., None],
            0.0,
        )
        first = np.asarray([pair[0] for pair in self.pair_components])
        second = np.asarray([pair[1] for pair in self.pair_components])
        distinct = jnp.asarray(
            (first != second).reshape((-1,) + (1,) * dimension), dtype=jnp.bool_
        )
        host_offsets = np.asarray(self.offsets, dtype=np.float64)
        lengths = np.sqrt(np.sum(host_offsets**2, axis=-1))
        radius = float(self.plan.interaction_range)
        units = jnp.asarray(host_offsets / lengths[:, None], dtype=dtype)
        kernel = jnp.asarray(1.0 - lengths / (radius + 1.0), dtype=dtype)
        spatial = tuple(range(1, dimension + 1))
        grid_axes = tuple(range(dimension))

        def accumulate(
            carry: tuple[Array, Array, Array], item: tuple[Array, Array, Array]
        ) -> tuple[tuple[Array, Array, Array], None]:
            force, active, largest = carry
            offset, unit, weight = item
            partner_indicator = jnp.roll(indicator, -offset, axis=spatial)
            partner_normals = jnp.roll(normals, -offset, axis=spatial)
            leaving = indicator * jnp.maximum(
                -ein.contract("m...d,d->m...", normals, unit), 0.0
            )
            arriving = partner_indicator * jnp.maximum(
                ein.contract("m...d,d->m...", partner_normals, unit), 0.0
            )
            terms = leaving[first] * arriving[second] + jnp.where(
                distinct, leaving[second] * arriving[first], 0.0
            )
            valid = self._pair_mask(offset, fluid)
            magnitude_ = weight * ein.contract("r,r...->...", pair_strengths, terms)
            magnitude_ = jnp.where(valid, magnitude_, 0.0)
            received = jnp.roll(magnitude_, offset, axis=grid_axes)
            force = force + (received - magnitude_)[..., None] * unit
            engaged = (terms > _ACTIVE_PAIR_WEIGHT) & (pair_strengths > 0.0).reshape(
                (-1,) + (1,) * dimension
            )
            active = active + jnp.sum(engaged & valid, dtype=jnp.int32)
            largest = jnp.maximum(largest, jnp.max(magnitude_))
            return (force, active, largest), None

        initial = (
            jnp.zeros((*self.grid_shape, dimension), dtype=dtype),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=dtype),
        )
        (force, active, largest), _ = jax.lax.scan(
            accumulate,
            initial,
            (jnp.asarray(self.offsets, dtype=jnp.int32), units, kernel),
        )
        return NearContactForce(force, active, largest)

    def evidence(
        self,
        force: NearContactForce,
        velocity: Array,
        fluid: Array,
        scaling: LatticeBoltzmannScaling,
        strength_valid: Array,
        /,
    ) -> NearContactRepulsionEvidence:
        """Convert one lattice evaluation into physical momentum and work evidence."""

        dtype = force.force_density.dtype
        dimension = self.velocity_set.dimension
        dx = scaling.cell_size.astype(dtype)
        dt = scaling.time_step.astype(dtype)
        rho0 = scaling.reference_density.astype(dtype)
        force_density_scale = rho0 * dx / dt**2
        volume = dx**dimension
        masked = jnp.where(fluid[..., None], force.force_density, 0.0)
        net = jnp.sum(masked, axis=tuple(range(dimension)))
        total = jnp.sum(jnp.sqrt(ein.contract("...d,...d->...", masked, masked)))
        residual = jnp.where(
            total > 0.0,
            jnp.sqrt(jnp.sum(net**2)) / jnp.where(total > 0.0, total, 1.0),
            0.0,
        )
        power = jnp.sum(ein.contract("...d,...d->...", masked, velocity))
        physical_power = power * force_density_scale * volume * dx / dt
        return NearContactRepulsionEvidence(
            net_force=net * force_density_scale * volume,
            net_force_residual=residual,
            power=physical_power,
            work=physical_power * dt,
            active_pair_count=force.active_pair_count,
            maximum_pair_force=force.maximum_pair_force * force_density_scale,
            strength_valid=strength_valid,
        )


__all__ = [
    "NearContactForce",
    "NearContactRepulsionEvidence",
    "NearContactRepulsionPlan",
    "PreparedNearContactRepulsion",
]
