#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


def _morton3(integer_coordinates: Array, depth: int, /) -> Array:
    coordinates = integer_coordinates.astype(jnp.uint32)
    key = jnp.zeros((coordinates.shape[0],), dtype=jnp.uint32)
    for bit in range(depth):
        for axis in range(3):
            key = key | (((coordinates[:, axis] >> bit) & 1) << (3 * bit + axis))
    return key


class NewtonianPairKernel(StrictModule, NonTrainableState):
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    cutoff: float | None = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        gravitational_constant: float,
        /,
        *,
        softening: float,
        cutoff: float | None = None,
    ):
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        cutoff_ = None if cutoff is None else float(cutoff)
        if (
            not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or (cutoff_ is not None and (not np.isfinite(cutoff_) or cutoff_ <= 0.0))
        ):
            raise ValueError("Newtonian pair kernel is invalid.")
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.cutoff = cutoff_
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "newtonian-pair-kernel",
                "gravitational_constant": gravity,
                "softening": epsilon,
                "cutoff": cutoff_,
            }
        )

    def acceleration(
        self,
        target_positions: ArrayLike,
        source_positions: ArrayLike,
        source_masses: ArrayLike,
        /,
        *,
        exclude_diagonal: bool = False,
    ) -> Array:
        targets = jnp.asarray(target_positions)
        sources = jnp.asarray(source_positions, dtype=targets.dtype)
        masses = jnp.asarray(source_masses, dtype=targets.dtype)
        if (
            targets.ndim != 2
            or sources.ndim != 2
            or targets.shape[1] != sources.shape[1]
            or masses.shape != (sources.shape[0],)
        ):
            raise ValueError("Pair-kernel source/target shapes are invalid.")
        displacement = sources[None, :, :] - targets[:, None, :]
        radius_squared = jnp.sum(displacement**2, axis=-1) + self.softening**2
        mask = jnp.ones(radius_squared.shape, dtype=bool)
        if exclude_diagonal:
            if targets.shape[0] != sources.shape[0]:
                raise ValueError(
                    "Diagonal exclusion requires equal source/target counts."
                )
            mask = mask & ~jnp.eye(targets.shape[0], dtype=bool)
        if self.cutoff is not None:
            mask = mask & (radius_squared <= self.cutoff**2 + self.softening**2)
        contribution = (
            self.gravitational_constant
            * masses[None, :, None]
            * displacement
            / radius_squared[..., None] ** 1.5
        )
        return jnp.sum(jnp.where(mask[..., None], contribution, 0.0), axis=1)


class ParticleGravityEvidence(StrictModule):
    net_force: Array
    maximum_acceleration: Array
    interaction_count: Array
    approximation_error: Array
    finite: Array
    successful: Array


class DirectParticleGravityPlan(StrictModule, NonTrainableState):
    kernel: NewtonianPairKernel

    def __init__(self, kernel: NewtonianPairKernel, /):
        self.kernel = kernel

    def evaluate(
        self,
        positions: ArrayLike,
        masses: ArrayLike,
        active_mask: ArrayLike | None = None,
        /,
    ) -> tuple[Array, ParticleGravityEvidence]:
        position = jnp.asarray(positions)
        mass = jnp.asarray(masses, dtype=position.dtype)
        active = (
            jnp.ones((position.shape[0],), dtype=bool)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        acceleration = self.kernel.acceleration(
            position,
            position,
            jnp.where(active, mass, 0.0),
            exclude_diagonal=True,
        )
        acceleration = jnp.where(active[:, None], acceleration, 0.0)
        finite = jnp.all(jnp.isfinite(acceleration))
        evidence = ParticleGravityEvidence(
            jnp.sum(mass[:, None] * acceleration, axis=0),
            jnp.max(jnp.sqrt(jnp.sum(acceleration**2, axis=-1))),
            jnp.sum(active) * jnp.maximum(jnp.sum(active) - 1, 0),
            jnp.asarray(0.0, dtype=position.dtype),
            finite,
            finite,
        )
        return acceleration, evidence


class DistributedParticleLayout(StrictModule, NonTrainableState):
    device_count: int = eqx.field(static=True)
    capacity_per_device: int = eqx.field(static=True)
    key_boundaries: Array
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        device_count: int,
        capacity_per_device: int,
        key_boundaries: ArrayLike,
        /,
    ):
        devices = int(device_count)
        capacity = int(capacity_per_device)
        boundaries = jnp.asarray(key_boundaries, dtype=jnp.uint32)
        if devices <= 0 or capacity <= 0 or boundaries.shape != (devices + 1,):
            raise ValueError("Distributed particle layout is invalid.")
        boundaries = eqx.error_if(
            boundaries,
            jnp.any(boundaries[1:] < boundaries[:-1]),
            "Distributed Morton boundaries must be increasing.",
        )
        self.device_count = devices
        self.capacity_per_device = capacity
        self.key_boundaries = boundaries
        self.layout_id = canonical_fingerprint(
            {
                "kind": "distributed-particle-layout",
                "device_count": devices,
                "capacity_per_device": capacity,
                "boundaries": np.asarray(boundaries).tolist(),
            }
        )

    def owners(self, morton_keys: ArrayLike, /) -> Array:
        keys = jnp.asarray(morton_keys, dtype=jnp.uint32)
        return jnp.clip(
            jnp.searchsorted(self.key_boundaries[1:], keys, side="right"),
            0,
            self.device_count - 1,
        )


class PreparedParticleOctree3D(StrictModule):
    positions: Array
    masses: Array
    active_mask: Array
    morton_keys: Array
    permutation: Array
    leaf_indices: Array
    leaf_mass: Array
    leaf_center_of_mass: Array
    leaf_quadrupole: Array
    leaf_centers: Array
    leaf_half_size: Array
    box_size: tuple[float, float, float] = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class ParticleOctreePlan3D(StrictModule, NonTrainableState):
    box_size: tuple[float, float, float] = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    leaf_count: int = eqx.field(static=True)
    leaf_centers: Array
    leaf_half_size: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, box_size: tuple[float, float, float], depth: int, /):
        lengths = tuple(float(value) for value in box_size)
        depth_ = int(depth)
        if len(lengths) != 3 or any(
            not np.isfinite(value) or value <= 0.0 for value in lengths
        ):
            raise ValueError("Particle octree requires a finite positive 3-D box.")
        if depth_ < 1 or depth_ > 10:
            raise ValueError("Particle octree depth must lie in [1,10].")
        count_axis = 1 << depth_
        integer = np.stack(
            np.meshgrid(
                np.arange(count_axis),
                np.arange(count_axis),
                np.arange(count_axis),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, 3))
        centers = (integer + 0.5) * np.asarray(lengths)[None, :] / count_axis
        self.box_size = lengths
        self.depth = depth_
        self.leaf_count = count_axis**3
        self.leaf_centers = jnp.asarray(centers)
        self.leaf_half_size = jnp.asarray(lengths) / (2.0 * count_axis)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "uniform-particle-octree-3d",
                "box_size": list(lengths),
                "depth": depth_,
            }
        )

    def prepare(
        self,
        positions: ArrayLike,
        masses: ArrayLike,
        active_mask: ArrayLike | None = None,
        /,
    ) -> PreparedParticleOctree3D:
        position = jnp.asarray(positions)
        mass = jnp.asarray(masses, dtype=position.dtype)
        active = (
            jnp.ones((position.shape[0],), dtype=bool)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        if (
            position.ndim != 2
            or position.shape[1] != 3
            or mass.shape != active.shape
            or mass.shape != (position.shape[0],)
        ):
            raise ValueError(
                "Particle octree positions, masses, and active mask disagree."
            )
        position = eqx.error_if(
            position,
            jnp.any(jnp.where(active[:, None], ~jnp.isfinite(position), False))
            | jnp.any(jnp.where(active, ~jnp.isfinite(mass), False))
            | jnp.any(jnp.where(active, mass <= 0.0, False))
            | jnp.any(jnp.where(active[:, None], position < 0.0, False))
            | jnp.any(
                jnp.where(
                    active[:, None],
                    position >= jnp.asarray(self.box_size),
                    False,
                )
            ),
            "Active octree particles must be finite, positive-mass, and inside the box.",
        )
        count_axis = 1 << self.depth
        integer = jnp.floor(position / jnp.asarray(self.box_size) * count_axis).astype(
            jnp.int32
        )
        integer = jnp.clip(integer, 0, count_axis - 1)
        keys = _morton3(integer, self.depth)
        inactive_key = jnp.asarray((1 << (3 * self.depth)) - 1, dtype=keys.dtype)
        sorted_keys = jnp.where(active, keys, inactive_key)
        permutation = jnp.lexsort((jnp.arange(position.shape[0]), sorted_keys))
        leaf_indices = (
            integer[:, 0] * count_axis**2 + integer[:, 1] * count_axis + integer[:, 2]
        )
        safe_mass = jnp.where(active, mass, 0.0)
        leaf_mass = (
            jnp.zeros((self.leaf_count,), dtype=mass.dtype)
            .at[leaf_indices]
            .add(safe_mass)
        )
        weighted_position = (
            jnp.zeros((self.leaf_count, 3), dtype=position.dtype)
            .at[leaf_indices]
            .add(safe_mass[:, None] * position)
        )
        safe_leaf_mass = jnp.where(leaf_mass > 0.0, leaf_mass, 1.0)
        center_of_mass = weighted_position / safe_leaf_mass[:, None]
        centered = position - center_of_mass[leaf_indices]
        outer = centered[:, :, None] * centered[:, None, :]
        radius_squared = jnp.sum(centered**2, axis=-1)
        quadrupole_particle = safe_mass[:, None, None] * (
            3.0 * outer - radius_squared[:, None, None] * jnp.eye(3, dtype=position.dtype)
        )
        quadrupole = (
            jnp.zeros((self.leaf_count, 3, 3), dtype=position.dtype)
            .at[leaf_indices]
            .add(quadrupole_particle)
        )
        return PreparedParticleOctree3D(
            position,
            mass,
            active,
            keys,
            permutation,
            leaf_indices,
            leaf_mass,
            center_of_mass,
            quadrupole,
            self.leaf_centers.astype(position.dtype),
            self.leaf_half_size.astype(position.dtype),
            self.box_size,
            self.depth,
            canonical_fingerprint(
                {
                    "kind": "prepared-particle-octree",
                    "plan": self.plan_id,
                    "capacity": position.shape[0],
                }
            ),
        )


class TreeGravityEvidence(StrictModule):
    net_force: Array
    maximum_acceleration: Array
    accepted_leaf_interactions: Array
    direct_particle_interactions: Array
    estimated_relative_error: Array
    finite: Array
    successful: Array


class TreeGravityResult(StrictModule):
    acceleration: Array
    evidence: TreeGravityEvidence
    successful: Array


class BarnesHutGravityPlan(StrictModule, NonTrainableState):
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    opening_angle: float = eqx.field(static=True)
    use_quadrupole: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gravitational_constant: float,
        /,
        *,
        softening: float,
        opening_angle: float = 0.5,
        use_quadrupole: bool = True,
    ):
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        theta = float(opening_angle)
        if (
            not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or not np.isfinite(theta)
            or theta <= 0.0
            or theta >= 1.0
        ):
            raise ValueError("Barnes-Hut policy is invalid.")
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.opening_angle = theta
        self.use_quadrupole = bool(use_quadrupole)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "barnes-hut-gravity",
                "gravitational_constant": gravity,
                "softening": epsilon,
                "opening_angle": theta,
                "use_quadrupole": bool(use_quadrupole),
            }
        )

    def evaluate(
        self,
        tree: PreparedParticleOctree3D,
        /,
        *,
        short_range_scale: float | None = None,
        cutoff: float | None = None,
    ) -> TreeGravityResult:
        position = tree.positions
        leaf_displacement = tree.leaf_center_of_mass[None, :, :] - position[:, None, :]
        leaf_distance_squared = jnp.sum(leaf_displacement**2, axis=-1) + self.softening**2
        leaf_distance = jnp.sqrt(leaf_distance_squared)
        size = 2.0 * jnp.max(tree.leaf_half_size)
        same_leaf = tree.leaf_indices[:, None] == jnp.arange(tree.leaf_mass.size)[None, :]
        accept = (
            (tree.leaf_mass[None, :] > 0.0)
            & ~same_leaf
            & (size / leaf_distance < self.opening_angle)
        )
        kernel = leaf_distance_squared ** (-1.5)
        if short_range_scale is not None:
            scale = jnp.asarray(short_range_scale, dtype=position.dtype)
            argument = leaf_distance / (2.0 * scale)
            kernel = kernel * (
                jax.scipy.special.erfc(argument)
                + leaf_distance / (scale * jnp.sqrt(jnp.pi)) * jnp.exp(-(argument**2))
            )
        if cutoff is not None:
            kernel = jnp.where(leaf_distance <= cutoff, kernel, 0.0)
            accept = accept & (leaf_distance <= cutoff)
        monopole = (
            self.gravitational_constant
            * tree.leaf_mass[None, :, None]
            * leaf_displacement
            * kernel[..., None]
        )
        far = jnp.where(accept[..., None], monopole, 0.0)
        if self.use_quadrupole:
            q_r = contract("lij,nlj->nli", tree.leaf_quadrupole, leaf_displacement)
            r_q_r = contract("nli,nli->nl", leaf_displacement, q_r)
            quadrupole = self.gravitational_constant * (
                2.5 * r_q_r[..., None] * leaf_displacement / leaf_distance[..., None] ** 7
                - q_r / leaf_distance[..., None] ** 5
            )
            far = far + jnp.where(accept[..., None], quadrupole, 0.0)
        source_displacement = position[None, :, :] - position[:, None, :]
        source_distance_squared = (
            jnp.sum(source_displacement**2, axis=-1) + self.softening**2
        )
        source_distance = jnp.sqrt(source_distance_squared)
        source_leaf_far = accept[:, tree.leaf_indices]
        direct_mask = tree.active_mask[None, :] & ~source_leaf_far
        direct_mask = direct_mask & ~jnp.eye(position.shape[0], dtype=bool)
        direct_kernel = source_distance_squared ** (-1.5)
        if short_range_scale is not None:
            scale = jnp.asarray(short_range_scale, dtype=position.dtype)
            argument = source_distance / (2.0 * scale)
            direct_kernel = direct_kernel * (
                jax.scipy.special.erfc(argument)
                + source_distance / (scale * jnp.sqrt(jnp.pi)) * jnp.exp(-(argument**2))
            )
        if cutoff is not None:
            direct_mask = direct_mask & (source_distance <= cutoff)
        direct = (
            self.gravitational_constant
            * tree.masses[None, :, None]
            * source_displacement
            * direct_kernel[..., None]
        )
        acceleration = jnp.sum(far, axis=1) + jnp.sum(
            jnp.where(direct_mask[..., None], direct, 0.0), axis=1
        )
        acceleration = jnp.where(tree.active_mask[:, None], acceleration, 0.0)
        finite = jnp.all(jnp.isfinite(acceleration))
        net_force = jnp.sum(tree.masses[:, None] * acceleration, axis=0)
        evidence = TreeGravityEvidence(
            net_force,
            jnp.max(jnp.sqrt(jnp.sum(acceleration**2, axis=-1))),
            jnp.sum(accept),
            jnp.sum(direct_mask),
            jnp.max(
                jnp.where(
                    accept,
                    (size / leaf_distance) ** (3 if self.use_quadrupole else 2),
                    0.0,
                )
            ),
            finite,
            finite,
        )
        return TreeGravityResult(acceleration, evidence, finite)


class CartesianExpansionSpace(StrictModule, NonTrainableState):
    order: int = eqx.field(static=True)
    exponents: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)

    def __init__(self, order: int, /):
        order_ = int(order)
        if order_ < 1 or order_ > 6:
            raise ValueError("Cartesian FMM order must lie in [1,6].")
        exponents = tuple(
            (i, j, k)
            for total in range(order_ + 1)
            for i in range(total + 1)
            for j in range(total - i + 1)
            for k in (total - i - j,)
        )
        self.order = order_
        self.exponents = exponents
        self.coefficient_count = len(exponents)


class CartesianFMMOperators(StrictModule, NonTrainableState):
    """First-order Cartesian P2M/M2M/M2L/L2L/L2P/P2P operators."""

    expansion: CartesianExpansionSpace
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)

    def __init__(
        self,
        expansion: CartesianExpansionSpace,
        gravitational_constant: float,
        softening: float,
        /,
    ):
        if expansion.order != 1:
            raise ValueError(
                "Current qualified Cartesian FMM operators require order one."
            )
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        if (
            not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
        ):
            raise ValueError("Cartesian FMM operator constants are invalid.")
        self.expansion = expansion
        self.gravitational_constant = gravity
        self.softening = epsilon

    def p2m(
        self,
        positions: ArrayLike,
        masses: ArrayLike,
        center: ArrayLike,
        /,
    ) -> Array:
        position = jnp.asarray(positions)
        mass = jnp.asarray(masses, dtype=position.dtype)
        center_ = jnp.asarray(center, dtype=position.dtype)
        relative = position - center_
        coefficients = []
        for i, j, k in self.expansion.exponents:
            coefficients.append(
                jnp.sum(
                    mass * relative[:, 0] ** i * relative[:, 1] ** j * relative[:, 2] ** k
                )
            )
        return jnp.stack(coefficients)

    def m2m(self, coefficients: ArrayLike, shift: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        shift_ = jnp.asarray(shift, dtype=values.dtype)
        output = []
        for alpha in self.expansion.exponents:
            total = jnp.asarray(0.0, dtype=values.dtype)
            for index, beta in enumerate(self.expansion.exponents):
                if all(beta[axis] <= alpha[axis] for axis in range(3)):
                    multiplier = 1.0
                    for axis in range(3):
                        if alpha[axis] - beta[axis] == 1:
                            multiplier = multiplier * shift_[axis]
                    total = total + multiplier * values[index]
            output.append(total)
        return jnp.stack(output)

    def m2l(
        self,
        multipole: ArrayLike,
        source_center: ArrayLike,
        target_center: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(multipole)
        source = jnp.asarray(source_center, dtype=values.dtype)
        target = jnp.asarray(target_center, dtype=values.dtype)
        displacement = target - source
        radius_squared = jnp.sum(displacement**2) + self.softening**2
        radius = jnp.sqrt(radius_squared)
        mass_index = self.expansion.exponents.index((0, 0, 0))
        dipole = jnp.stack(
            tuple(
                values[self.expansion.exponents.index(exponent)]
                for exponent in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            )
        )
        mass = values[mass_index]
        potential = -self.gravitational_constant * (
            mass / radius + jnp.dot(dipole, displacement) / radius**3
        )
        gradient = self.gravitational_constant * (
            mass * displacement / radius**3
            + dipole / radius**3
            - 3.0 * jnp.dot(dipole, displacement) * displacement / radius**5
        )
        local = jnp.zeros_like(values).at[mass_index].set(potential)
        for axis, exponent in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
            local = local.at[self.expansion.exponents.index(exponent)].set(gradient[axis])
        return local

    def l2l(self, local: ArrayLike, shift: ArrayLike, /) -> Array:
        values = jnp.asarray(local)
        shift_ = jnp.asarray(shift, dtype=values.dtype)
        mass_index = self.expansion.exponents.index((0, 0, 0))
        gradient = jnp.stack(
            tuple(
                values[self.expansion.exponents.index(exponent)]
                for exponent in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            )
        )
        return values.at[mass_index].set(values[mass_index] + jnp.dot(gradient, shift_))

    def l2p(self, local: ArrayLike, displacement: ArrayLike, /) -> tuple[Array, Array]:
        values = jnp.asarray(local)
        offset = jnp.asarray(displacement, dtype=values.dtype)
        mass_index = self.expansion.exponents.index((0, 0, 0))
        gradient = jnp.stack(
            tuple(
                values[self.expansion.exponents.index(exponent)]
                for exponent in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            )
        )
        return values[mass_index] + jnp.dot(gradient, offset), -gradient

    def p2p(
        self,
        target: ArrayLike,
        source_positions: ArrayLike,
        source_masses: ArrayLike,
        /,
    ) -> Array:
        target_ = jnp.asarray(target)
        source = jnp.asarray(source_positions, dtype=target_.dtype)
        mass = jnp.asarray(source_masses, dtype=target_.dtype)
        displacement = source - target_
        radius_squared = jnp.sum(displacement**2, axis=-1) + self.softening**2
        return jnp.sum(
            self.gravitational_constant
            * mass[:, None]
            * displacement
            / radius_squared[:, None] ** 1.5,
            axis=0,
        )


class FMMEvidence(StrictModule):
    near_interactions: Array
    far_interactions: Array
    estimated_relative_error: Array
    finite: Array
    successful: Array


class UniformFMMPlan(StrictModule, NonTrainableState):
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    expansion: CartesianExpansionSpace
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gravitational_constant: float,
        expansion: CartesianExpansionSpace,
        /,
        *,
        softening: float,
    ):
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        if (
            not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
        ):
            raise ValueError("Uniform FMM policy is invalid.")
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.expansion = expansion
        self.plan_id = canonical_fingerprint(
            {
                "kind": "uniform-cartesian-fmm",
                "gravitational_constant": gravity,
                "softening": epsilon,
                "order": expansion.order,
            }
        )

    def evaluate(self, tree: PreparedParticleOctree3D, /) -> TreeGravityResult:
        leaf_centers = tree.leaf_centers
        displacement = tree.leaf_center_of_mass[None, :, :] - leaf_centers[:, None, :]
        distance_squared = jnp.sum(displacement**2, axis=-1) + self.softening**2
        distance = jnp.sqrt(distance_squared)
        cell_size = 2.0 * jnp.max(tree.leaf_half_size)
        far = (distance > 2.0 * cell_size) & (tree.leaf_mass[None, :] > 0.0)
        far = far & ~jnp.eye(tree.leaf_mass.size, dtype=bool)
        leaf_local_acceleration = jnp.sum(
            jnp.where(
                far[..., None],
                self.gravitational_constant
                * tree.leaf_mass[None, :, None]
                * displacement
                / distance[..., None] ** 3,
                0.0,
            ),
            axis=1,
        )
        local = leaf_local_acceleration[tree.leaf_indices]
        particle_displacement = tree.positions[None, :, :] - tree.positions[:, None, :]
        particle_distance_squared = (
            jnp.sum(particle_displacement**2, axis=-1) + self.softening**2
        )
        source_far = far[tree.leaf_indices[:, None], tree.leaf_indices[None, :]]
        near = (
            tree.active_mask[None, :]
            & ~source_far
            & ~jnp.eye(tree.positions.shape[0], dtype=bool)
        )
        direct = (
            self.gravitational_constant
            * tree.masses[None, :, None]
            * particle_displacement
            / particle_distance_squared[..., None] ** 1.5
        )
        acceleration = local + jnp.sum(jnp.where(near[..., None], direct, 0.0), axis=1)
        acceleration = jnp.where(tree.active_mask[:, None], acceleration, 0.0)
        finite = jnp.all(jnp.isfinite(acceleration))
        ratio = cell_size / jnp.maximum(distance, cell_size)
        error = jnp.max(jnp.where(far, ratio ** (self.expansion.order + 1), 0.0))
        evidence = TreeGravityEvidence(
            jnp.sum(tree.masses[:, None] * acceleration, axis=0),
            jnp.max(jnp.sqrt(jnp.sum(acceleration**2, axis=-1))),
            jnp.sum(far),
            jnp.sum(near),
            error,
            finite,
            finite,
        )
        return TreeGravityResult(acceleration, evidence, finite)


class PeriodicEwaldEvidence(StrictModule):
    real_space_acceleration: Array
    reciprocal_acceleration: Array
    net_force: Array
    finite: Array
    successful: Array


class PeriodicEwaldResult(StrictModule):
    acceleration: Array
    evidence: PeriodicEwaldEvidence
    successful: Array


class PeriodicEwaldForcePlan(StrictModule, NonTrainableState):
    """Small-N softened-neutral periodic Ewald acceleration reference."""

    box_size: tuple[float, ...] = eqx.field(static=True)
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    real_offsets: Array
    wavevectors: Array
    volume: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        box_size: tuple[float, ...],
        gravitational_constant: float,
        /,
        *,
        softening: float,
        alpha: float,
        real_shells: int = 2,
        reciprocal_modes: int = 4,
    ):
        lengths = tuple(float(value) for value in box_size)
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        alpha_ = float(alpha)
        real = int(real_shells)
        reciprocal = int(reciprocal_modes)
        if (
            not lengths
            or any(not np.isfinite(value) or value <= 0.0 for value in lengths)
            or not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or not np.isfinite(alpha_)
            or alpha_ <= 0.0
            or real < 0
            or reciprocal < 1
        ):
            raise ValueError("Periodic Ewald policy is invalid.")
        dimension = len(lengths)
        integer_offsets = np.asarray(
            tuple(product(range(-real, real + 1), repeat=dimension)), dtype=float
        )
        reciprocal_indices = np.asarray(
            tuple(
                index
                for index in product(range(-reciprocal, reciprocal + 1), repeat=dimension)
                if any(value != 0 for value in index)
            ),
            dtype=float,
        )
        wavevectors = 2.0 * np.pi * reciprocal_indices / np.asarray(lengths)[None, :]
        self.box_size = lengths
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.alpha = alpha_
        self.real_offsets = jnp.asarray(integer_offsets * np.asarray(lengths)[None, :])
        self.wavevectors = jnp.asarray(wavevectors)
        self.volume = float(np.prod(lengths))
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-ewald-force",
                "box_size": list(lengths),
                "gravitational_constant": gravity,
                "softening": epsilon,
                "alpha": alpha_,
                "real_shells": real,
                "reciprocal_modes": reciprocal,
            }
        )

    def evaluate(self, positions: ArrayLike, masses: ArrayLike, /) -> PeriodicEwaldResult:
        position = jnp.asarray(positions)
        mass = jnp.asarray(masses, dtype=position.dtype)
        if (
            position.ndim != 2
            or position.shape[1] != len(self.box_size)
            or mass.shape != (position.shape[0],)
        ):
            raise ValueError("Periodic Ewald positions/masses have incompatible shapes.")
        position = eqx.error_if(
            position,
            jnp.any(~jnp.isfinite(position))
            | jnp.any(~jnp.isfinite(mass))
            | jnp.any(mass <= 0.0),
            "Periodic Ewald inputs must be finite with positive masses.",
        )
        target = position[:, None, None, :]
        source = position[None, :, None, :] + self.real_offsets[None, None, :, :]
        displacement = source - target
        distance_squared = jnp.sum(displacement**2, axis=-1) + self.softening**2
        distance = jnp.sqrt(distance_squared)
        zero_offset = jnp.all(self.real_offsets == 0.0, axis=-1)
        self_pair = (
            jnp.eye(position.shape[0], dtype=bool)[:, :, None]
            & zero_offset[None, None, :]
        )
        screening = jax.scipy.special.erfc(self.alpha * distance) + (
            2.0
            * self.alpha
            * distance
            / jnp.sqrt(jnp.pi)
            * jnp.exp(-((self.alpha * distance) ** 2))
        )
        inverse_cube = jnp.where(self_pair, 0.0, screening / distance**3)
        real_acceleration = jnp.sum(
            self.gravitational_constant
            * mass[None, :, None, None]
            * displacement
            * inverse_cube[..., None],
            axis=(1, 2),
        )
        k = self.wavevectors.astype(position.dtype)
        k_squared = jnp.sum(k**2, axis=-1)
        source_phase = contract("kd,nd->kn", k, position)
        density_real = contract("n,kn->k", mass, jnp.cos(source_phase))
        density_imag = -contract("n,kn->k", mass, jnp.sin(source_phase))
        target_phase = source_phase.T
        real_product = -density_real[None, :] * jnp.sin(target_phase) - density_imag[
            None, :
        ] * jnp.cos(target_phase)
        coefficient = (
            4.0
            * jnp.pi
            * self.gravitational_constant
            / self.volume
            * jnp.exp(-k_squared / (4.0 * self.alpha**2))
            / k_squared
        )
        reciprocal_acceleration = contract("k,nk,kd->nd", coefficient, real_product, k)
        acceleration = real_acceleration + reciprocal_acceleration
        net_force = jnp.sum(mass[:, None] * acceleration, axis=0)
        finite = jnp.all(jnp.isfinite(acceleration))
        evidence = PeriodicEwaldEvidence(
            real_acceleration,
            reciprocal_acceleration,
            net_force,
            finite,
            finite,
        )
        return PeriodicEwaldResult(acceleration, evidence, finite)


class PeriodicBarnesHutPlan(StrictModule, NonTrainableState):
    """Barnes-Hut plus an exact small-N Ewald-minus-direct periodic correction."""

    barnes_hut: BarnesHutGravityPlan
    ewald: Any

    def __init__(self, barnes_hut: BarnesHutGravityPlan, ewald: Any, /):
        if (
            barnes_hut.gravitational_constant != ewald.gravitational_constant
            or barnes_hut.softening != ewald.softening
        ):
            raise ValueError("Barnes-Hut and Ewald kernels must match.")
        self.barnes_hut = barnes_hut
        self.ewald = ewald

    def evaluate(self, tree: PreparedParticleOctree3D, /) -> TreeGravityResult:
        approximate = self.barnes_hut.evaluate(tree)
        position = tree.positions
        displacement = position[None, :, :] - position[:, None, :]
        squared = jnp.sum(displacement**2, axis=-1) + self.barnes_hut.softening**2
        direct = jnp.sum(
            jnp.where(
                (tree.active_mask[None, :] & ~jnp.eye(position.shape[0], dtype=bool))[
                    ..., None
                ],
                self.barnes_hut.gravitational_constant
                * tree.masses[None, :, None]
                * displacement
                / squared[..., None] ** 1.5,
                0.0,
            ),
            axis=1,
        )
        periodic = self.ewald.evaluate(position, tree.masses)
        acceleration = approximate.acceleration + periodic.acceleration - direct
        finite = (
            approximate.successful
            & periodic.successful
            & jnp.all(jnp.isfinite(acceleration))
        )
        evidence = TreeGravityEvidence(
            jnp.sum(tree.masses[:, None] * acceleration, axis=0),
            jnp.max(jnp.sqrt(jnp.sum(acceleration**2, axis=-1))),
            approximate.evidence.accepted_leaf_interactions,
            approximate.evidence.direct_particle_interactions,
            approximate.evidence.estimated_relative_error,
            finite,
            finite,
        )
        return TreeGravityResult(acceleration, evidence, finite)


class MeshComplementCalibrationEvidence(StrictModule):
    maximum_absolute_residual: Array
    rms_residual: Array
    tolerance_met: Array
    finite: Array
    successful: Array


class MeshComplementCalibrationPlan(StrictModule, NonTrainableState):
    tolerance: float = eqx.field(static=True)

    def __init__(self, tolerance: float, /):
        value = float(tolerance)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Mesh-complement tolerance must be finite and positive.")
        self.tolerance = value

    def qualify(
        self,
        reference_acceleration: ArrayLike,
        long_range_acceleration: ArrayLike,
        short_range_acceleration: ArrayLike,
        /,
    ) -> MeshComplementCalibrationEvidence:
        reference = jnp.asarray(reference_acceleration)
        long_range = jnp.asarray(long_range_acceleration, dtype=reference.dtype)
        short_range = jnp.asarray(short_range_acceleration, dtype=reference.dtype)
        if long_range.shape != reference.shape or short_range.shape != reference.shape:
            raise ValueError("Mesh-complement accelerations must have equal shapes.")
        residual = long_range + short_range - reference
        norm = jnp.sqrt(jnp.sum(residual**2, axis=-1))
        maximum = jnp.max(norm)
        rms = jnp.sqrt(jnp.mean(norm**2))
        finite = jnp.all(jnp.isfinite(residual))
        tolerance_met = maximum <= self.tolerance
        return MeshComplementCalibrationEvidence(
            maximum, rms, tolerance_met, finite, finite & tolerance_met
        )


class TreePMSplitPolicy(StrictModule, NonTrainableState):
    split_scale: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    compensation_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, split_scale: float, cutoff: float, compensation_id: str, /):
        split = float(split_scale)
        cutoff_ = float(cutoff)
        compensation = str(compensation_id).strip()

        if (
            not np.isfinite(split)
            or split <= 0.0
            or not np.isfinite(cutoff_)
            or cutoff_ <= split
            or not compensation
        ):
            raise ValueError("TreePM split policy is invalid.")
        self.split_scale = split
        self.cutoff = cutoff_
        self.compensation_id = compensation
        self.policy_id = canonical_fingerprint(
            {
                "kind": "treepm-split-policy",
                "split_scale": split,
                "cutoff": cutoff_,
                "compensation_id": compensation,
            }
        )


class TreePMResult(StrictModule):
    long_range_acceleration: Array
    short_range_acceleration: Array
    total_acceleration: Array
    short_evidence: TreeGravityEvidence
    finite: Array
    successful: Array


class TreePMPlan(StrictModule, NonTrainableState):
    short_range: BarnesHutGravityPlan
    split: TreePMSplitPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(self, short_range: BarnesHutGravityPlan, split: TreePMSplitPolicy, /):
        self.short_range = short_range
        self.split = split
        self.plan_id = canonical_fingerprint(
            {
                "kind": "single-device-treepm",
                "short_range": short_range.plan_id,
                "split": split.policy_id,
            }
        )

    def evaluate(
        self,
        tree: PreparedParticleOctree3D,
        long_range_acceleration: ArrayLike,
        /,
    ) -> TreePMResult:
        long_range = jnp.asarray(long_range_acceleration, dtype=tree.positions.dtype)
        if long_range.shape != tree.positions.shape:
            raise ValueError("TreePM long-range acceleration must match particles.")
        short = self.short_range.evaluate(
            tree,
            short_range_scale=self.split.split_scale,
            cutoff=self.split.cutoff,
        )
        total = long_range + short.acceleration
        finite = jnp.all(jnp.isfinite(total))
        return TreePMResult(
            long_range,
            short.acceleration,
            total,
            short.evidence,
            finite,
            finite & short.successful,
        )


__all__ = [
    "BarnesHutGravityPlan",
    "CartesianExpansionSpace",
    "CartesianFMMOperators",
    "DirectParticleGravityPlan",
    "DistributedParticleLayout",
    "FMMEvidence",
    "MeshComplementCalibrationEvidence",
    "MeshComplementCalibrationPlan",
    "NewtonianPairKernel",
    "ParticleGravityEvidence",
    "ParticleOctreePlan3D",
    "PeriodicBarnesHutPlan",
    "PeriodicEwaldEvidence",
    "PeriodicEwaldForcePlan",
    "PeriodicEwaldResult",
    "PreparedParticleOctree3D",
    "TreeGravityEvidence",
    "TreeGravityResult",
    "TreePMPlan",
    "TreePMResult",
    "TreePMSplitPolicy",
    "UniformFMMPlan",
]
