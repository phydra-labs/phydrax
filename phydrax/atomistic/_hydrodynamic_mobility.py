#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    FunctionLinearOperator,
    OperatorProperties,
)
from ._system import PreparedAtomisticSystem


class AbstractHydrodynamicMobilityPlan(StrictModule, NonTrainableState):
    mobility_id: AbstractAttribute[str]
    maximum_particles: AbstractAttribute[int]
    __strict_abstract__ = True

    @abc.abstractmethod
    def prepare(
        self, system: PreparedAtomisticSystem, active_slots: ArrayLike, /
    ) -> AbstractPreparedHydrodynamicMobility:
        raise NotImplementedError


class AbstractPreparedHydrodynamicMobility(StrictModule, NonTrainableState):
    plan: AbstractAttribute[AbstractHydrodynamicMobilityPlan]
    system: AbstractAttribute[PreparedAtomisticSystem]
    active_slots: AbstractAttribute[Array]
    coordinate_space: AbstractAttribute[ArraySpace]
    prepared_id: AbstractAttribute[str]
    route_id: AbstractAttribute[str]
    __strict_abstract__ = True

    @abc.abstractmethod
    def configuration_valid(self, positions: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def operator(self, positions: ArrayLike, /) -> AbstractLinearOperator:
        raise NotImplementedError


class ConstantIsotropicMobilityPlan(AbstractHydrodynamicMobilityPlan):
    mobility: float = eqx.field(static=True)
    maximum_particles: int = eqx.field(static=True)
    mobility_id: str = eqx.field(static=True)

    def __init__(self, mobility: float, /, *, maximum_particles: int):
        value = float(mobility)
        maximum = int(maximum_particles)
        if not math.isfinite(value) or value <= 0.0 or maximum <= 0:
            raise ValueError("Constant mobility and particle capacity must be positive.")
        self.mobility = value
        self.maximum_particles = maximum
        self.mobility_id = canonical_fingerprint(
            {
                "kind": "constant-isotropic-hydrodynamic-mobility",
                "mobility": value,
                "maximum_particles": maximum,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, active_slots: ArrayLike, /
    ) -> PreparedConstantIsotropicMobility:
        return PreparedConstantIsotropicMobility(self, system, active_slots)


class PreparedConstantIsotropicMobility(AbstractPreparedHydrodynamicMobility):
    plan: ConstantIsotropicMobilityPlan
    system: PreparedAtomisticSystem
    active_slots: Array
    coordinate_space: ArraySpace
    prepared_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ConstantIsotropicMobilityPlan,
        system: PreparedAtomisticSystem,
        active_slots: ArrayLike,
        /,
    ):
        slots = _active_slots(plan.maximum_particles, system, active_slots)
        space = ArraySpace(
            (int(slots.size), 3),
            dtype=system.plan.coordinate_dtype,
            space_id=f"constant-mobility:{plan.mobility_id}:coordinates",
        )
        self.plan = plan
        self.system = system
        self.active_slots = slots
        self.coordinate_space = space
        self.route_id = canonical_fingerprint(
            {
                "kind": "constant-mobility-route",
                "system": system.prepared_id,
                "active_slots": np.asarray(slots).tolist(),
            }
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-constant-isotropic-mobility",
                "plan": plan.mobility_id,
                "system": system.prepared_id,
                "route": self.route_id,
            }
        )

    def configuration_valid(self, positions: ArrayLike, /) -> Array:
        value = self.coordinate_space.validate(jnp.asarray(positions))
        return jnp.all(jnp.isfinite(value))

    def operator(self, positions: ArrayLike, /) -> AbstractLinearOperator:
        value = self.coordinate_space.validate(jnp.asarray(positions))
        del value

        def action(vector):
            return self.plan.mobility * self.coordinate_space.validate(vector)

        return FunctionLinearOperator(
            action,
            source=self.coordinate_space,
            target=self.coordinate_space,
            transpose_action=action,
            properties=_spd_properties(),
            operator_id=f"{self.prepared_id}:operator",
        )


class FreeSpaceRPYMobilityPlan(AbstractHydrodynamicMobilityPlan):
    hydrodynamic_radius: float = eqx.field(static=True)
    dynamic_viscosity: float = eqx.field(static=True)
    maximum_particles: int = eqx.field(static=True)
    mobility_id: str = eqx.field(static=True)

    def __init__(
        self,
        hydrodynamic_radius: float,
        dynamic_viscosity: float,
        /,
        *,
        maximum_particles: int,
    ):
        radius = float(hydrodynamic_radius)
        viscosity = float(dynamic_viscosity)
        maximum = int(maximum_particles)
        if (
            not math.isfinite(radius)
            or radius <= 0.0
            or not math.isfinite(viscosity)
            or viscosity <= 0.0
            or maximum <= 0
        ):
            raise ValueError("RPY radius, viscosity, and capacity must be positive.")
        self.hydrodynamic_radius = radius
        self.dynamic_viscosity = viscosity
        self.maximum_particles = maximum
        self.mobility_id = canonical_fingerprint(
            {
                "kind": "free-space-equal-radius-rpy-mobility",
                "hydrodynamic_radius": radius,
                "dynamic_viscosity": viscosity,
                "maximum_particles": maximum,
                "boundary": "unbounded-three-dimensional",
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, active_slots: ArrayLike, /
    ) -> PreparedFreeSpaceRPYMobility:
        if system.cell is not None:
            raise ValueError("Free-space RPY does not admit periodic cells.")
        return PreparedFreeSpaceRPYMobility(self, system, active_slots)


class PreparedFreeSpaceRPYMobility(AbstractPreparedHydrodynamicMobility):
    plan: FreeSpaceRPYMobilityPlan
    system: PreparedAtomisticSystem
    active_slots: Array
    coordinate_space: ArraySpace
    prepared_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: FreeSpaceRPYMobilityPlan,
        system: PreparedAtomisticSystem,
        active_slots: ArrayLike,
        /,
    ):
        slots = _active_slots(plan.maximum_particles, system, active_slots)
        space = ArraySpace(
            (int(slots.size), 3),
            dtype=system.plan.coordinate_dtype,
            space_id=f"rpy:{plan.mobility_id}:coordinates",
        )
        self.plan = plan
        self.system = system
        self.active_slots = slots
        self.coordinate_space = space
        self.route_id = canonical_fingerprint(
            {
                "kind": "free-space-rpy-all-pairs-route",
                "system": system.prepared_id,
                "active_slots": np.asarray(slots).tolist(),
            }
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-free-space-rpy-mobility",
                "plan": plan.mobility_id,
                "system": system.prepared_id,
                "route": self.route_id,
            }
        )

    def configuration_valid(self, positions: ArrayLike, /) -> Array:
        position = self.coordinate_space.validate(jnp.asarray(positions))
        displacement = position[:, None, :] - position[None, :, :]
        squared = jnp.sum(displacement * displacement, axis=-1)
        distinct = ~jnp.eye(position.shape[0], dtype=bool)
        return jnp.all(jnp.isfinite(position)) & jnp.all(
            jnp.where(distinct, squared > 0.0, True)
        )

    def operator(self, positions: ArrayLike, /) -> AbstractLinearOperator:
        position = self.coordinate_space.validate(jnp.asarray(positions))
        radius = jnp.asarray(self.plan.hydrodynamic_radius, dtype=position.dtype)
        viscosity = jnp.asarray(self.plan.dynamic_viscosity, dtype=position.dtype)
        count = int(position.shape[0])
        displacement = position[:, None, :] - position[None, :, :]
        squared = jnp.sum(displacement * displacement, axis=-1)
        identity_pair = jnp.eye(count, dtype=bool)
        configuration_valid = self.configuration_valid(position)
        distance = jnp.sqrt(squared)
        safe_distance = jnp.where(identity_pair | (distance <= 0.0), 1.0, distance)
        direction = jnp.where(
            (~identity_pair & (distance > 0.0))[..., None],
            displacement / safe_distance[..., None],
            0.0,
        )
        outer = direction[..., :, None] * direction[..., None, :]
        identity = jnp.eye(3, dtype=position.dtype)
        self_mobility = 1.0 / (6.0 * jnp.pi * viscosity * radius)
        separated = (1.0 / (8.0 * jnp.pi * viscosity * safe_distance))[
            ..., None, None
        ] * (
            identity
            + outer
            + (2.0 * radius * radius / (3.0 * safe_distance * safe_distance))[
                ..., None, None
            ]
            * (identity - 3.0 * outer)
        )
        overlap_ratio = jnp.where(identity_pair, 0.0, distance / radius)
        overlapping = self_mobility * (
            (1.0 - 9.0 * overlap_ratio / 32.0)[..., None, None] * identity
            + (3.0 * overlap_ratio / 32.0)[..., None, None] * outer
        )
        blocks = jnp.where(
            identity_pair[..., None, None],
            self_mobility * identity,
            jnp.where(
                (distance >= 2.0 * radius)[..., None, None],
                separated,
                overlapping,
            ),
        )
        safe_blocks = identity_pair[..., None, None] * self_mobility * identity
        blocks = jnp.where(configuration_valid, blocks, safe_blocks)

        def action(vector):
            value = self.coordinate_space.validate(vector)
            return contract("ijab,jb->ia", blocks, value)

        return FunctionLinearOperator(
            action,
            source=self.coordinate_space,
            target=self.coordinate_space,
            transpose_action=action,
            properties=_spd_properties(),
            operator_id=f"{self.prepared_id}:operator",
        )


def materialize_mobility(
    mobility: AbstractPreparedHydrodynamicMobility,
    positions: ArrayLike,
    /,
    *,
    maximum_dofs: int,
) -> Array:
    if not isinstance(mobility, AbstractPreparedHydrodynamicMobility):
        raise TypeError("mobility must be AbstractPreparedHydrodynamicMobility.")
    dofs = mobility.coordinate_space.size
    if dofs > int(maximum_dofs):
        raise ValueError("Mobility materialization exceeds maximum_dofs.")
    value = mobility.coordinate_space.validate(jnp.asarray(positions))
    value = eqx.error_if(
        value,
        ~mobility.configuration_valid(value),
        "Hydrodynamic mobility configuration is outside its certified route.",
    )
    operator = mobility.operator(value)
    identity = jnp.eye(dofs, dtype=mobility.coordinate_space.dtype)
    columns = jax.vmap(
        lambda column: mobility.coordinate_space.flatten(
            operator.mv(mobility.coordinate_space.unflatten(column))
        ),
        in_axes=1,
        out_axes=1,
    )(identity)
    return columns


def _active_slots(
    maximum_particles: int,
    system: PreparedAtomisticSystem,
    value: ArrayLike,
    /,
) -> Array:
    if not isinstance(system, PreparedAtomisticSystem):
        raise TypeError("system must be PreparedAtomisticSystem.")
    slots = np.asarray(value)
    if (
        slots.ndim != 1
        or not np.issubdtype(slots.dtype, np.integer)
        or slots.size == 0
        or slots.size > maximum_particles
        or np.any(slots < 0)
        or np.any(slots >= system.capacity)
        or np.unique(slots).size != slots.size
        or not np.all(np.asarray(system.active_mask)[slots])
    ):
        raise ValueError("Hydrodynamic active slots are invalid or exceed capacity.")
    particle_ids = np.asarray(system.plan.particle_ids)[slots]
    order = np.argsort(particle_ids, kind="stable")
    return jnp.asarray(slots[order], dtype=jnp.int32)


def _spd_properties() -> OperatorProperties:
    return OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
        },
    )


__all__ = [
    "AbstractHydrodynamicMobilityPlan",
    "AbstractPreparedHydrodynamicMobility",
    "ConstantIsotropicMobilityPlan",
    "FreeSpaceRPYMobilityPlan",
    "PreparedConstantIsotropicMobility",
    "PreparedFreeSpaceRPYMobility",
    "materialize_mobility",
]
