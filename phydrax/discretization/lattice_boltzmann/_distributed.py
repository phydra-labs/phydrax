#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..finite_difference._distributed import (
    DistributedHaloSchedule,
    HaloExchangeDescriptor,
)
from ._execution import (
    _realize_lattice_boltzmann,
    lattice_boltzmann_equivalence,
    LatticeBoltzmannRealizationResult,
    ReferenceLatticeBoltzmannExecutionPlan,
    with_lattice_boltzmann_equivalence,
)
from ._lattice import LatticeBoltzmannVelocitySet


class LatticeBoltzmannHaloRoute(StrictModule, NonTrainableState):
    """One trailing-Q population and its incoming face/edge/corner neighbor."""

    descriptor: HaloExchangeDescriptor | None
    direction_index: int = eqx.field(static=True)
    velocity_offset: tuple[int, ...] = eqx.field(static=True)
    source_offset: tuple[int, ...] = eqx.field(static=True)
    codimension: int = eqx.field(static=True)
    local: bool = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        direction_index: int,
        velocity_offset: tuple[int, ...],
        descriptor: HaloExchangeDescriptor | None,
        /,
    ):
        direction = int(direction_index)
        velocity = tuple(int(value) for value in velocity_offset)
        source = tuple(-value for value in velocity)
        local = all(value == 0 for value in velocity)
        if (
            direction < 0
            or not velocity
            or any(value not in (-1, 0, 1) for value in velocity)
        ):
            raise ValueError("LBM halo route direction or offset is invalid.")
        if local != (descriptor is None):
            raise ValueError("Only the rest population may omit a halo descriptor.")
        if descriptor is not None and descriptor.offset != source:
            raise ValueError("LBM halo descriptor must match the incoming source offset.")
        self.descriptor = descriptor
        self.direction_index = direction
        self.velocity_offset = velocity
        self.source_offset = source
        self.codimension = sum(value != 0 for value in velocity)
        self.local = local
        self.route_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-halo-route",
                "direction": direction,
                "velocity_offset": list(velocity),
                "source_offset": list(source),
                "descriptor": None if descriptor is None else descriptor.descriptor_id,
            }
        )


class LatticeBoltzmannHaloSchedule(StrictModule, NonTrainableState):
    """Neutral direction adapter over the generic multi-axis halo schedule."""

    velocity_set: LatticeBoltzmannVelocitySet
    schedule: DistributedHaloSchedule
    routes: tuple[LatticeBoltzmannHaloRoute, ...]
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        velocity_set: LatticeBoltzmannVelocitySet,
        schedule: DistributedHaloSchedule,
        /,
    ):
        if not isinstance(velocity_set, LatticeBoltzmannVelocitySet):
            raise TypeError("velocity_set must be a LatticeBoltzmannVelocitySet.")
        if not isinstance(schedule, DistributedHaloSchedule):
            raise TypeError("schedule must be a DistributedHaloSchedule.")
        if len(schedule.global_shape) != velocity_set.dimension:
            raise ValueError("Halo schedule rank must match the LBM velocity dimension.")
        descriptors = {descriptor.offset: descriptor for descriptor in schedule.exchanges}
        routes = []
        for direction, velocity in enumerate(velocity_set.velocity_tuples):
            source = tuple(-value for value in velocity)
            descriptor = (
                None if all(value == 0 for value in velocity) else descriptors.get(source)
            )
            if descriptor is None and any(value != 0 for value in velocity):
                raise ValueError(
                    f"Generic halo schedule does not cover LBM velocity offset {velocity}."
                )
            routes.append(LatticeBoltzmannHaloRoute(direction, velocity, descriptor))
        if tuple(route.direction_index for route in routes) != tuple(
            range(velocity_set.population_count)
        ):
            raise RuntimeError("LBM halo routes do not cover every population direction.")
        self.velocity_set = velocity_set
        self.schedule = schedule
        self.routes = tuple(routes)
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-halo-schedule",
                "lattice": velocity_set.lattice_id,
                "schedule": schedule.schedule_id,
                "routes": [route.route_id for route in routes],
            }
        )

    def route(self, direction_index: int, /) -> LatticeBoltzmannHaloRoute:
        direction = int(direction_index)
        if direction < 0 or direction >= len(self.routes):
            raise ValueError("LBM halo direction index is out of range.")
        return self.routes[direction]

    def select(self, populations: Array, direction_index: int, /) -> Array:
        """Select one direction while preserving the public trailing-Q axis."""

        if not eqx.is_array(populations):
            raise TypeError("LBM halo exchange accepts JAX arrays only.")
        if populations.shape[-1] != self.velocity_set.population_count:
            raise ValueError("LBM halo populations must use the trailing Q axis.")
        direction = self.route(direction_index).direction_index
        return populations[..., direction : direction + 1]

    def ppermute_route(
        self,
        payload: Array,
        direction_index: int,
        /,
    ) -> Array:
        """Send one selected-Q payload along its face/edge/corner collective route."""

        if not eqx.is_array(payload):
            raise TypeError("LBM collective halo exchange accepts JAX arrays only.")
        if payload.ndim == 0 or payload.shape[-1] != 1:
            raise ValueError("LBM collective payload must preserve one trailing Q slot.")
        if payload.dtype != jnp.dtype(self.schedule.precision.field_dtype):
            raise TypeError("LBM halo dtype must match communication precision.")
        route = self.route(direction_index)
        exchanged = payload
        for axis, direction in enumerate(route.velocity_offset):
            if direction:
                exchanged = self.schedule.ppermute_halo(
                    exchanged,
                    axis,
                    direction,
                )
        return exchanged

    def select_and_ppermute(
        self,
        populations: Array,
        direction_index: int,
        /,
    ) -> Array:
        """Select a trailing-Q population and apply only its required collectives."""

        return self.ppermute_route(
            self.select(populations, direction_index),
            direction_index,
        )

    def exchange_route_reference(
        self,
        blocks: Array,
        direction_index: int,
        /,
    ) -> Array:
        """Return only the selected incoming face/edge/corner halo payload."""

        if not eqx.is_array(blocks):
            raise TypeError("LBM reference halo exchange accepts JAX arrays only.")
        expected = (
            *self.schedule.partition_shape,
            *self.schedule.local_shape,
            self.velocity_set.population_count,
        )
        if blocks.shape != expected:
            raise ValueError(f"LBM reference halo blocks must have shape {expected}.")
        if blocks.dtype != jnp.dtype(self.schedule.precision.field_dtype):
            raise TypeError("LBM halo dtype must match the generic schedule precision.")
        route = self.route(direction_index)
        extended_shape = tuple(
            local + lower + upper
            for local, lower, upper in zip(
                self.schedule.local_shape,
                self.schedule.halo_plan.lower_widths,
                self.schedule.halo_plan.upper_widths,
                strict=True,
            )
        )
        output_shape = (*self.schedule.partition_shape, *extended_shape, 1)
        if route.local:
            return jnp.zeros(output_shape, dtype=blocks.dtype)
        exchanged = self.schedule.exchange_reference(blocks[..., route.direction_index])
        slices = []
        for offset, local, lower, upper in zip(
            route.source_offset,
            self.schedule.local_shape,
            self.schedule.halo_plan.lower_widths,
            self.schedule.halo_plan.upper_widths,
            strict=True,
        ):
            if offset < 0:
                slices.append(slice(0, lower))
            elif offset > 0:
                slices.append(slice(lower + local, lower + local + upper))
            else:
                slices.append(slice(lower, lower + local))
        mask = np.zeros(extended_shape, dtype=bool)
        mask[tuple(slices)] = True
        broadcast_mask = jnp.asarray(mask).reshape(
            (1,) * len(self.schedule.partition_shape) + extended_shape
        )
        return jnp.where(broadcast_mask, exchanged, 0.0)[..., None]

    def exchange_reference(self, blocks: Array, /) -> Array:
        """Assemble local values and exactly routed incoming halos for every Q."""

        if not eqx.is_array(blocks):
            raise TypeError("LBM reference halo exchange accepts JAX arrays only.")
        expected = (
            *self.schedule.partition_shape,
            *self.schedule.local_shape,
            self.velocity_set.population_count,
        )
        if blocks.shape != expected:
            raise ValueError(f"LBM reference halo blocks must have shape {expected}.")
        extended_shape = tuple(
            local + lower + upper
            for local, lower, upper in zip(
                self.schedule.local_shape,
                self.schedule.halo_plan.lower_widths,
                self.schedule.halo_plan.upper_widths,
                strict=True,
            )
        )
        output = jnp.zeros(
            (
                *self.schedule.partition_shape,
                *extended_shape,
                self.velocity_set.population_count,
            ),
            dtype=blocks.dtype,
        )
        partition_slices = (slice(None),) * len(self.schedule.partition_shape)
        interior = tuple(
            slice(lower, lower + local)
            for local, lower in zip(
                self.schedule.local_shape,
                self.schedule.halo_plan.lower_widths,
                strict=True,
            )
        )
        output = output.at[partition_slices + interior + (slice(None),)].set(blocks)
        for route in self.routes:
            if route.local:
                continue
            payload = self.exchange_route_reference(blocks, route.direction_index)[..., 0]
            current = output[..., route.direction_index]
            output = output.at[..., route.direction_index].set(current + payload)
        return output


class LatticeBoltzmannShardingMetadata(StrictModule, NonTrainableState):
    """Exact spatial partitioning with an explicitly unpartitioned trailing Q axis."""

    global_population_shape: tuple[int, ...] = eqx.field(static=True)
    local_population_shape: tuple[int, ...] = eqx.field(static=True)
    partition_shape: tuple[int, ...] = eqx.field(static=True)
    mesh_axis_names: tuple[str, ...] = eqx.field(static=True)
    population_axis_partitioned: bool = eqx.field(static=True)
    route_count: int = eqx.field(static=True)
    metadata_id: str = eqx.field(static=True)


class ShardedLatticeBoltzmannExecutionPlan(StrictModule, NonTrainableState):
    """Fixed NamedSharding realization with optional reference qualification."""

    reference: ReferenceLatticeBoltzmannExecutionPlan
    halo: LatticeBoltzmannHaloSchedule
    population_sharding: NamedSharding = eqx.field(static=True)
    metadata: LatticeBoltzmannShardingMetadata
    backend: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference: ReferenceLatticeBoltzmannExecutionPlan,
        halo: LatticeBoltzmannHaloSchedule,
        /,
        *,
        backend: str = "jax",
    ):
        if not isinstance(reference, ReferenceLatticeBoltzmannExecutionPlan):
            raise TypeError("reference must be a reference LBM execution plan.")
        if not isinstance(halo, LatticeBoltzmannHaloSchedule):
            raise TypeError("halo must be LatticeBoltzmannHaloSchedule.")
        if reference.velocity_set.lattice_id != halo.velocity_set.lattice_id:
            raise ValueError("Reference execution and halo velocity sets do not match.")
        if backend != "jax":
            raise ValueError("Sharded LBM execution supports only the JAX backend.")
        spatial_spec = tuple(
            name if count > 1 else None
            for name, count in zip(
                halo.schedule.mesh_axis_names,
                halo.schedule.partition_shape,
                strict=True,
            )
        )
        sharding = NamedSharding(
            halo.schedule.sharding.mesh,
            PartitionSpec(*spatial_spec, None),
        )
        global_shape = (
            *halo.schedule.global_shape,
            halo.velocity_set.population_count,
        )
        local_shape = (
            *halo.schedule.local_shape,
            halo.velocity_set.population_count,
        )
        metadata_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-sharding-metadata",
                "global_population_shape": list(global_shape),
                "local_population_shape": list(local_shape),
                "partition_shape": list(halo.schedule.partition_shape),
                "mesh_axis_names": list(halo.schedule.mesh_axis_names),
                "population_axis_partitioned": False,
                "routes": len(halo.routes),
            }
        )
        metadata = LatticeBoltzmannShardingMetadata(
            global_shape,
            local_shape,
            halo.schedule.partition_shape,
            halo.schedule.mesh_axis_names,
            False,
            len(halo.routes),
            metadata_id,
        )
        self.reference = reference
        self.halo = halo
        self.population_sharding = sharding
        self.metadata = metadata
        self.backend = "jax"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sharded-lattice-boltzmann-execution",
                "reference": reference.plan_id,
                "halo": halo.schedule_id,
                "metadata": metadata_id,
                "backend": "jax",
            }
        )

    def shard(self, populations: Array, /) -> Array:
        if not eqx.is_array(populations):
            raise TypeError("Sharded LBM execution accepts JAX arrays only.")
        if populations.shape != self.metadata.global_population_shape:
            raise ValueError("LBM populations do not match the sharding global shape.")
        if populations.dtype != jnp.dtype(self.halo.schedule.precision.field_dtype):
            raise TypeError(
                "LBM population dtype must match halo communication precision."
            )
        return jax.device_put(populations, self.population_sharding)

    def realize(
        self,
        initial_populations: Array,
        /,
        *,
        step_count: int,
        step_size: Any,
        args: Any = None,
        t0: Any = 0.0,
        rtol: float = 0.0,
        atol: float = 0.0,
        verify_equivalence: bool = True,
    ) -> LatticeBoltzmannRealizationResult:
        reference = (
            self.reference.realize(
                initial_populations,
                step_count=step_count,
                step_size=step_size,
                args=args,
                t0=t0,
            )
            if verify_equivalence
            else None
        )
        candidate = _realize_lattice_boltzmann(
            self.reference.step,
            self.reference.velocity_set,
            self.shard(initial_populations),
            step_count=step_count,
            step_size=step_size,
            args=args,
            t0=t0,
            plan_id=self.plan_id,
            step_id=self.reference.step_id,
            execution_kind="sharded",
        )
        if reference is None:
            return candidate
        evidence = lattice_boltzmann_equivalence(
            reference,
            candidate,
            rtol=rtol,
            atol=atol,
        )
        return with_lattice_boltzmann_equivalence(candidate, evidence)


__all__ = [
    "LatticeBoltzmannHaloRoute",
    "LatticeBoltzmannHaloSchedule",
    "LatticeBoltzmannShardingMetadata",
    "ShardedLatticeBoltzmannExecutionPlan",
]
