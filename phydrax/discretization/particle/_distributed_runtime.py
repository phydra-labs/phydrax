#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity owner-computes particle exchange on a real JAX mesh."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


if TYPE_CHECKING:
    from ...solver._particle_gravity import DistributedParticleLayout


class DistributedParticleState(StrictModule):
    """Owner-ordered fixed-capacity particles; inactive slots use stable ID ``-1``."""

    positions: Array
    momenta: Array
    masses: Array
    stable_ids: Array
    logical_slots: Array
    active_mask: Array
    rng_counters: Array
    scale_factor: Array
    owner: Array
    runtime_id: str = eqx.field(static=True)


class ParticleExchangeEvidence(StrictModule):
    send_count: Array
    receive_count: Array
    migration_count: Array
    send_capacity: int = eqx.field(static=True)
    receive_capacity: int = eqx.field(static=True)
    finite: Array
    ids_unique: Array
    successful: Array
    runtime_id: str = eqx.field(static=True)


class DistributedParticleMigrationResult(StrictModule):
    state: DistributedParticleState
    evidence: ParticleExchangeEvidence


class DistributedParticleGhostState(StrictModule):
    positions: Array
    masses: Array
    stable_ids: Array
    source_owner: Array
    valid: Array
    left_count: Array
    right_count: Array
    capacity_per_side: int = eqx.field(static=True)
    successful: Array
    runtime_id: str = eqx.field(static=True)


class DistributedParticleRuntimePlan(StrictModule, NonTrainableState):
    """Serializable capacity and periodic spatial-ownership contract."""

    layout: DistributedParticleLayout
    box_size: tuple[float, ...] = eqx.field(static=True)
    send_capacity: int = eqx.field(static=True)
    receive_capacity: int = eqx.field(static=True)
    ghost_capacity: int = eqx.field(static=True)
    ghost_width: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: DistributedParticleLayout,
        box_size: Sequence[float],
        /,
        *,
        send_capacity: int | None = None,
        receive_capacity: int | None = None,
        ghost_capacity: int | None = None,
        ghost_width: float = 0.0,
    ):
        # DistributedParticleLayout lives in the solver layer. Keep this
        # discretization owner structurally typed to avoid a solver import cycle.
        lengths = tuple(float(value) for value in box_size)
        if not lengths or any(
            not np.isfinite(value) or value <= 0.0 for value in lengths
        ):
            raise ValueError("Particle runtime box_size must be finite and positive.")
        send = layout.capacity_per_device if send_capacity is None else int(send_capacity)
        receive = (
            layout.capacity_per_device
            if receive_capacity is None
            else int(receive_capacity)
        )
        ghosts = (
            layout.capacity_per_device if ghost_capacity is None else int(ghost_capacity)
        )
        width = float(ghost_width)
        if send <= 0 or receive <= 0 or receive != layout.capacity_per_device:
            raise ValueError(
                "Particle receive capacity must equal layout.capacity_per_device and send capacity must be positive."
            )
        boundaries = np.asarray(layout.key_boundaries, dtype=np.uint64)
        minimum_key_width = int(np.min(np.diff(boundaries)))
        minimum_slab_width = (
            lengths[0] * minimum_key_width / float(np.iinfo(np.uint32).max)
        )
        if (
            ghosts <= 0
            or ghosts > layout.capacity_per_device
            or not np.isfinite(width)
            or width < 0.0
            or minimum_key_width <= 0
            or width > minimum_slab_width
        ):
            raise ValueError("Particle ghost capacity/width is invalid.")
        self.layout = layout
        self.box_size = lengths
        self.send_capacity = send
        self.receive_capacity = receive
        self.ghost_capacity = ghosts
        self.ghost_width = width
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-particle-runtime-plan",
                "layout": layout.layout_id,
                "box_size": list(lengths),
                "send_capacity": send,
                "receive_capacity": receive,
                "ghost_capacity": ghosts,
                "ghost_width": width,
                "ownership": "periodic-axis-0-slabs",
            }
        )

    def prepare(
        self, mesh: Mesh, axis_name: str, /
    ) -> "PreparedDistributedParticleRuntime":
        return PreparedDistributedParticleRuntime(self, mesh, axis_name)


class PreparedDistributedParticleRuntime(StrictModule, NonTrainableState):
    """Real all-to-all migration and periodic ppermute ghost exchange."""

    plan: DistributedParticleRuntimePlan
    mesh: Mesh = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    vector_sharding: NamedSharding = eqx.field(static=True)
    scalar_sharding: NamedSharding = eqx.field(static=True)
    replicated_sharding: NamedSharding = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self, plan: DistributedParticleRuntimePlan, mesh: Mesh, axis_name: str, /
    ):
        if not isinstance(plan, DistributedParticleRuntimePlan):
            raise TypeError("plan must be DistributedParticleRuntimePlan.")
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be a JAX Mesh.")
        axis = str(axis_name)
        if axis not in mesh.axis_names or len(mesh.axis_names) != 1:
            raise ValueError("Particle runtime requires one named mesh axis.")
        if mesh.shape[axis] != plan.layout.device_count:
            raise ValueError("Particle layout and execution mesh device counts differ.")
        self.plan = plan
        self.mesh = mesh
        self.axis_name = axis
        self.vector_sharding = NamedSharding(mesh, PartitionSpec(axis, None))
        self.scalar_sharding = NamedSharding(mesh, PartitionSpec(axis))
        self.replicated_sharding = NamedSharding(mesh, PartitionSpec())
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-distributed-particle-runtime",
                "plan": plan.plan_id,
                "axis_name": axis,
                "mesh_shape": mesh.shape[axis],
                "device_keys": [
                    [int(device.process_index), int(device.id)]
                    for device in mesh.devices.flat
                ],
            }
        )

    @property
    def total_capacity(self) -> int:
        return self.plan.layout.device_count * self.plan.layout.capacity_per_device

    def _owners(self, positions: Array, active: Array, /) -> Array:
        length = jnp.asarray(self.plan.box_size[0], dtype=positions.dtype)
        normalized = jnp.mod(positions[..., 0], length) / length
        maximum = jnp.asarray(np.iinfo(np.uint32).max, dtype=positions.dtype)
        keys = jnp.floor(normalized * maximum).astype(jnp.uint32)
        owner = self.plan.layout.owners(keys).astype(jnp.int32)
        return jnp.where(active, owner, 0)

    def initialize(
        self,
        positions: ArrayLike,
        momenta: ArrayLike,
        masses: ArrayLike,
        stable_ids: ArrayLike,
        active_mask: ArrayLike,
        scale_factor: ArrayLike,
        /,
        *,
        rng_counters: ArrayLike | None = None,
    ) -> DistributedParticleState:
        position = jnp.asarray(positions)
        momentum = jnp.asarray(momenta, dtype=position.dtype)
        mass = jnp.asarray(masses, dtype=position.dtype)
        ids = jnp.asarray(stable_ids, dtype=jnp.int64)
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
        expected = (self.total_capacity, len(self.plan.box_size))
        if position.shape != expected or momentum.shape != expected:
            raise ValueError(
                f"Particle position/momentum arrays must have shape {expected}."
            )
        if (
            mass.shape != (self.total_capacity,)
            or ids.shape != mass.shape
            or active.shape != mass.shape
        ):
            raise ValueError("Particle scalar arrays must have total-capacity shape.")
        counters = (
            jnp.zeros((self.total_capacity,), dtype=jnp.uint64)
            if rng_counters is None
            else jnp.asarray(rng_counters, dtype=jnp.uint64)
        )
        if counters.shape != mass.shape:
            raise ValueError("rng_counters must have total-capacity shape.")
        scale = jnp.asarray(scale_factor, dtype=position.dtype)
        if scale.shape != ():
            raise ValueError("Particle scale_factor must be scalar.")
        logical = jnp.arange(self.total_capacity, dtype=jnp.int32)
        stable_order = jnp.lexsort((ids, ~active))
        position = position[stable_order]
        momentum = momentum[stable_order]
        mass = mass[stable_order]
        ids = ids[stable_order]
        active = active[stable_order]
        counters = counters[stable_order]
        logical = logical[stable_order]
        owner = self._owners(position, active)
        one_hot = (
            jax.nn.one_hot(owner, self.plan.layout.device_count, dtype=jnp.int32)
            * active[:, None]
        )
        rank_table = jnp.cumsum(one_hot, axis=0) - 1
        rank = jnp.sum(rank_table * one_hot, axis=-1)
        counts = jnp.sum(one_hot, axis=0)
        valid = active & (rank < self.plan.layout.capacity_per_device)
        target = owner * self.plan.layout.capacity_per_device + rank
        sentinel = self.total_capacity
        target = jnp.where(valid, target, sentinel)

        def pack(value, fill):
            output = jnp.full(
                (self.total_capacity + 1,) + value.shape[1:], fill, value.dtype
            )
            output = output.at[target].set(value)
            return output[: self.total_capacity]

        packed_positions = pack(position, 0.0)
        packed_momenta = pack(momentum, 0.0)
        packed_masses = pack(mass, 0.0)
        packed_ids = pack(ids, -1)
        packed_slots = pack(logical, -1)
        packed_active = pack(active, False)
        packed_rng = pack(counters, 0)
        packed_owner = jnp.repeat(
            jnp.arange(self.plan.layout.device_count, dtype=jnp.int32),
            self.plan.layout.capacity_per_device,
        )
        capacity_ok = jnp.all(counts <= self.plan.layout.capacity_per_device)
        ids_unique = ~jnp.any(active[1:] & active[:-1] & (ids[1:] == ids[:-1]))
        finite = (
            jnp.all(jnp.isfinite(position) | ~active[:, None])
            & jnp.all(jnp.isfinite(momentum) | ~active[:, None])
            & jnp.all(jnp.isfinite(mass) | ~active)
            & jnp.isfinite(scale)
        )
        packed_positions = eqx.error_if(
            packed_positions,
            ~capacity_ok | ~finite | ~ids_unique,
            "Initial distributed particle ownership exceeds capacity, contains duplicate stable IDs, or is non-finite.",
        )
        return DistributedParticleState(
            jax.device_put(packed_positions, self.vector_sharding),
            jax.device_put(packed_momenta, self.vector_sharding),
            jax.device_put(packed_masses, self.scalar_sharding),
            jax.device_put(packed_ids, self.scalar_sharding),
            jax.device_put(packed_slots, self.scalar_sharding),
            jax.device_put(packed_active, self.scalar_sharding),
            jax.device_put(packed_rng, self.scalar_sharding),
            jax.device_put(scale, self.replicated_sharding),
            jax.device_put(packed_owner, self.scalar_sharding),
            self.runtime_id,
        )

    def _require_state(self, state: DistributedParticleState, /) -> None:
        if not isinstance(state, DistributedParticleState):
            raise TypeError("state must be DistributedParticleState.")
        if state.runtime_id != self.runtime_id:
            raise ValueError("Distributed particle state belongs to a different runtime.")

    def migrate(
        self,
        state: DistributedParticleState,
        proposed_positions: ArrayLike,
        proposed_momenta: ArrayLike,
        /,
        *,
        proposed_rng_counters: ArrayLike | None = None,
        end_scale_factor: ArrayLike | None = None,
    ) -> DistributedParticleMigrationResult:
        """Route every active particle exactly once and atomically roll back overflow."""

        self._require_state(state)
        positions = jax.device_put(jnp.asarray(proposed_positions), self.vector_sharding)
        momenta = jax.device_put(jnp.asarray(proposed_momenta), self.vector_sharding)
        if (
            positions.shape != state.positions.shape
            or momenta.shape != state.momenta.shape
        ):
            raise ValueError("Proposed particle payloads must preserve fixed capacity.")
        rng = (
            state.rng_counters
            if proposed_rng_counters is None
            else jax.device_put(
                jnp.asarray(proposed_rng_counters, dtype=jnp.uint64),
                self.scalar_sharding,
            )
        )
        if rng.shape != state.rng_counters.shape:
            raise ValueError("Proposed RNG counters must preserve fixed capacity.")
        scale = (
            state.scale_factor
            if end_scale_factor is None
            else jnp.asarray(end_scale_factor, dtype=state.scale_factor.dtype)
        )
        if scale.shape != ():
            raise ValueError("end_scale_factor must be scalar.")
        p = self.plan.layout.device_count
        c = self.plan.layout.capacity_per_device
        s = self.plan.send_capacity
        axis = self.axis_name
        runtime_id = self.runtime_id
        box_length = self.plan.box_size[0]
        key_boundaries = self.plan.layout.key_boundaries

        def exchange(
            old_position,
            old_momentum,
            old_mass,
            old_ids,
            old_slots,
            old_active,
            old_rng,
            old_owner,
            new_position,
            new_momentum,
            new_rng,
        ):
            del old_owner
            length = jnp.asarray(box_length, dtype=new_position.dtype)
            finite_particle = (
                jnp.all(jnp.isfinite(new_position), axis=-1)
                & jnp.all(jnp.isfinite(new_momentum), axis=-1)
                & jnp.isfinite(old_mass)
            )
            normalized = jnp.mod(new_position[:, 0], length) / length
            maximum = jnp.asarray(np.iinfo(np.uint32).max, dtype=new_position.dtype)
            keys = jnp.floor(normalized * maximum).astype(jnp.uint32)
            destination = jnp.clip(
                jnp.searchsorted(key_boundaries[1:], keys, side="right"),
                0,
                p - 1,
            ).astype(jnp.int32)
            valid = old_active & finite_particle
            hot = jax.nn.one_hot(destination, p, dtype=jnp.int32) * valid[:, None]
            rank_table = jnp.cumsum(hot, axis=0) - 1
            rank = jnp.sum(rank_table * hot, axis=-1)
            send_count = jnp.sum(hot, axis=0)
            send_ok = jnp.all(send_count <= s)
            target = destination * s + rank
            sentinel = p * s
            target = jnp.where(valid & (rank < s), target, sentinel)

            def send_buffer(value, fill):
                buffer = jnp.full((p * s + 1,) + value.shape[1:], fill, value.dtype)
                buffer = buffer.at[target].set(value)
                return buffer[: p * s].reshape((p, s) + value.shape[1:])

            send_position = send_buffer(new_position, 0.0)
            send_momentum = send_buffer(new_momentum, 0.0)
            send_mass = send_buffer(old_mass, 0.0)
            send_ids = send_buffer(old_ids, -1)
            send_slots = send_buffer(old_slots, -1)
            send_rng = send_buffer(new_rng, 0)
            send_valid = send_buffer(valid, False)

            def route(value):
                return jax.lax.all_to_all(
                    value,
                    axis,
                    split_axis=0,
                    concat_axis=0,
                    tiled=True,
                )

            received_position = route(send_position).reshape((-1, new_position.shape[-1]))
            received_momentum = route(send_momentum).reshape((-1, new_momentum.shape[-1]))
            received_mass = route(send_mass).reshape((-1,))
            received_ids = route(send_ids).reshape((-1,))
            received_slots = route(send_slots).reshape((-1,))
            received_rng = route(send_rng).reshape((-1,))
            received_valid = route(send_valid).reshape((-1,))
            received_count = jnp.sum(received_valid, dtype=jnp.int32)
            receive_ok = received_count <= c
            padded_valid = jnp.concatenate(
                (received_valid, jnp.zeros((c,), dtype=jnp.bool_))
            )
            padded_ids = jnp.concatenate(
                (received_ids, jnp.full((c,), -1, dtype=received_ids.dtype))
            )
            maximum_id = jnp.iinfo(received_ids.dtype).max
            order = jnp.argsort(
                jnp.where(padded_valid, padded_ids, maximum_id), stable=True
            )[:c]

            def compact(value, fill):
                padding = jnp.full((c,) + value.shape[1:], fill, value.dtype)
                return jnp.concatenate((value, padding), axis=0)[order]

            candidate_position = compact(received_position, 0.0)
            candidate_momentum = compact(received_momentum, 0.0)
            candidate_mass = compact(received_mass, 0.0)
            candidate_ids = compact(received_ids, -1)
            candidate_slots = compact(received_slots, -1)
            candidate_rng = compact(received_rng, 0)
            candidate_active = compact(received_valid, False)
            audit_destination = jnp.mod(candidate_ids, p).astype(jnp.int32)
            audit_hot = (
                jax.nn.one_hot(audit_destination, p, dtype=jnp.int32)
                * candidate_active[:, None]
            )
            audit_rank_table = jnp.cumsum(audit_hot, axis=0) - 1
            audit_rank = jnp.sum(audit_rank_table * audit_hot, axis=-1)
            audit_target = jnp.where(
                candidate_active, audit_destination * c + audit_rank, p * c
            )

            def audit_buffer(value, fill):
                buffer = jnp.full((p * c + 1,), fill, value.dtype)
                return buffer.at[audit_target].set(value)[: p * c].reshape((p, c))

            audit_ids = route(audit_buffer(candidate_ids, -1)).reshape((-1,))
            audit_valid = route(audit_buffer(candidate_active, False)).reshape((-1,))
            audit_order = jnp.lexsort((audit_ids, ~audit_valid))
            sorted_audit_ids = audit_ids[audit_order]
            sorted_audit_valid = audit_valid[audit_order]
            ids_unique_bucket = ~jnp.any(
                sorted_audit_valid[1:]
                & sorted_audit_valid[:-1]
                & (sorted_audit_ids[1:] == sorted_audit_ids[:-1])
            )
            ids_unique = jax.lax.pmin(ids_unique_bucket.astype(jnp.int32), axis) == 1
            local_ok = (
                send_ok & receive_ok & jnp.all(finite_particle | ~old_active) & ids_unique
            )
            successful = jax.lax.pmin(local_ok.astype(jnp.int32), axis) == 1
            current_index = jax.lax.axis_index(axis).astype(jnp.int32)
            candidate_owner = jnp.full((c,), current_index, dtype=jnp.int32)
            migration_local = jnp.sum(
                valid & (destination != current_index), dtype=jnp.int32
            )
            migration_count = jax.lax.psum(migration_local, axis)
            total_receive = jax.lax.psum(received_count, axis)
            maximum_send = jax.lax.pmax(jnp.max(send_count), axis)
            finite = (
                jax.lax.pmin(
                    jnp.all(finite_particle | ~old_active).astype(jnp.int32), axis
                )
                == 1
            )

            def commit(candidate, previous):
                mask = successful.reshape((1,) * candidate.ndim)
                return jnp.where(mask, candidate, previous)

            return (
                commit(candidate_position, old_position),
                commit(candidate_momentum, old_momentum),
                commit(candidate_mass, old_mass),
                commit(candidate_ids, old_ids),
                commit(candidate_slots, old_slots),
                commit(candidate_active, old_active),
                commit(candidate_rng, old_rng),
                commit(candidate_owner, jnp.full((c,), current_index, jnp.int32)),
                maximum_send,
                total_receive,
                migration_count,
                finite,
                ids_unique,
                successful,
            )

        in_specs = (
            PartitionSpec(axis, None),
            PartitionSpec(axis, None),
            PartitionSpec(axis),
            PartitionSpec(axis),
            PartitionSpec(axis),
            PartitionSpec(axis),
            PartitionSpec(axis),
            PartitionSpec(axis),
            PartitionSpec(axis, None),
            PartitionSpec(axis, None),
            PartitionSpec(axis),
        )
        out_specs = (
            PartitionSpec(axis, None),
            PartitionSpec(axis, None),
            PartitionSpec(axis),
            PartitionSpec(axis),
            PartitionSpec(axis),
            PartitionSpec(axis),
            PartitionSpec(axis),
            PartitionSpec(axis),
            PartitionSpec(),
            PartitionSpec(),
            PartitionSpec(),
            PartitionSpec(),
            PartitionSpec(),
            PartitionSpec(),
        )
        routed = jax.shard_map(
            exchange,
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            state.positions,
            state.momenta,
            state.masses,
            state.stable_ids,
            state.logical_slots,
            state.active_mask,
            state.rng_counters,
            state.owner,
            positions,
            momenta,
            rng,
        )
        (
            out_position,
            out_momentum,
            out_mass,
            out_ids,
            out_slots,
            out_active,
            out_rng,
            out_owner,
            send_count,
            receive_count,
            migration_count,
            finite,
            ids_unique,
            successful,
        ) = routed
        accepted_scale = jnp.where(successful, scale, state.scale_factor)
        output = DistributedParticleState(
            out_position,
            out_momentum,
            out_mass,
            out_ids,
            out_slots,
            out_active,
            out_rng,
            accepted_scale,
            out_owner,
            runtime_id,
        )
        evidence = ParticleExchangeEvidence(
            send_count,
            receive_count,
            migration_count,
            s,
            c,
            finite,
            ids_unique,
            successful,
            runtime_id,
        )
        return DistributedParticleMigrationResult(output, evidence)

    def exchange_ghosts(
        self, state: DistributedParticleState, /
    ) -> DistributedParticleGhostState:
        """Exchange fixed-capacity left/right slab ghosts through periodic ppermute."""

        self._require_state(state)
        p = self.plan.layout.device_count
        g = self.plan.ghost_capacity
        width = self.plan.ghost_width
        length = self.plan.box_size[0]
        key_boundaries = self.plan.layout.key_boundaries
        axis = self.axis_name
        left_permutation = tuple((source, (source - 1) % p) for source in range(p))
        right_permutation = tuple((source, (source + 1) % p) for source in range(p))

        def exchange(position, mass, ids, active):
            owner = jax.lax.axis_index(axis).astype(jnp.int32)
            maximum = jnp.asarray(np.iinfo(np.uint32).max, dtype=position.dtype)
            lower = (
                jnp.asarray(length, position.dtype)
                * key_boundaries[owner].astype(position.dtype)
                / maximum
            )
            upper = (
                jnp.asarray(length, position.dtype)
                * key_boundaries[owner + 1].astype(position.dtype)
                / maximum
            )
            x = jnp.mod(position[:, 0], length)
            tolerance = jnp.asarray(width, position.dtype) + 8.0 * jnp.finfo(
                position.dtype
            ).eps * jnp.asarray(length, position.dtype)
            left_mask = active & ((x - lower) <= tolerance)
            right_mask = active & ((upper - x) <= tolerance)
            maximum_id = jnp.iinfo(ids.dtype).max

            def pack(mask, values, fill):
                order = jnp.argsort(jnp.where(mask, ids, maximum_id), stable=True)[:g]
                selected = values[order]
                selected_mask = mask[order]
                payload_mask = selected_mask.reshape(
                    selected_mask.shape + (1,) * (values.ndim - 1)
                )
                return jnp.where(payload_mask, selected, jnp.asarray(fill, values.dtype))

            left_position = pack(left_mask, position, 0.0)
            right_position = pack(right_mask, position, 0.0)
            left_mass = pack(left_mask, mass, 0.0)
            right_mass = pack(right_mask, mass, 0.0)
            left_ids = pack(left_mask, ids, -1)
            right_ids = pack(right_mask, ids, -1)
            left_valid = pack(left_mask, active, False)
            right_valid = pack(right_mask, active, False)
            left_count = jnp.sum(left_mask, dtype=jnp.int32)
            right_count = jnp.sum(right_mask, dtype=jnp.int32)
            local_ok = (left_count <= g) & (right_count <= g)
            successful = jax.lax.pmin(local_ok.astype(jnp.int32), axis) == 1

            from_left_position = jax.lax.ppermute(right_position, axis, right_permutation)
            from_right_position = jax.lax.ppermute(left_position, axis, left_permutation)
            from_left_mass = jax.lax.ppermute(right_mass, axis, right_permutation)
            from_right_mass = jax.lax.ppermute(left_mass, axis, left_permutation)
            from_left_ids = jax.lax.ppermute(right_ids, axis, right_permutation)
            from_right_ids = jax.lax.ppermute(left_ids, axis, left_permutation)
            from_left_valid = jax.lax.ppermute(right_valid, axis, right_permutation)
            from_right_valid = jax.lax.ppermute(left_valid, axis, left_permutation)
            from_left_count = jax.lax.ppermute(right_count, axis, right_permutation)
            from_right_count = jax.lax.ppermute(left_count, axis, left_permutation)
            source_left = jnp.full((g,), (owner - 1) % p, jnp.int32)
            source_right = jnp.full((g,), (owner + 1) % p, jnp.int32)
            return (
                jnp.concatenate((from_left_position, from_right_position), axis=0),
                jnp.concatenate((from_left_mass, from_right_mass), axis=0),
                jnp.concatenate((from_left_ids, from_right_ids), axis=0),
                jnp.concatenate((source_left, source_right), axis=0),
                jnp.concatenate((from_left_valid, from_right_valid), axis=0),
                from_left_count[None],
                from_right_count[None],
                successful,
            )

        outputs = jax.shard_map(
            exchange,
            mesh=self.mesh,
            in_specs=(
                PartitionSpec(self.axis_name, None),
                PartitionSpec(self.axis_name),
                PartitionSpec(self.axis_name),
                PartitionSpec(self.axis_name),
            ),
            out_specs=(
                PartitionSpec(self.axis_name, None),
                PartitionSpec(self.axis_name),
                PartitionSpec(self.axis_name),
                PartitionSpec(self.axis_name),
                PartitionSpec(self.axis_name),
                PartitionSpec(self.axis_name),
                PartitionSpec(self.axis_name),
                PartitionSpec(),
            ),
            check_vma=False,
        )(state.positions, state.masses, state.stable_ids, state.active_mask)
        return DistributedParticleGhostState(
            *outputs[:7],
            g,
            outputs[7],
            self.runtime_id,
        )

    def logical_arrays(self, state: DistributedParticleState, /) -> dict[str, Array]:
        """Return globally sharded logical-order arrays without a host gather."""

        self._require_state(state)
        n = self.total_capacity
        sentinel = n
        index = jnp.where(state.active_mask, state.logical_slots, sentinel)

        def scatter(values, fill):
            target = jnp.full((n + 1,) + values.shape[1:], fill, values.dtype)
            target = target.at[index].set(values)
            return target[:n]

        return {
            "positions": jax.device_put(
                scatter(state.positions, 0.0), self.vector_sharding
            ),
            "momenta": jax.device_put(scatter(state.momenta, 0.0), self.vector_sharding),
            "masses": jax.device_put(scatter(state.masses, 0.0), self.scalar_sharding),
            "stable_ids": jax.device_put(
                scatter(state.stable_ids, -1), self.scalar_sharding
            ),
            "active_mask": jax.device_put(
                scatter(state.active_mask, False), self.scalar_sharding
            ),
            "rng_counters": jax.device_put(
                scatter(state.rng_counters, 0), self.scalar_sharding
            ),
        }

    def owner_order(
        self, logical_values: ArrayLike, state: DistributedParticleState, /
    ) -> Array:
        """Route one logical particle payload back to current owner slots."""

        self._require_state(state)
        values = jnp.asarray(logical_values)
        if values.shape[0] != self.total_capacity:
            raise ValueError("Logical particle payload must begin with total capacity.")
        safe = jnp.clip(state.logical_slots, 0, self.total_capacity - 1)
        selected = values[safe]
        mask = state.active_mask.reshape(
            state.active_mask.shape + (1,) * (values.ndim - 1)
        )
        selected = jnp.where(mask, selected, jnp.zeros((), dtype=values.dtype))
        sharding = self.scalar_sharding if selected.ndim == 1 else self.vector_sharding
        return jax.device_put(selected, sharding)


__all__ = [
    "DistributedParticleGhostState",
    "DistributedParticleMigrationResult",
    "DistributedParticleRuntimePlan",
    "DistributedParticleState",
    "ParticleExchangeEvidence",
    "PreparedDistributedParticleRuntime",
]
