#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Particle decompositions: legacy Cartesian slabs and fractional owner grids.

`ParticleDomainDecompositionPlan` is the historical slab contract along the
first axis of an axis-aligned `ParticleBox`. `FractionalOwnerPartition` is the
lattice-aware owner contract: owners tile the fractional coordinates of a
`PeriodicCell`, so oblique and partially periodic cells are partitioned in
their own coordinates rather than as diagonal boxes. Periodic image aliases
are integer lattice translations of an owned atom; they never create a new
physical atom identity.
"""

from __future__ import annotations

from itertools import product
from typing import final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...backends.distributed import JaxCollectiveProvider
from ...ein import contract
from ...typing import parse
from .._periodic_cell import PeriodicCell
from ..spatial._distributed_relations import _bucket_ranks, _pack
from ._pairwise import ParticleBox
from ._precision import ParticlePrecisionPolicy


ParticleBackendKind: TypeAlias = Literal["pure-jax", "pallas", "triton"]
ParticleDeterminism: TypeAlias = Literal["fast", "deterministic", "compensated"]


class ParticleBackendPolicy(StrictModule, NonTrainableState):
    backend: ParticleBackendKind = eqx.field(static=True)
    determinism: ParticleDeterminism = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        backend: ParticleBackendKind = "pure-jax",
        /,
        *,
        determinism: ParticleDeterminism = "deterministic",
    ) -> None:
        backend = parse(backend, ParticleBackendKind, "backend")
        determinism = parse(determinism, ParticleDeterminism, "determinism")
        self.backend = backend
        self.determinism = determinism
        self.policy_id = canonical_fingerprint(
            {
                "kind": "particle-backend-policy",
                "backend": backend,
                "determinism": determinism,
            }
        )


class ParticleKernelRequestPlan(StrictModule, NonTrainableState):
    distance: bool = eqx.field(static=True)
    direction: bool = eqx.field(static=True)
    kernel_value: bool = eqx.field(static=True)
    kernel_gradient: bool = eqx.field(static=True)
    smoothing_derivative: bool = eqx.field(static=True)
    materialize: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        distance: bool = True,
        direction: bool = False,
        kernel_value: bool = False,
        kernel_gradient: bool = True,
        smoothing_derivative: bool = False,
        materialize: bool = False,
    ) -> None:
        self.distance = bool(distance)
        self.direction = bool(direction)
        self.kernel_value = bool(kernel_value)
        self.kernel_gradient = bool(kernel_gradient)
        self.smoothing_derivative = bool(smoothing_derivative)
        self.materialize = bool(materialize)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-kernel-request",
                "distance": distance,
                "direction": direction,
                "kernel_value": kernel_value,
                "kernel_gradient": kernel_gradient,
                "smoothing_derivative": smoothing_derivative,
                "materialize": materialize,
            }
        )


class MixedPrecisionCertification(StrictModule):
    finite: Array
    relative_error: Array
    tolerance: Array
    successful: Array
    evidence_id: str = eqx.field(static=True)


def certify_particle_precision(
    reference: ArrayLike,
    candidate: ArrayLike,
    precision: ParticlePrecisionPolicy,
    /,
    *,
    tolerance: float,
) -> MixedPrecisionCertification:
    reference_ = precision.certification(reference)
    candidate_ = precision.certification(candidate)
    scale = jnp.maximum(jnp.max(jnp.abs(reference_)), jnp.finfo(reference_.dtype).tiny)
    error = jnp.max(jnp.abs(candidate_ - reference_)) / scale
    finite = jnp.all(jnp.isfinite(candidate_))
    successful = finite & (error <= tolerance)
    evidence_id = canonical_fingerprint(
        {
            "kind": "particle-mixed-precision-certification",
            "precision": precision.policy_id,
            "tolerance": tolerance,
            "shape": list(reference_.shape),
        }
    )
    return MixedPrecisionCertification(
        finite, error, jnp.asarray(tolerance, reference_.dtype), successful, evidence_id
    )


class ParticleDomainDecompositionPlan(StrictModule, NonTrainableState):
    partitions: int = eqx.field(static=True)
    halo_radius: float = eqx.field(static=True)
    box: ParticleBox
    plan_id: str = eqx.field(static=True)

    def __init__(self, partitions: int, halo_radius: float, box: ParticleBox, /) -> None:
        if partitions <= 0 or halo_radius <= 0.0:
            raise ValueError("Particle decomposition parameters are invalid.")
        self.partitions = int(partitions)
        self.halo_radius = float(halo_radius)
        self.box = box
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-domain-decomposition",
                "partitions": partitions,
                "halo_radius": halo_radius,
                "box": box.box_id,
            }
        )


class ParticleHaloState(StrictModule, NonTrainableState):
    owner: Array
    owned_mask: Array
    halo_mask: Array
    local_mask: Array
    migration_count: Array
    halo_count: Array
    successful: Array


def prepare_particle_halos(
    plan: ParticleDomainDecompositionPlan,
    position: ArrayLike,
    active_mask: ArrayLike,
    /,
) -> ParticleHaloState:
    position_ = jnp.asarray(position)
    active = jnp.asarray(active_mask, bool)
    relative = (position_[:, 0] - plan.box.lower[0]) / plan.box.lengths[0]
    owner = jnp.clip(
        jnp.floor(relative * plan.partitions).astype(jnp.int32), 0, plan.partitions - 1
    )
    owned = jnp.arange(plan.partitions)[:, None] == owner[None, :]
    edges = (
        plan.box.lower[0]
        + plan.box.lengths[0] * jnp.arange(plan.partitions + 1) / plan.partitions
    )
    distance_to_left = jnp.abs(position_[:, 0][None, :] - edges[:-1, None])
    distance_to_right = jnp.abs(position_[:, 0][None, :] - edges[1:, None])
    if plan.box.periodic_axes[0]:
        length = plan.box.lengths[0]
        distance_to_left = jnp.mod(distance_to_left, length)
        distance_to_right = jnp.mod(distance_to_right, length)
        distance_to_left = jnp.minimum(distance_to_left, length - distance_to_left)
        distance_to_right = jnp.minimum(distance_to_right, length - distance_to_right)
    halo = (
        (distance_to_left <= plan.halo_radius) | (distance_to_right <= plan.halo_radius)
    ) & ~owned
    halo = halo & active[None, :]
    owned = owned & active[None, :]
    local = owned | halo
    return ParticleHaloState(
        owner,
        owned,
        halo,
        local,
        jnp.zeros((), jnp.int32),
        jnp.sum(halo, dtype=jnp.int32),
        jnp.all(jnp.isfinite(position_)),
    )


def halo_update(values: ArrayLike, halo: ParticleHaloState, /) -> Array:
    value = jnp.asarray(values)
    return jnp.where(
        halo.local_mask.reshape(halo.local_mask.shape + (1,) * (value.ndim - 1)),
        value[None, ...],
        0.0,
    )


def halo_sum(local_values: ArrayLike, halo: ParticleHaloState, /) -> Array:
    local = jnp.asarray(local_values)
    if local.shape[:2] != halo.local_mask.shape:
        raise ValueError("Halo-local values must begin with (partition, particle).")
    return jnp.sum(local, axis=0)


def migrate_particle_halos(
    plan: ParticleDomainDecompositionPlan,
    previous: ParticleHaloState,
    position: ArrayLike,
    active_mask: ArrayLike,
    /,
) -> ParticleHaloState:
    current = prepare_particle_halos(plan, position, active_mask)
    migration = jnp.sum(
        (current.owner != previous.owner) & jnp.asarray(active_mask, bool)
    )
    return ParticleHaloState(
        current.owner,
        current.owned_mask,
        current.halo_mask,
        current.local_mask,
        migration,
        current.halo_count,
        current.successful,
    )


class ImageAliasPackets(NamedTuple):
    """Periodic image aliases received by one owner region.

    Row ``q * packet_capacity + p`` came from owner ``q``. An alias names the
    physical atom by its owner and slot and positions its image at the stored
    source coordinate plus ``translations @ H``.
    """

    owners: Array
    slots: Array
    translations: Array
    positions: Array
    valid: Array
    maximum_load: Array


@final
class FractionalOwnerPartition(StrictModule, NonTrainableState):
    """Declared owner grid over the fractional coordinates of a periodic cell.

    ``counts[a]`` owners split lattice axis ``a``. Owner ``o`` has the
    C-ordered grid index of ``counts`` and the half-open fractional region
    ``[i_a / counts[a], (i_a + 1) / counts[a])`` per axis. Periodic axes are
    assigned on wrapped coordinates in ``[0, 1)``; on nonperiodic lattice axes
    the first and last owner regions extend to infinity, so every finite atom
    has exactly one owner. Components orthogonal to a lower-rank lattice are
    never partitioned.

    An atom at wrapped fractional coordinate ``s`` may act as a source for a
    receiver of owner ``o`` within Cartesian radius ``R`` only through an image
    ``s + m`` with ``m`` an integer translation (zero on nonperiodic axes) that
    lies inside owner ``o``'s region widened by ``R * ||H^{-1}[:, a]||`` on
    every axis ``a``. `alias_mask` evaluates exactly this conservative test;
    completeness over ``m`` is the caller's `PeriodicImageStencil` contract.
    """

    cell: PeriodicCell
    counts: tuple[int, ...] = eqx.field(static=True)
    owner_count: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(self, cell: PeriodicCell, counts: tuple[int, ...], /) -> None:
        if not isinstance(cell, PeriodicCell):
            raise TypeError("cell must be a PeriodicCell.")
        grid = tuple(counts)
        if len(grid) != cell.rank:
            raise ValueError("counts must hold one owner count per lattice axis.")
        for count in grid:
            if isinstance(count, (bool, np.bool_)) or not isinstance(
                count, (int, np.integer)
            ):
                raise TypeError("Owner counts must be integers.")
            if count < 1:
                raise ValueError("Owner counts must be positive.")
        normalized = tuple(int(count) for count in grid)
        self.cell = cell
        self.counts = normalized
        self.owner_count = int(np.prod(normalized))
        self.partition_id = canonical_fingerprint(
            {
                "kind": "fractional-owner-partition",
                "cell": cell.cell_id,
                "counts": list(normalized),
            }
        )

    def _grid_index(self) -> np.ndarray:
        """Host ``(owner_count, rank)`` grid index of every owner in C order."""
        return np.asarray(
            tuple(product(*(range(count) for count in self.counts))), dtype=np.int32
        ).reshape((self.owner_count, self.cell.rank))

    def owners(self, fractional: ArrayLike, /) -> Array:
        """Owner of each row of wrapped fractional coordinates ``(..., rank)``."""
        value = jnp.asarray(fractional)
        if not value.shape or value.shape[-1] != self.cell.rank:
            raise ValueError("Fractional coordinates must end in the lattice rank.")
        counts = jnp.asarray(self.counts, dtype=jnp.int32)
        index = jnp.clip(
            jnp.floor(value * counts.astype(value.dtype)).astype(jnp.int32),
            0,
            counts - 1,
        )
        strides = np.cumprod((1,) + self.counts[:0:-1])[::-1].astype(np.int32)
        return jnp.sum(index * jnp.asarray(strides), axis=-1, dtype=jnp.int32)

    def region_bounds(self, dtype: DTypeLike, /) -> tuple[Array, Array]:
        """Fractional ``(lower, upper)`` bounds, each ``(owner_count, rank)``."""
        index = self._grid_index()
        counts = np.asarray(self.counts, dtype=np.float64)
        lower = index / counts
        upper = (index + 1) / counts
        open_axes = ~np.asarray(self.cell.periodic_axes, dtype=np.bool_)
        lower = np.where(open_axes & (index == 0), -np.inf, lower)
        upper = np.where(open_axes & (index == counts - 1), np.inf, upper)
        return jnp.asarray(lower, dtype=dtype), jnp.asarray(upper, dtype=dtype)

    def alias_mask(
        self, fractional: ArrayLike, shifts: ArrayLike, reach: ArrayLike, /
    ) -> Array:
        """Which ``(atom, translation, owner)`` images may reach owner receivers.

        ``fractional`` is ``(atoms, rank)`` wrapped coordinates, ``shifts`` the
        ``(images, rank)`` integer stencil, and ``reach`` the per-axis fractional
        widening ``R * ||H^{-1}[:, a]||``. The working set is
        ``atoms x images x owners x rank``: bounded by the owner grid and the
        declared stencil, never by the global atom count.
        """
        value = jnp.asarray(fractional)
        stencil = jnp.asarray(shifts)
        widening = jnp.asarray(reach, dtype=value.dtype)
        rank = self.cell.rank
        if value.ndim != 2 or value.shape[1] != rank:
            raise ValueError("fractional must have shape (atoms, rank).")
        if stencil.ndim != 2 or stencil.shape[1] != rank:
            raise ValueError("shifts must have shape (images, rank).")
        if widening.shape != (rank,):
            raise ValueError("reach must hold one widening per lattice axis.")
        lower, upper = self.region_bounds(value.dtype)
        images = value[:, None, :] + stencil.astype(value.dtype)[None, :, :]
        inside = (images[:, :, None, :] >= lower - widening) & (
            images[:, :, None, :] <= upper + widening
        )
        return jnp.all(inside, axis=-1)

    def local_alias_exchange(
        self,
        positions: Array,
        active: Array,
        cell_vectors: Array,
        inverse_vectors: Array,
        shifts: Array,
        radius: float,
        axis_name: str,
        packet_capacity: int,
        /,
    ) -> ImageAliasPackets:
        """Exchange periodic image aliases inside one mapped owner region.

        Every active owned atom sends, to every owner whose widened region its
        image ``s + m`` reaches, one alias ``(slot, k)`` with ``k = m - floor(s)``
        on periodic axes, so the alias position is ``x + k @ H`` relative to the
        stored (possibly unwrapped) coordinate ``x``. The owner itself receives
        only nonzero translations of its own atoms. Aliases carry the stable
        owner/slot of the physical atom, never a new identity. Packets per
        destination are capped at ``packet_capacity``; the returned load reports
        refusal instead of truncating silently.
        """
        owners = self.owner_count
        provider = JaxCollectiveProvider(axis_name)
        me = jax.lax.axis_index(axis_name).astype(jnp.int32)
        dtype = positions.dtype
        matrix = cell_vectors.astype(dtype)
        inverse = inverse_vectors.astype(dtype)
        fractional = contract(
            "ni,ia->na", positions - self.cell.origin.astype(dtype), inverse
        )
        periodic = self.cell.periodic_mask
        images = jnp.where(periodic, jnp.floor(fractional), 0.0)
        wrapped = fractional - images
        reach = radius * jnp.sqrt(jnp.sum(inverse * inverse, axis=0))
        stencil = shifts.astype(jnp.int32)
        reached = self.alias_mask(wrapped, stencil, reach)
        central = jnp.all(stencil == 0, axis=1)
        own_owner = jnp.arange(owners, dtype=jnp.int32) == me
        reached = (
            reached
            & active[:, None, None]
            & ~(central[None, :, None] & own_owner[None, None, :])
        )
        count, images_count = reached.shape[:2]
        translations = stencil[None, :, :] - images.astype(jnp.int32)[:, None, :]
        image_positions = positions[:, None, :] + contract(
            "nsa,ad->nsd", translations.astype(dtype), matrix
        )
        shape = (count, images_count, owners)
        destination = jnp.broadcast_to(
            jnp.arange(owners, dtype=jnp.int32)[None, None, :], shape
        ).reshape((-1,))
        valid = reached.reshape((-1,))
        rank, loads = _bucket_ranks(destination, valid, owners)
        fits = valid & (rank < packet_capacity)

        def route(values: Array, fill: ArrayLike) -> Array:
            """Pack per-(atom, translation) values for every destination owner."""
            expanded = jnp.broadcast_to(
                values[:, :, None], shape + values.shape[2:]
            ).reshape((-1,) + values.shape[2:])
            packet = _pack(
                expanded, destination, rank, fits, owners, packet_capacity, fill
            )
            received = provider.all_to_all(packet, split_axis=0, concat_axis=0)
            return received.reshape((owners * packet_capacity,) + values.shape[2:])

        slots = jnp.broadcast_to(
            jnp.arange(count, dtype=jnp.int32)[:, None], (count, images_count)
        )
        delivered = provider.all_to_all(
            _pack(fits, destination, rank, fits, owners, packet_capacity, False),
            split_axis=0,
            concat_axis=0,
        ).reshape((owners * packet_capacity,))
        senders = jnp.broadcast_to(
            jnp.arange(owners, dtype=jnp.int32)[:, None], (owners, packet_capacity)
        ).reshape((-1,))
        return ImageAliasPackets(
            owners=senders,
            slots=route(slots, 0),
            translations=route(translations, 0),
            positions=route(image_positions, 0),
            valid=delivered,
            maximum_load=jnp.max(loads),
        )


class ParticleLoadBalanceReport(StrictModule):
    owned_particles: Array
    halo_particles: Array
    weighted_work: Array
    imbalance: Array


def particle_load_balance_report(
    halo: ParticleHaloState,
    pair_counts: ArrayLike,
    iteration_counts: ArrayLike,
    /,
) -> ParticleLoadBalanceReport:
    owned = jnp.sum(halo.owned_mask, axis=1)
    halos = jnp.sum(halo.halo_mask, axis=1)
    work = owned + halos + jnp.asarray(pair_counts) + jnp.asarray(iteration_counts)
    imbalance = jnp.max(work) / jnp.maximum(jnp.mean(work), 1.0)
    return ParticleLoadBalanceReport(owned, halos, work, imbalance)


__all__ = [
    "FractionalOwnerPartition",
    "ImageAliasPackets",
    "MixedPrecisionCertification",
    "ParticleBackendPolicy",
    "ParticleDomainDecompositionPlan",
    "ParticleHaloState",
    "ParticleKernelRequestPlan",
    "ParticleLoadBalanceReport",
    "certify_particle_precision",
    "halo_sum",
    "halo_update",
    "migrate_particle_halos",
    "particle_load_balance_report",
    "prepare_particle_halos",
]
