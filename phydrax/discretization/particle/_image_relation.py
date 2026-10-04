#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation


IMAGE_INT32_LIMIT = 2**31 - 1


def symmetric_int32(value: Array, /) -> tuple[Array, Array]:
    """Return integer ``value`` as int32 and whether it lies in ``|v| <= 2**31 - 1``.

    The symmetric range keeps negation (``reversed``) exact.  Out-of-range
    entries are replaced by zero and reported, never clipped or wrapped.
    """
    info = jnp.iinfo(value.dtype)
    representable = jnp.ones(value.shape, dtype=jnp.bool_)
    if info.max > IMAGE_INT32_LIMIT:
        representable &= value <= jnp.asarray(IMAGE_INT32_LIMIT, dtype=value.dtype)
    if info.min < -IMAGE_INT32_LIMIT:
        representable &= value >= jnp.asarray(-IMAGE_INT32_LIMIT, dtype=value.dtype)
    return jnp.where(representable, value, 0).astype(jnp.int32), representable


def checked_int32_add(left: Array, right: Array, /) -> tuple[Array, Array]:
    """Add symmetric-range int32 arrays; report sums outside ``|v| <= 2**31 - 1``."""
    overflow = ((left > 0) & (right > IMAGE_INT32_LIMIT - left)) | (
        (left < 0) & (right < -IMAGE_INT32_LIMIT - left)
    )
    return left + right, overflow


def checked_int32_shift(base: Array, plus: Array, minus: Array, /) -> tuple[Array, Array]:
    """Return ``base + plus - minus`` for symmetric-range int32 and exact overflow.

    The verdict concerns the true result, not an intermediate: when
    ``plus - minus`` leaves the range its terms share a sign, so the sum is
    regrouped as ``(base + plus) - minus``.  Overflowing entries keep ``base``,
    which is never a fabricated translation.
    """
    difference, difference_overflow = checked_int32_add(plus, -minus)
    direct, direct_overflow = checked_int32_add(base, difference)
    partial, partial_overflow = checked_int32_add(base, plus)
    regrouped, regrouped_overflow = checked_int32_add(partial, -minus)
    overflow = jnp.where(
        difference_overflow, partial_overflow | regrouped_overflow, direct_overflow
    )
    value = jnp.where(difference_overflow, regrouped, direct)
    return jnp.where(overflow, base, value), overflow


def representation_shifts(
    relation: ParticleImageRelation, offsets: ArrayLike, /
) -> tuple[Array, Array]:
    """Return ``n + offsets[receiver] - offsets[source]`` and per-route overflow.

    Overflow marks valid routes whose offsets or re-expressed shift leave the
    symmetric int32 image representation; those routes keep their old shift.
    """
    delta = jnp.asarray(offsets)
    expected = (relation.case_count * relation.particle_capacity, relation.lattice_rank)
    if delta.shape != expected or not jnp.issubdtype(delta.dtype, jnp.integer):
        raise ValueError(f"offsets must be integers with shape {expected}.")
    delta, representable = symmetric_int32(delta)
    shifts, overflow = checked_int32_shift(
        relation.image_shifts,
        delta[relation.receiver_indices],
        delta[relation.source_indices],
    )
    overflow = jnp.any(overflow, axis=-1) | ~jnp.all(
        representable[relation.receiver_indices] & representable[relation.source_indices],
        axis=-1,
    )
    overflow = relation.valid & overflow
    shifts = jnp.where(overflow[:, None], relation.image_shifts, shifts)
    return jnp.where(relation.valid[:, None], shifts, 0), overflow


class ParticleImageRelation(StrictModule, NonTrainableState):
    """Directed same-set periodic image routes with explicit integer translations.

    Route ``e`` connects source particle ``s_e`` to receiver ``t_e`` in case
    ``c_e`` through the integer lattice translation ``n_e``.  Its native
    displacement is ``d_e = x[t_e] - x[s_e] + n_e @ H[c_e]`` for row lattice
    vectors ``H``.  Route identity is the stable ``(source id, receiver id, n,
    case)`` tuple, never a slot or a distance coincidence.  Every nonzero self
    image is admissible; ``(s == t, n == 0)`` is forbidden.  Unlike
    ``ParticlePairRelation`` this relation is directed and repeated
    ``(source, receiver)`` pairs with distinct ``n`` are distinct routes; it is
    not a pair-once classical relation.

    Particle indices are flat case-major ``case * particle_capacity + local``.
    Shifts on nonperiodic axes are zero by construction of the producing search.
    """

    relation: EdgeRelation
    source_particle_ids: Array
    receiver_particle_ids: Array
    image_shifts: Array
    route_cases: Array
    case_count: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    lattice_rank: int = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    relation_schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation: EdgeRelation,
        source_particle_ids: ArrayLike,
        receiver_particle_ids: ArrayLike,
        image_shifts: ArrayLike,
        route_cases: ArrayLike,
        /,
        *,
        case_count: int,
        particle_capacity: int,
        support_id: str,
        relation_schema_id: str,
    ) -> None:
        if not isinstance(relation, EdgeRelation):
            raise TypeError("relation must be an EdgeRelation.")
        cases = int(case_count)
        capacity = int(particle_capacity)
        if cases <= 0 or capacity <= 0:
            raise ValueError("case_count and particle_capacity must be positive.")
        if relation.source_size != cases * capacity or relation.target_size != (
            cases * capacity
        ):
            raise ValueError(
                "Image relation endpoints must index the flat case-major particles."
            )
        routes = relation.capacity
        source_ids = jnp.asarray(source_particle_ids)
        receiver_ids = jnp.asarray(receiver_particle_ids)
        shifts = jnp.asarray(image_shifts)
        route_case = jnp.asarray(route_cases)
        if source_ids.shape != (routes,) or receiver_ids.shape != (routes,):
            raise ValueError("Image route particle IDs must align with routes.")
        if shifts.ndim != 2 or shifts.shape[0] != routes:
            raise ValueError("image_shifts must have shape (routes, lattice rank).")
        if route_case.shape != (routes,):
            raise ValueError("route_cases must align with routes.")
        for name, value in (
            ("source_particle_ids", source_ids),
            ("receiver_particle_ids", receiver_ids),
            ("image_shifts", shifts),
            ("route_cases", route_case),
        ):
            if not jnp.issubdtype(value.dtype, jnp.integer):
                raise TypeError(f"{name} must contain integers.")
        schema = str(relation_schema_id)
        support = str(support_id)
        if not schema or not support:
            raise ValueError("Image relation support and schema IDs must be non-empty.")
        valid = relation.valid
        # Range checks precede the int32 casts so wide inputs refuse instead of
        # wrapping into a different translation or case.
        shifts, shift_representable = symmetric_int32(shifts)
        unrepresentable = valid & ~jnp.all(shift_representable, axis=-1)
        miscased = valid & (
            (route_case < 0)
            | (route_case >= cases)
            | (relation.source_indices // capacity != route_case)
            | (relation.target_indices // capacity != route_case)
        )
        route_case = jnp.where(miscased, 0, route_case).astype(jnp.int32)
        self_zero = valid & (
            (relation.source_indices == relation.target_indices)
            & jnp.all(shifts == 0, axis=-1)
            & ~unrepresentable
        )
        if not isinstance(self_zero, jax.core.Tracer):
            if bool(np.any(unrepresentable)):
                raise ValueError(
                    "Image shifts must lie in the symmetric int32 range |n| <= 2**31 - 1."
                )
            if bool(np.any(self_zero)):
                raise ValueError(
                    "Image relations exclude the zero translation of a particle to itself."
                )
            if bool(np.any(miscased)):
                raise ValueError(
                    "Valid image routes must lie in a case 0 <= c < case_count that "
                    "owns both endpoints."
                )
        # Traced construction (an image search rebuild) keeps runtime refusals;
        # invalid padding routes are excluded by ``valid`` and never trip them.
        checked_valid = eqx.error_if(
            valid,
            jnp.any(unrepresentable),
            "Image shifts must lie in the symmetric int32 range |n| <= 2**31 - 1.",
        )
        checked_valid = eqx.error_if(
            checked_valid,
            jnp.any(self_zero),
            "Image relations exclude the zero translation of a particle to itself.",
        )
        checked_valid = eqx.error_if(
            checked_valid,
            jnp.any(miscased),
            "Valid image routes must lie in a case that owns both endpoints.",
        )
        self.relation = relation.with_valid(checked_valid)
        self.source_particle_ids = source_ids
        self.receiver_particle_ids = receiver_ids
        self.image_shifts = shifts
        self.route_cases = route_case
        self.case_count = cases
        self.particle_capacity = capacity
        self.lattice_rank = int(shifts.shape[1])
        self.support_id = support
        self.relation_schema_id = schema

    @property
    def source_indices(self) -> Array:
        return self.relation.source_indices

    @property
    def receiver_indices(self) -> Array:
        return self.relation.target_indices

    @property
    def valid(self) -> Array:
        return self.relation.valid

    @property
    def capacity(self) -> int:
        return self.relation.capacity

    def translations(self, cell_vectors: ArrayLike, /) -> Array:
        """Return ``n_e @ H[c_e]`` for ``cell_vectors`` of shape ``(r, d)`` or ``(case, r, d)``."""
        vectors = jnp.asarray(cell_vectors)
        if vectors.ndim == 2:
            vectors = vectors[None]
        if vectors.ndim != 3 or vectors.shape[:2] != (
            self.case_count,
            self.lattice_rank,
        ):
            raise ValueError(
                "cell_vectors must have shape (lattice rank, d) or (case, lattice rank, d)."
            )
        routed = vectors[jnp.where(self.valid, self.route_cases, 0)]
        return contract(
            "ei,eid->ed",
            self.image_shifts.astype(vectors.dtype),
            routed,
            backend="jax",
        )

    def displacement(self, positions: ArrayLike, cell_vectors: ArrayLike, /) -> Array:
        """Return ``x[receiver] - x[source] + n @ H``, zero on invalid routes."""
        value = jnp.asarray(positions)
        if value.ndim != 2 or value.shape[0] != self.case_count * self.particle_capacity:
            raise ValueError("positions must have flat case-major particle shape.")
        translation = self.translations(jnp.asarray(cell_vectors, dtype=value.dtype))
        raw = value[self.receiver_indices] - value[self.source_indices] + translation
        return jnp.where(self.valid[:, None], raw, 0.0)

    def reversed(self) -> "ParticleImageRelation":
        """Map every route ``(source, receiver, n)`` to ``(receiver, source, -n)``.

        Route order is preserved; the result is not receiver-major sorted.
        """
        return ParticleImageRelation(
            self.relation.transpose(),
            self.receiver_particle_ids,
            self.source_particle_ids,
            -self.image_shifts,
            self.route_cases,
            case_count=self.case_count,
            particle_capacity=self.particle_capacity,
            support_id=self.support_id,
            relation_schema_id=self.relation_schema_id,
        )

    def with_representation_offsets(
        self, offsets: ArrayLike, /
    ) -> "ParticleImageRelation":
        """Re-express routes for positions moved by whole lattice translations.

        If each particle position becomes ``x'_p = x_p - offsets_p @ H``, the same
        physical image routes need ``n' = n + offsets[receiver] - offsets[source]``.
        Membership and order are unchanged; no rebuild or sort occurs.  A valid
        route whose offsets or ``n'`` leave ``|n| <= 2**31 - 1`` is refused (on
        the host, or by a runtime error when traced); status-carrying callers use
        ``ParticleImageNeighborhoodState.with_representation_offsets``.
        """
        shifts, overflow = representation_shifts(self, offsets)
        message = (
            "Re-expressed image shifts must lie in the symmetric int32 range "
            "|n| <= 2**31 - 1."
        )
        if not isinstance(overflow, jax.core.Tracer) and bool(np.any(overflow)):
            raise ValueError(message)
        shifts = eqx.error_if(shifts, jnp.any(overflow), message)
        return eqx.tree_at(lambda relation: relation.image_shifts, self, shifts)


class ParticleImageRelationEvidence(StrictModule, NonTrainableState):
    """Per-case requirements and complete failure evidence for one image search.

    ``required_*`` are the counts the geometry demands, independent of the
    capacity that truncated storage.  Capacity failures (cell occupancy,
    images, edges, degree) are repairable by a declared larger capacity.
    ``stencil_overflow`` (the lattice left the prepared deformation envelope of
    the candidate stencil), ``domain_violation``, ``nonfinite`` (non-finite
    active positions, a non-finite lattice, or a failed lattice solve, the last
    two independent of the active set) and ``representation_overflow`` (an
    active wrap count or route image shift outside ``|n| <= 2**31 - 1``) are
    geometric/scientific failures and never justify a capacity retry.
    """

    active_particles: Array
    required_cell_occupancy: Array
    required_edges: Array
    stored_edges: Array
    required_degree: Array
    required_images: Array
    required_offset_extents: Array
    cell_overflow: Array
    stencil_overflow: Array
    image_overflow: Array
    edge_overflow: Array
    degree_overflow: Array
    domain_violation: Array
    nonfinite: Array
    representation_overflow: Array

    @property
    def capacity_failure(self) -> Array:
        return (
            self.cell_overflow
            | self.image_overflow
            | self.edge_overflow
            | self.degree_overflow
        )

    @property
    def scientific_failure(self) -> Array:
        return (
            self.stencil_overflow
            | self.domain_violation
            | self.nonfinite
            | self.representation_overflow
        )

    @property
    def successful(self) -> Array:
        return ~(self.capacity_failure | self.scientific_failure)


__all__ = ["ParticleImageRelation", "ParticleImageRelationEvidence"]
