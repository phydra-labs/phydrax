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

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import integer
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

    def take_routes(
        self, compaction: ImageRouteCompaction, /, *, relation_schema_id: str
    ) -> ParticleImageRelation:
        """Return the routes selected by ``compaction`` as a new image relation.

        Identity, image shift and case of every kept route are copied exactly;
        padding routes are invalid and owned by their compaction case, with
        endpoints at that case's first particle.
        """
        if not isinstance(compaction, ImageRouteCompaction):
            raise TypeError("compaction must be an ImageRouteCompaction.")
        if compaction.case_count != self.case_count:
            raise ValueError("Compaction case count differs from the relation.")
        indices = compaction.indices
        valid = compaction.valid & self.valid[indices]
        padding = compaction.cases * self.particle_capacity
        size = self.case_count * self.particle_capacity
        return ParticleImageRelation(
            EdgeRelation(
                jnp.where(valid, self.source_indices[indices], padding),
                jnp.where(valid, self.receiver_indices[indices], padding),
                source_size=size,
                target_size=size,
                valid=valid,
            ),
            jnp.where(valid, self.source_particle_ids[indices], 0),
            jnp.where(valid, self.receiver_particle_ids[indices], 0),
            jnp.where(valid[:, None], self.image_shifts[indices], 0),
            compaction.cases,
            case_count=self.case_count,
            particle_capacity=self.particle_capacity,
            support_id=self.support_id,
            relation_schema_id=relation_schema_id,
        )

    def reversal_map(self) -> ImageReversalMap:
        """Match every route ``(s, t, n, c)`` to its reversal ``(t, s, -n, c)``.

        Matching uses material endpoint IDs, the integer image sign and the
        case, never storage slots or distances, so it is independent of route
        order.  A route is ``paired`` when its exact reversal is stored; a case
        is ``closed`` when every valid route in it is paired.  A complete radius
        search is closed because ``|x_t - x_s + n H| = |x_s - x_t - n H|``;
        truncated storage is not.  Nonzero self images ``(s, s, n)`` pair with
        the distinct route ``(s, s, -n)``.
        """
        invalid = (~self.valid).astype(jnp.int32)
        cases = jnp.where(self.valid, self.route_cases, 0)
        sources = jnp.where(self.valid, self.source_particle_ids, 0)
        receivers = jnp.where(self.valid, self.receiver_particle_ids, 0)
        shifts = jnp.where(self.valid[:, None], self.image_shifts, 0)
        rank = self.lattice_rank
        forward = tuple(shifts[:, axis] for axis in reversed(range(rank))) + (
            sources,
            receivers,
            cases,
            invalid,
        )
        backward = tuple(-shifts[:, axis] for axis in reversed(range(rank))) + (
            receivers,
            sources,
            cases,
            invalid,
        )
        forward_order = jax.lax.stop_gradient(jnp.lexsort(forward)).astype(jnp.int32)
        backward_order = jax.lax.stop_gradient(jnp.lexsort(backward)).astype(jnp.int32)
        partner = (
            jnp.zeros((self.capacity,), dtype=jnp.int32)
            .at[backward_order]
            .set(forward_order)
        )
        matched = self.valid & self.valid[partner]
        for key, reversed_key in zip(forward, backward, strict=True):
            matched = matched & (key[partner] == reversed_key)
        partner = jnp.where(matched, partner, jnp.arange(self.capacity, dtype=jnp.int32))
        unpaired = (
            jnp.zeros((self.case_count,), dtype=jnp.int32)
            .at[cases]
            .add((self.valid & ~matched).astype(jnp.int32))
        )
        conflict = self.valid & (
            (self.source_particle_ids == self.receiver_particle_ids)
            != (self.source_indices == self.receiver_indices)
        )
        conflicts = (
            jnp.zeros((self.case_count,), dtype=jnp.int32)
            .at[cases]
            .add(conflict.astype(jnp.int32))
        )
        return ImageReversalMap(
            partner,
            matched,
            unpaired == 0,
            conflicts > 0,
            self.relation_schema_id,
        )

    def canonical_half(
        self, /, *, capacity: int | None = None, case_capacity: int | None = None
    ) -> ParticleImageHalfView:
        """Return the exact canonical half of this directed relation.

        A route ``(s, t, n)`` is canonical when ``id_s < id_t``, or when
        ``id_s == id_t`` and the first nonzero component of ``n`` is positive.
        Exactly one of every reversal pair is canonical, including the two
        distinct routes ``(s, s, n)`` and ``(s, s, -n)`` of a nonzero self image.
        For a symmetric pair function ``phi(d) == phi(-d)`` the complete
        directed sum is twice the half sum; this identity does not hold for
        arbitrary directed message functions.  Exactly one of ``capacity``
        (one route buffer) or ``case_capacity`` (a fixed segment per case) is
        required; overflow, an unclosed parent and endpoint-ID conflicts are
        reported per case.
        """
        reversal = self.reversal_map()
        canonical = self.valid & (
            (self.source_particle_ids < self.receiver_particle_ids)
            | (
                (self.source_particle_ids == self.receiver_particle_ids)
                & (_first_nonzero_sign(self.image_shifts) > 0)
            )
        )
        compaction = compact_image_routes(
            canonical,
            self.route_cases,
            case_count=self.case_count,
            capacity=capacity,
            case_capacity=case_capacity,
        )
        schema = canonical_fingerprint(
            {
                "kind": "canonical-half-image-relation",
                "parent": self.relation_schema_id,
                "capacity": capacity,
                "case_capacity": case_capacity,
            }
        )
        half = self.take_routes(compaction, relation_schema_id=schema)
        full_routes = jnp.where(half.valid, compaction.indices, 0)
        return ParticleImageHalfView(
            half,
            full_routes,
            jnp.where(half.valid, reversal.partner[full_routes], 0),
            half.valid & (half.source_indices == half.receiver_indices),
            compaction.required,
            compaction.stored,
            compaction.overflow,
            ~reversal.closed,
            reversal.identity_conflict,
            self.relation_schema_id,
        )


def _first_nonzero_sign(shifts: Array, /) -> Array:
    """Sign of the first nonzero component of each integer image shift."""
    sign = jnp.zeros(shifts.shape[:1], dtype=jnp.int32)
    for axis in reversed(range(shifts.shape[1])):
        component = shifts[:, axis]
        sign = jnp.where(component != 0, jnp.sign(component).astype(jnp.int32), sign)
    return sign


class ImageRouteCompaction(StrictModule, NonTrainableState):
    """Fixed-capacity selection of routes from a parent image relation.

    ``indices`` are parent route positions; ``valid`` marks kept routes and
    ``cases`` owns every slot (padding included).  ``required`` counts the
    selected parent routes per case independently of storage, ``stored``
    those kept, and ``overflow`` marks cases whose selection did not fit.
    ``case_capacity`` is the fixed segment width of a per-case layout, or
    ``None`` for one shared buffer whose kept routes preserve parent order.
    """

    indices: Array
    valid: Array
    cases: Array
    required: Array
    stored: Array
    overflow: Array
    case_count: int = eqx.field(static=True)
    case_capacity: int | None = eqx.field(static=True)


def compact_image_routes(
    mask: ArrayLike,
    route_cases: ArrayLike,
    /,
    *,
    case_count: int,
    capacity: int | None = None,
    case_capacity: int | None = None,
) -> ImageRouteCompaction:
    """Compact masked routes into one buffer or into fixed per-case segments.

    Exactly one of ``capacity`` (shared buffer, parent order preserved) and
    ``case_capacity`` (segment ``c`` holds case ``c``'s routes in parent
    order) is required.  Selection beyond the capacity is never silent: it is
    reported per case by ``overflow`` with the true ``required`` count.
    """
    selected = jnp.asarray(mask, dtype=jnp.bool_)
    owners = jnp.asarray(route_cases)
    cases = int(case_count)
    if selected.ndim != 1 or owners.shape != selected.shape:
        raise ValueError("mask and route_cases must be aligned rank-one arrays.")
    if cases <= 0:
        raise ValueError("case_count must be positive.")
    if (capacity is None) == (case_capacity is None):
        raise ValueError("Exactly one of capacity and case_capacity is required.")
    routes = selected.shape[0]
    safe_cases = jnp.where(selected, owners, 0).astype(jnp.int32)
    required = (
        jnp.zeros((cases,), dtype=jnp.int32)
        .at[safe_cases]
        .add(selected.astype(jnp.int32))
    )
    if capacity is not None:
        size = int(capacity)
        if size < 0 or size > IMAGE_INT32_LIMIT:
            raise ValueError("capacity must be a nonnegative int32 count.")
        indices = jax.lax.stop_gradient(
            jnp.nonzero(selected, size=size, fill_value=0)[0]
        ).astype(jnp.int32)
        total = jnp.sum(selected, dtype=jnp.int32)
        valid = jnp.arange(size, dtype=jnp.int32) < jnp.minimum(total, size)
        out_cases = jnp.where(valid, safe_cases[indices], cases - 1)
        stored = (
            jnp.zeros((cases,), dtype=jnp.int32)
            .at[out_cases]
            .add(valid.astype(jnp.int32))
        )
        return ImageRouteCompaction(
            jnp.where(valid, indices, 0),
            valid,
            out_cases,
            required,
            stored,
            stored < required,
            cases,
            None,
        )
    width = integer(case_capacity, "case_capacity")
    if width <= 0 or cases * width > IMAGE_INT32_LIMIT:
        raise ValueError("case_capacity must be a positive int32 segment width.")
    key = jnp.where(selected, safe_cases, cases)
    order = jax.lax.stop_gradient(jnp.argsort(key, stable=True)).astype(jnp.int32)
    sorted_cases = key[order]
    offsets = jnp.cumsum(required) - required
    rank = (
        jnp.arange(routes, dtype=jnp.int32)
        - offsets[jnp.minimum(sorted_cases, cases - 1)]
    )
    keep = (sorted_cases < cases) & (rank < width)
    target = jnp.where(keep, sorted_cases * width + rank, cases * width)
    indices = (
        jnp.zeros((cases * width,), dtype=jnp.int32).at[target].set(order, mode="drop")
    )
    valid = jnp.zeros((cases * width,), dtype=jnp.bool_).at[target].set(True, mode="drop")
    return ImageRouteCompaction(
        indices,
        valid,
        jnp.repeat(jnp.arange(cases, dtype=jnp.int32), width),
        required,
        jnp.minimum(required, width),
        required > width,
        cases,
        width,
    )


class ImageReversalMap(StrictModule, NonTrainableState):
    """Exact route reversal ``(s, t, n, c) -> (t, s, -n, c)`` over stored routes.

    ``partner[e]`` is the stored reversal of route ``e`` where ``paired[e]``,
    else ``e`` itself.  ``closed[c]`` holds when every valid route of case
    ``c`` is paired.  ``identity_conflict[c]`` marks equal endpoint IDs on
    distinct particle slots (or the converse), which makes material identity
    ambiguous and every half view of that case invalid.
    """

    partner: Array
    paired: Array
    closed: Array
    identity_conflict: Array
    relation_schema_id: str = eqx.field(static=True)


class ParticleImageHalfView(StrictModule, NonTrainableState):
    """Canonical half of a directed image relation with its counting convention.

    ``relation`` holds one route of every reversal pair (see
    ``ParticleImageRelation.canonical_half``) in parent order.
    ``full_routes`` and ``partner_routes`` give, per half route, the parent
    route and its stored reversal.  Each half route represents exactly
    ``directed_multiplicity == 2`` directed routes, including nonzero self
    images, whose ``self_image`` flag is set.  ``overflow`` is a capacity
    failure; ``reversal_open`` (the parent is not reversal closed, e.g.
    truncated) and ``identity_conflict`` are scientific failures under which
    the half-sum identity does not hold.
    """

    relation: ParticleImageRelation
    full_routes: Array
    partner_routes: Array
    self_image: Array
    required_routes: Array
    stored_routes: Array
    overflow: Array
    reversal_open: Array
    identity_conflict: Array
    parent_relation_schema_id: str = eqx.field(static=True)
    counting_convention: str = eqx.field(static=True, default="canonical-half")
    directed_multiplicity: int = eqx.field(static=True, default=2)

    @property
    def capacity_failure(self) -> Array:
        return self.overflow

    @property
    def scientific_failure(self) -> Array:
        return self.reversal_open | self.identity_conflict

    @property
    def successful(self) -> Array:
        return ~(self.capacity_failure | self.scientific_failure)


class ParticleImageRelationEvidence(StrictModule, NonTrainableState):
    """Per-case requirements and complete failure evidence for one image search.

    ``required_*`` are the counts the geometry demands, independent of the
    capacity that truncated storage.  Capacity failures (cell occupancy,
    images, edges, degree, and the intermediate cluster ``tile_overflow``)
    are repairable by a declared larger capacity.  ``stencil_overflow`` (the
    lattice left the prepared deformation envelope of the candidate stencil),
    ``domain_violation``, ``nonfinite`` (non-finite active positions, a
    non-finite lattice, or a failed lattice solve, the last two independent
    of the active set), ``representation_overflow`` (an active wrap count or
    route image shift outside ``|n| <= 2**31 - 1``, or a required route/degree
    count beyond the int32 count range),
    ``cluster_extent_overflow`` (a cluster left the prepared bounding-radius
    envelope of a tile search) and ``receiver_violation`` (a requested
    receiver ID absent, inactive or repeated) are geometric/scientific
    failures and never justify a capacity retry.  ``required_tile_pairs`` is
    zero for searches without intermediate tiles.
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
    required_tile_pairs: Array
    tile_overflow: Array
    cluster_extent_overflow: Array
    receiver_violation: Array

    @property
    def capacity_failure(self) -> Array:
        return (
            self.cell_overflow
            | self.image_overflow
            | self.edge_overflow
            | self.degree_overflow
            | self.tile_overflow
        )

    @property
    def scientific_failure(self) -> Array:
        return (
            self.stencil_overflow
            | self.domain_violation
            | self.nonfinite
            | self.representation_overflow
            | self.cluster_extent_overflow
            | self.receiver_violation
        )

    @property
    def successful(self) -> Array:
        return ~(self.capacity_failure | self.scientific_failure)


__all__ = [
    "compact_image_routes",
    "ImageReversalMap",
    "ImageRouteCompaction",
    "ParticleImageHalfView",
    "ParticleImageRelation",
    "ParticleImageRelationEvidence",
]
