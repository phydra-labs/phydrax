#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike, DTypeLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation
from ...sparse._streamed import PreparedStreamedRelation, StreamedRelationPlan
from ...typing import checked
from .._core import DiscretizationCapability, DiscretizationKey, PreparationReport
from .._periodic_cell import lattice_right_inverse_with_status, PeriodicCell
from ._cell_list import CellListParticleNeighborhoodPlan
from ._core import ParticleDiscretization
from ._image_neighborhood import (
    AbstractParticleImageNeighborhoodPlan,
    AbstractPreparedParticleImageNeighborhood,
    image_certificate,
    ImageCertificate,
    ParticleImageCapacity,
    ParticleImageNeighborhoodState,
)
from ._image_relation import checked_int32_add, symmetric_int32
from ._metric_cell_list import MetricCellListParticleNeighborhoodPlan
from ._neighborhood import (
    AbstractParticleNeighborhoodPlan,
    AbstractPreparedParticleNeighborhood,
    ParticleNeighborhoodState,
)
from ._pairwise import ParticleBox
from ._precision import ParticleRealization


def _same_active_mask(active: Array, reference: Array, /) -> Array:
    if isinstance(active, Tracer) or isinstance(reference, Tracer):
        return jnp.all(active == reference)
    # Frozen inference masks are immutable preparation, not runtime work.
    with jax.ensure_compile_time_eval():
        return jnp.all(active == reference)


class _PairReuseTerms(NamedTuple):
    value: Array
    active: Array
    vectors: Array
    particle_maximum: Array
    cell_deformation: Array
    maximum: Array
    threshold: Array
    finite: Array
    certified: Array


class _ImageReuseTerms(NamedTuple):
    value: Array
    active: Array
    vectors: Array
    delta: Array
    current_counts: Array
    particle_maximum: Array
    cell_deformation: Array
    spent: Array
    margin: Array
    skin: Array
    finite: Array
    representable: Array
    certified: Array


class ParticleVerletState(StrictModule, NonTrainableState):
    neighborhood: ParticleNeighborhoodState
    reference_position: Array
    reference_active_mask: Array
    reference_cell_vectors: Array
    epoch: Array
    rebuilt: Array
    rebuild_count: Array
    maximum_reference_displacement: Array
    maximum_cell_deformation: Array
    certificate_margin: Array
    successful: Array
    prepared_verlet_id: str = eqx.field(static=True)


class VerletParticleNeighborhoodPlan(AbstractParticleNeighborhoodPlan):
    """Certificate-based relation cache composed with one authority neighborhood."""

    base: AbstractParticleNeighborhoodPlan
    interaction_radius: float = eqx.field(static=True)
    skin: float = eqx.field(static=True)
    box: ParticleBox | PeriodicCell | None
    backend: ParticleRealization = eqx.field(static=True)
    key: DiscretizationKey
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        base: AbstractParticleNeighborhoodPlan,
        interaction_radius: float,
        skin: float,
        /,
        *,
        name: str = "verlet-particle-neighborhood",
        plan_id: str | None = None,
    ) -> None:
        interaction = float(interaction_radius)
        skin_ = float(skin)
        if not np.isfinite(interaction) or interaction <= 0.0:
            raise ValueError("interaction_radius must be finite and positive.")
        if not np.isfinite(skin_) or skin_ <= 0.0:
            raise ValueError("skin must be finite and positive.")
        if isinstance(
            base,
            (CellListParticleNeighborhoodPlan, MetricCellListParticleNeighborhoodPlan),
        ) and base.search_radius < (interaction + skin_):
            raise ValueError(
                "Cell-list search radius must cover interaction_radius plus skin."
            )
        key = DiscretizationKey(
            name,
            base.key.role,
            domain_labels=base.key.domain_labels + ("verlet_cache",),
        )
        identifier = canonical_fingerprint(
            {
                "kind": "verlet-particle-neighborhood-plan",
                "base": base.plan_id,
                "interaction_radius": interaction,
                "skin": skin_,
                "key": key.key_id,
            }
        )
        self.base = base
        self.interaction_radius = interaction
        self.skin = skin_
        self.box = base.box
        self.backend = base.backend
        self.key = key
        self.plan_id = identifier if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedVerletParticleNeighborhood:
        return PreparedVerletParticleNeighborhood(self, particles)


class PreparedVerletParticleNeighborhood(AbstractPreparedParticleNeighborhood):
    plan: VerletParticleNeighborhoodPlan
    base: AbstractPreparedParticleNeighborhood
    key: DiscretizationKey
    box: ParticleBox | PeriodicCell | None
    backend: ParticleRealization = eqx.field(static=True)
    pair_capacity: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    default_active_mask: Array
    particle_discretization_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    artifact_kind: str = eqx.field(static=True)
    preparation: PreparationReport
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, plan: VerletParticleNeighborhoodPlan, particles: ParticleDiscretization, /
    ) -> None:
        base = plan.base.prepare(particles)
        preparation = PreparationReport(
            capabilities=tuple(
                set(base.preparation.capabilities)
                | {DiscretizationCapability.TOPOLOGY_REFRESH_FIXED_CAPACITY}
            ),
            diagnostics=base.preparation.diagnostics
            + (
                "candidate routes cached under a displacement certificate",
                "relation rebuild threshold is skin/2",
                "history remapping is required only after a rebuild",
            ),
            resource_counts={
                **dict(base.preparation.resource_counts),
                "reference_position_values": (
                    particles.capacity * particles.ambient_dimension
                ),
            },
        )
        self.plan = plan
        self.base = base
        self.key = plan.key
        self.box = plan.box
        self.backend = plan.backend
        self.pair_capacity = base.pair_capacity
        self.particle_capacity = particles.capacity
        self.default_active_mask = particles.active_mask
        self.particle_discretization_id = particles.prepared_id
        self.numeric_version = particles.numeric_version
        self.artifact_kind = "verlet-particle-neighborhood"
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-verlet-particle-neighborhood",
                "plan": plan.plan_id,
                "base": base.prepared_id,
                "particles": particles.prepared_id,
                "preparation": preparation.report_id,
            }
        )

    def build(
        self, positions: ArrayLike, /, *, active_mask: ArrayLike | None = None
    ) -> ParticleNeighborhoodState:
        """Build authority routes without cache reuse."""
        active = self._active(active_mask)
        return self.base.build(positions, active_mask=active)

    def _resolved_cell_vectors(
        self, dtype: DTypeLike, cell_vectors: ArrayLike | None, /
    ) -> Array:
        if cell_vectors is not None:
            value = jnp.asarray(cell_vectors, dtype=dtype)
        elif isinstance(self.box, PeriodicCell):
            value = self.box.vectors.astype(dtype)
        elif isinstance(self.box, ParticleBox):
            value = jnp.diag(self.box.lengths.astype(dtype))
        else:
            value = jnp.zeros((0, 0), dtype=dtype)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("Verlet cell vectors must be a square matrix.")
        return value

    def initialize(
        self,
        positions: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
    ) -> ParticleVerletState:
        value = self._positions(positions)
        active = self._active(active_mask)
        vectors = self._resolved_cell_vectors(value.dtype, cell_vectors)
        neighborhood = self.base.build(value, active_mask=active)
        successful = (
            neighborhood.successful
            & jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(vectors))
        )
        return ParticleVerletState(
            neighborhood,
            value,
            active,
            vectors,
            jnp.zeros((), dtype=jnp.int32),
            jnp.asarray(True),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.zeros((), dtype=value.dtype),
            jnp.zeros((), dtype=value.dtype),
            jnp.asarray(0.5 * self.plan.skin, dtype=value.dtype),
            successful,
            self.prepared_id,
        )

    def _reuse_terms(
        self,
        positions: ArrayLike,
        previous: ParticleVerletState,
        active_mask: ArrayLike | None,
        cell_vectors: ArrayLike | None,
        /,
    ) -> _PairReuseTerms:
        if previous.prepared_verlet_id != self.prepared_id:
            raise ValueError("Verlet state belongs to another prepared neighborhood.")
        value = self._positions(positions)
        active = self._active(active_mask)
        vectors = self._resolved_cell_vectors(value.dtype, cell_vectors)
        if vectors.shape != previous.reference_cell_vectors.shape:
            raise ValueError("Verlet cell shape changed from its prepared state.")
        displacement = value - previous.reference_position
        if self.box is not None:
            displacement = self.box.minimum_image(displacement)
        distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        particle_maximum = jnp.max(distance)
        cell_delta = vectors - previous.reference_cell_vectors
        cell_deformation = jnp.sqrt(jnp.sum(cell_delta * cell_delta))
        maximum = particle_maximum + cell_deformation
        threshold = jnp.asarray(0.5 * self.plan.skin, dtype=value.dtype)
        finite = (
            jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(vectors))
            & jnp.isfinite(maximum)
        )
        certified = (
            previous.successful
            & finite
            & (maximum <= threshold)
            & _same_active_mask(active, previous.reference_active_mask)
        )
        return _PairReuseTerms(
            value,
            active,
            vectors,
            particle_maximum,
            cell_deformation,
            maximum,
            threshold,
            finite,
            certified,
        )

    @checked
    def certifies(
        self,
        positions: ArrayLike,
        previous: ParticleVerletState,
        /,
        *,
        active_mask: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
    ) -> Array:
        """Return whether ``previous``'s cached pair epoch is exact here.

        The predicate ``update`` uses: previous success, finite inputs, an
        unchanged active set and particle plus cell motion within ``skin/2``.
        """
        return self._reuse_terms(positions, previous, active_mask, cell_vectors).certified

    @checked
    def update(
        self,
        positions: ArrayLike,
        previous: ParticleVerletState,
        /,
        *,
        active_mask: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
    ) -> ParticleVerletState:
        terms = self._reuse_terms(positions, previous, active_mask, cell_vectors)
        value = terms.value
        active = terms.active
        vectors = terms.vectors
        particle_maximum = terms.particle_maximum
        cell_deformation = terms.cell_deformation
        maximum = terms.maximum
        threshold = terms.threshold
        finite = terms.finite
        rebuild = ~terms.certified

        def rebuild_routes(_: None) -> ParticleVerletState:
            neighborhood = self.base.build(value, active_mask=active)
            successful = neighborhood.successful & finite
            return ParticleVerletState(
                neighborhood,
                value,
                active,
                vectors,
                previous.epoch + jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(True),
                previous.rebuild_count + jnp.asarray(1, dtype=jnp.int32),
                particle_maximum,
                cell_deformation,
                threshold,
                successful,
                self.prepared_id,
            )

        def reuse_routes(_: None) -> ParticleVerletState:
            return ParticleVerletState(
                previous.neighborhood,
                previous.reference_position,
                previous.reference_active_mask,
                previous.reference_cell_vectors,
                previous.epoch,
                jnp.asarray(False),
                previous.rebuild_count,
                particle_maximum,
                cell_deformation,
                threshold - maximum,
                previous.successful & finite,
                self.prepared_id,
            )

        return jax.lax.cond(rebuild, rebuild_routes, reuse_routes, operand=None)

    def _active(self, active_mask: ArrayLike | None, /) -> Array:
        if active_mask is None:
            return self.default_active_mask
        value = jnp.asarray(active_mask, dtype=jnp.bool_)
        if value.shape != (self.particle_capacity,):
            raise ValueError("active_mask must have particle-capacity shape.")
        return self.default_active_mask & value

    def _positions(self, positions: ArrayLike, /) -> Array:
        value = jnp.asarray(positions)
        expected = (
            (self.particle_capacity, self.base.plan.box.ambient_dimension)
            if self.base.plan.box is not None
            else None
        )
        if value.ndim != 2 or value.shape[0] != self.particle_capacity:
            raise ValueError(
                "Verlet positions must have shape (particle_capacity, dimension)."
            )
        if expected is not None and value.shape != expected:
            raise ValueError(f"Verlet positions must have shape {expected}.")
        return value


@eqx.filter_jit
def _epoch_schedule(
    plan: StreamedRelationPlan,
    relation: EdgeRelation,
    owner_id: str,
    epoch: Array,
    receiver_valid: Array,
) -> PreparedStreamedRelation:
    """Prepare one epoch's receiver-major schedule with a fixed traced layout.

    Compiled preparation gives eager and in-loop schedules one PyTree layout,
    so reused and rebuilt epochs are interchangeable under ``lax.cond``.  The
    image relation is already canonically ordered, so stable route IDs are the
    route positions.
    """
    return plan.prepare(
        relation,
        owner_id=owner_id,
        epoch=epoch,
        stable_route_ids=jnp.arange(relation.capacity, dtype=jnp.int32),
        receiver_valid=receiver_valid,
    )


class ParticleImageVerletState(StrictModule, NonTrainableState):
    """Cached image relation with an image-aware rebuild certificate.

    ``reference`` holds routes expressed for ``reference_position``.  Positions
    later rewrapped by whole lattice translations keep the same physical routes
    through ``representation_offsets``; ``neighborhood`` re-expresses them for
    the current positions without a rebuild or sort.
    """

    reference: ParticleImageNeighborhoodState
    reference_position: Array
    reference_active_mask: Array
    reference_cell_vectors: Array
    reference_image_counts: Array
    representation_offsets: Array
    epoch: Array
    rebuilt: Array
    rebuild_count: Array
    maximum_reference_displacement: Array
    maximum_cell_deformation: Array
    stencil_margin: Array
    certificate_margin: Array
    successful: Array
    schedule: PreparedStreamedRelation | None
    prepared_verlet_id: str = eqx.field(static=True)

    @property
    def neighborhood(self) -> ParticleImageNeighborhoodState:
        return self.reference.with_representation_offsets(self.representation_offsets)

    @property
    def capacity_failure(self) -> Array:
        return self.reference.capacity_failure

    @property
    def scientific_failure(self) -> Array:
        return self.reference.scientific_failure | (
            ~self.successful & ~self.reference.capacity_failure
        )


class ImageVerletParticleNeighborhoodPlan(StrictModule, NonTrainableState):
    """Skin cache over an image-aware search, valid beyond the unique-image radius.

    ``streamed`` optionally prepares the receiver-major streamed schedule of
    the cached directed relation once per rebuild epoch; reused epochs keep it
    because shifts change only by exact image-count offsets, never membership.
    """

    base: AbstractParticleImageNeighborhoodPlan
    streamed: StreamedRelationPlan | None
    interaction_radius: float = eqx.field(static=True)
    skin: float = eqx.field(static=True)
    key: DiscretizationKey
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        base: AbstractParticleImageNeighborhoodPlan,
        interaction_radius: float,
        skin: float,
        /,
        *,
        name: str = "image-verlet-particle-neighborhood",
        streamed: StreamedRelationPlan | None = None,
        plan_id: str | None = None,
    ) -> None:
        interaction = float(interaction_radius)
        skin_ = float(skin)
        if not np.isfinite(interaction) or interaction <= 0.0:
            raise ValueError("interaction_radius must be finite and positive.")
        if not np.isfinite(skin_) or skin_ <= 0.0:
            raise ValueError("skin must be finite and positive.")
        if base.search_radius < interaction + skin_:
            raise ValueError(
                "Image search radius must cover interaction_radius plus skin."
            )
        key = DiscretizationKey(
            name,
            base.key.role,
            domain_labels=base.key.domain_labels + ("verlet_cache",),
        )
        identifier = canonical_fingerprint(
            {
                "kind": "image-verlet-particle-neighborhood-plan",
                "base": base.plan_id,
                "interaction_radius": interaction,
                "skin": skin_,
                "streamed": None if streamed is None else streamed.plan_id,
                "key": key.key_id,
            }
        )
        self.base = base
        self.streamed = streamed
        self.interaction_radius = interaction
        self.skin = skin_
        self.key = key
        self.plan_id = identifier if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    @property
    def cell(self) -> PeriodicCell:
        return self.base.cell

    def with_capacity(
        self, capacity: ParticleImageCapacity, /
    ) -> ImageVerletParticleNeighborhoodPlan:
        return ImageVerletParticleNeighborhoodPlan(
            self.base.with_capacity(capacity),
            self.interaction_radius,
            self.skin,
            name=self.key.name,
            streamed=self.streamed,
        )

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedImageVerletParticleNeighborhood:
        return PreparedImageVerletParticleNeighborhood(self, particles)


class PreparedImageVerletParticleNeighborhood(StrictModule, NonTrainableState):
    """Image-aware Verlet lifecycle with a certificate for stored and absent images.

    For row lattice ``H`` the cached search used radius ``R_s >= R + skin`` in
    its wrapped build frame with complete integer extents ``K``.  A cached
    relation stays exact while (a) every image within ``R`` under the current
    cell still has ``|n_i| <= K_i`` (fractional spread plus ``R * reach``) and
    (b) ``2 max |dx| + sum_i K_i |dH_i| <= skin`` bounds the motion of every
    route in that stencil, stored or absent.  Wrapped coordinates are handled
    by exact image-count differences, not by keeping stale shifts.
    """

    plan: ImageVerletParticleNeighborhoodPlan
    base: AbstractPreparedParticleImageNeighborhood
    key: DiscretizationKey
    preparation: PreparationReport
    particle_capacity: int = eqx.field(static=True)
    particle_discretization_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    artifact_kind: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: ImageVerletParticleNeighborhoodPlan,
        particles: ParticleDiscretization,
        /,
    ) -> None:
        base = plan.base.prepare(particles)
        preparation = PreparationReport(
            capabilities=tuple(
                set(base.preparation.capabilities)
                | {DiscretizationCapability.TOPOLOGY_REFRESH_FIXED_CAPACITY}
            ),
            diagnostics=base.preparation.diagnostics
            + (
                "image routes cached under a stored-and-absent image certificate",
                "cell deformation is charged per complete stencil coefficient",
                "rewrapped positions update shifts by exact image-count differences",
            ),
            resource_counts={
                **dict(base.preparation.resource_counts),
                "reference_position_values": (
                    particles.capacity * particles.ambient_dimension
                ),
            },
        )
        self.plan = plan
        self.base = base
        self.key = plan.key
        self.preparation = preparation
        self.particle_capacity = particles.capacity
        self.particle_discretization_id = particles.prepared_id
        self.numeric_version = particles.numeric_version
        self.artifact_kind = "image-verlet-particle-neighborhood"
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-image-verlet-particle-neighborhood",
                "plan": plan.plan_id,
                "base": base.prepared_id,
                "particles": particles.prepared_id,
                "preparation": preparation.report_id,
            }
        )

    @property
    def cell(self) -> PeriodicCell:
        return self.base.cell

    @property
    def box(self) -> PeriodicCell:
        return self.base.cell

    @property
    def capacity(self) -> ParticleImageCapacity:
        return self.base.capacity

    @property
    def resource_evidence_id(self) -> str:
        return self.preparation.report_id

    def _positions(self, positions: ArrayLike, /) -> Array:
        value = jnp.asarray(positions)
        expected = (self.particle_capacity, self.cell.ambient_dimension)
        if value.shape != expected:
            raise ValueError(f"Verlet positions must have shape {expected}.")
        return value

    def _image_counts(
        self, image_counts: ArrayLike | None, /
    ) -> tuple[Array, Array] | None:
        """Return int32 counts and per-particle representability, range-checked
        before the cast so wide counts refuse instead of wrapping."""
        if image_counts is None:
            return None
        counts = jnp.asarray(image_counts)
        expected = (self.particle_capacity, self.cell.rank)
        if counts.shape != expected or not jnp.issubdtype(counts.dtype, jnp.integer):
            raise ValueError(f"image_counts must be integers with shape {expected}.")
        periodic = self.cell.periodic_mask
        counts, representable = symmetric_int32(counts)
        return (
            jnp.where(periodic, counts, 0),
            jnp.all(representable | ~periodic, axis=-1),
        )

    def _certificate(
        self,
        reference: ParticleImageNeighborhoodState,
        reference_position: Array,
        reference_vectors: Array,
        continuous: Array,
        vectors: Array,
        active: Array,
        /,
    ) -> ImageCertificate:
        return image_certificate(
            reference_position[None],
            continuous[None],
            reference_vectors[None],
            vectors[None],
            self.cell.origin.astype(vectors.dtype)[None],
            reference.wrap_counts[None],
            reference.stencil_extents,
            active[None],
            self.cell.periodic_mask[None],
            self.plan.interaction_radius,
        )

    def _fresh_state(
        self,
        value: Array,
        active: Array,
        vectors: Array,
        counts: Array,
        epoch: Array,
        rebuild_count: Array,
        admissible: Array,
        /,
    ) -> ParticleImageVerletState:
        reference = self.base.build(value, active_mask=active, cell_vectors=vectors)
        margin = self._certificate(
            reference, value, vectors, value, vectors, active
        ).coverage_margin[0]
        zero = jnp.zeros((), dtype=value.dtype)
        return ParticleImageVerletState(
            reference,
            value,
            active,
            vectors,
            counts,
            jnp.zeros_like(counts),
            epoch,
            jnp.asarray(True),
            rebuild_count,
            zero,
            zero,
            margin,
            jnp.asarray(self.plan.skin, dtype=value.dtype),
            reference.successful & admissible & (margin > 0.0),
            None
            if self.plan.streamed is None
            else _epoch_schedule(
                self.plan.streamed,
                reference.relation.relation,
                reference.relation_schema_id,
                epoch,
                active,
            ),
            self.prepared_id,
        )

    def initialize(
        self,
        positions: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
        image_counts: ArrayLike | None = None,
    ) -> ParticleImageVerletState:
        """Build the first image epoch.

        ``image_counts`` (``positions + image_counts @ H`` = unwrapped) makes
        later rewrapping exact; without it, rewrapping is inferred under
        half-cell continuity between updates.  Active counts outside
        ``|n| <= 2**31 - 1`` make the state unsuccessful.
        """
        value = self._positions(positions)
        active = self.base.resolved_active_mask(active_mask)
        vectors = self.base.resolved_cell_vectors(value.dtype, cell_vectors)
        resolved = self._image_counts(image_counts)
        if resolved is None:
            counts = jnp.zeros((self.particle_capacity, self.cell.rank), dtype=jnp.int32)
            representable = jnp.asarray(True)
        else:
            counts, rows = resolved
            representable = jnp.all(rows | ~active)
        finite = jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(vectors))
        return self._fresh_state(
            value,
            active,
            vectors,
            counts,
            jnp.zeros((), dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            finite & representable,
        )

    def _reuse_terms(
        self,
        positions: ArrayLike,
        previous: ParticleImageVerletState,
        active_mask: ArrayLike | None,
        cell_vectors: ArrayLike | None,
        image_counts: ArrayLike | None,
        /,
    ) -> _ImageReuseTerms:
        if previous.prepared_verlet_id != self.prepared_id:
            raise ValueError("Verlet state belongs to another prepared neighborhood.")
        value = self._positions(positions)
        active = self.base.resolved_active_mask(active_mask)
        vectors = self.base.resolved_cell_vectors(value.dtype, cell_vectors)
        resolved = self._image_counts(image_counts)
        periodic = self.cell.periodic_mask
        reference_counts = previous.reference_image_counts
        if resolved is None:
            raw_inverse, solved = lattice_right_inverse_with_status(vectors)
            inverse = jnp.where(solved, raw_inverse, 0.0)
            moved = contract(
                "nd,dr->nr",
                value - previous.reference_position,
                inverse,
                backend="jax",
            )
            safe = jnp.where(jnp.isfinite(moved), moved, 0.0)
            steps = jax.lax.stop_gradient(jnp.round(safe))
            rows = jnp.all(~periodic | (jnp.abs(steps) < 2.0**31), axis=-1)
            delta = jnp.where(
                periodic & rows[:, None] & active[:, None], -steps, 0.0
            ).astype(jnp.int32)
            current_counts, count_overflow = checked_int32_add(reference_counts, delta)
            current_counts = jnp.where(count_overflow, reference_counts, current_counts)
            rows = rows & ~jnp.any(count_overflow, axis=-1)
            delta = jnp.where(count_overflow, 0, delta)
            delta_rows = rows
        else:
            current_counts, rows = resolved
            delta, delta_overflow = checked_int32_add(current_counts, -reference_counts)
            delta = jnp.where(delta_overflow | ~active[:, None], 0, delta)
            delta_rows = rows & ~jnp.any(delta_overflow, axis=-1)
        # Current counts must be representable for a rebuild; reuse further
        # needs every count difference and re-expressed route shift in range.
        representable = jnp.all(rows | ~active)
        reexpressed = previous.reference.with_representation_offsets(delta)
        reusable = jnp.all(delta_rows | ~active) & ~jnp.any(
            reexpressed.evidence.representation_overflow
        )
        continuous = value + contract(
            "nr,rd->nd", delta.astype(value.dtype), vectors, backend="jax"
        )
        certificate = self._certificate(
            previous.reference,
            previous.reference_position,
            previous.reference_cell_vectors,
            continuous,
            vectors,
            active,
        )
        particle_maximum = certificate.maximum_displacement[0]
        cell_deformation = certificate.cell_deformation[0]
        spent = 2.0 * particle_maximum + cell_deformation
        margin = certificate.coverage_margin[0]
        finite = (
            jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(vectors))
            & jnp.isfinite(spent)
        )
        skin = jnp.asarray(self.plan.skin, dtype=value.dtype)
        same_active = _same_active_mask(active, previous.reference_active_mask)
        certified = (
            previous.successful
            & finite
            & representable
            & reusable
            & (spent <= skin)
            & (margin > 0.0)
            & same_active
        )
        return _ImageReuseTerms(
            value,
            active,
            vectors,
            delta,
            current_counts,
            particle_maximum,
            cell_deformation,
            spent,
            margin,
            skin,
            finite,
            representable,
            certified,
        )

    @checked
    def certifies(
        self,
        positions: ArrayLike,
        previous: ParticleImageVerletState,
        /,
        *,
        active_mask: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
        image_counts: ArrayLike | None = None,
    ) -> Array:
        """Return whether ``previous``'s cached image epoch is exact here.

        True only when the stored-and-absent image certificate holds: the
        previous epoch succeeded, inputs are finite, the active set is
        unchanged, ``2 max|dx| + sum_i K_i |dH_i| <= skin`` in the build frame
        (rewrapping applied through exact or inferred image counts), and every
        image within the interaction radius stays inside the enumerated
        stencil under ``cell_vectors`` (a failed lattice solve fails it).  This
        is the exact predicate ``update`` uses to decide reuse versus rebuild,
        so fixed-topology consumers can refuse a stale epoch without rebuilding.
        """
        return self._reuse_terms(
            positions, previous, active_mask, cell_vectors, image_counts
        ).certified

    @checked
    def update(
        self,
        positions: ArrayLike,
        previous: ParticleImageVerletState,
        /,
        *,
        active_mask: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
        image_counts: ArrayLike | None = None,
    ) -> ParticleImageVerletState:
        """Reuse the cached image epoch while ``certifies`` holds, else rebuild."""
        terms = self._reuse_terms(
            positions, previous, active_mask, cell_vectors, image_counts
        )
        value = terms.value
        active = terms.active
        vectors = terms.vectors
        delta = terms.delta
        current_counts = terms.current_counts
        particle_maximum = terms.particle_maximum
        cell_deformation = terms.cell_deformation
        spent = terms.spent
        margin = terms.margin
        skin = terms.skin
        finite = terms.finite
        representable = terms.representable
        rebuild = ~terms.certified

        def rebuild_routes(_: None) -> ParticleImageVerletState:
            fresh = self._fresh_state(
                value,
                active,
                vectors,
                current_counts,
                previous.epoch + jnp.asarray(1, dtype=jnp.int32),
                previous.rebuild_count + jnp.asarray(1, dtype=jnp.int32),
                finite & representable,
            )
            return eqx.tree_at(
                lambda state: (
                    state.maximum_reference_displacement,
                    state.maximum_cell_deformation,
                ),
                fresh,
                (particle_maximum, cell_deformation),
            )

        def reuse_routes(_: None) -> ParticleImageVerletState:
            return ParticleImageVerletState(
                previous.reference,
                previous.reference_position,
                previous.reference_active_mask,
                previous.reference_cell_vectors,
                previous.reference_image_counts,
                delta,
                previous.epoch,
                jnp.asarray(False),
                previous.rebuild_count,
                particle_maximum,
                cell_deformation,
                margin,
                skin - spent,
                previous.successful & finite,
                previous.schedule,
                self.prepared_id,
            )

        return jax.lax.cond(rebuild, rebuild_routes, reuse_routes, operand=None)


__all__ = [
    "ImageVerletParticleNeighborhoodPlan",
    "ParticleImageVerletState",
    "ParticleVerletState",
    "PreparedImageVerletParticleNeighborhood",
    "PreparedVerletParticleNeighborhood",
    "VerletParticleNeighborhoodPlan",
]
