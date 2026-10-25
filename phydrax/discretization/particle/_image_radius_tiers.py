#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._image_neighborhood import ImageCertificate, ParticleImageNeighborhoodState
from ._image_relation import (
    compact_image_routes,
    ParticleImageRelation,
    representation_shifts,
)


_INT32_LIMIT = 2**31 - 1


class ImageRadiusTier(StrictModule, NonTrainableState):
    """One ordered physical cutoff with its own route and degree storage.

    ``maximum_case_edges`` selects per-case segments of that width (bounded
    in total by ``maximum_edges``); otherwise all cases share one buffer.
    """

    name: str = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    maximum_case_edges: int | None = eqx.field(static=True)
    tier_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        radius: float,
        /,
        *,
        maximum_edges: int,
        maximum_degree: int,
        maximum_case_edges: int | None = None,
    ) -> None:
        label = str(name)
        value = float(radius)
        edges = int(maximum_edges)
        degree = int(maximum_degree)
        if not label:
            raise ValueError("Tier name must be nonempty.")
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Tier radius must be finite and positive.")
        if edges <= 0 or edges > _INT32_LIMIT or degree <= 0 or degree > _INT32_LIMIT:
            raise ValueError("Tier edge and degree capacities must be positive int32.")
        case_edges = None if maximum_case_edges is None else int(maximum_case_edges)
        if case_edges is not None and (case_edges <= 0 or case_edges > edges):
            raise ValueError(
                "maximum_case_edges must be positive and no larger than maximum_edges."
            )
        self.name = label
        self.radius = value
        self.maximum_edges = edges
        self.maximum_degree = degree
        self.maximum_case_edges = case_edges
        self.tier_id = canonical_fingerprint(
            {
                "kind": "image-radius-tier",
                "name": label,
                "radius": value,
                "maximum_edges": edges,
                "maximum_degree": degree,
                "maximum_case_edges": case_edges,
            }
        )


class ImageRadiusTierPlan(StrictModule, NonTrainableState):
    """Ordered radius tiers sharing one certified outer image search.

    Tier ``t`` stores, as its fixed-topology candidate set, every outer route
    whose build-geometry length is below ``radius_t + skin``; its physical
    cutoff mask uses current geometry and ``radius_t``.  The outer search must
    reach ``outer_radius = max radius + skin``.  Each tier keeps independent
    counts, storage and status, so one tier overflowing never invalidates
    another.
    """

    tiers: tuple[ImageRadiusTier, ...]
    skin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, tiers: tuple[ImageRadiusTier, ...], /, *, skin: float) -> None:
        values = tuple(tiers)
        if not values:
            raise ValueError("A radius tier plan requires at least one tier.")
        for tier in values:
            if not isinstance(tier, ImageRadiusTier):
                raise TypeError("Tiers must be ImageRadiusTier instances.")
        radii = [tier.radius for tier in values]
        if any(later <= earlier for earlier, later in zip(radii[:-1], radii[1:])):
            raise ValueError("Tier radii must be strictly increasing.")
        if len({tier.name for tier in values}) != len(values):
            raise ValueError("Tier names must be unique.")
        margin = float(skin)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("skin must be finite and nonnegative.")
        self.tiers = values
        self.skin = margin
        self.plan_id = canonical_fingerprint(
            {
                "kind": "image-radius-tier-plan",
                "tiers": [tier.tier_id for tier in values],
                "skin": margin,
            }
        )

    @property
    def outer_radius(self) -> float:
        return self.tiers[-1].radius + self.skin

    def prepare(
        self, /, *, source_radius: float, case_count: int
    ) -> PreparedImageRadiusTiers:
        return PreparedImageRadiusTiers(
            self, source_radius=source_radius, case_count=case_count
        )


class ImageTierEvidence(StrictModule, NonTrainableState):
    """Per-case counts and status of one radius tier.

    ``edge_overflow``/``degree_overflow`` are the tier's own capacity
    failures; source failures are carried from the outer search.
    ``representation_overflow`` marks re-expressed shifts outside the int32
    image range.
    """

    required_edges: Array
    stored_edges: Array
    required_degree: Array
    edge_overflow: Array
    degree_overflow: Array
    source_capacity_failure: Array
    source_scientific_failure: Array
    representation_overflow: Array

    @property
    def capacity_failure(self) -> Array:
        return self.edge_overflow | self.degree_overflow | self.source_capacity_failure

    @property
    def scientific_failure(self) -> Array:
        return self.source_scientific_failure | self.representation_overflow

    @property
    def successful(self) -> Array:
        return ~(self.capacity_failure | self.scientific_failure)


class ImageRadiusTierView(StrictModule, NonTrainableState):
    """Fixed-topology candidate routes of one tier and its physical cutoff.

    ``relation`` holds the outer routes shorter than ``candidate_radius`` at
    build geometry, in outer (receiver-major) order; ``parent_routes`` maps
    them to outer routes.  ``cutoff_mask`` applies the physical ``radius`` to
    current geometry.
    """

    relation: ParticleImageRelation
    parent_routes: Array
    evidence: ImageTierEvidence
    name: str = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    candidate_radius: float = eqx.field(static=True)
    tier_id: str = eqx.field(static=True)

    def cutoff_mask(self, positions: ArrayLike, cell_vectors: ArrayLike, /) -> Array:
        """Routes strictly inside the physical radius under current geometry."""
        displacement = self.relation.displacement(positions, cell_vectors)
        squared = jnp.sum(displacement * displacement, axis=-1)
        return self.relation.valid & jax.lax.stop_gradient(
            squared < self.radius * self.radius
        )

    def with_representation_offsets(self, offsets: ArrayLike, /) -> ImageRadiusTierView:
        """Re-express shifts for positions ``x' = x - offsets @ H`` (see the relation)."""
        shifts, overflow = representation_shifts(self.relation, offsets)
        relation = self.relation
        failed = (
            jnp.zeros((relation.case_count,), dtype=jnp.int32)
            .at[relation.route_cases]
            .add(overflow.astype(jnp.int32))
        ) > 0
        return eqx.tree_at(
            lambda view: (
                view.relation.image_shifts,
                view.evidence.representation_overflow,
            ),
            self,
            (shifts, self.evidence.representation_overflow | failed),
        )


class ImageRadiusTierViews(StrictModule, NonTrainableState):
    """All tier views formed from one outer image relation epoch."""

    tiers: tuple[ImageRadiusTierView, ...]
    skin: float = eqx.field(static=True)
    source_relation_schema_id: str = eqx.field(static=True)

    def view(self, name: str, /) -> ImageRadiusTierView:
        for tier in self.tiers:
            if tier.name == name:
                return tier
        raise KeyError(f"No radius tier named {name!r}.")

    def with_representation_offsets(self, offsets: ArrayLike, /) -> ImageRadiusTierViews:
        return ImageRadiusTierViews(
            tuple(tier.with_representation_offsets(offsets) for tier in self.tiers),
            self.skin,
            self.source_relation_schema_id,
        )

    def complete(self, certificate: ImageCertificate, /) -> Array:
        """Per ``(tier, case)`` completeness under a lifecycle certificate.

        ``certificate`` must be evaluated at the largest tier radius against
        the outer epoch's build frame; it then bounds every tier's stored and
        absent routes.  A tier is complete when the certificate holds within
        the tier skin and the tier itself succeeded.
        """
        valid = certificate.valid(self.skin)
        return jnp.stack([valid & tier.evidence.successful for tier in self.tiers])


class PreparedImageRadiusTiers(StrictModule, NonTrainableState):
    """Radius tiers bound to an outer image search of ``source_radius``."""

    plan: ImageRadiusTierPlan
    source_radius: float = eqx.field(static=True)
    case_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: ImageRadiusTierPlan, /, *, source_radius: float, case_count: int
    ) -> None:
        if not isinstance(plan, ImageRadiusTierPlan):
            raise TypeError("plan must be an ImageRadiusTierPlan.")
        radius = float(source_radius)
        cases = int(case_count)
        if not np.isfinite(radius) or radius < plan.outer_radius:
            raise ValueError(
                "The outer search radius must cover the largest tier radius plus skin."
            )
        if cases <= 0:
            raise ValueError("case_count must be positive.")
        for tier in plan.tiers:
            if (
                tier.maximum_case_edges is not None
                and cases * tier.maximum_case_edges > tier.maximum_edges
            ):
                raise ValueError(
                    f"Tier {tier.name!r} per-case segments exceed maximum_edges."
                )
        self.plan = plan
        self.source_radius = radius
        self.case_count = cases
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-image-radius-tiers",
                "plan": plan.plan_id,
                "source_radius": radius,
                "case_count": cases,
            }
        )

    def views(
        self, neighborhood: ParticleImageNeighborhoodState, /
    ) -> ImageRadiusTierViews:
        """Form every tier's candidate set from one outer neighborhood epoch.

        Membership uses the epoch's build geometry (``reference_positions`` and
        ``cell_vectors``), so the tier topology is fixed for the epoch.
        """
        if not isinstance(neighborhood, ParticleImageNeighborhoodState):
            raise TypeError("neighborhood must be a ParticleImageNeighborhoodState.")
        relation = neighborhood.relation
        if relation.case_count != self.case_count:
            raise ValueError("Neighborhood case count differs from the prepared tiers.")
        if neighborhood.search_radius < self.source_radius:
            raise ValueError("Neighborhood search radius is below the prepared radius.")
        displacement = relation.displacement(
            neighborhood.reference_positions, neighborhood.cell_vectors
        )
        squared = jax.lax.stop_gradient(jnp.sum(displacement * displacement, axis=-1))
        cases = relation.case_count
        size = cases * relation.particle_capacity
        source = neighborhood.evidence
        views = []
        for tier in self.plan.tiers:
            candidate_radius = tier.radius + self.plan.skin
            mask = relation.valid & (squared < candidate_radius * candidate_radius)
            compaction = compact_image_routes(
                mask,
                relation.route_cases,
                case_count=cases,
                capacity=tier.maximum_edges if tier.maximum_case_edges is None else None,
                case_capacity=tier.maximum_case_edges,
            )
            degree = (
                jnp.zeros((size,), dtype=jnp.int32)
                .at[jnp.where(mask, relation.receiver_indices, 0)]
                .add(mask.astype(jnp.int32))
            )
            required_degree = jnp.max(degree.reshape((cases, -1)), axis=-1)
            tier_relation = relation.take_routes(
                compaction,
                relation_schema_id=canonical_fingerprint(
                    {
                        "kind": "image-radius-tier-relation",
                        "parent": relation.relation_schema_id,
                        "tier": tier.tier_id,
                        "tiers": self.prepared_id,
                    }
                ),
            )
            views.append(
                ImageRadiusTierView(
                    tier_relation,
                    jnp.where(tier_relation.valid, compaction.indices, 0),
                    ImageTierEvidence(
                        compaction.required,
                        compaction.stored,
                        required_degree,
                        compaction.overflow,
                        required_degree > tier.maximum_degree,
                        source.capacity_failure,
                        source.scientific_failure,
                        jnp.zeros((cases,), dtype=jnp.bool_),
                    ),
                    tier.name,
                    tier.radius,
                    candidate_radius,
                    tier.tier_id,
                )
            )
        return ImageRadiusTierViews(
            tuple(views), self.plan.skin, relation.relation_schema_id
        )


__all__ = [
    "ImageRadiusTier",
    "ImageRadiusTierPlan",
    "ImageRadiusTierView",
    "ImageRadiusTierViews",
    "ImageTierEvidence",
    "PreparedImageRadiusTiers",
]
