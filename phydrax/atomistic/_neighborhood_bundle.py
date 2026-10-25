#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical atomistic neighborhood requirement, preparation and lifecycle owner.

Every scientific consumer of atomistic geometry declares one purely scientific
``AtomisticNeighborhoodRequirement`` (owning binding, role, cutoff, selection,
receivers, exclusions) whose ``requirement_id`` is its identity.  An
``AtomisticNeighborhoodBundlePlan`` binds each requirement to exactly one
explicitly declared source and, for bounded requirements, one role capacity
ladder; sharing a source is certified at preparation from entity, coverage,
counting and receiver-selection compatibility, never from equal cutoffs or
class names.

A bounded (finite-cutoff) source is a native ``SegmentedImageVerletPlan`` over
a native image route search: routes carry explicit integer translations,
certificates cover stored and absent images under displacement and cell
deformation, and per-case selective rebuild, exact rewrapping and per-case
commit are native.  A complete (``cutoff=None``) pair source enumerates every
unordered pair of a finite system through the native dense pair owner under an
explicit pair bound; unbounded periodic support is refused.  Role views are
pure functions of the accepted source epochs and the certified geometry:

* classical minimum-image and interaction-site pairs are canonical halves with
  runtime unique-image evidence (a pair with two images, or a self image,
  inside the cutoff is a scientific failure, never silently a full image sum);
* explicitly counted image-pair sums are canonical halves of every image route
  with ``directed_multiplicity == 2`` and reversal/self-image evidence;
* directed image graphs and full directed coordination keep every image route;
  a model-declared ``ImageNeighborSelectionPolicy`` is scientific selection and
  is reported separately from storage overflow;
* complete pairs keep every active unordered pair without a cutoff.

Every role carries a scalar topology revision and, for directed roles with a
declared ``StreamedRelationPlan``, its own prepared streamed schedule; both
advance only when that role's topology changes.  Reciprocal-space artifacts
stay term-owned.  Candidate refresh is pure; capacity growth is a host
lifecycle transaction that replays the same attempt from the accepted state
with stable epochs and route identities.

Prepared bundles and their states have content-free treedefs: engines index
entity support schemas, statics are layout (``layout_id``), and the exact
material digests are dynamic identity words (``prepared_identity``,
``bundle_identity``) compared in compiled code, so every material of one
layout shares one executable.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from enum import StrEnum
from typing import final, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    AbstractImageRouteSearch,
    CellListImageRouteSearch,
    ClusterTileImageRouteSearch,
    DenseImageRouteSearch,
    DenseParticleNeighborhoodPlan,
    ImageNeighborSelectionPolicy,
    ImageRadiusTier,
    ImageRadiusTierPlan,
    ImageReceiverRows,
    ImageReceiverSubset,
    ImageSelectionCapacity,
    ImageSelectionEvidence,
    ImageTilePolicy,
    ParticleDiscretization,
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    ParticleImageRelation,
    ParticlePairRelation,
    PeriodicCell,
    PreparedDenseParticleNeighborhood,
    PreparedImageNeighborSelection,
    PreparedImageRadiusTiers,
    SegmentedImageVerletCertificate,
    SegmentedImageVerletPlan,
    SegmentedImageVerletState,
    TiledImageRouteSearch,
)
from ..discretization.particle._image_neighborhood import RadiusBoundary, SelectionMeasure
from ..discretization.particle._image_relation import (
    checked_int32_add,
    representation_shifts,
)
from ..sparse import EdgeRelation, PreparedStreamedRelation, StreamedRelationPlan
from ..typing import checked, parse
from ._sites import AtomisticInteractionSiteState, AtomisticSiteDomain
from ._system import PreparedAtomisticSystem


_INT32_LIMIT = 2**31 - 1


class _ImageSearchInputs(TypedDict):
    active: Array
    stable_ids: Array
    vectors: Array
    origin: Array
    periodic: Array
    image_counts: Array | None
    receivers: ImageReceiverSubset | None


class AtomisticNeighborhoodRole(StrEnum):
    """Closed scientific role of one atomistic neighborhood requirement."""

    CLASSICAL_MINIMUM_IMAGE_PAIRS = "classical-minimum-image-pairs"
    DIRECTED_IMAGE_GRAPH = "directed-image-graph"
    IMAGE_PAIR_SUM = "image-pair-sum"
    COORDINATION = "coordination"
    INTERACTION_SITE_PAIRS = "interaction-site-pairs"


class AtomisticPairCounting(StrEnum):
    """Route counting convention of a role view.

    ``DIRECTED`` keeps every directed image route (both orientations, every
    repeated and nonzero self image).  ``IMAGE_HALF`` keeps one route of every
    reversal pair; each represents two directed routes.  ``MINIMUM_IMAGE_HALF``
    is an image half additionally certified to hold at most one image per
    unordered pair and no self image inside the cutoff.  ``COMPLETE_PAIRS``
    is every unordered pair of active entities of a finite (nonperiodic)
    system without any cutoff; each represents two directed routes.
    """

    DIRECTED = "directed"
    IMAGE_HALF = "image-half"
    MINIMUM_IMAGE_HALF = "minimum-image-half"
    COMPLETE_PAIRS = "complete-pairs"


class AtomisticPairExclusions(StrEnum):
    """Pair exclusion convention applied by a classical pair view."""

    NONE = "none"
    TOPOLOGY_EXCEPTIONS = "topology-exceptions"


_ROLE_ENTITY = {
    AtomisticNeighborhoodRole.CLASSICAL_MINIMUM_IMAGE_PAIRS: (
        AtomisticSiteDomain.PHYSICAL_ATOMS
    ),
    AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH: AtomisticSiteDomain.PHYSICAL_ATOMS,
    AtomisticNeighborhoodRole.IMAGE_PAIR_SUM: AtomisticSiteDomain.PHYSICAL_ATOMS,
    AtomisticNeighborhoodRole.COORDINATION: AtomisticSiteDomain.PHYSICAL_ATOMS,
    AtomisticNeighborhoodRole.INTERACTION_SITE_PAIRS: (
        AtomisticSiteDomain.INTERACTION_SITES
    ),
}
_ROLE_COUNTING = {
    AtomisticNeighborhoodRole.CLASSICAL_MINIMUM_IMAGE_PAIRS: (
        AtomisticPairCounting.MINIMUM_IMAGE_HALF
    ),
    AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH: AtomisticPairCounting.DIRECTED,
    AtomisticNeighborhoodRole.IMAGE_PAIR_SUM: AtomisticPairCounting.IMAGE_HALF,
    AtomisticNeighborhoodRole.COORDINATION: AtomisticPairCounting.DIRECTED,
    AtomisticNeighborhoodRole.INTERACTION_SITE_PAIRS: (
        AtomisticPairCounting.MINIMUM_IMAGE_HALF
    ),
}
_SOURCE_ENTITIES = (
    AtomisticSiteDomain.PHYSICAL_ATOMS,
    AtomisticSiteDomain.INTERACTION_SITES,
)


def _nonempty(value: str, name: str, /) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{name} must be nonempty.")
    return text


def _positive_finite(value: float, name: str, /) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _nonnegative_finite(value: float, name: str, /) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return number


@final
class AtomisticRoleCapacity(StrictModule, NonTrainableState):
    """Fixed storage of one role view, distinct from its scientific definition.

    ``maximum_routes`` is one shared route buffer or, with
    ``maximum_case_routes``, the total bound of fixed per-case segments.
    ``maximum_degree`` bounds the directed candidate degree per receiver.
    """

    maximum_routes: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    maximum_case_routes: int | None = eqx.field(static=True)
    capacity_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_routes: int,
        maximum_degree: int,
        maximum_case_routes: int | None = None,
    ) -> None:
        routes = int(maximum_routes)
        degree = int(maximum_degree)
        if routes <= 0 or routes > _INT32_LIMIT or degree <= 0 or degree > _INT32_LIMIT:
            raise ValueError("Role route and degree capacities must be positive int32.")
        case_routes = None if maximum_case_routes is None else int(maximum_case_routes)
        if case_routes is not None and not 0 < case_routes <= routes:
            raise ValueError(
                "maximum_case_routes must be positive and no larger than maximum_routes."
            )
        self.maximum_routes = routes
        self.maximum_degree = degree
        self.maximum_case_routes = case_routes
        self.capacity_id = canonical_fingerprint(
            {
                "kind": "atomistic-role-capacity",
                "maximum_routes": routes,
                "maximum_degree": degree,
                "maximum_case_routes": case_routes,
            }
        )

    def dominates(self, other: AtomisticRoleCapacity, /) -> bool:
        if (self.maximum_case_routes is None) != (other.maximum_case_routes is None):
            return False
        return (
            self.maximum_routes >= other.maximum_routes
            and self.maximum_degree >= other.maximum_degree
            and (
                self.maximum_case_routes is None
                or other.maximum_case_routes is None
                or self.maximum_case_routes >= other.maximum_case_routes
            )
        )

    def covers(self, required_routes: ArrayLike, required_degree: ArrayLike, /) -> bool:
        """Host decision whether per-case requirements fit this storage."""
        routes = np.asarray(required_routes)
        degree = np.asarray(required_degree)
        fits = (
            int(np.sum(routes)) <= self.maximum_routes
            and int(np.max(degree, initial=0)) <= self.maximum_degree
        )
        if self.maximum_case_routes is not None:
            fits = fits and int(np.max(routes, initial=0)) <= self.maximum_case_routes
        return fits


_COMPLETE_ROLES = (
    AtomisticNeighborhoodRole.CLASSICAL_MINIMUM_IMAGE_PAIRS,
    AtomisticNeighborhoodRole.INTERACTION_SITE_PAIRS,
)


@final
class AtomisticNeighborhoodRequirement(StrictModule, NonTrainableState):
    """One purely scientific neighborhood request of an owning binding.

    ``owner`` is the explicit owning binding name (the program term binding or
    another consumer such as a CV or crystal-contact owner); it is part of the
    identity, so distinct consumers with otherwise equal requests keep
    distinct views.  ``cutoff`` is the physical radius under current geometry;
    ``cutoff=None`` declares the complete finite all-pairs relation of a
    physical or interaction-site pair role (for example direct Coulomb or
    all-site terms), admitted only on systems without periodic support.
    ``selection`` is a scientific nearest-neighbor rule of a directed graph,
    ``receiver_group`` a declared receiver-row subset of a directed role and
    ``exclusions`` the physical or interaction-site pair exception convention.
    Storage is not
    part of a requirement: role capacity ladders are bound by the bundle plan,
    so ``requirement_id`` is stable across capacity growth.  Several
    requirements of one owner and role coexist when any scientific field
    differs.
    """

    owner: str = eqx.field(static=True)
    role: AtomisticNeighborhoodRole = eqx.field(static=True)
    cutoff: float | None = eqx.field(static=True)
    selection: ImageNeighborSelectionPolicy | None
    receiver_group: str | None = eqx.field(static=True)
    exclusions: AtomisticPairExclusions = eqx.field(static=True)
    requirement_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        owner: str,
        role: AtomisticNeighborhoodRole,
        cutoff: float | None,
        /,
        *,
        selection: ImageNeighborSelectionPolicy | None = None,
        receiver_group: str | None = None,
        exclusions: AtomisticPairExclusions = AtomisticPairExclusions.NONE,
    ) -> None:
        name = _nonempty(owner, "owner")
        kind = parse(role, AtomisticNeighborhoodRole, "role")
        radius = None if cutoff is None else _positive_finite(cutoff, "cutoff")
        exclusion = parse(exclusions, AtomisticPairExclusions, "exclusions")
        if radius is None and kind not in _COMPLETE_ROLES:
            raise ValueError(
                "Only physical or interaction-site pair roles declare the complete "
                "finite all-pairs relation (cutoff=None)."
            )
        counting = _ROLE_COUNTING[kind]
        if selection is not None:
            if kind is not AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH:
                raise ValueError(
                    "Nearest-neighbor selection belongs only to a directed image graph."
                )
            if radius is not None and selection.radius > radius:
                raise ValueError("The selection radius exceeds the graph cutoff.")
        group = None if receiver_group is None else _nonempty(receiver_group, "group")
        if group is not None and counting is not AtomisticPairCounting.DIRECTED:
            raise ValueError(
                "Receiver subsets break reversal closure; only directed roles declare them."
            )
        if exclusion is AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS and kind not in (
            AtomisticNeighborhoodRole.CLASSICAL_MINIMUM_IMAGE_PAIRS,
            AtomisticNeighborhoodRole.INTERACTION_SITE_PAIRS,
        ):
            raise ValueError(
                "Topology exception scales apply only to classical physical or "
                "interaction-site pairs."
            )
        self.owner = name
        self.role = kind
        self.cutoff = radius
        self.selection = selection
        self.receiver_group = group
        self.exclusions = exclusion
        self.requirement_id = canonical_fingerprint(
            {
                "kind": "atomistic-neighborhood-requirement",
                "owner": name,
                "role": kind.value,
                "cutoff": radius,
                "selection": None if selection is None else selection.policy_id,
                "receiver_group": group,
                "exclusions": exclusion.value,
            }
        )

    @property
    def label(self) -> str:
        """Readable ``owner:role`` label for diagnostics, never an identity."""
        return f"{self.owner}:{self.role.value}"

    @property
    def complete(self) -> bool:
        """Whether this is the complete finite all-pairs relation."""
        return self.cutoff is None

    @property
    def entity(self) -> AtomisticSiteDomain:
        return _ROLE_ENTITY[self.role]

    @property
    def counting(self) -> AtomisticPairCounting:
        if self.cutoff is None:
            return AtomisticPairCounting.COMPLETE_PAIRS
        return _ROLE_COUNTING[self.role]

    def bounded_cutoff(self) -> float:
        """The finite cutoff of a bounded requirement."""
        if self.cutoff is None:
            raise ValueError(f"{self.label} is a complete all-pairs requirement.")
        return self.cutoff


@final
class AtomisticRoleCapacityLadder(StrictModule, NonTrainableState):
    """Declared, strictly growing role-view storage bound to one requirement.

    Only the first entry is used until a capacity-only replay selects a later
    one; the ladder never grows past its last entry.
    """

    requirement_id: str = eqx.field(static=True)
    entries: tuple[AtomisticRoleCapacity, ...]
    ladder_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        requirement: AtomisticNeighborhoodRequirement,
        entries: tuple[AtomisticRoleCapacity, ...],
        /,
    ) -> None:
        values = tuple(entries)
        if not values:
            raise ValueError("A role capacity ladder requires at least one entry.")
        for previous, entry in zip(values[:-1], values[1:], strict=True):
            if not entry.dominates(previous) or entry.capacity_id == previous.capacity_id:
                raise ValueError("Role capacity ladder entries must strictly grow.")
        self.requirement_id = requirement.requirement_id
        self.entries = values
        self.ladder_id = canonical_fingerprint(
            {
                "kind": "atomistic-role-capacity-ladder",
                "requirement": requirement.requirement_id,
                "entries": [entry.capacity_id for entry in values],
            }
        )


class AbstractAtomisticImageSearchPolicy(StrictModule, NonTrainableState):
    """Native image route search route and its explicit resource bounds."""

    policy_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def search(
        self,
        radius: float,
        vectors: np.ndarray,
        periodic: np.ndarray,
        capacity: ParticleImageCapacity,
        /,
        *,
        particle_capacity: int,
    ) -> AbstractImageRouteSearch:
        """Prepare the case-batched search for reference ``vectors[case, r, d]``."""
        raise NotImplementedError


@final
class AtomisticCellListImageSearch(AbstractAtomisticImageSearchPolicy):
    """Scalable fractional-lattice cell-list image search."""

    maximum_candidate_slots: int = eqx.field(static=True)
    deformation_margin: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self, *, maximum_candidate_slots: int, deformation_margin: float = 0.0
    ) -> None:
        slots = int(maximum_candidate_slots)
        if slots <= 0:
            raise ValueError("maximum_candidate_slots must be positive.")
        margin = _nonnegative_finite(deformation_margin, "deformation_margin")
        self.maximum_candidate_slots = slots
        self.deformation_margin = margin
        self.policy_id = canonical_fingerprint(
            {
                "kind": "atomistic-cell-list-image-search",
                "maximum_candidate_slots": slots,
                "deformation_margin": margin,
            }
        )

    def search(
        self,
        radius: float,
        vectors: np.ndarray,
        periodic: np.ndarray,
        capacity: ParticleImageCapacity,
        /,
        *,
        particle_capacity: int,
    ) -> AbstractImageRouteSearch:
        del periodic
        return CellListImageRouteSearch(
            radius,
            vectors,
            capacity,
            particle_capacity=particle_capacity,
            maximum_candidate_slots=self.maximum_candidate_slots,
            deformation_margin=self.deformation_margin,
        )


@final
class AtomisticDenseImageSearch(AbstractAtomisticImageSearchPolicy):
    """Bounded dense image reference for validation and small systems."""

    maximum_dense_routes: int = eqx.field(static=True)
    deformation_margin: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self, *, maximum_dense_routes: int, deformation_margin: float = 0.0
    ) -> None:
        routes = int(maximum_dense_routes)
        if routes <= 0:
            raise ValueError("maximum_dense_routes must be positive.")
        margin = _nonnegative_finite(deformation_margin, "deformation_margin")
        self.maximum_dense_routes = routes
        self.deformation_margin = margin
        self.policy_id = canonical_fingerprint(
            {
                "kind": "atomistic-dense-image-search",
                "maximum_dense_routes": routes,
                "deformation_margin": margin,
            }
        )

    def search(
        self,
        radius: float,
        vectors: np.ndarray,
        periodic: np.ndarray,
        capacity: ParticleImageCapacity,
        /,
        *,
        particle_capacity: int,
    ) -> AbstractImageRouteSearch:
        return DenseImageRouteSearch(
            radius,
            vectors,
            periodic,
            capacity,
            particle_capacity=particle_capacity,
            maximum_dense_routes=self.maximum_dense_routes,
            deformation_margin=self.deformation_margin,
        )


@final
class AtomisticTiledImageSearch(AbstractAtomisticImageSearchPolicy):
    """Bounded scalar/tiled all-pairs image reference with an explicit tile workspace.

    Tests every receiver, source and image candidate of the dense reference
    by ``lax`` traversal of ``tiles``; one tile, not ``N**2``, is materialized.
    """

    tiles: ImageTilePolicy
    deformation_margin: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, *, tiles: ImageTilePolicy, deformation_margin: float = 0.0
    ) -> None:
        margin = _nonnegative_finite(deformation_margin, "deformation_margin")
        self.tiles = tiles
        self.deformation_margin = margin
        self.policy_id = canonical_fingerprint(
            {
                "kind": "atomistic-tiled-image-search",
                "tiles": tiles.policy_id,
                "deformation_margin": margin,
            }
        )

    def search(
        self,
        radius: float,
        vectors: np.ndarray,
        periodic: np.ndarray,
        capacity: ParticleImageCapacity,
        /,
        *,
        particle_capacity: int,
    ) -> AbstractImageRouteSearch:
        return TiledImageRouteSearch(
            radius,
            vectors,
            periodic,
            capacity,
            particle_capacity=particle_capacity,
            tiles=self.tiles,
            deformation_margin=self.deformation_margin,
        )


@final
class AtomisticClusterTileImageSearch(AbstractAtomisticImageSearchPolicy):
    """Morton cluster-tile image search with an independently charged tile budget."""

    cluster_width: int = eqx.field(static=True)
    maximum_cluster_radius: float = eqx.field(static=True)
    tile_capacity: ParticleImageCapacity
    maximum_candidate_slots: int = eqx.field(static=True)
    deformation_margin: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        cluster_width: int,
        maximum_cluster_radius: float,
        tile_capacity: ParticleImageCapacity,
        maximum_candidate_slots: int,
        deformation_margin: float = 0.0,
    ) -> None:
        width = int(cluster_width)
        slots = int(maximum_candidate_slots)
        if width <= 0 or slots <= 0:
            raise ValueError(
                "cluster_width and maximum_candidate_slots must be positive."
            )
        bound = _nonnegative_finite(maximum_cluster_radius, "maximum_cluster_radius")
        margin = _nonnegative_finite(deformation_margin, "deformation_margin")
        if tile_capacity.maximum_case_edges is None:
            raise ValueError("tile_capacity must declare maximum_case_edges.")
        self.cluster_width = width
        self.maximum_cluster_radius = bound
        self.tile_capacity = tile_capacity
        self.maximum_candidate_slots = slots
        self.deformation_margin = margin
        self.policy_id = canonical_fingerprint(
            {
                "kind": "atomistic-cluster-tile-image-search",
                "cluster_width": width,
                "maximum_cluster_radius": bound,
                "tile_capacity": tile_capacity.capacity_id,
                "maximum_candidate_slots": slots,
                "deformation_margin": margin,
            }
        )

    def search(
        self,
        radius: float,
        vectors: np.ndarray,
        periodic: np.ndarray,
        capacity: ParticleImageCapacity,
        /,
        *,
        particle_capacity: int,
    ) -> AbstractImageRouteSearch:
        del periodic
        return ClusterTileImageRouteSearch(
            radius,
            vectors,
            capacity,
            particle_capacity=particle_capacity,
            cluster_width=self.cluster_width,
            maximum_cluster_radius=self.maximum_cluster_radius,
            tile_capacity=self.tile_capacity,
            maximum_candidate_slots=self.maximum_candidate_slots,
            deformation_margin=self.deformation_margin,
        )


@final
class AtomisticNeighborhoodSource(StrictModule, NonTrainableState):
    """One declared certified image source search and the requirements it serves.

    ``serves`` binds bounded (finite-cutoff) requirements explicitly, stored by
    ``requirement_id``; the plan certifies every binding.  ``capacities`` is
    the native search capacity ladder whose entries must use the per-case
    segment layout (``maximum_case_edges``).  The search radius defaults to
    the largest served cutoff plus ``skin``.  ``receiver_group`` and
    ``receiver_rows`` declare a receiver-row subset whose stable IDs are
    supplied at runtime.  An ``INTERACTION_SITES`` source searches realized
    site coordinates and is the independent site geometry of virtual-site or
    coarse coordinate maps.
    """

    name: str = eqx.field(static=True)
    entity: AtomisticSiteDomain = eqx.field(static=True)
    serves: tuple[str, ...] = eqx.field(static=True)
    search: AbstractAtomisticImageSearchPolicy
    capacities: ParticleImageCapacityLadder
    skin: float = eqx.field(static=True)
    search_radius: float | None = eqx.field(static=True)
    receiver_group: str | None = eqx.field(static=True)
    receiver_rows: int | None = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        name: str,
        entity: AtomisticSiteDomain,
        serves: tuple[AtomisticNeighborhoodRequirement, ...],
        search: AbstractAtomisticImageSearchPolicy,
        capacities: ParticleImageCapacityLadder,
        /,
        *,
        skin: float,
        search_radius: float | None = None,
        receiver_group: str | None = None,
        receiver_rows: int | None = None,
    ) -> None:
        label = _nonempty(name, "name")
        domain = parse(entity, AtomisticSiteDomain, "entity")
        if domain not in _SOURCE_ENTITIES:
            raise ValueError(
                "Neighborhood sources search physical atoms or interaction sites."
            )
        if any(value.complete for value in serves):
            raise ValueError(
                "Complete all-pairs requirements are served by an "
                "AtomisticCompletePairSource, not by an image search."
            )
        identifiers = tuple(sorted(value.requirement_id for value in serves))
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("serves must bind at least one unique requirement.")
        margin = _positive_finite(skin, "skin")
        radius = (
            None
            if search_radius is None
            else _positive_finite(search_radius, "search_radius")
        )
        group = None if receiver_group is None else _nonempty(receiver_group, "group")
        rows = None if receiver_rows is None else int(receiver_rows)
        if (group is None) != (rows is None) or (rows is not None and rows <= 0):
            raise ValueError(
                "receiver_group and a positive receiver_rows are declared together."
            )
        for entry in capacities.entries:
            if entry.maximum_case_edges is None:
                raise ValueError(
                    "Source capacities use the per-case segment layout "
                    "(maximum_case_edges) required for case-selective rebuild."
                )
        self.name = label
        self.entity = domain
        self.serves = identifiers
        self.search = search
        self.capacities = capacities
        self.skin = margin
        self.search_radius = radius
        self.receiver_group = group
        self.receiver_rows = rows
        self.source_id = canonical_fingerprint(
            {
                "kind": "atomistic-neighborhood-source",
                "name": label,
                "entity": domain.value,
                "serves": list(identifiers),
                "search": search.policy_id,
                "capacities": capacities.ladder_id,
                "skin": margin,
                "search_radius": radius,
                "receiver_group": group,
                "receiver_rows": rows,
            }
        )


@final
class AtomisticCompletePairSource(StrictModule, NonTrainableState):
    """Complete finite all-pairs source of ``cutoff=None`` pair requirements.

    Every unordered pair of one case's entities is enumerated once by the
    native ``DenseParticleNeighborhoodPlan`` (stable-ID ordered endpoints);
    the runtime active set masks it.  ``maximum_pairs`` is the explicit
    per-case allocation bound ``N (N - 1) / 2`` must fit, so the complete
    relation is never an unbounded hidden allocation.  The topology needs no
    motion certificate; it changes only with the active set.
    """

    name: str = eqx.field(static=True)
    entity: AtomisticSiteDomain = eqx.field(static=True)
    serves: tuple[str, ...] = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        entity: AtomisticSiteDomain,
        serves: tuple[AtomisticNeighborhoodRequirement, ...],
        /,
        *,
        maximum_pairs: int,
    ) -> None:
        label = _nonempty(name, "name")
        domain = parse(entity, AtomisticSiteDomain, "entity")
        if domain not in _SOURCE_ENTITIES:
            raise ValueError(
                "Neighborhood sources search physical atoms or interaction sites."
            )
        if any(not value.complete for value in serves):
            raise ValueError(
                "A complete-pair source serves only cutoff=None requirements."
            )
        identifiers = tuple(sorted(value.requirement_id for value in serves))
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("serves must bind at least one unique requirement.")
        pairs = int(maximum_pairs)
        if pairs < 0 or pairs > _INT32_LIMIT:
            raise ValueError("maximum_pairs must be a nonnegative int32 count.")
        self.name = label
        self.entity = domain
        self.serves = identifiers
        self.maximum_pairs = pairs
        self.source_id = canonical_fingerprint(
            {
                "kind": "atomistic-complete-pair-source",
                "name": label,
                "entity": domain.value,
                "serves": list(identifiers),
                "maximum_pairs": pairs,
            }
        )


AtomisticSource = AtomisticNeighborhoodSource | AtomisticCompletePairSource


def _canonical_requirements(
    requirements: tuple[AtomisticNeighborhoodRequirement, ...], /
) -> tuple[
    tuple[AtomisticNeighborhoodRequirement, ...],
    tuple[str, ...],
    dict[str, AtomisticNeighborhoodRequirement],
]:
    """Unique requirements in canonical ``(owner, role, requirement_id)`` slot order."""
    ordered = tuple(
        sorted(
            requirements,
            key=lambda value: (value.owner, value.role.value, value.requirement_id),
        )
    )
    identifiers = tuple(value.requirement_id for value in ordered)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Neighborhood requirements must be unique.")
    return ordered, identifiers, {value.requirement_id: value for value in ordered}


def _slot_role_ladders(
    ordered: tuple[AtomisticNeighborhoodRequirement, ...],
    by_id: Mapping[str, AtomisticNeighborhoodRequirement],
    role_capacities: tuple[AtomisticRoleCapacityLadder, ...],
    /,
) -> tuple[AtomisticRoleCapacityLadder | None, ...]:
    """Slot-aligned role ladders: one per bounded requirement, none when complete."""
    ladders: dict[str, AtomisticRoleCapacityLadder] = {}
    for ladder in role_capacities:
        requirement = by_id.get(ladder.requirement_id)
        if requirement is None:
            raise ValueError("A role capacity ladder binds an unknown requirement.")
        if requirement.complete:
            raise ValueError(
                f"{requirement.label} is complete; its storage is its source's "
                "bounded enumeration, not a role capacity ladder."
            )
        if ladder.requirement_id in ladders:
            raise ValueError(f"{requirement.label} has two role capacity ladders.")
        ladders[ladder.requirement_id] = ladder
    unbound = [
        value.label
        for value in ordered
        if not value.complete and value.requirement_id not in ladders
    ]
    if unbound:
        raise ValueError(f"Requirements {unbound!r} have no role capacity ladder.")
    return tuple(ladders.get(value.requirement_id) for value in ordered)


def _source_owners(
    sources: tuple[AtomisticSource, ...],
    identifiers: tuple[str, ...],
    by_id: Mapping[str, AtomisticNeighborhoodRequirement],
    /,
) -> tuple[tuple[AtomisticSource, ...], tuple[int, ...]]:
    """Name-ordered sources and the slot-aligned index of each requirement's source."""
    declared = tuple(sorted(sources, key=lambda value: value.name))
    names = tuple(value.name for value in declared)
    if len(set(names)) != len(names):
        raise ValueError("Neighborhood sources must have unique names.")
    owner_of: dict[str, int] = {}
    for index, source in enumerate(declared):
        for identifier in source.serves:
            if identifier not in by_id:
                raise ValueError(f"Source {source.name!r} serves an unknown requirement.")
            if identifier in owner_of:
                raise ValueError(
                    f"Requirement {by_id[identifier].label} is served by two sources."
                )
            owner_of[identifier] = index
    missing = [by_id[value].label for value in identifiers if value not in owner_of]
    if missing:
        raise ValueError(f"Requirements {missing!r} are not served by any source.")
    return declared, tuple(owner_of[value] for value in identifiers)


def _admit_binding(
    requirement: AtomisticNeighborhoodRequirement,
    ladder: AtomisticRoleCapacityLadder | None,
    source: AtomisticSource,
    cases: int,
    /,
) -> None:
    """Admit one requirement's source entity, receiver closure and role segments."""
    site_on_atoms = (
        requirement.role is AtomisticNeighborhoodRole.INTERACTION_SITE_PAIRS
        and source.entity is AtomisticSiteDomain.PHYSICAL_ATOMS
    )
    if requirement.entity is not source.entity and not site_on_atoms:
        raise ValueError(
            f"Source {source.name!r} searches {source.entity.value}; "
            f"{requirement.label} requires {requirement.entity.value}."
        )
    if isinstance(source, AtomisticCompletePairSource):
        return
    if requirement.counting is AtomisticPairCounting.DIRECTED:
        if requirement.receiver_group != source.receiver_group:
            raise ValueError(
                f"{requirement.label} and source {source.name!r} declare "
                "different receiver selections."
            )
    elif source.receiver_group is not None:
        raise ValueError(
            f"Half-counted {requirement.label} needs a reversal-closed "
            f"source; {source.name!r} searches a receiver subset."
        )
    if ladder is None:
        raise RuntimeError("A bounded requirement lacks its ladder.")
    for capacity in ladder.entries:
        if (
            capacity.maximum_case_routes is not None
            and cases * capacity.maximum_case_routes > capacity.maximum_routes
        ):
            raise ValueError(
                f"{requirement.label} per-case segments exceed maximum_routes."
            )


def _resolved_source_radii(
    source: AtomisticSource,
    served: list[
        tuple[AtomisticNeighborhoodRequirement, AtomisticRoleCapacityLadder | None]
    ],
    cases: int,
    /,
) -> tuple[float | None, float | None]:
    """Admit one source's bindings and resolve its ``(search, interaction)`` radii.

    A complete-pair source has no radii; its pairs over every case must fit
    the int32 route index space.  An image search radius (declared, or the
    largest served cutoff plus skin) covers every served cutoff plus skin and
    every selection radius, strictly beyond a closed selection boundary.
    """
    for requirement, ladder in served:
        _admit_binding(requirement, ladder, source, cases)
    if isinstance(source, AtomisticCompletePairSource):
        if cases * source.maximum_pairs > _INT32_LIMIT:
            raise ValueError(
                f"Source {source.name!r} pairs over {cases} cases exceed the "
                "int32 route index space."
            )
        return None, None
    maximum = max(requirement.bounded_cutoff() for requirement, _ in served)
    radius = (
        maximum + source.skin if source.search_radius is None else source.search_radius
    )
    for requirement, _ in served:
        cutoff = requirement.bounded_cutoff()
        if not cutoff + source.skin <= radius:
            raise ValueError(
                f"Source {source.name!r} radius {radius} does not cover "
                f"{requirement.label} cutoff {cutoff} plus skin."
            )
        policy = requirement.selection
        if policy is not None and not (
            radius > policy.radius
            if policy.radius_boundary == "closed"
            else radius >= policy.radius
        ):
            raise ValueError(
                f"Source {source.name!r} does not cover the selection of "
                f"{requirement.label}."
            )
    for entry in source.capacities.entries:
        entry.route_buffer(cases)
    return float(radius), float(maximum)


@final
class AtomisticNeighborhoodBundlePlan(StrictModule, NonTrainableState):
    """Position-independent requirements, certified source bindings and resources.

    Requirements are ordered canonically by ``(owner, role, requirement_id)``
    and sources by name, so view slots do not depend on declaration order.
    Every requirement is served by exactly one source: bounded requirements by
    an image ``AtomisticNeighborhoodSource`` and complete (``cutoff=None``)
    requirements by an ``AtomisticCompletePairSource``.  Each bounded
    requirement has exactly one ``role_capacities`` ladder; complete
    requirements have none (their storage is the source's exact bounded
    enumeration).  A binding is admitted only when the source searches the
    requirement's entity, covers ``cutoff + skin``, is reversal closed for half
    counting (no receiver subset) and declares the same receiver subset as a
    directed role.  The only cross-entity binding is interaction-site pairs
    served by a physical-atom source, admitted at preparation only under a
    proven identity coordinate map (no unproved aliasing); independent
    ``INTERACTION_SITES`` sources are always admissible.  ``frame`` is the
    explicit nonperiodic search domain of a finite system with image sources.
    """

    requirements: tuple[AtomisticNeighborhoodRequirement, ...]
    sources: tuple[AtomisticSource, ...]
    role_capacities: tuple[AtomisticRoleCapacityLadder | None, ...]
    frame: PeriodicCell | None
    case_count: int = eqx.field(static=True)
    role_sources: tuple[int, ...] = eqx.field(static=True)
    search_radii: tuple[float | None, ...] = eqx.field(static=True)
    interaction_radii: tuple[float | None, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        requirements: tuple[AtomisticNeighborhoodRequirement, ...],
        sources: tuple[AtomisticSource, ...],
        role_capacities: tuple[AtomisticRoleCapacityLadder, ...],
        /,
        *,
        case_count: int = 1,
        frame: PeriodicCell | None = None,
    ) -> None:
        cases = int(case_count)
        if cases <= 0:
            raise ValueError("case_count must be positive.")
        ordered, identifiers, by_id = _canonical_requirements(requirements)
        ordered_ladders = _slot_role_ladders(ordered, by_id, role_capacities)
        declared, role_sources = _source_owners(sources, identifiers, by_id)
        search_radii: list[float | None] = []
        interaction_radii: list[float | None] = []
        for index, source in enumerate(declared):
            served = [
                (requirement, ladder)
                for requirement, ladder, owner in zip(
                    ordered, ordered_ladders, role_sources, strict=True
                )
                if owner == index
            ]
            search_radius, interaction_radius = _resolved_source_radii(
                source, served, cases
            )
            search_radii.append(search_radius)
            interaction_radii.append(interaction_radius)
        if frame is not None and any(frame.periodic_axes):
            raise ValueError("A finite-system search frame must be nonperiodic.")
        self.requirements = ordered
        self.sources = declared
        self.role_capacities = ordered_ladders
        self.frame = frame
        self.case_count = cases
        self.role_sources = role_sources
        self.search_radii = tuple(search_radii)
        self.interaction_radii = tuple(interaction_radii)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-neighborhood-bundle-plan",
                "requirements": list(identifiers),
                "sources": [value.source_id for value in declared],
                "role_capacities": [
                    None if value is None else value.ladder_id
                    for value in ordered_ladders
                ],
                "case_count": cases,
                "frame": None if frame is None else frame.cell_id,
            }
        )

    def slot(self, requirement: AtomisticNeighborhoodRequirement, /) -> int:
        """Return the static view slot of exactly ``requirement``."""
        for index, value in enumerate(self.requirements):
            if value.requirement_id == requirement.requirement_id:
                return index
        raise KeyError(f"No neighborhood requirement {requirement.label}.")

    def prepare(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        graph_plans: Mapping[str, StreamedRelationPlan] | None = None,
    ) -> PreparedAtomisticNeighborhoodBundle:
        """Prepare against ``system``.

        ``graph_plans`` optionally maps directed requirements' ``requirement_id``
        to the ``StreamedRelationPlan`` of their graph schedule.
        """
        aligned = None
        if graph_plans is not None:
            known = {value.requirement_id for value in self.requirements}
            if set(graph_plans) - known:
                raise ValueError("graph_plans names requirements outside this plan.")
            aligned = tuple(
                graph_plans.get(value.requirement_id) for value in self.requirements
            )
        return PreparedAtomisticNeighborhoodBundle(self, system, graph_plans=aligned)

    def to_dict(self) -> dict[str, object]:
        """Canonical pickle-free JSON recipe replayed only through constructors.

        ``from_dict`` rebuilds every requirement, selection, source, search
        policy, capacity ladder and frame through its owning constructor and
        refuses a recipe that does not reproduce ``plan_id``.
        """
        frame = self.frame
        return {
            "kind": _PLAN_RECIPE_KIND,
            "requirements": [_requirement_dict(value) for value in self.requirements],
            "sources": [_source_dict(value) for value in self.sources],
            "role_capacities": [
                {
                    "requirement": ladder.requirement_id,
                    "entries": [_role_capacity_dict(entry) for entry in ladder.entries],
                }
                for ladder in self.role_capacities
                if ladder is not None
            ],
            "case_count": self.case_count,
            "frame": None
            if frame is None
            else {
                "dtype": str(np.asarray(frame.vectors).dtype),
                "vectors": np.asarray(frame.vectors).tolist(),
                "origin": np.asarray(frame.origin).tolist(),
                "periodic_axes": list(frame.periodic_axes),
                "certified_condition_number": frame.certified_condition_number,
            },
            "plan_id": self.plan_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object], /) -> AtomisticNeighborhoodBundlePlan:
        """Replay a ``to_dict`` recipe through the owning constructors."""
        record = _recipe_record(value, _PLAN_FIELDS, "bundle plan")
        if record["kind"] != _PLAN_RECIPE_KIND:
            raise ValueError("Recipe is not an atomistic neighborhood bundle plan.")
        requirements = tuple(
            _requirement_from(item) for item in _recipe_list(record["requirements"])
        )
        by_id = {item.requirement_id: item for item in requirements}
        sources = tuple(
            _source_from(item, by_id) for item in _recipe_list(record["sources"])
        )
        ladders = []
        for item in _recipe_list(record["role_capacities"]):
            ladder = _recipe_record(item, {"requirement", "entries"}, "role ladder")
            requirement = by_id.get(str(ladder["requirement"]))
            if requirement is None:
                raise ValueError("A role ladder recipe names an unknown requirement.")
            ladders.append(
                AtomisticRoleCapacityLadder(
                    requirement,
                    tuple(
                        _role_capacity_from(entry)
                        for entry in _recipe_list(ladder["entries"])
                    ),
                )
            )
        frame_record = record["frame"]
        frame = None
        if frame_record is not None:
            fields = _recipe_record(
                frame_record,
                {
                    "dtype",
                    "vectors",
                    "origin",
                    "periodic_axes",
                    "certified_condition_number",
                },
                "frame",
            )
            dtype = np.dtype(str(fields["dtype"]))
            frame = PeriodicCell(
                np.asarray(fields["vectors"], dtype=dtype),
                origin=np.asarray(fields["origin"], dtype=dtype),
                periodic_axes=tuple(
                    bool(axis) for axis in _recipe_list(fields["periodic_axes"])
                ),
                maximum_condition_number=float(
                    _recipe_number(fields["certified_condition_number"])
                ),
            )
        plan = cls(
            requirements,
            sources,
            tuple(ladders),
            case_count=int(_recipe_number(record["case_count"])),
            frame=frame,
        )
        if plan.plan_id != record["plan_id"]:
            raise ValueError(
                "The neighborhood bundle recipe is not reproduced by its constructors."
            )
        return plan


_PLAN_RECIPE_KIND = "atomistic-neighborhood-bundle-plan"
_PLAN_FIELDS = frozenset(
    {
        "kind",
        "requirements",
        "sources",
        "role_capacities",
        "case_count",
        "frame",
        "plan_id",
    }
)


def _recipe_record(
    value: object, fields: frozenset[str] | set[str], label: str, /
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"The {label} recipe fields are not canonical.")
    return value


def _recipe_list(value: object, /) -> list[object]:
    if not isinstance(value, list | tuple):
        raise ValueError("A recipe sequence must be a list.")
    return list(value)


def _recipe_number(value: object, /) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("A recipe number must be an integer or float.")
    return value


def _optional_number(value: object, /) -> float | None:
    return None if value is None else float(_recipe_number(value))


def _optional_text(value: object, /) -> str | None:
    return None if value is None else str(value)


def _selection_dict(policy: ImageNeighborSelectionPolicy, /) -> dict[str, object]:
    return {
        "radius": policy.radius,
        "maximum_neighbors": policy.maximum_neighbors,
        "degeneracy_tolerance": policy.degeneracy_tolerance,
        "measure": policy.measure,
        "strict": policy.strict,
        "radius_boundary": policy.radius_boundary,
        "coincidence_threshold": policy.coincidence_threshold,
    }


def _selection_from(value: object, /) -> ImageNeighborSelectionPolicy:
    record = _recipe_record(
        value,
        {
            "radius",
            "maximum_neighbors",
            "degeneracy_tolerance",
            "measure",
            "strict",
            "radius_boundary",
            "coincidence_threshold",
        },
        "selection",
    )
    measure = parse(record["measure"], SelectionMeasure, "selection measure")
    boundary = parse(
        record["radius_boundary"], RadiusBoundary, "selection radius boundary"
    )
    return ImageNeighborSelectionPolicy(
        float(_recipe_number(record["radius"])),
        int(_recipe_number(record["maximum_neighbors"])),
        degeneracy_tolerance=float(_recipe_number(record["degeneracy_tolerance"])),
        measure=measure,
        strict=bool(record["strict"]),
        radius_boundary=boundary,
        coincidence_threshold=_optional_number(record["coincidence_threshold"]),
    )


def _requirement_dict(value: AtomisticNeighborhoodRequirement, /) -> dict[str, object]:
    return {
        "owner": value.owner,
        "role": value.role.value,
        "cutoff": value.cutoff,
        "selection": None
        if value.selection is None
        else _selection_dict(value.selection),
        "receiver_group": value.receiver_group,
        "exclusions": value.exclusions.value,
    }


def _requirement_from(value: object, /) -> AtomisticNeighborhoodRequirement:
    record = _recipe_record(
        value,
        {"owner", "role", "cutoff", "selection", "receiver_group", "exclusions"},
        "requirement",
    )
    return AtomisticNeighborhoodRequirement(
        str(record["owner"]),
        AtomisticNeighborhoodRole(str(record["role"])),
        _optional_number(record["cutoff"]),
        selection=None
        if record["selection"] is None
        else _selection_from(record["selection"]),
        receiver_group=_optional_text(record["receiver_group"]),
        exclusions=AtomisticPairExclusions(str(record["exclusions"])),
    )


def _image_capacity_dict(value: ParticleImageCapacity, /) -> dict[str, object]:
    return {
        "maximum_particles_per_cell": value.maximum_particles_per_cell,
        "maximum_edges": value.maximum_edges,
        "maximum_degree": value.maximum_degree,
        "maximum_images": value.maximum_images,
        "maximum_case_edges": value.maximum_case_edges,
    }


def _image_capacity_from(value: object, /) -> ParticleImageCapacity:
    record = _recipe_record(
        value,
        {
            "maximum_particles_per_cell",
            "maximum_edges",
            "maximum_degree",
            "maximum_images",
            "maximum_case_edges",
        },
        "image capacity",
    )
    case_edges = record["maximum_case_edges"]
    return ParticleImageCapacity(
        maximum_particles_per_cell=int(
            _recipe_number(record["maximum_particles_per_cell"])
        ),
        maximum_edges=int(_recipe_number(record["maximum_edges"])),
        maximum_degree=int(_recipe_number(record["maximum_degree"])),
        maximum_images=int(_recipe_number(record["maximum_images"])),
        maximum_case_edges=None
        if case_edges is None
        else int(_recipe_number(case_edges)),
    )


def _role_capacity_dict(value: AtomisticRoleCapacity, /) -> dict[str, object]:
    return {
        "maximum_routes": value.maximum_routes,
        "maximum_degree": value.maximum_degree,
        "maximum_case_routes": value.maximum_case_routes,
    }


def _role_capacity_from(value: object, /) -> AtomisticRoleCapacity:
    record = _recipe_record(
        value,
        {"maximum_routes", "maximum_degree", "maximum_case_routes"},
        "role capacity",
    )
    case_routes = record["maximum_case_routes"]
    return AtomisticRoleCapacity(
        maximum_routes=int(_recipe_number(record["maximum_routes"])),
        maximum_degree=int(_recipe_number(record["maximum_degree"])),
        maximum_case_routes=None
        if case_routes is None
        else int(_recipe_number(case_routes)),
    )


def _search_dict(value: AbstractAtomisticImageSearchPolicy, /) -> dict[str, object]:
    match value:
        case AtomisticCellListImageSearch():
            return {
                "kind": "cell-list",
                "maximum_candidate_slots": value.maximum_candidate_slots,
                "deformation_margin": value.deformation_margin,
            }
        case AtomisticDenseImageSearch():
            return {
                "kind": "dense",
                "maximum_dense_routes": value.maximum_dense_routes,
                "deformation_margin": value.deformation_margin,
            }
        case AtomisticTiledImageSearch():
            return {
                "kind": "tiled",
                "receiver_width": value.tiles.receiver_width,
                "source_width": value.tiles.source_width,
                "maximum_tile_candidates": value.tiles.maximum_tile_candidates,
                "deformation_margin": value.deformation_margin,
            }
        case AtomisticClusterTileImageSearch():
            return {
                "kind": "cluster-tile",
                "cluster_width": value.cluster_width,
                "maximum_cluster_radius": value.maximum_cluster_radius,
                "tile_capacity": _image_capacity_dict(value.tile_capacity),
                "maximum_candidate_slots": value.maximum_candidate_slots,
                "deformation_margin": value.deformation_margin,
            }
        case _:
            raise TypeError(
                f"{type(value).__name__} has no canonical neighborhood recipe."
            )


def _search_from(value: object, /) -> AbstractAtomisticImageSearchPolicy:
    if not isinstance(value, Mapping):
        raise ValueError("The search recipe must be a mapping.")
    match value.get("kind"):
        case "cell-list":
            record = _recipe_record(
                value,
                {"kind", "maximum_candidate_slots", "deformation_margin"},
                "cell-list search",
            )
            return AtomisticCellListImageSearch(
                maximum_candidate_slots=int(
                    _recipe_number(record["maximum_candidate_slots"])
                ),
                deformation_margin=float(_recipe_number(record["deformation_margin"])),
            )
        case "dense":
            record = _recipe_record(
                value,
                {"kind", "maximum_dense_routes", "deformation_margin"},
                "dense search",
            )
            return AtomisticDenseImageSearch(
                maximum_dense_routes=int(_recipe_number(record["maximum_dense_routes"])),
                deformation_margin=float(_recipe_number(record["deformation_margin"])),
            )
        case "tiled":
            record = _recipe_record(
                value,
                {
                    "kind",
                    "receiver_width",
                    "source_width",
                    "maximum_tile_candidates",
                    "deformation_margin",
                },
                "tiled search",
            )
            return AtomisticTiledImageSearch(
                tiles=ImageTilePolicy(
                    receiver_width=int(_recipe_number(record["receiver_width"])),
                    source_width=int(_recipe_number(record["source_width"])),
                    maximum_tile_candidates=int(
                        _recipe_number(record["maximum_tile_candidates"])
                    ),
                ),
                deformation_margin=float(_recipe_number(record["deformation_margin"])),
            )
        case "cluster-tile":
            record = _recipe_record(
                value,
                {
                    "kind",
                    "cluster_width",
                    "maximum_cluster_radius",
                    "tile_capacity",
                    "maximum_candidate_slots",
                    "deformation_margin",
                },
                "cluster-tile search",
            )
            return AtomisticClusterTileImageSearch(
                cluster_width=int(_recipe_number(record["cluster_width"])),
                maximum_cluster_radius=float(
                    _recipe_number(record["maximum_cluster_radius"])
                ),
                tile_capacity=_image_capacity_from(record["tile_capacity"]),
                maximum_candidate_slots=int(
                    _recipe_number(record["maximum_candidate_slots"])
                ),
                deformation_margin=float(_recipe_number(record["deformation_margin"])),
            )
        case _:
            raise ValueError("The search recipe kind is unknown.")


def _source_dict(value: AtomisticSource, /) -> dict[str, object]:
    if isinstance(value, AtomisticCompletePairSource):
        return {
            "kind": "complete-pairs",
            "name": value.name,
            "entity": value.entity.value,
            "serves": list(value.serves),
            "maximum_pairs": value.maximum_pairs,
        }
    return {
        "kind": "image",
        "name": value.name,
        "entity": value.entity.value,
        "serves": list(value.serves),
        "search": _search_dict(value.search),
        "capacities": [_image_capacity_dict(entry) for entry in value.capacities.entries],
        "skin": value.skin,
        "search_radius": value.search_radius,
        "receiver_group": value.receiver_group,
        "receiver_rows": value.receiver_rows,
    }


def _source_from(
    value: object, requirements: Mapping[str, AtomisticNeighborhoodRequirement], /
) -> AtomisticSource:
    if not isinstance(value, Mapping):
        raise ValueError("The source recipe must be a mapping.")

    def served(items: object, /) -> tuple[AtomisticNeighborhoodRequirement, ...]:
        identifiers = [str(item) for item in _recipe_list(items)]
        if any(item not in requirements for item in identifiers):
            raise ValueError("A source recipe serves an unknown requirement.")
        return tuple(requirements[item] for item in identifiers)

    match value.get("kind"):
        case "complete-pairs":
            record = _recipe_record(
                value,
                {"kind", "name", "entity", "serves", "maximum_pairs"},
                "complete-pair source",
            )
            return AtomisticCompletePairSource(
                str(record["name"]),
                AtomisticSiteDomain(str(record["entity"])),
                served(record["serves"]),
                maximum_pairs=int(_recipe_number(record["maximum_pairs"])),
            )
        case "image":
            record = _recipe_record(
                value,
                {
                    "kind",
                    "name",
                    "entity",
                    "serves",
                    "search",
                    "capacities",
                    "skin",
                    "search_radius",
                    "receiver_group",
                    "receiver_rows",
                },
                "image source",
            )
            rows = record["receiver_rows"]
            return AtomisticNeighborhoodSource(
                str(record["name"]),
                AtomisticSiteDomain(str(record["entity"])),
                served(record["serves"]),
                _search_from(record["search"]),
                ParticleImageCapacityLadder(
                    tuple(
                        _image_capacity_from(entry)
                        for entry in _recipe_list(record["capacities"])
                    )
                ),
                skin=float(_recipe_number(record["skin"])),
                search_radius=_optional_number(record["search_radius"]),
                receiver_group=_optional_text(record["receiver_group"]),
                receiver_rows=None if rows is None else int(_recipe_number(rows)),
            )
        case _:
            raise ValueError("The source recipe kind is unknown.")


@final
class AtomisticNeighborhoodSourceProof(StrictModule, NonTrainableState):
    """Certified compatibility facts of one prepared source and its consumers.

    For an image source ``coverage_margins[k] = search_radius - skin -
    cutoff_k >= 0`` for every served requirement.  A complete-pair source has
    no radius, skin or margins (``None``): its coverage is the complete
    enumeration.  ``countings`` record each consumer's counting convention;
    ``site_alias`` holds when interaction-site pairs are served by the
    physical-atom search under a proven identity coordinate map.  Every fact
    is layout: ``engine_id`` is the content-free engine digest (an engine
    indexes its entity support schema, never a prepared material support) and
    ``proof_id`` digests these facts; the material is bound by the owning
    bundle's ``prepared_identity``.
    """

    source: str = eqx.field(static=True)
    entity: AtomisticSiteDomain = eqx.field(static=True)
    served: tuple[str, ...] = eqx.field(static=True)
    countings: tuple[str, ...] = eqx.field(static=True)
    cutoffs: tuple[float | None, ...] = eqx.field(static=True)
    coverage_margins: tuple[float | None, ...] = eqx.field(static=True)
    search_radius: float | None = eqx.field(static=True)
    interaction_radius: float | None = eqx.field(static=True)
    skin: float | None = eqx.field(static=True)
    receiver_group: str | None = eqx.field(static=True)
    site_alias: bool = eqx.field(static=True)
    capacity_id: str = eqx.field(static=True)
    engine_id: str = eqx.field(static=True)
    proof_id: str = eqx.field(static=True)

    @property
    def shared(self) -> bool:
        return len(self.served) > 1


@final
class PreparedAtomisticCompletePairs(StrictModule, NonTrainableState):
    """Case-tiled complete unordered pair enumeration of one complete source.

    ``relation`` holds every pair ``(smaller ID -> larger ID)`` of every case
    with zero image shifts of the bundle lattice rank; validity is applied at
    runtime from the accepted active set.  The enumeration of one particle
    capacity is layout: endpoint slots and stable particle IDs are dynamic
    leaves and the relation indexes the declared ``support_id`` schema, so
    ``prepared_id`` is a content-free layout digest.  The dense preparation's
    material digest is never retained; the owning bundle binds the material
    through its ``prepared_identity``.
    """

    relation: ParticleImageRelation
    case_count: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: AtomisticCompletePairSource,
        dense: PreparedDenseParticleNeighborhood,
        /,
        *,
        case_count: int,
        lattice_rank: int,
        support_id: str,
    ) -> None:
        if dense.plan.maximum_pairs != source.maximum_pairs:
            raise ValueError("The dense enumeration is not prepared for this source.")
        support = _nonempty(support_id, "support_id")
        pairs = dense.pair_relation
        cases = int(case_count)
        rank = int(lattice_rank)
        count = dense.particle_capacity
        left = np.asarray(pairs.left_indices, dtype=np.int64)
        right = np.asarray(pairs.right_indices, dtype=np.int64)
        per_case = left.size
        offsets = np.repeat(np.arange(cases, dtype=np.int64) * count, per_case)
        size = cases * count
        schema = canonical_fingerprint(
            {
                "kind": "atomistic-complete-pair-relation",
                "source": source.source_id,
                "dense": dense.plan.plan_id,
                "support": support,
                "particle_capacity": count,
                "case_count": cases,
                "lattice_rank": rank,
            }
        )
        self.relation = ParticleImageRelation(
            EdgeRelation(
                (np.tile(left, cases) + offsets).astype(np.int32),
                (np.tile(right, cases) + offsets).astype(np.int32),
                source_size=size,
                target_size=size,
            ),
            np.tile(np.asarray(pairs.left_particle_ids), cases),
            np.tile(np.asarray(pairs.right_particle_ids), cases),
            np.zeros((cases * per_case, rank), dtype=np.int32),
            np.repeat(np.arange(cases, dtype=np.int32), per_case),
            case_count=cases,
            particle_capacity=count,
            support_id=support,
            relation_schema_id=schema,
        )
        self.case_count = cases
        self.particle_capacity = count
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-atomistic-complete-pairs", "relation": schema}
        )


def _material_complete_pairs_id(
    source: AtomisticCompletePairSource,
    dense: PreparedDenseParticleNeighborhood,
    case_count: int,
    lattice_rank: int,
    /,
) -> str:
    """Exact material digest of a complete-pair engine (bundle identity payload).

    The canonical bundle digest has always named the engine through the dense
    preparation's material digest; it is reproduced on the host for the
    bundle's dynamic ``prepared_identity`` only and never stored statically.
    """
    relation = canonical_fingerprint(
        {
            "kind": "atomistic-complete-pair-relation",
            "source": source.source_id,
            "dense": dense.prepared_id,
            "case_count": int(case_count),
            "lattice_rank": int(lattice_rank),
        }
    )
    return canonical_fingerprint(
        {"kind": "prepared-atomistic-complete-pairs", "relation": relation}
    )


@final
class AtomisticCompletePairState(StrictModule, NonTrainableState):
    """Accepted epoch of one complete-pair source.

    ``reference_active`` is the active set the epoch's topology was formed
    from; a changed active set is a new epoch (or ``stale`` when rebuild is
    not permitted).  Nonfinite active coordinates are a scientific failure.
    There is no capacity failure: the complete enumeration was admitted at
    preparation.  ``prepared_pairs_id`` is the engine's content-free layout
    digest.
    """

    reference_active: Array
    initialized: Array
    case_epochs: Array
    rebuilt: Array
    rebuild_counts: Array
    stale: Array
    nonfinite: Array
    prepared_pairs_id: str = eqx.field(static=True)

    @property
    def capacity_failure(self) -> Array:
        return jnp.zeros_like(self.initialized)

    @property
    def scientific_failure(self) -> Array:
        return self.initialized & self.nonfinite

    @property
    def successful(self) -> Array:
        return self.initialized & ~self.stale & ~self.nonfinite


AtomisticSourceState = SegmentedImageVerletState | AtomisticCompletePairState
AtomisticSourceEngine = SegmentedImageVerletPlan | PreparedAtomisticCompletePairs


@final
class AtomisticNeighborhoodGeometry(StrictModule, NonTrainableState):
    """Case-batched current geometry a bundle state is certified for.

    Positions are ``(case, N, d)`` system particle coordinates and, when a
    site source exists, ``(case, S, d)`` realized interaction-site
    coordinates.  ``cell_vectors`` are ``(case, r, d)`` row lattices (the
    prepared cell or finite frame by default; ``r == 0`` for a finite system
    served only by complete pairs).  Optional integer image counts make
    ``positions + counts @ H`` the unwrapped coordinates so rewrapping
    re-expresses route shifts exactly.  Build with
    ``PreparedAtomisticNeighborhoodBundle.geometry``.
    """

    positions: Array
    active: Array
    cell_vectors: Array
    image_counts: Array | None
    site_positions: Array | None
    site_active: Array | None
    site_image_counts: Array | None
    site_successful: Array

    def __init__(
        self,
        positions: Array,
        active: Array,
        cell_vectors: Array,
        image_counts: Array | None,
        site_positions: Array | None,
        site_active: Array | None,
        site_image_counts: Array | None,
        site_successful: Array,
        /,
    ) -> None:
        if positions.ndim != 3 or active.shape != positions.shape[:2]:
            raise ValueError("Geometry positions and active masks are inconsistent.")
        cases = positions.shape[0]
        if cell_vectors.ndim != 3 or cell_vectors.shape[0] != cases:
            raise ValueError("Geometry cell vectors must be (case, rank, d).")
        if site_successful.shape != (cases,):
            raise ValueError("site_successful must be one flag per case.")
        if (site_positions is None) != (site_active is None):
            raise ValueError("Site positions and active masks are supplied together.")
        if site_positions is None and site_image_counts is not None:
            raise ValueError("Site image counts require site positions.")
        self.positions = positions
        self.active = active
        self.cell_vectors = cell_vectors
        self.image_counts = image_counts
        self.site_positions = site_positions
        self.site_active = site_active
        self.site_image_counts = site_image_counts
        self.site_successful = site_successful


@final
class AtomisticRoleEvidence(StrictModule, NonTrainableState):
    """Per-case status of one role view with capacity and scientific causes split.

    Capacity failures (role ``route_overflow``/``degree_overflow`` and source
    search ``source_capacity_failure``) are repairable by a declared larger
    capacity.  Scientific failures (source domain/stencil/finite/representation
    failures, an open reversal closure of a half view, endpoint identity
    conflicts, a repeated minimum image or self image inside a classical
    cutoff, re-expression overflow and a failed site realization) never are.
    ``stale`` cases were retained without a valid certificate because rebuild
    was not permitted; ``initialized`` is false for cases that own no epoch.
    ``capped_receivers`` counts receivers truncated by declared scientific
    selection, which is not a failure.  ``selection_changed`` marks a case
    whose frozen-epoch binding (``PreparedAtomisticNeighborhoodBundle.bind``)
    could not certify that the declared selection at the current geometry is
    the retained one; such a case is also ``stale``, never silently reused.
    """

    required_routes: Array
    stored_routes: Array
    required_degree: Array
    route_overflow: Array
    degree_overflow: Array
    source_capacity_failure: Array
    source_scientific_failure: Array
    reversal_open: Array
    identity_conflict: Array
    repeated_minimum_image: Array
    representation_overflow: Array
    entity_failure: Array
    capped_receivers: Array
    initialized: Array
    stale: Array
    epochs: Array
    rebuilt: Array
    selection_changed: Array
    selection: ImageSelectionEvidence | None = None

    @property
    def capacity_failure(self) -> Array:
        return self.route_overflow | self.degree_overflow | self.source_capacity_failure

    @property
    def scientific_failure(self) -> Array:
        return (
            self.source_scientific_failure
            | self.reversal_open
            | self.identity_conflict
            | self.repeated_minimum_image
            | self.representation_overflow
            | self.entity_failure
        )

    @property
    def uninitialized(self) -> Array:
        return ~self.initialized

    @property
    def successful(self) -> Array:
        return (
            self.initialized
            & ~self.stale
            & ~self.capacity_failure
            & ~self.scientific_failure
        )


@final
class AtomisticNeighborhoodView(StrictModule, NonTrainableState):
    """Resolved routes of one requirement in the current image representation.

    ``relation`` routes ``(source, receiver, n)`` have displacement
    ``x[receiver] - x[source] + n @ H`` for flat case-major entity positions.
    ``mask`` marks routes inside the physical cutoff (or selected) at the
    certified geometry; for complete pairs it is every active pair.  Half and
    complete views keep one route per unordered pair (or reversal pair) and
    represent ``directed_multiplicity == 2`` directed routes, ``self_image``
    marking nonzero self images.  Classical, site and complete pair views also
    expose the pair-once ``pair_relation`` (self images excluded) in their own
    entity space and, under topology exclusions, per-pair Lennard-Jones and
    electrostatic exception scales.  Site-pair scales are looked up through
    each site's declared physical particle (``physical_dof_indices``) in the
    system's pair key space; pairs with a virtual endpoint keep unit scales.
    ``cutoff`` is ``None`` for complete pairs.
    """

    relation: ParticleImageRelation
    mask: Array
    self_image: Array
    receiver_rows: ImageReceiverRows | None
    pair_relation: ParticlePairRelation | None
    lennard_jones_scales: Array | None
    electrostatic_scales: Array | None
    evidence: AtomisticRoleEvidence
    owner: str = eqx.field(static=True)
    role: AtomisticNeighborhoodRole = eqx.field(static=True)
    entity: AtomisticSiteDomain = eqx.field(static=True)
    counting: AtomisticPairCounting = eqx.field(static=True)
    selected: bool = eqx.field(static=True)
    directed_multiplicity: int = eqx.field(static=True)
    cutoff: float | None = eqx.field(static=True)
    source: str = eqx.field(static=True)
    requirement_id: str = eqx.field(static=True)

    def displacement(self, positions: ArrayLike, cell_vectors: ArrayLike, /) -> Array:
        """Differentiable route displacement for flat or ``(case, P, d)`` positions."""
        value = jnp.asarray(positions)
        if value.ndim == 3:
            value = value.reshape((-1, value.shape[-1]))
        return self.relation.displacement(value, cell_vectors)

    def cutoff_mask(self, positions: ArrayLike, cell_vectors: ArrayLike, /) -> Array:
        """Routes strictly inside the cutoff under ``positions``/``cell_vectors``.

        For a selected graph the selection itself stays the one certified at
        refresh; only the radius test is re-applied.  A complete view has no
        cutoff: every valid route is returned.
        """
        if self.cutoff is None:
            return self.relation.valid
        displacement = self.displacement(positions, cell_vectors)
        squared = jax.lax.stop_gradient(jnp.sum(displacement * displacement, axis=-1))
        return self.relation.valid & (squared < self.cutoff * self.cutoff)


@final
class AtomisticNeighborhoodBundleState(StrictModule, NonTrainableState):
    """The single accepted atomistic neighborhood state of one prepared bundle.

    ``sources`` are the native per-case source epochs (segmented image Verlet
    epochs with build frame, wrap counts, representation offsets,
    certificates and stale/admissible status, or complete-pair epochs);
    ``views`` are the role views aligned with the plan's requirement slots.
    ``graph_epochs[slot]`` is each role's scalar topology revision: it
    advances exactly when the role's valid route endpoints, route validity or
    receiver endpoint mask change (never on rewrapping, which changes only
    image shifts).  ``graph_schedules[slot]`` is the role's prepared streamed
    schedule bound to that revision where a graph plan was declared.
    ``geometry`` is the certified current geometry.  Every numeric leaf is
    dynamic, and no static field is content-derived: ``plan_id`` (the declared
    bundle plan) and ``layout_id`` (the prepared bundle's content-free
    executable layout) are static, while the exact prepared-bundle digest the
    state is bound to is the dynamic ``bundle_identity`` (``bundle_id``
    decodes it on the host).  States of every material of one layout share
    one treedef; a consumer refuses a foreign material numerically with
    ``bundle_identity.equals(bundle.prepared_identity)``.
    """

    sources: tuple[AtomisticSourceState, ...]
    views: tuple[AtomisticNeighborhoodView, ...]
    graph_schedules: tuple[PreparedStreamedRelation | None, ...]
    graph_epochs: Array
    geometry: AtomisticNeighborhoodGeometry
    bundle_identity: IdentityWords
    plan_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    @property
    def bundle_id(self) -> str:
        """Host decoding of the bound prepared-bundle digest."""
        return self.bundle_identity.fingerprint()

    def view(self, slot: int, /) -> AtomisticNeighborhoodView:
        return self.views[int(slot)]

    def _stack(self, values: tuple[Array, ...], /) -> Array:
        if not values:
            return jnp.zeros((0, self.geometry.positions.shape[0]), dtype=jnp.bool_)
        return jnp.stack(values)

    @property
    def _geometry_successful(self) -> Array:
        valid = jnp.all(
            ~self.geometry.active
            | jnp.all(jnp.isfinite(self.geometry.positions), axis=-1),
            axis=-1,
        ) & jnp.all(jnp.isfinite(self.geometry.cell_vectors), axis=(-2, -1))
        return valid & self.geometry.site_successful

    @property
    def successful(self) -> Array:
        """Per case: every role view successful."""
        return self._geometry_successful & jnp.all(
            self._stack(tuple(view.evidence.successful for view in self.views)), axis=0
        )

    @property
    def capacity_failure(self) -> Array:
        return jnp.any(
            self._stack(tuple(view.evidence.capacity_failure for view in self.views)),
            axis=0,
        )

    @property
    def scientific_failure(self) -> Array:
        return ~self._geometry_successful | jnp.any(
            self._stack(tuple(view.evidence.scientific_failure for view in self.views)),
            axis=0,
        )

    @property
    def uninitialized(self) -> Array:
        return jnp.any(
            self._stack(tuple(view.evidence.uninitialized for view in self.views)), axis=0
        )

    @property
    def stale(self) -> Array:
        return jnp.any(
            self._stack(tuple(view.evidence.stale for view in self.views)), axis=0
        )

    @property
    def selection_changed(self) -> Array:
        """Per case: a frozen-epoch binding could not certify a retained selection."""
        return jnp.any(
            self._stack(tuple(view.evidence.selection_changed for view in self.views)),
            axis=0,
        )

    @property
    def capacity_only(self) -> Array:
        """Per case: failures exist and every one is a repairable capacity failure."""
        return (
            self.capacity_failure
            & ~self.scientific_failure
            & ~self.stale
            & ~self.uninitialized
        )

    @property
    def epochs(self) -> Array:
        """``(source, case)`` topology epochs of the accepted sources."""
        if not self.sources:
            return jnp.zeros((0, self.geometry.positions.shape[0]), dtype=jnp.int32)
        return jnp.stack([source.case_epochs for source in self.sources])

    @property
    def rebuilt(self) -> Array:
        if not self.sources:
            return jnp.zeros((0, self.geometry.positions.shape[0]), dtype=jnp.bool_)
        return jnp.stack([source.rebuilt for source in self.sources])


def _identity_site_map(
    system: PreparedAtomisticSystem, particles: ParticleDiscretization, /
) -> bool:
    """Host proof that the interaction sites are exactly the system particles."""
    plan = system.coordinate_map.plan
    sites = plan.sites
    if plan.virtual_rules or sites.capacity != system.capacity:
        return False
    active = np.asarray(particles.active_mask, dtype=np.bool_)
    indices = np.asarray(plan.physical_dof_indices)
    return (
        np.array_equal(np.asarray(sites.site_ids), np.asarray(particles.particle_ids))
        and np.array_equal(np.asarray(sites.active_mask, dtype=np.bool_), active)
        and np.array_equal(np.asarray(sites.physical_mask, dtype=np.bool_), active)
        and np.array_equal(indices[active], np.flatnonzero(active))
    )


def _site_support_schema(system: PreparedAtomisticSystem, /) -> str:
    """Indexed-support schema of this layout's interaction-site slots.

    The site analogue of ``PreparedAtomisticSystem.support_schema_id``: site
    sources and their relations index it, never a prepared site material.
    """
    coordinate_map = system.coordinate_map
    return canonical_fingerprint(
        {
            "kind": "atomistic-interaction-site-support-schema",
            "entity": "interaction_site",
            "capacity": coordinate_map.plan.sites.capacity,
            "ambient_dimension": _DIMENSION,
            "coordinate_dtype": coordinate_map.coordinate_dtype,
        }
    )


def _within(
    relation: ParticleImageRelation, positions: Array, vectors: Array, cutoff: float, /
) -> Array:
    displacement = relation.displacement(positions, vectors)
    squared = jax.lax.stop_gradient(jnp.sum(displacement * displacement, axis=-1))
    return relation.valid & (squared < cutoff * cutoff)


def _repeated_minimum_image(
    relation: ParticleImageRelation, within: Array, self_image: Array, /
) -> Array:
    """Per case: an unordered pair with two images, or a self image, inside cutoff."""
    cases = relation.case_count
    owner = jnp.where(within, relation.route_cases, cases)
    sources = jnp.where(within, relation.source_particle_ids, 0)
    receivers = jnp.where(within, relation.receiver_particle_ids, 0)
    order = jax.lax.stop_gradient(jnp.lexsort((receivers, sources, owner)))
    ordered_owner = owner[order]
    ordered_sources = sources[order]
    ordered_receivers = receivers[order]
    ordered_within = within[order]
    repeated = (
        ordered_within[1:]
        & ordered_within[:-1]
        & (ordered_owner[1:] == ordered_owner[:-1])
        & (ordered_sources[1:] == ordered_sources[:-1])
        & (ordered_receivers[1:] == ordered_receivers[:-1])
    )
    repeats = (
        jnp.zeros((cases,), dtype=jnp.int32)
        .at[jnp.where(repeated, ordered_owner[1:], 0)]
        .add(repeated.astype(jnp.int32))
    )
    inside_self = within & self_image
    selves = (
        jnp.zeros((cases,), dtype=jnp.int32)
        .at[jnp.where(inside_self, relation.route_cases, 0)]
        .add(inside_self.astype(jnp.int32))
    )
    return (repeats > 0) | (selves > 0)


def _case_counts(relation: ParticleImageRelation, mask: Array, /) -> Array:
    return (
        jnp.zeros((relation.case_count,), dtype=jnp.int32)
        .at[jnp.where(mask, relation.route_cases, 0)]
        .add(mask.astype(jnp.int32))
    )


def _same_topology(
    first: ParticleImageRelation,
    first_receivers: Array,
    second: ParticleImageRelation,
    second_receivers: Array,
    /,
) -> Array:
    """Whether two equally shaped relations have the same schedule topology.

    Schedules depend on valid route endpoints, validity and the receiver
    endpoint mask; image shifts (rewrapping) never change them.  This is a
    device comparison of aligned arrays, never a content hash.
    """
    valid = first.valid
    return (
        jnp.all(valid == second.valid)
        & jnp.all(~valid | (first.source_indices == second.source_indices))
        & jnp.all(~valid | (first.receiver_indices == second.receiver_indices))
        & jnp.all(first_receivers == second_receivers)
    )


def _select_leading(mask: Array, new: Array | None, old: Array | None, /) -> Array | None:
    if new is None or old is None:
        if new is not None or old is not None:
            raise ValueError("Candidate and accepted geometry supply different inputs.")
        return None
    shape = mask.shape + (1,) * (new.ndim - mask.ndim)
    return jnp.where(mask.reshape(shape), new, old)


def _complete_pair_routes(
    engine: PreparedAtomisticCompletePairs,
    source: AtomisticCompletePairState,
    cases: int,
    /,
) -> tuple[ParticleImageRelation, Array, Array, Array, Array]:
    """Complete pairs active in the accepted epoch.

    Returns the relation masked by the epoch's reference active set, its
    route mask, (absent) self images, per-case pair counts and maximum
    per-case incident degree.
    """
    base = engine.relation
    reference = source.reference_active.reshape((-1,))
    relation = eqx.tree_at(
        lambda value: value.relation,
        base,
        base.relation.with_valid(
            reference[base.source_indices] & reference[base.receiver_indices]
        ),
    )
    mask = relation.valid
    self_image = jnp.zeros_like(mask)
    counts = _case_counts(relation, mask)
    incident = (
        jnp.zeros((cases * engine.particle_capacity,), dtype=jnp.int32)
        .at[relation.receiver_indices]
        .add(mask.astype(jnp.int32))
        .at[relation.source_indices]
        .add(mask.astype(jnp.int32))
    )
    degree = jnp.max(incident.reshape((cases, -1)), axis=-1)
    return relation, mask, self_image, counts, degree


def _role_receiver_rows(
    source: AtomisticSourceState,
    relation: ParticleImageRelation,
    parent: Array | None,
    /,
) -> ImageReceiverRows | None:
    """Receiver rows of a role's routes, remapped through their parent routes."""
    if (
        not isinstance(source, SegmentedImageVerletState)
        or source.receiver_rows is None
        or parent is None
    ):
        return None
    native = source.receiver_rows
    return ImageReceiverRows(
        native.particle_indices,
        native.valid,
        jnp.where(
            relation.valid,
            native.route_rows[parent],
            relation.route_cases * native.row_capacity,
        ),
        native.row_capacity,
        native.case_count,
    )


_DIMENSION = 3


def _bundle_search_cell(
    plan: AtomisticNeighborhoodBundlePlan, system: PreparedAtomisticSystem, /
) -> tuple[PeriodicCell | None, bool]:
    """The searched cell or finite frame and whether it has periodic support.

    A periodic system is searched in its own prepared cell and a finite system
    with image sources in the plan's explicit nonperiodic frame; complete
    all-pairs sources are refused under periodic support.
    """
    complete_sources = any(
        isinstance(source, AtomisticCompletePairSource) for source in plan.sources
    )
    image_sources = any(
        isinstance(source, AtomisticNeighborhoodSource) for source in plan.sources
    )
    cell = system.cell
    if cell is None:
        cell = plan.frame
        if cell is None and image_sources:
            raise ValueError(
                "A finite system with image sources needs the plan's explicit "
                "nonperiodic frame."
            )
    elif plan.frame is not None:
        raise ValueError("A periodic system is searched in its own prepared cell.")
    if cell is not None and cell.ambient_dimension != _DIMENSION:
        raise ValueError("Atomistic neighborhoods are three-dimensional.")
    periodic = cell is not None and any(cell.periodic_axes)
    if complete_sources and periodic:
        raise ValueError(
            "Complete all-pairs (cutoff=None) requirements have unbounded "
            "periodic support; declare a finite cutoff or an explicit image sum."
        )
    return cell, periodic


def _capacity_levels(
    plan: AtomisticNeighborhoodBundlePlan,
    source_levels: tuple[int, ...] | None,
    role_levels: tuple[int, ...] | None,
    /,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Current source and role ladder levels, defaulting to every first entry."""
    sources = plan.sources
    requirements = plan.requirements
    source_levels_ = (
        (0,) * len(sources) if source_levels is None else tuple(source_levels)
    )
    role_levels_ = (0,) * len(requirements) if role_levels is None else tuple(role_levels)
    if len(source_levels_) != len(sources) or any(
        not 0
        <= level
        < (
            1
            if isinstance(source, AtomisticCompletePairSource)
            else len(source.capacities.entries)
        )
        for level, source in zip(source_levels_, sources, strict=True)
    ):
        raise ValueError("source_levels must index each source capacity ladder.")
    if len(role_levels_) != len(requirements) or any(
        not 0 <= level < (1 if ladder is None else len(ladder.entries))
        for level, ladder in zip(role_levels_, plan.role_capacities, strict=True)
    ):
        raise ValueError("role_levels must index each requirement capacity ladder.")
    return source_levels_, role_levels_


def _role_graph_plans(
    requirements: tuple[AtomisticNeighborhoodRequirement, ...],
    graph_plans: tuple[StreamedRelationPlan | None, ...] | None,
    /,
) -> tuple[StreamedRelationPlan | None, ...]:
    """Slot-aligned graph schedule plans; only directed roles own one."""
    graph = (None,) * len(requirements) if graph_plans is None else tuple(graph_plans)
    if len(graph) != len(requirements):
        raise ValueError("graph_plans must align with the requirement slots.")
    for requirement, graph_plan in zip(requirements, graph, strict=True):
        if (
            graph_plan is not None
            and requirement.counting is not AtomisticPairCounting.DIRECTED
        ):
            raise ValueError(
                f"{requirement.label} is not a directed role; only directed "
                "roles own graph schedules."
            )
    return graph


def _admit_site_material(
    plan: AtomisticNeighborhoodBundlePlan,
    system: PreparedAtomisticSystem,
    cell: PeriodicCell | None,
    periodic: bool,
    identity_sites: bool,
    /,
) -> np.ndarray:
    """Admit this material's minimum-image cutoffs and site bindings.

    A physical-atom source serves interaction-site pairs only under a proven
    identity coordinate map, and topology exceptions are refused with active
    virtual sites.  Returns the coordinate map's physical-site mask.
    """
    for index, requirement in enumerate(plan.requirements):
        source = plan.sources[plan.role_sources[index]]
        if (
            requirement.counting is AtomisticPairCounting.MINIMUM_IMAGE_HALF
            and periodic
            and cell is not None
            and not requirement.bounded_cutoff() < cell.unique_image_radius
        ):
            raise ValueError(
                f"Classical minimum-image {requirement.label} cutoff "
                f"{requirement.cutoff} violates the cell unique-image radius "
                f"{cell.unique_image_radius}; use an explicit image-pair sum."
            )
        if requirement.entity is not source.entity and not identity_sites:
            raise ValueError(
                f"{requirement.label} is bound to the physical-atom source "
                f"{source.name!r}, but the interaction sites are not a proven "
                "identity image of the system particles; bind it to an "
                "independent INTERACTION_SITES source."
            )
    site_plan = system.coordinate_map.plan
    site_physical = np.asarray(site_plan.sites.physical_mask, dtype=np.bool_)
    virtual_sites = np.asarray(site_plan.sites.active_mask, dtype=np.bool_) & (
        ~site_physical
    )
    if (
        any(
            requirement.role is AtomisticNeighborhoodRole.INTERACTION_SITE_PAIRS
            and requirement.exclusions is AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS
            for requirement in plan.requirements
        )
        and np.any(virtual_sites)
        and system.topology.exception_keys.shape[0] > 0
    ):
        raise ValueError(
            "Virtual interaction sites require an explicit site-space pair-exception "
            "contract; topology exceptions are physical-particle pair scales."
        )
    return site_physical


def _complete_pair_engine(
    source: AtomisticCompletePairSource,
    entities: ParticleDiscretization,
    support: str,
    cases: int,
    rank: int,
    /,
) -> tuple[PreparedAtomisticCompletePairs, str, str]:
    """Bounded complete enumeration, its capacity digest and material engine digest."""
    dense = DenseParticleNeighborhoodPlan(
        source.maximum_pairs, name=f"atomistic-complete-pairs-{source.name}"
    ).prepare(entities)
    complete = PreparedAtomisticCompletePairs(
        source,
        dense,
        case_count=cases,
        lattice_rank=rank,
        support_id=support,
    )
    capacity_id = canonical_fingerprint(
        {
            "kind": "atomistic-complete-pair-capacity",
            "maximum_pairs": source.maximum_pairs,
        }
    )
    return complete, capacity_id, _material_complete_pairs_id(source, dense, cases, rank)


def _image_source_engine(
    plan: AtomisticNeighborhoodBundlePlan,
    index: int,
    source: AtomisticNeighborhoodSource,
    level: int,
    cell: PeriodicCell | None,
    entities: ParticleDiscretization,
    support: str,
    /,
) -> tuple[SegmentedImageVerletPlan, str, float, float, str]:
    """Segmented image Verlet engine of image source ``index`` at ``level``.

    Returns the engine indexing ``support``, its capacity digest, the resolved
    search and interaction radii, and the exact material engine digest of the
    canonical bundle payload: the same plan over the prepared material support.
    """
    if cell is None:
        raise RuntimeError("An image source requires a lattice or frame.")
    cases = plan.case_count
    vectors = np.broadcast_to(
        np.asarray(cell.vectors)[None], (cases,) + tuple(cell.vectors.shape)
    )
    periodic_rows = np.broadcast_to(
        np.asarray(cell.periodic_mask, dtype=np.bool_)[None],
        (cases, cell.rank),
    )
    capacity = source.capacities.entries[level]
    radius = plan.search_radii[index]
    interaction = plan.interaction_radii[index]
    if radius is None or interaction is None:
        raise RuntimeError("An image source lacks its resolved radii.")
    search = source.search.search(
        radius,
        np.array(vectors),
        np.array(periodic_rows),
        capacity,
        particle_capacity=entities.capacity,
    )
    if search.radius != radius or search.case_count != cases:
        raise ValueError(f"Source {source.name!r} search binding is inconsistent.")
    verlet = SegmentedImageVerletPlan(
        search,
        interaction,
        source.skin,
        support_id=support,
        receiver_rows=source.receiver_rows,
    )
    material_engine_id = SegmentedImageVerletPlan(
        search,
        interaction,
        source.skin,
        support_id=entities.support.support_id,
        receiver_rows=source.receiver_rows,
    ).plan_id
    return verlet, capacity.capacity_id, radius, interaction, material_engine_id


def _prepare_source(
    plan: AtomisticNeighborhoodBundlePlan,
    index: int,
    level: int,
    cell: PeriodicCell | None,
    entities: ParticleDiscretization,
    support: str,
    /,
) -> tuple[AtomisticSourceEngine, AtomisticNeighborhoodSourceProof, str]:
    """Engine, layout proof and material proof digest of source ``index``.

    The layout proof names the content-free engine indexing ``support``; the
    material proof digests the same facts naming the exact material engine
    and feeds only the bundle's dynamic ``prepared_identity``.
    """
    source = plan.sources[index]
    served = [
        requirement
        for requirement, owner in zip(plan.requirements, plan.role_sources, strict=True)
        if owner == index
    ]
    site_alias = bool(
        source.entity is AtomisticSiteDomain.PHYSICAL_ATOMS
        and any(requirement.entity is not source.entity for requirement in served)
    )
    engine: AtomisticSourceEngine
    radius: float | None
    interaction: float | None
    skin: float | None
    margins: list[float | None]
    if isinstance(source, AtomisticCompletePairSource):
        complete, capacity_id, material_engine_id = _complete_pair_engine(
            source,
            entities,
            support,
            plan.case_count,
            0 if cell is None else cell.rank,
        )
        engine = complete
        radius = interaction = skin = None
        margins = [None for _ in served]
        receiver_group = None
        engine_id = complete.prepared_id
    else:
        verlet, capacity_id, radius, interaction, material_engine_id = (
            _image_source_engine(plan, index, source, level, cell, entities, support)
        )
        engine = verlet
        skin = source.skin
        margins = [radius - skin - requirement.bounded_cutoff() for requirement in served]
        receiver_group = source.receiver_group
        engine_id = verlet.plan_id
    facts = {
        "source": source.name,
        "entity": source.entity.value,
        "served": [requirement.requirement_id for requirement in served],
        "countings": [requirement.counting.value for requirement in served],
        "cutoffs": [requirement.cutoff for requirement in served],
        "coverage_margins": margins,
        "search_radius": radius,
        "interaction_radius": interaction,
        "skin": skin,
        "receiver_group": receiver_group,
        "site_alias": site_alias,
        "capacity": capacity_id,
        "engine": engine_id,
    }
    proof = AtomisticNeighborhoodSourceProof(
        source.name,
        source.entity,
        tuple(requirement.requirement_id for requirement in served),
        tuple(requirement.counting.value for requirement in served),
        tuple(requirement.cutoff for requirement in served),
        tuple(margins),
        radius,
        interaction,
        skin,
        receiver_group,
        site_alias,
        capacity_id,
        engine_id,
        canonical_fingerprint({"kind": "atomistic-neighborhood-source-proof", **facts}),
    )
    material_proof = canonical_fingerprint(
        {
            "kind": "atomistic-neighborhood-source-proof",
            **facts,
            "engine": material_engine_id,
        }
    )
    return engine, proof, material_proof


def _prepare_role(
    plan: AtomisticNeighborhoodBundlePlan,
    index: int,
    level: int,
    graph_plan: StreamedRelationPlan | None,
    engine: AtomisticSourceEngine,
    engine_id: str,
    /,
) -> tuple[
    str | None, PreparedImageRadiusTiers | None, PreparedImageNeighborSelection | None
]:
    """Graph-schedule owner and tier or selection view of role ``index``.

    Complete-pair roles use their source's enumeration directly; a selected
    role prepares its selection in its role storage and an unselected role
    one radius tier.
    """
    requirement = plan.requirements[index]
    source_index = plan.role_sources[index]
    cases = plan.case_count
    ladder = plan.role_capacities[index]
    role_capacity = None if ladder is None else ladder.entries[level]
    owner = (
        None
        if graph_plan is None
        else canonical_fingerprint(
            {
                "kind": "atomistic-neighborhood-graph-owner",
                "requirement": requirement.requirement_id,
                "role_capacity": None
                if role_capacity is None
                else role_capacity.capacity_id,
                "engine": engine_id,
                "case_count": cases,
            }
        )
    )
    if isinstance(engine, PreparedAtomisticCompletePairs):
        return owner, None, None
    source = plan.sources[source_index]
    radius = plan.search_radii[source_index]
    if (
        role_capacity is None
        or radius is None
        or not isinstance(source, AtomisticNeighborhoodSource)
    ):
        raise RuntimeError("A bounded role lacks its storage or source radius.")
    if requirement.selection is not None:
        selection = PreparedImageNeighborSelection(
            requirement.selection,
            ImageSelectionCapacity(
                maximum_routes=role_capacity.maximum_routes,
                maximum_degree=role_capacity.maximum_degree,
                maximum_case_routes=role_capacity.maximum_case_routes,
            ),
            source_radius=radius,
            case_count=cases,
        )
        return owner, None, selection
    if requirement.counting is AtomisticPairCounting.DIRECTED:
        edges = role_capacity.maximum_routes
        degree = role_capacity.maximum_degree
        case_edges = role_capacity.maximum_case_routes
    else:
        # Half views compact a candidate tier that is a subset of the
        # source routes; its storage is the source's declared storage.
        search_capacity = engine.search.capacity
        edges = engine.search.route_capacity
        degree = search_capacity.maximum_degree
        case_edges = search_capacity.maximum_case_edges
    tier = ImageRadiusTier(
        requirement.label,
        requirement.bounded_cutoff(),
        maximum_edges=edges,
        maximum_degree=degree,
        maximum_case_edges=case_edges,
    )
    tiers = ImageRadiusTierPlan((tier,), skin=source.skin).prepare(
        source_radius=radius, case_count=cases
    )
    return owner, tiers, None


@final
class PreparedAtomisticNeighborhoodBundle(StrictModule, NonTrainableState):
    """Bundle plan bound to one prepared system, cell/frame and capacity levels.

    Preparation resolves every requirement to a static view slot and source,
    proves source sharing, prepares one native ``SegmentedImageVerletPlan``
    per image source and one bounded complete enumeration per complete-pair
    source, the per-role tier/selection views and the optional per-role graph
    schedule plans (``graph_plans`` aligned with the requirement slots, only
    for directed roles).  ``initialize``, ``refresh``, ``certify`` and
    ``select_cases`` are pure and traceable; ``grow_capacities`` and
    ``rebase`` are host lifecycle transactions.  Source sharing never implies
    graph schedule sharing: every role owns its own schedule and revision.

    The treedef is content-free, so bundles of every material of one layout
    share one executable.  Engines and their relations index entity support
    schemas (``system.support_schema_id`` for physical atoms,
    ``site_support_id`` for interaction sites), so engine, proof, graph-owner
    and relation-schema identities are layout.  Material particle/site IDs,
    active masks and site-to-particle indices are dynamic arrays.
    ``site_alias`` is the declared plan fact that a physical-atom source serves
    interaction-site pairs (a code-path selector); preparation refuses a
    material whose coordinate map is not a proven identity.  ``layout_id``
    (static) digests the plan, case count, levels and capacities, layout
    proofs, tiers, selections, graph plans, cell and ``system.layout_id``.  The
    exact canonical prepared digest (system material, cell, levels, material
    source proofs, tiers, selections, graph plans) is the dynamic
    ``prepared_identity``; ``prepared_id`` decodes it on the host.
    """

    plan: AtomisticNeighborhoodBundlePlan
    system: PreparedAtomisticSystem
    cell: PeriodicCell | None
    engines: tuple[AtomisticSourceEngine, ...]
    tiers: tuple[PreparedImageRadiusTiers | None, ...]
    selections: tuple[PreparedImageNeighborSelection | None, ...]
    graph_plans: tuple[StreamedRelationPlan | None, ...]
    proofs: tuple[AtomisticNeighborhoodSourceProof, ...]
    particle_ids: Array
    particle_active: Array
    site_ids: Array
    site_active: Array
    site_physical_indices: Array
    prepared_identity: IdentityWords
    source_levels: tuple[int, ...] = eqx.field(static=True)
    role_levels: tuple[int, ...] = eqx.field(static=True)
    graph_owners: tuple[str | None, ...] = eqx.field(static=True)
    site_alias: bool = eqx.field(static=True)
    site_support_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: AtomisticNeighborhoodBundlePlan,
        system: PreparedAtomisticSystem,
        /,
        *,
        graph_plans: tuple[StreamedRelationPlan | None, ...] | None = None,
        source_levels: tuple[int, ...] | None = None,
        role_levels: tuple[int, ...] | None = None,
    ) -> None:
        sources = plan.sources
        requirements = plan.requirements
        cell, periodic = _bundle_search_cell(plan, system)
        source_levels_, role_levels_ = _capacity_levels(plan, source_levels, role_levels)
        graph = _role_graph_plans(requirements, graph_plans)
        # Host preparation of the material supports: their digests feed only
        # the dynamic prepared identity; engines index the support schemas.
        particles = system.particle_discretization()
        site_particles = system.site_discretization()
        identity_sites = _identity_site_map(system, particles)
        site_physical = _admit_site_material(plan, system, cell, periodic, identity_sites)
        # The only cross-entity binding the plan admits is interaction-site
        # pairs on a physical-atom source; declaring it selects the physical
        # code path and is proven for this material above.
        alias = any(
            requirement.entity is not sources[plan.role_sources[index]].entity
            for index, requirement in enumerate(requirements)
        )
        site_support = _site_support_schema(system)
        cases = plan.case_count
        engines: list[AtomisticSourceEngine] = []
        proofs: list[AtomisticNeighborhoodSourceProof] = []
        material_proofs: list[str] = []
        for index, source in enumerate(sources):
            site_source = source.entity is AtomisticSiteDomain.INTERACTION_SITES
            engine, proof, material_proof = _prepare_source(
                plan,
                index,
                source_levels_[index],
                cell,
                site_particles if site_source else particles,
                site_support if site_source else system.support_schema_id,
            )
            engines.append(engine)
            proofs.append(proof)
            material_proofs.append(material_proof)
        tiers: list[PreparedImageRadiusTiers | None] = []
        selections: list[PreparedImageNeighborSelection | None] = []
        owners: list[str | None] = []
        for index, graph_plan in enumerate(graph):
            source_index = plan.role_sources[index]
            owner, tier, selection = _prepare_role(
                plan,
                index,
                role_levels_[index],
                graph_plan,
                engines[source_index],
                proofs[source_index].engine_id,
            )
            owners.append(owner)
            tiers.append(tier)
            selections.append(selection)
        role_capacity_ids = [
            None if ladder is None else ladder.entries[level].capacity_id
            for ladder, level in zip(plan.role_capacities, role_levels_, strict=True)
        ]
        tier_ids = [None if value is None else value.prepared_id for value in tiers]
        selection_ids = [
            None if value is None else value.selection_id for value in selections
        ]
        graph_plan_ids = [None if value is None else value.plan_id for value in graph]
        cell_id = None if cell is None else cell.cell_id
        prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-atomistic-neighborhood-bundle",
                "plan": plan.plan_id,
                "system": system.prepared_id,
                "cell": cell_id,
                "source_levels": list(source_levels_),
                "role_levels": list(role_levels_),
                "role_capacities": role_capacity_ids,
                "proofs": material_proofs,
                "tiers": tier_ids,
                "selections": selection_ids,
                "graph_plans": graph_plan_ids,
            }
        )
        layout = canonical_fingerprint(
            {
                "kind": "atomistic-neighborhood-bundle-layout",
                "plan": plan.plan_id,
                "case_count": cases,
                "system": system.layout_id,
                "cell": cell_id,
                "source_levels": list(source_levels_),
                "role_levels": list(role_levels_),
                "role_capacities": role_capacity_ids,
                "proofs": [proof.proof_id for proof in proofs],
                "tiers": tier_ids,
                "selections": selection_ids,
                "graph_plans": graph_plan_ids,
                "site_alias": alias,
                "site_support": site_support,
            }
        )
        self.plan = plan
        self.system = system
        self.cell = cell
        self.engines = tuple(engines)
        self.tiers = tuple(tiers)
        self.selections = tuple(selections)
        self.graph_plans = graph
        self.proofs = tuple(proofs)
        self.particle_ids = particles.particle_ids
        self.particle_active = jnp.asarray(particles.active_mask, dtype=jnp.bool_)
        self.site_ids = site_particles.particle_ids
        self.site_active = jnp.asarray(site_particles.active_mask, dtype=jnp.bool_)
        self.site_physical_indices = jnp.asarray(
            np.where(
                site_physical,
                np.asarray(system.coordinate_map.plan.physical_dof_indices),
                -1,
            ),
            dtype=jnp.int32,
        )
        self.prepared_identity = prepared_identity
        self.source_levels = source_levels_
        self.role_levels = role_levels_
        self.graph_owners = tuple(owners)
        self.site_alias = alias
        self.site_support_id = site_support
        self.layout_id = layout

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical prepared-bundle digest."""
        return self.prepared_identity.fingerprint()

    @property
    def lattice_rank(self) -> int:
        return 0 if self.cell is None else self.cell.rank

    def slot(self, requirement: AtomisticNeighborhoodRequirement, /) -> int:
        return self.plan.slot(requirement)

    def view(
        self,
        state: AtomisticNeighborhoodBundleState,
        requirement: AtomisticNeighborhoodRequirement,
        /,
    ) -> AtomisticNeighborhoodView:
        """Return the view of exactly ``requirement`` (layout-checked, traceable)."""
        self._require_state(state)
        return state.views[self.plan.slot(requirement)]

    def graph_schedule(
        self,
        state: AtomisticNeighborhoodBundleState,
        requirement: AtomisticNeighborhoodRequirement,
        /,
    ) -> PreparedStreamedRelation:
        """Return the streamed schedule bound to ``requirement``'s graph revision."""
        self._require_state(state)
        schedule = state.graph_schedules[self.plan.slot(requirement)]
        if schedule is None:
            raise ValueError(f"{requirement.label} declares no graph plan.")
        return schedule

    def graph_epoch(
        self,
        state: AtomisticNeighborhoodBundleState,
        requirement: AtomisticNeighborhoodRequirement,
        /,
    ) -> Array:
        """Scalar topology revision of ``requirement``'s view."""
        self._require_state(state)
        return state.graph_epochs[self.plan.slot(requirement)]

    def capacity(self, slot: int, /) -> AtomisticRoleCapacity:
        """Current role storage of a bounded ``slot``."""
        index = int(slot)
        ladder = self.plan.role_capacities[index]
        if ladder is None:
            raise ValueError("A complete role has no role capacity ladder.")
        return ladder.entries[self.role_levels[index]]

    def source_capacity(self, source: int, /) -> ParticleImageCapacity:
        """Current search storage of image source ``source``."""
        index = int(source)
        declared = self.plan.sources[index]
        if not isinstance(declared, AtomisticNeighborhoodSource):
            raise ValueError("A complete-pair source has no image capacity ladder.")
        return declared.capacities.entries[self.source_levels[index]]

    def _site_source(self, index: int, /) -> bool:
        return self.plan.sources[index].entity is AtomisticSiteDomain.INTERACTION_SITES

    def geometry(
        self,
        positions: ArrayLike,
        /,
        *,
        cell_vectors: ArrayLike | None = None,
        image_counts: ArrayLike | None = None,
        active: ArrayLike | None = None,
        sites: AtomisticInteractionSiteState | None = None,
        site_image_counts: ArrayLike | None = None,
    ) -> AtomisticNeighborhoodGeometry:
        """Validate and bind current case-batched geometry.

        ``sites`` is a realized interaction-site state with a leading case axis
        on its positions; it is required exactly when a site source exists.
        Sites realized by another coordinate map are refused through
        ``site_successful`` (identity words), never by a traced string check.
        Without a cell or frame the lattice has rank zero.
        """
        dtype = jnp.dtype(self.system.plan.coordinate_dtype)
        cases = self.plan.case_count
        count = self.system.capacity
        dimension = _DIMENSION
        rank = self.lattice_rank
        value = jnp.asarray(positions, dtype=dtype)
        if value.shape != (cases, count, dimension):
            raise ValueError(f"positions must have shape {(cases, count, dimension)}.")
        if cell_vectors is None:
            vectors = (
                jnp.zeros((cases, 0, dimension), dtype=dtype)
                if self.cell is None
                else jnp.broadcast_to(
                    self.cell.vectors.astype(dtype)[None], (cases, rank, dimension)
                )
            )
        else:
            vectors = jnp.asarray(cell_vectors, dtype=dtype)
            if vectors.shape != (cases, rank, dimension):
                raise ValueError(
                    f"cell_vectors must have shape {(cases, rank, dimension)}."
                )
        counts = None
        if image_counts is not None:
            counts = jnp.asarray(image_counts)
            if counts.shape != (cases, count, rank) or not jnp.issubdtype(
                counts.dtype, jnp.integer
            ):
                raise ValueError("image_counts must be integers of shape (case, N, r).")
        mask = jnp.broadcast_to(self.particle_active[None], (cases, count))
        if active is not None:
            requested = jnp.asarray(active, dtype=jnp.bool_)
            if requested.shape != (cases, count):
                raise ValueError("active must have shape (case, N).")
            mask = mask & requested
        needs_sites = any(
            self._site_source(index) for index in range(len(self.plan.sources))
        )
        site_count = self.site_ids.shape[0]
        if sites is None:
            if needs_sites:
                raise ValueError("A site source requires realized interaction sites.")
            if site_image_counts is not None:
                raise ValueError("site_image_counts require interaction sites.")
            return AtomisticNeighborhoodGeometry(
                value,
                mask,
                vectors,
                counts,
                None,
                None,
                None,
                jnp.ones((cases,), dtype=jnp.bool_),
            )
        if not needs_sites:
            raise ValueError("No source searches interaction sites.")
        if not isinstance(sites, AtomisticInteractionSiteState):
            raise TypeError("sites must be an AtomisticInteractionSiteState.")
        site_positions = jnp.asarray(sites.positions, dtype=dtype)
        if site_positions.shape != (cases, site_count, dimension):
            raise ValueError(
                f"Site positions must have shape {(cases, site_count, dimension)}."
            )
        site_mask = jnp.broadcast_to(
            jnp.asarray(sites.active_mask, dtype=jnp.bool_), (cases, site_count)
        ) & jnp.broadcast_to(self.site_active[None], (cases, site_count))
        site_counts = None
        if site_image_counts is not None:
            site_counts = jnp.asarray(site_image_counts)
            if site_counts.shape != (cases, site_count, rank) or not jnp.issubdtype(
                site_counts.dtype, jnp.integer
            ):
                raise ValueError(
                    "site_image_counts must be integers of shape (case, S, r)."
                )
        # Sites realized by another coordinate map are a failed site
        # realization (a scientific failure of every site role), decided by
        # identity words so the binding stays traceable.
        same_map = sites.coordinate_map_identity.equals(
            self.system.coordinate_map.prepared_identity
        )
        site_successful = jnp.asarray(sites.successful, dtype=jnp.bool_) & same_map
        return AtomisticNeighborhoodGeometry(
            value,
            mask,
            vectors,
            counts,
            site_positions,
            site_mask,
            site_counts,
            jnp.broadcast_to(site_successful, (cases,)),
        )

    def _require_geometry(self, geometry: AtomisticNeighborhoodGeometry, /) -> None:
        if not isinstance(geometry, AtomisticNeighborhoodGeometry):
            raise TypeError("geometry must be an AtomisticNeighborhoodGeometry.")
        expected = (self.plan.case_count, self.system.capacity, _DIMENSION)
        if geometry.positions.shape != expected:
            raise ValueError("Geometry belongs to another system or case layout.")
        if geometry.cell_vectors.shape[1:] != (self.lattice_rank, _DIMENSION):
            raise ValueError("Geometry lattice differs from the prepared cell.")

    def _require_state(self, state: AtomisticNeighborhoodBundleState, /) -> None:
        """Static layout admission; the material is bound by ``bundle_identity``."""
        if not isinstance(state, AtomisticNeighborhoodBundleState):
            raise TypeError("state must be an AtomisticNeighborhoodBundleState.")
        if state.layout_id != self.layout_id:
            raise ValueError(
                "Bundle state belongs to another prepared bundle layout; rebase it first."
            )

    def _entity(
        self, index: int, geometry: AtomisticNeighborhoodGeometry, /
    ) -> tuple[Array, Array, Array | None]:
        """Positions, active mask and image counts of source ``index``'s entities."""
        if self._site_source(index):
            positions = geometry.site_positions
            active = geometry.site_active
            if positions is None or active is None:
                raise ValueError("A site source requires realized interaction sites.")
            return positions, active, geometry.site_image_counts
        return geometry.positions, geometry.active, geometry.image_counts

    def _search_inputs(
        self,
        index: int,
        geometry: AtomisticNeighborhoodGeometry,
        receivers: Mapping[str, ImageReceiverSubset] | None,
        /,
    ) -> tuple[Array, _ImageSearchInputs]:
        source = self.plan.sources[index]
        cell = self.cell
        if not isinstance(source, AtomisticNeighborhoodSource) or cell is None:
            raise RuntimeError("Search inputs belong to image sources.")
        cases = self.plan.case_count
        positions, active, counts = self._entity(index, geometry)
        ids = self.site_ids if self._site_source(index) else self.particle_ids
        subset = None
        if source.receiver_group is not None:
            if receivers is None or source.receiver_group not in receivers:
                raise ValueError(
                    f"Source {source.name!r} needs receivers for group "
                    f"{source.receiver_group!r}."
                )
            subset = receivers[source.receiver_group]
        return positions, {
            "active": active,
            "stable_ids": jnp.broadcast_to(ids[None], (cases, ids.shape[0])),
            "vectors": geometry.cell_vectors,
            "origin": jnp.broadcast_to(
                cell.origin.astype(positions.dtype)[None], (cases, _DIMENSION)
            ),
            "periodic": jnp.broadcast_to(cell.periodic_mask[None], (cases, cell.rank)),
            "image_counts": counts,
            "receivers": subset,
        }

    def _complete_inputs(
        self, index: int, geometry: AtomisticNeighborhoodGeometry, /
    ) -> tuple[Array, Array]:
        positions, active, _ = self._entity(index, geometry)
        finite = jnp.all(jnp.isfinite(positions), axis=-1)
        return active, jnp.any(active & ~finite, axis=-1)

    def _require_receivers(
        self, receivers: Mapping[str, ImageReceiverSubset] | None, /
    ) -> None:
        groups = {
            source.receiver_group
            for source in self.plan.sources
            if isinstance(source, AtomisticNeighborhoodSource)
            and source.receiver_group is not None
        }
        supplied = set() if receivers is None else set(receivers)
        if supplied != groups:
            raise ValueError(
                f"Receiver subsets {sorted(supplied)!r} differ from the declared "
                f"groups {sorted(groups)!r}."
            )

    def _role_receivers(
        self, index: int, sources: tuple[AtomisticSourceState, ...], /
    ) -> Array:
        """Flat receiver endpoint mask of the epoch topology serving role ``index``."""
        source = sources[self.plan.role_sources[index]]
        if isinstance(source, AtomisticCompletePairState):
            return source.reference_active.reshape((-1,))
        return source.reference_active_mask.reshape((-1,))

    def _physical_site_pairs(
        self,
        pairs: ParticlePairRelation,
        requirement: AtomisticNeighborhoodRequirement,
        /,
    ) -> tuple[ParticlePairRelation, Array]:
        """Map interaction-site pairs to canonical physical particle pairs.

        Each site's physical particle is the coordinate map's declared
        ``physical_dof_indices`` entry (``-1`` for virtual sites).  Endpoints
        are re-ordered by physical stable ID, which need not follow the site
        ID order, and case offsets are preserved.  Routes with a virtual
        endpoint are invalid here; two physical sites declaring the same
        particle are returned as per-route identity conflicts.
        """
        site_count = self.site_ids.shape[0]
        count = self.system.capacity
        cases = self.plan.case_count
        left_case = pairs.left_indices // site_count
        right_case = pairs.right_indices // site_count
        left = self.site_physical_indices[pairs.left_indices % site_count]
        right = self.site_physical_indices[pairs.right_indices % site_count]
        endpoints = pairs.valid & (left >= 0) & (right >= 0)
        conflict = endpoints & ((left == right) | (left_case != right_case))
        physical = endpoints & ~conflict
        left_ids = self.particle_ids[jnp.maximum(left, 0)]
        right_ids = self.particle_ids[jnp.maximum(right, 0)]
        swap = left_ids > right_ids
        offset = left_case * count
        low = offset + jnp.where(swap, right, left)
        high = offset + jnp.where(swap, left, right)
        support = self.system.support_schema_id
        relation = ParticlePairRelation(
            EdgeRelation(
                jnp.where(physical, low, 0),
                jnp.where(physical, high, 0),
                source_size=cases * count,
                target_size=cases * count,
                valid=physical,
            ),
            jnp.where(physical, jnp.minimum(left_ids, right_ids), 0),
            jnp.where(physical, jnp.maximum(left_ids, right_ids), 1),
            source_support_id=support,
            target_support_id=support,
            same_set=True,
            unordered=True,
            relation_schema_id=canonical_fingerprint(
                {
                    "kind": "atomistic-site-physical-pair-relation",
                    "parent": pairs.relation_schema_id,
                    "requirement": requirement.requirement_id,
                    "support": support,
                }
            ),
        )
        return relation, conflict

    def _pair_once_view(
        self,
        requirement: AtomisticNeighborhoodRequirement,
        relation: ParticleImageRelation,
        self_image: Array,
        sites: bool,
        initialized: Array,
        identity_conflict: Array,
        /,
    ) -> tuple[ParticlePairRelation, Array | None, Array | None, Array]:
        """Pair-once relation of a pair role and its topology exception scales.

        Self images are invalid pairs.  Under topology exceptions, site pairs
        are first mapped to physical particle pairs; keyed physical pairs take
        the topology's Lennard-Jones and electrostatic scales, and unresolved
        keys or site identity conflicts extend ``identity_conflict``.
        """
        support = relation.support_id
        pair_relation = ParticlePairRelation(
            relation.relation.with_valid(~self_image),
            relation.source_particle_ids,
            relation.receiver_particle_ids,
            source_support_id=support,
            target_support_id=support,
            same_set=True,
            unordered=True,
            relation_schema_id=canonical_fingerprint(
                {
                    "kind": "atomistic-pair-once-relation",
                    "parent": relation.relation_schema_id,
                    "requirement": requirement.requirement_id,
                }
            ),
        )
        if requirement.exclusions is not AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS:
            return pair_relation, None, None, identity_conflict
        if sites:
            physical, conflict = self._physical_site_pairs(pair_relation, requirement)
        else:
            physical, conflict = (
                pair_relation,
                jnp.zeros_like(pair_relation.valid),
            )
        keys = self.system.pair_key_space.keys(physical)
        lj, coulomb = self.system.topology.pair_scales(keys.keys)
        scaled = physical.valid & keys.valid
        # A valid site pair without two physical endpoints (a virtual
        # site, admitted only without topology exceptions) keeps unit
        # scales; it is never matched against physical exception keys.
        unscaled = pair_relation.valid & ~physical.valid & ~conflict
        lennard_jones = jnp.where(scaled, lj, jnp.where(unscaled, 1.0, 0.0))
        electrostatic = jnp.where(scaled, coulomb, jnp.where(unscaled, 1.0, 0.0))
        identity_conflict = (
            identity_conflict
            | (initialized & ~keys.successful)
            | (initialized & (_case_counts(relation, conflict) > 0))
        )
        return pair_relation, lennard_jones, electrostatic, identity_conflict

    def _view(
        self,
        index: int,
        source: AtomisticSourceState,
        geometry: AtomisticNeighborhoodGeometry,
        /,
    ) -> AtomisticNeighborhoodView:
        """Resolve role ``index`` from its source epoch under ``geometry``.

        The selected, directed-tier and half-tier image branches stay inline
        as one ordered kernel: each binds the same route and evidence values
        (relation, cutoff mask, self images, counts, overflows) consumed
        positionally by ``AtomisticRoleEvidence``, and their traced operation
        order (the tier's own cutoff mask for directed views, the
        stop-gradient ``_within`` mask for half views) is the view's numerical
        contract.  Complete-pair masking, the pair-once/exception phase and
        receiver-row remapping are separate phases.
        """
        requirement = self.plan.requirements[index]
        source_index = self.plan.role_sources[index]
        engine = self.engines[source_index]
        cases = self.plan.case_count
        sites = self._site_source(source_index)
        positions = geometry.site_positions if sites else geometry.positions
        if positions is None:
            raise ValueError("A site source requires realized interaction sites.")
        flat = positions.reshape((-1, positions.shape[-1]))
        vectors = geometry.cell_vectors
        initialized = source.initialized
        false = jnp.zeros((cases,), dtype=jnp.bool_)
        capped = jnp.zeros((cases,), dtype=jnp.int32)
        reversal_open = identity_conflict = repeated = representation = false
        route_overflow = degree_overflow = false
        parent: Array | None = None
        pair_relation = None
        lennard_jones = electrostatic = None
        selection = self.selections[index]
        selection_evidence: ImageSelectionEvidence | None = None
        if isinstance(engine, PreparedAtomisticCompletePairs):
            if not isinstance(source, AtomisticCompletePairState):
                raise TypeError("A complete-pair source carries a complete-pair state.")
            relation, mask, self_image, required, degree = _complete_pair_routes(
                engine, source, cases
            )
            stored = required
        elif selection is not None:
            if not isinstance(source, SegmentedImageVerletState):
                raise TypeError("An image source carries a segmented Verlet state.")
            candidate = source.neighborhood
            chosen = selection.select(
                candidate.relation,
                flat,
                vectors,
                evidence=candidate.evidence,
                certified=initialized & ~source.stale,
            )
            relation = chosen.relation
            parent = chosen.parent_routes
            mask = relation.valid
            self_image = relation.valid & (
                relation.source_indices == relation.receiver_indices
            )
            evidence = chosen.evidence
            selection_evidence = evidence
            required = evidence.selected_routes
            stored = evidence.stored_routes
            degree = evidence.selected_degree
            route_overflow = evidence.route_overflow
            degree_overflow = evidence.degree_overflow
            capped = evidence.capped_receivers
            representation = candidate.evidence.representation_overflow
        else:
            tiers = self.tiers[index]
            if tiers is None or not isinstance(source, SegmentedImageVerletState):
                raise RuntimeError("An unselected image role lacks its prepared tier.")
            offsets = source.representation_offsets.reshape((-1, vectors.shape[1]))
            tier = (
                tiers.views(source.reference)
                .tiers[0]
                .with_representation_offsets(offsets)
            )
            representation = tier.evidence.representation_overflow
            degree = tier.evidence.required_degree
            if requirement.counting is AtomisticPairCounting.DIRECTED:
                relation = tier.relation
                parent = tier.parent_routes
                mask = tier.cutoff_mask(flat, vectors)
                self_image = relation.valid & (
                    relation.source_indices == relation.receiver_indices
                )
                required = tier.evidence.required_edges
                stored = tier.evidence.stored_edges
                route_overflow = tier.evidence.edge_overflow
                degree_overflow = tier.evidence.degree_overflow
            else:
                capacity = self.capacity(index)
                half = tier.relation.canonical_half(
                    capacity=(
                        capacity.maximum_routes
                        if capacity.maximum_case_routes is None
                        else None
                    ),
                    case_capacity=capacity.maximum_case_routes,
                )
                relation = half.relation
                mask = _within(relation, flat, vectors, requirement.bounded_cutoff())
                self_image = half.self_image
                required = half.required_routes
                stored = half.stored_routes
                route_overflow = half.overflow
                degree_overflow = degree > capacity.maximum_degree
                # Truncated source storage is not reversal closed; that is the
                # source capacity failure, not a scientific one.
                reversal_open = half.reversal_open & ~source.capacity_failure
                identity_conflict = half.identity_conflict
                if requirement.counting is AtomisticPairCounting.MINIMUM_IMAGE_HALF:
                    repeated = _repeated_minimum_image(relation, mask, self_image)
        if requirement.counting in (
            AtomisticPairCounting.MINIMUM_IMAGE_HALF,
            AtomisticPairCounting.COMPLETE_PAIRS,
        ):
            pair_relation, lennard_jones, electrostatic, identity_conflict = (
                self._pair_once_view(
                    requirement,
                    relation,
                    self_image,
                    sites,
                    initialized,
                    identity_conflict,
                )
            )
        rows = _role_receiver_rows(source, relation, parent)
        entity_failure = initialized & ~geometry.site_successful if sites else false
        evidence = AtomisticRoleEvidence(
            required,
            stored,
            degree,
            initialized & route_overflow,
            initialized & degree_overflow,
            source.capacity_failure,
            source.scientific_failure,
            initialized & reversal_open,
            initialized & identity_conflict,
            initialized & repeated,
            initialized & representation,
            entity_failure,
            capped,
            initialized,
            initialized & source.stale,
            source.case_epochs,
            source.rebuilt,
            false,
            selection=selection_evidence,
        )
        counting = requirement.counting
        return AtomisticNeighborhoodView(
            relation,
            mask,
            self_image,
            rows,
            pair_relation,
            lennard_jones,
            electrostatic,
            evidence,
            requirement.owner,
            requirement.role,
            requirement.entity,
            counting,
            selection is not None,
            1 if counting is AtomisticPairCounting.DIRECTED else 2,
            requirement.cutoff,
            self.plan.sources[source_index].name,
            requirement.requirement_id,
        )

    def _views(
        self,
        sources: tuple[AtomisticSourceState, ...],
        geometry: AtomisticNeighborhoodGeometry,
        /,
    ) -> tuple[AtomisticNeighborhoodView, ...]:
        return tuple(
            self._view(index, sources[self.plan.role_sources[index]], geometry)
            for index in range(len(self.plan.requirements))
        )

    def _schedule(
        self,
        index: int,
        view: AtomisticNeighborhoodView,
        receivers: Array,
        epoch: Array,
        /,
    ) -> PreparedStreamedRelation | None:
        """Prepare one role's capacity-bounded schedule at revision ``epoch``.

        Capacity-bounded preparation has one array layout eagerly and inside a
        compiled rebuild, so retained and refreshed schedules are
        interchangeable under ``lax.cond``.  Route positions are the stable
        route IDs of the canonically ordered view relation.
        """
        graph_plan = self.graph_plans[index]
        owner = self.graph_owners[index]
        if graph_plan is None or owner is None:
            return None
        relation = view.relation
        return graph_plan.prepare(
            relation.relation,
            owner_id=owner,
            epoch=epoch,
            stable_route_ids=jnp.arange(relation.capacity, dtype=jnp.int32),
            receiver_valid=receivers,
            capacity_bounded=True,
        )

    def _fresh_graphs(
        self,
        views: tuple[AtomisticNeighborhoodView, ...],
        sources: tuple[AtomisticSourceState, ...],
        epochs: Array,
        /,
    ) -> tuple[PreparedStreamedRelation | None, ...]:
        return tuple(
            self._schedule(
                index, view, self._role_receivers(index, sources), epochs[index]
            )
            for index, view in enumerate(views)
        )

    def _state(
        self,
        sources: tuple[AtomisticSourceState, ...],
        views: tuple[AtomisticNeighborhoodView, ...],
        schedules: tuple[PreparedStreamedRelation | None, ...],
        epochs: Array,
        geometry: AtomisticNeighborhoodGeometry,
        identity: IdentityWords,
        /,
    ) -> AtomisticNeighborhoodBundleState:
        """Assemble a state bound to ``identity``.

        Traced transitions carry the input state's identity forward, so a
        foreign-material state of this layout is never re-labelled as this
        bundle's material by a refresh, bind or case selection.
        """
        return AtomisticNeighborhoodBundleState(
            sources,
            views,
            schedules,
            epochs,
            geometry,
            identity,
            self.plan.plan_id,
            self.layout_id,
        )

    def _carried_identity(
        self,
        candidate: AtomisticNeighborhoodBundleState,
        accepted: AtomisticNeighborhoodBundleState,
        /,
    ) -> IdentityWords:
        """Identity of a per-case merge: this bundle's only if both inputs carry it.

        A foreign input's own identity is carried otherwise, so the merged
        state stays refused by every consumer that checks the words.
        """
        foreign = ~candidate.bundle_identity.equals(self.prepared_identity)
        return IdentityWords(
            jnp.where(
                foreign[..., None],
                candidate.bundle_identity.words,
                accepted.bundle_identity.words,
            )
        )

    def initialize(
        self,
        geometry: AtomisticNeighborhoodGeometry,
        /,
        *,
        receivers: Mapping[str, ImageReceiverSubset] | None = None,
        cases: ArrayLike | None = None,
    ) -> AtomisticNeighborhoodBundleState:
        """Form every source for the ``cases`` mask (default every case).

        Cases outside the mask own no epoch and report ``uninitialized``.
        Every role starts at graph revision zero.
        """
        self._require_geometry(geometry)
        self._require_receivers(receivers)
        count = self.plan.case_count
        selected = (
            jnp.ones((count,), dtype=jnp.bool_)
            if cases is None
            else jnp.broadcast_to(jnp.asarray(cases, dtype=jnp.bool_), (count,))
        )
        sources: list[AtomisticSourceState] = []
        for index, engine in enumerate(self.engines):
            if isinstance(engine, PreparedAtomisticCompletePairs):
                active, nonfinite = self._complete_inputs(index, geometry)
                sources.append(
                    AtomisticCompletePairState(
                        active & selected[:, None],
                        selected,
                        jnp.zeros((count,), dtype=jnp.int32),
                        selected,
                        selected.astype(jnp.int32),
                        jnp.zeros((count,), dtype=jnp.bool_),
                        nonfinite & selected,
                        engine.prepared_id,
                    )
                )
                continue
            positions, inputs = self._search_inputs(index, geometry, receivers)
            sources.append(engine.initialize(positions, **inputs, cases=selected))
        state_sources = tuple(sources)
        views = self._views(state_sources, geometry)
        epochs = jnp.zeros((len(views),), dtype=jnp.int32)
        return self._state(
            state_sources,
            views,
            self._fresh_graphs(views, state_sources, epochs),
            epochs,
            geometry,
            self.prepared_identity,
        )

    def certify(
        self,
        state: AtomisticNeighborhoodBundleState,
        geometry: AtomisticNeighborhoodGeometry,
        /,
        *,
        receivers: Mapping[str, ImageReceiverSubset] | None = None,
    ) -> tuple[SegmentedImageVerletCertificate | None, ...]:
        """Per-source stored-and-absent image reuse verdicts, without rebuilding.

        Complete-pair sources need no motion certificate (``None``): their
        topology changes only with the active set.
        """
        self._require_state(state)
        self._require_geometry(geometry)
        self._require_receivers(receivers)
        certificates: list[SegmentedImageVerletCertificate | None] = []
        for index, engine in enumerate(self.engines):
            previous = state.sources[index]
            if isinstance(engine, PreparedAtomisticCompletePairs):
                certificates.append(None)
                continue
            if not isinstance(previous, SegmentedImageVerletState):
                raise TypeError("An image source carries a segmented Verlet state.")
            positions, inputs = self._search_inputs(index, geometry, receivers)
            certificates.append(engine.certify(positions, previous, **inputs))
        return tuple(certificates)

    def refresh(
        self,
        state: AtomisticNeighborhoodBundleState,
        geometry: AtomisticNeighborhoodGeometry,
        /,
        *,
        receivers: Mapping[str, ImageReceiverSubset] | None = None,
        allow_rebuild: ArrayLike | None = None,
        force_rebuild: ArrayLike | None = None,
    ) -> AtomisticNeighborhoodBundleState:
        """Pure candidate refresh of every role under the proposed geometry.

        Each image source independently retains its certified cases bitwise
        (with exact rewrap offsets) and rebuilds the rest where
        ``allow_rebuild`` permits; ``force_rebuild`` rebuilds certified cases
        too.  A complete-pair case starts a new epoch only when its active set
        changes (or is forced).  A case needing a forbidden rebuild is
        retained as ``stale``.  A role's graph revision advances, and its
        schedule is re-prepared, only when the role's topology actually
        changed.  ``state`` is never modified.
        """
        self._require_state(state)
        self._require_geometry(geometry)
        self._require_receivers(receivers)
        count = self.plan.case_count
        allow = (
            jnp.ones((count,), dtype=jnp.bool_)
            if allow_rebuild is None
            else jnp.broadcast_to(jnp.asarray(allow_rebuild, dtype=jnp.bool_), (count,))
        )
        force = (
            jnp.zeros((count,), dtype=jnp.bool_)
            if force_rebuild is None
            else jnp.broadcast_to(jnp.asarray(force_rebuild, dtype=jnp.bool_), (count,))
        )
        sources: list[AtomisticSourceState] = []
        for index, engine in enumerate(self.engines):
            previous = state.sources[index]
            if isinstance(engine, PreparedAtomisticCompletePairs):
                if not isinstance(previous, AtomisticCompletePairState):
                    raise TypeError("A complete-pair source carries its own state.")
                active, nonfinite = self._complete_inputs(index, geometry)
                changed = ~previous.initialized | jnp.any(
                    active != previous.reference_active, axis=-1
                )
                rebuild = allow & (force | changed)
                initialized = previous.initialized | rebuild
                sources.append(
                    AtomisticCompletePairState(
                        jnp.where(rebuild[:, None], active, previous.reference_active),
                        initialized,
                        jnp.where(
                            rebuild, previous.case_epochs + 1, previous.case_epochs
                        ),
                        rebuild,
                        previous.rebuild_counts + rebuild.astype(jnp.int32),
                        previous.initialized & ~rebuild & changed,
                        nonfinite & initialized,
                        engine.prepared_id,
                    )
                )
                continue
            if not isinstance(previous, SegmentedImageVerletState):
                raise TypeError("An image source carries a segmented Verlet state.")
            positions, inputs = self._search_inputs(index, geometry, receivers)
            sources.append(
                engine.update(
                    positions,
                    previous,
                    **inputs,
                    allow_rebuild=allow,
                    force_rebuild=force,
                )
            )
        state_sources = tuple(sources)
        views = self._views(state_sources, geometry)
        schedules: list[PreparedStreamedRelation | None] = []
        epochs: list[Array] = []
        for index, view in enumerate(views):
            receivers_now = self._role_receivers(index, state_sources)
            changed = ~_same_topology(
                view.relation,
                receivers_now,
                state.views[index].relation,
                self._role_receivers(index, state.sources),
            )
            epoch = state.graph_epochs[index] + changed.astype(jnp.int32)
            epochs.append(epoch)
            retained = state.graph_schedules[index]
            if retained is None:
                schedules.append(None)
                continue
            schedules.append(
                jax.lax.cond(
                    changed,
                    lambda index=index, view=view, mask=receivers_now, epoch=epoch: (
                        self._schedule(index, view, mask, epoch)
                    ),
                    lambda retained=retained: retained,
                )
            )
        return self._state(
            state_sources,
            views,
            tuple(schedules),
            jnp.stack(epochs) if epochs else jnp.zeros((0,), dtype=jnp.int32),
            geometry,
            state.bundle_identity,
        )

    def bind(
        self,
        state: AtomisticNeighborhoodBundleState,
        geometry: AtomisticNeighborhoodGeometry,
        /,
        *,
        receivers: Mapping[str, ImageReceiverSubset] | None = None,
    ) -> AtomisticNeighborhoodBundleState:
        """Bind current geometry to the accepted epochs of ``state``; never rebuild.

        The frozen-epoch binding of a fixed candidate topology for varying
        coordinates and cells.  No search, rebuild, reselection or topology sort
        is traced: each image source certifies its stored and absent images
        under the current positions, cell and image representation
        (``SegmentedImageVerletPlan.retain``) and re-expresses its routes with
        exact rewrap offsets; a complete-pair source keeps its accepted active
        set.  Every role keeps its accepted routes, re-expressed for the
        current representation, and recomputes its physical cutoff mask (and
        the classical unique-image evidence) at the current geometry.  A
        selected directed graph keeps its frozen selection only where a
        sort-free certificate proves the declared rule selects exactly the same
        routes at the current geometry; otherwise the case reports
        ``selection_changed``.  Every uncertified case is ``stale`` and
        unsuccessful.  Graph revisions and schedules are retained bitwise, and
        the bound geometry is the program's exact evaluation point.
        """
        self._require_state(state)
        self._require_geometry(geometry)
        self._require_receivers(receivers)
        sources: list[AtomisticSourceState] = []
        for index, engine in enumerate(self.engines):
            previous = state.sources[index]
            if isinstance(engine, PreparedAtomisticCompletePairs):
                if not isinstance(previous, AtomisticCompletePairState):
                    raise TypeError("A complete-pair source carries its own state.")
                active, nonfinite = self._complete_inputs(index, geometry)
                changed = jnp.any(active != previous.reference_active, axis=-1)
                sources.append(
                    AtomisticCompletePairState(
                        previous.reference_active,
                        previous.initialized,
                        previous.case_epochs,
                        jnp.zeros_like(previous.rebuilt),
                        previous.rebuild_counts,
                        previous.stale | (previous.initialized & changed),
                        nonfinite & previous.initialized,
                        engine.prepared_id,
                    )
                )
                continue
            if not isinstance(previous, SegmentedImageVerletState):
                raise TypeError("An image source carries a segmented Verlet state.")
            positions, inputs = self._search_inputs(index, geometry, receivers)
            sources.append(engine.retain(positions, previous, **inputs))
        state_sources = tuple(sources)
        views = tuple(
            self._bound_view(
                index,
                view,
                state.sources[self.plan.role_sources[index]],
                state_sources[self.plan.role_sources[index]],
                geometry,
            )
            for index, view in enumerate(state.views)
        )
        return self._state(
            state_sources,
            views,
            state.graph_schedules,
            state.graph_epochs,
            geometry,
            state.bundle_identity,
        )

    def _bound_view(
        self,
        index: int,
        accepted: AtomisticNeighborhoodView,
        previous: AtomisticSourceState,
        source: AtomisticSourceState,
        geometry: AtomisticNeighborhoodGeometry,
        /,
    ) -> AtomisticNeighborhoodView:
        """Re-express one accepted role view at the bound geometry without a sort."""
        requirement = self.plan.requirements[index]
        source_index = self.plan.role_sources[index]
        cases = self.plan.case_count
        sites = self._site_source(source_index)
        positions = geometry.site_positions if sites else geometry.positions
        if positions is None:
            raise ValueError("A site source requires realized interaction sites.")
        flat = positions.reshape((-1, positions.shape[-1]))
        vectors = geometry.cell_vectors
        initialized = source.initialized
        false = jnp.zeros((cases,), dtype=jnp.bool_)
        relation = accepted.relation
        overflow = false
        if isinstance(source, SegmentedImageVerletState):
            if not isinstance(previous, SegmentedImageVerletState):
                raise TypeError("An image source carries a segmented Verlet state.")
            # Re-expression is additive: the accepted view carries the accepted
            # offsets, so only their change since acceptance is applied.
            delta, delta_overflow = checked_int32_add(
                source.representation_offsets, -previous.representation_offsets
            )
            shifts, route_overflow = representation_shifts(
                relation,
                jnp.where(delta_overflow, 0, delta).reshape((-1, delta.shape[-1])),
            )
            relation = eqx.tree_at(lambda value: value.image_shifts, relation, shifts)
            overflow = (_case_counts(relation, route_overflow) > 0) | jnp.any(
                delta_overflow, axis=(1, 2)
            )
        selection_changed = false
        if accepted.selected:
            unchanged = self._selection_unchanged(index, relation, source, flat, vectors)
            selection_changed = initialized & ~unchanged
            mask = relation.valid
        elif requirement.cutoff is None:
            mask = relation.valid
        else:
            mask = _within(relation, flat, vectors, requirement.bounded_cutoff())
        old = accepted.evidence
        repeated = old.repeated_minimum_image
        if requirement.counting is AtomisticPairCounting.MINIMUM_IMAGE_HALF:
            repeated = initialized & _repeated_minimum_image(
                relation, mask, accepted.self_image
            )
        evidence = AtomisticRoleEvidence(
            old.required_routes,
            old.stored_routes,
            old.required_degree,
            old.route_overflow,
            old.degree_overflow,
            source.capacity_failure,
            source.scientific_failure,
            old.reversal_open,
            old.identity_conflict,
            repeated,
            old.representation_overflow | (initialized & overflow),
            initialized & ~geometry.site_successful if sites else false,
            old.capped_receivers,
            initialized,
            initialized & (source.stale | selection_changed),
            source.case_epochs,
            source.rebuilt,
            selection_changed,
            selection=old.selection,
        )
        return eqx.tree_at(
            lambda value: (value.relation, value.mask, value.evidence),
            accepted,
            (relation, mask, evidence),
        )

    def _selection_unchanged(
        self,
        index: int,
        selected: ParticleImageRelation,
        source: AtomisticSourceState,
        positions: Array,
        vectors: Array,
        /,
    ) -> Array:
        """Per case: the declared selection at this geometry is exactly ``selected``.

        A sort-free certificate over the certified candidate epoch, with
        measures computed exactly as ``PreparedImageNeighborSelection.select``.
        For one receiver with eligible candidates ``E``, retained routes ``S``
        and ``k = maximum_neighbors``: an uncapped receiver (``|E| <= k``)
        needs ``S == E``.  A strict capped receiver needs ``|S| == k`` and
        exactly ``k`` eligible routes at or below the canonical maximum
        ``(m, source ID, n)`` of ``S``.  A degeneracy-band receiver needs
        ``S = {e in E : m_e <= T}`` for ``T = max_S m``, fewer than ``k + 1``
        retained routes with ``m + tolerance < T`` (so the band anchor
        ``m_(k)`` admits ``T``) and at least ``k + 1`` with ``m + tolerance``
        below the next eligible value (so it excludes the rest).  Every
        condition is a segment count or extremum and every retained route must
        be eligible.
        """
        prepared = self.selections[index]
        if prepared is None or not isinstance(source, SegmentedImageVerletState):
            raise RuntimeError("A selected role lacks its prepared selection or epoch.")
        policy = prepared.policy
        candidates = source.neighborhood.relation
        cases = candidates.case_count
        size = cases * candidates.particle_capacity

        def measured(relation: ParticleImageRelation, /) -> tuple[Array, Array]:
            displacement = relation.displacement(positions, vectors)
            squared = jax.lax.stop_gradient(jnp.sum(displacement * displacement, axis=-1))
            bound = policy.radius * policy.radius
            eligible = relation.valid & (
                squared <= bound
                if policy.radius_boundary == "closed"
                else squared < bound
            )
            if policy.coincidence_threshold is not None:
                eligible = eligible & (squared > policy.coincidence_threshold)
            measure = jnp.where(
                eligible,
                squared if policy.measure == "squared-distance" else jnp.sqrt(squared),
                0.0,
            )
            return eligible, measure

        def count(mask: Array, receivers: Array, /) -> Array:
            return (
                jnp.zeros((size,), dtype=jnp.int32)
                .at[jnp.where(mask, receivers, 0)]
                .add(mask.astype(jnp.int32))
            )

        eligible, measure = measured(candidates)
        kept = selected.valid
        kept_eligible, kept_measure = measured(selected)
        receivers = candidates.receiver_indices
        kept_receivers = selected.receiver_indices
        eligible_count = count(eligible, receivers)
        kept_count = count(kept, kept_receivers)
        stray = count(kept & ~kept_eligible, kept_receivers) > 0
        neighbors = policy.maximum_neighbors
        if policy.strict:
            rank = candidates.lattice_rank
            kept_keys = (
                kept_measure,
                selected.source_particle_ids,
                *(selected.image_shifts[:, axis] for axis in range(rank)),
            )
            candidate_keys = (
                measure,
                candidates.source_particle_ids,
                *(candidates.image_shifts[:, axis] for axis in range(rank)),
            )
            tied = kept
            below = jnp.zeros_like(eligible)
            equal = jnp.ones_like(eligible)
            for kept_key, candidate_key in zip(kept_keys, candidate_keys, strict=True):
                lowest = (
                    jnp.asarray(-jnp.inf, dtype=kept_key.dtype)
                    if jnp.issubdtype(kept_key.dtype, jnp.floating)
                    else jnp.asarray(jnp.iinfo(kept_key.dtype).min, dtype=kept_key.dtype)
                )
                top = (
                    jnp.full((size,), lowest, dtype=kept_key.dtype)
                    .at[jnp.where(tied, kept_receivers, 0)]
                    .max(jnp.where(tied, kept_key, lowest))
                )
                tied = tied & (kept_key == top[kept_receivers])
                bound = top[receivers].astype(candidate_key.dtype)
                below = below | (equal & (candidate_key < bound))
                equal = equal & (candidate_key == bound)
            prefix = count(eligible & (below | equal), receivers)
            capped_ok = (kept_count == neighbors) & (prefix == neighbors)
        else:
            tolerance = jnp.asarray(policy.degeneracy_tolerance, dtype=measure.dtype)
            infinite = jnp.asarray(jnp.inf, dtype=measure.dtype)
            top = (
                jnp.full((size,), -infinite, dtype=measure.dtype)
                .at[jnp.where(kept, kept_receivers, 0)]
                .max(jnp.where(kept, kept_measure, -infinite))
            )
            inside = eligible & (measure <= top[receivers])
            outside = eligible & ~inside
            following = (
                jnp.full((size,), infinite, dtype=measure.dtype)
                .at[jnp.where(outside, receivers, 0)]
                .min(jnp.where(outside, measure, infinite))
            )
            band = kept_measure + tolerance
            short = count(kept & (band < top[kept_receivers]), kept_receivers)
            separated = count(kept & (band < following[kept_receivers]), kept_receivers)
            capped_ok = (
                (count(inside, receivers) == kept_count)
                & (short <= neighbors)
                & (separated >= neighbors + 1)
            )
        receiver_ok = ~stray & jnp.where(
            eligible_count > neighbors, capped_ok, kept_count == eligible_count
        )
        return jnp.all(receiver_ok.reshape((cases, -1)), axis=-1)

    def select_cases(
        self,
        accept: ArrayLike,
        candidate: AtomisticNeighborhoodBundleState,
        accepted: AtomisticNeighborhoodBundleState,
        /,
    ) -> AtomisticNeighborhoodBundleState:
        """Commit ``candidate`` for accepted cases and keep ``accepted`` elsewhere.

        A role whose merged topology equals the accepted (or candidate) view
        keeps that state's schedule and revision bitwise; a genuinely mixed
        topology is re-prepared at the next revision.
        """
        self._require_state(candidate)
        self._require_state(accepted)
        mask = jnp.broadcast_to(
            jnp.asarray(accept, dtype=jnp.bool_), (self.plan.case_count,)
        )
        sources: list[AtomisticSourceState] = []
        for engine, new, old in zip(
            self.engines, candidate.sources, accepted.sources, strict=True
        ):
            if isinstance(engine, PreparedAtomisticCompletePairs):
                if not isinstance(new, AtomisticCompletePairState) or not isinstance(
                    old, AtomisticCompletePairState
                ):
                    raise TypeError("A complete-pair source carries its own state.")
                sources.append(
                    AtomisticCompletePairState(
                        jnp.where(
                            mask[:, None], new.reference_active, old.reference_active
                        ),
                        jnp.where(mask, new.initialized, old.initialized),
                        jnp.where(mask, new.case_epochs, old.case_epochs),
                        jnp.where(mask, new.rebuilt, old.rebuilt),
                        jnp.where(mask, new.rebuild_counts, old.rebuild_counts),
                        jnp.where(mask, new.stale, old.stale),
                        jnp.where(mask, new.nonfinite, old.nonfinite),
                        engine.prepared_id,
                    )
                )
                continue
            if not isinstance(new, SegmentedImageVerletState) or not isinstance(
                old, SegmentedImageVerletState
            ):
                raise TypeError("An image source carries a segmented Verlet state.")
            sources.append(engine.select_cases(mask, new, old))
        fresh, kept = candidate.geometry, accepted.geometry
        geometry = AtomisticNeighborhoodGeometry(
            jnp.where(mask[:, None, None], fresh.positions, kept.positions),
            jnp.where(mask[:, None], fresh.active, kept.active),
            jnp.where(mask[:, None, None], fresh.cell_vectors, kept.cell_vectors),
            _select_leading(mask, fresh.image_counts, kept.image_counts),
            _select_leading(mask, fresh.site_positions, kept.site_positions),
            _select_leading(mask, fresh.site_active, kept.site_active),
            _select_leading(mask, fresh.site_image_counts, kept.site_image_counts),
            jnp.where(mask, fresh.site_successful, kept.site_successful),
        )
        state_sources = tuple(sources)
        views = self._views(state_sources, geometry)
        schedules: list[PreparedStreamedRelation | None] = []
        epochs: list[Array] = []
        for index, view in enumerate(views):
            receivers_now = self._role_receivers(index, state_sources)
            keep_accepted = _same_topology(
                view.relation,
                receivers_now,
                accepted.views[index].relation,
                self._role_receivers(index, accepted.sources),
            )
            keep_candidate = _same_topology(
                view.relation,
                receivers_now,
                candidate.views[index].relation,
                self._role_receivers(index, candidate.sources),
            )
            old_epoch = accepted.graph_epochs[index]
            new_epoch = candidate.graph_epochs[index]
            mixed_epoch = jnp.maximum(old_epoch, new_epoch) + 1
            epochs.append(
                jnp.where(
                    keep_accepted,
                    old_epoch,
                    jnp.where(keep_candidate, new_epoch, mixed_epoch),
                )
            )
            old_schedule = accepted.graph_schedules[index]
            new_schedule = candidate.graph_schedules[index]
            if old_schedule is None or new_schedule is None:
                schedules.append(None)
                continue
            branch = jnp.where(keep_accepted, 0, jnp.where(keep_candidate, 1, 2))
            schedules.append(
                jax.lax.switch(
                    branch,
                    (
                        lambda old_schedule=old_schedule: old_schedule,
                        lambda new_schedule=new_schedule: new_schedule,
                        lambda index=index, view=view, receivers=receivers_now, epoch=(mixed_epoch): (
                            self._schedule(index, view, receivers, epoch)
                        ),
                    ),
                )
            )
        return self._state(
            state_sources,
            views,
            tuple(schedules),
            jnp.stack(epochs) if epochs else jnp.zeros((0,), dtype=jnp.int32),
            geometry,
            self._carried_identity(candidate, accepted),
        )

    def grow_capacities(
        self, candidate: AtomisticNeighborhoodBundleState, /
    ) -> PreparedAtomisticNeighborhoodBundle:
        """Host lifecycle choice of the next declared capacities for a failed attempt.

        Refuses unless every failure of ``candidate`` is a capacity failure.
        An image source with a capacity failure moves to the first later
        ladder entry covering its observed requirements; role storage grows
        only for roles whose source search succeeded (their counts are then
        exact).  Graph plans carry over.  Replay the same attempt with
        ``rebase(accepted)`` followed by ``refresh``.
        """
        self._require_state(candidate)
        if bool(np.any(np.asarray(candidate.scientific_failure))):
            raise ValueError(
                "Neighborhood failure is scientific or geometric, not capacity-only; "
                "refusing a capacity replay."
            )
        if bool(np.any(np.asarray(candidate.stale))):
            raise ValueError(
                "A stale (rebuild-forbidden) neighborhood is not a capacity failure."
            )
        if not bool(np.any(np.asarray(candidate.capacity_failure))):
            raise ValueError("The candidate neighborhood has no capacity failure.")
        source_levels = list(self.source_levels)
        role_levels = list(self.role_levels)
        grown = set()
        for index, (source, state) in enumerate(
            zip(self.plan.sources, candidate.sources, strict=True)
        ):
            if not isinstance(source, AtomisticNeighborhoodSource) or not isinstance(
                state, SegmentedImageVerletState
            ):
                continue
            if not bool(np.any(np.asarray(state.capacity_failure))):
                continue
            initialized = np.asarray(state.initialized, dtype=np.bool_)

            def initialized_only(leaf: Array) -> np.ndarray:
                values = np.asarray(leaf)
                shape = initialized.shape + (1,) * (values.ndim - initialized.ndim)
                return np.where(initialized.reshape(shape), values, np.zeros_like(values))

            evidence = jax.tree.map(initialized_only, state.reference.evidence)
            ladder = source.capacities
            entry = ladder.select(evidence, ladder.entries[source_levels[index]])
            source_levels[index] = [value.capacity_id for value in ladder.entries].index(
                entry.capacity_id
            )
            grown.add(index)
        for index, (requirement, view) in enumerate(
            zip(self.plan.requirements, candidate.views, strict=True)
        ):
            role_ladder = self.plan.role_capacities[index]
            if role_ladder is None or self.plan.role_sources[index] in grown:
                continue
            evidence = view.evidence
            overflow = np.asarray(evidence.route_overflow | evidence.degree_overflow)
            if not bool(np.any(overflow)):
                continue
            required = np.asarray(evidence.required_routes)
            degree = np.asarray(evidence.required_degree)
            entries = role_ladder.entries
            for level in range(role_levels[index] + 1, len(entries)):
                if entries[level].covers(required, degree):
                    role_levels[index] = level
                    break
            else:
                raise RuntimeError(
                    f"Role capacity ladder of {requirement.label} is exhausted for "
                    "the observed requirements."
                )
        return PreparedAtomisticNeighborhoodBundle(
            self.plan,
            self.system,
            graph_plans=self.graph_plans,
            source_levels=tuple(source_levels),
            role_levels=tuple(role_levels),
        )

    def rebase(
        self, accepted: AtomisticNeighborhoodBundleState, /
    ) -> AtomisticNeighborhoodBundleState:
        """Express an accepted state of another capacity level under this bundle.

        Sources whose prepared engine is unchanged (every complete-pair source
        and every image source without grown storage) are carried bitwise.  A
        source with grown storage is searched again at its accepted epoch's own
        build frame (reference positions, lattice, wrap counts, active set and
        receivers), which reproduces the same route identities; epochs,
        rebuild counts, representation offsets and lifecycle margins are
        retained.  Role views are re-formed under the current role storage at
        the accepted geometry; graph revisions are retained and schedules are
        re-prepared for the possibly regrown relation layouts.  A state of this
        very layout is returned unchanged when it is bound to this bundle's
        material and refused otherwise: a material change is never a rebase.
        """
        if not isinstance(accepted, AtomisticNeighborhoodBundleState):
            raise TypeError("accepted must be an AtomisticNeighborhoodBundleState.")
        if accepted.plan_id != self.plan.plan_id:
            raise ValueError("Accepted state belongs to another bundle plan.")
        if accepted.layout_id == self.layout_id:
            if accepted.bundle_id != self.prepared_id:
                raise ValueError(
                    "Accepted state belongs to another material of this bundle "
                    "layout; a material change is never a rebase."
                )
            return accepted
        if len(accepted.sources) != len(self.engines):
            raise ValueError("Accepted state source layout differs from the plan.")
        sources: list[AtomisticSourceState] = []
        for engine, old in zip(self.engines, accepted.sources, strict=True):
            if isinstance(engine, PreparedAtomisticCompletePairs):
                if not isinstance(old, AtomisticCompletePairState):
                    raise TypeError("A complete-pair source carries its own state.")
                sources.append(old)
                continue
            if not isinstance(old, SegmentedImageVerletState):
                raise TypeError("An image source carries a segmented Verlet state.")
            if old.prepared_verlet_id == engine.plan_id:
                sources.append(old)
                continue
            reference = old.reference
            cases, particles = engine.case_count, engine.particle_capacity
            receivers = (
                None
                if engine.receiver_rows is None
                else ImageReceiverSubset(
                    old.reference_receiver_ids, old.reference_receiver_valid
                )
            )
            fresh = engine.initialize(
                reference.reference_positions.reshape((cases, particles, -1)),
                active=old.reference_active_mask,
                stable_ids=old.reference_particle_ids,
                vectors=reference.cell_vectors,
                origin=old.reference_origin,
                periodic=old.reference_periodic,
                image_counts=old.reference_image_counts,
                receivers=receivers,
                cases=old.initialized,
            )
            sources.append(
                eqx.tree_at(
                    lambda state: (
                        state.representation_offsets,
                        state.case_epochs,
                        state.rebuilt,
                        state.rebuild_counts,
                        state.stale,
                        state.maximum_reference_displacement,
                        state.maximum_cell_deformation,
                        state.stencil_margin,
                        state.certificate_margin,
                        state.admissible,
                    ),
                    fresh,
                    (
                        old.representation_offsets,
                        old.case_epochs,
                        old.rebuilt,
                        old.rebuild_counts,
                        old.stale,
                        old.maximum_reference_displacement,
                        old.maximum_cell_deformation,
                        old.stencil_margin,
                        old.certificate_margin,
                        old.admissible,
                    ),
                )
            )
        state_sources = tuple(sources)
        views = self._views(state_sources, accepted.geometry)
        return self._state(
            state_sources,
            views,
            self._fresh_graphs(views, state_sources, accepted.graph_epochs),
            accepted.graph_epochs,
            accepted.geometry,
            self.prepared_identity,
        )


__all__ = [
    "AbstractAtomisticImageSearchPolicy",
    "AtomisticCellListImageSearch",
    "AtomisticClusterTileImageSearch",
    "AtomisticCompletePairSource",
    "AtomisticCompletePairState",
    "AtomisticDenseImageSearch",
    "AtomisticNeighborhoodBundlePlan",
    "AtomisticNeighborhoodBundleState",
    "AtomisticNeighborhoodGeometry",
    "AtomisticNeighborhoodRequirement",
    "AtomisticNeighborhoodRole",
    "AtomisticNeighborhoodSource",
    "AtomisticNeighborhoodSourceProof",
    "AtomisticNeighborhoodView",
    "AtomisticPairCounting",
    "AtomisticPairExclusions",
    "AtomisticRoleCapacity",
    "AtomisticRoleCapacityLadder",
    "AtomisticRoleEvidence",
    "AtomisticTiledImageSearch",
    "PreparedAtomisticCompletePairs",
    "PreparedAtomisticNeighborhoodBundle",
]
