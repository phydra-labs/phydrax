#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Static admission of distributed atomistic domain execution.

An `AtomisticDomainPlan` binds one periodic system's canonical point
ownership (devices or reference lanes), the declared fractional owner
partition of its cell, one `AtomisticDomainRolePlan` per neighborhood role
(each with its own cutoff, selection, image stencil, streamed relation tiling
and capacities), the spatial strategy, the owner reduction order, and the
optional graph-partition and reciprocal-distribution plans. Every capacity is
explicit; exceeding one is a refused topology, migration, or message, never a
truncated relation.
"""

from __future__ import annotations

from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..backends.distributed import JaxCollectiveProvider
from ..discretization._periodic_cell import (
    _complete_image_shifts,
    PeriodicCell,
    PeriodicImageStencil,
)
from ..discretization.particle._distributed import FractionalOwnerPartition
from ..discretization.particle._image_neighborhood import ImageNeighborSelectionPolicy
from ..discretization.spatial._distributed_relations import DistributedOwnershipPlan
from ..sparse._streamed import StreamedRelationPlan
from ..typing import parse
from ._domain_state import AtomisticDomainStrategy
from ._domain_terms import AbstractAtomisticDomainTerm, require_domain_requirement
from ._graph_partition import AtomisticGraphPartitionPlan
from ._neighborhood_bundle import (
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
)
from ._reciprocal_distribution import AtomisticReciprocalDistributionPlan
from ._system import PreparedAtomisticSystem


AtomisticDomainReductionMode: TypeAlias = Literal["fast", "deterministic", "compensated"]

_AMBIENT_DIMENSION = 3


def _integer(name: str, value: int, /, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _length(name: str, value: float, /, *, positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be real.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return result


@final
class AtomisticDomainReductionPolicy(StrictModule, NonTrainableState):
    """Declared owner reduction order of domain energies, partials and evidence.

    ``fast`` uses an unordered sum; ``deterministic`` a sequential sum in owner
    order; ``compensated`` a Kahan sum in owner order. ``ordered_sum`` reduces
    an owner-stacked value outside an owner region; ``owner_sum`` produces the
    same replicated value inside one (an all-gather followed by the ordered
    sum). ``owner_sum`` is an evidence/scalar reduction, not the differentiable
    context transpose of system stages (``owner_system_sum`` owns that).
    """

    mode: AtomisticDomainReductionMode = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, mode: AtomisticDomainReductionMode = "deterministic", /) -> None:
        resolved = parse(mode, AtomisticDomainReductionMode, "mode")
        self.mode = resolved
        self.policy_id = canonical_fingerprint(
            {"kind": "atomistic-domain-reduction", "mode": resolved}
        )

    def ordered_sum(self, stacked: Array, /) -> Array:
        """Sum over the leading axis in the declared order."""
        array = jnp.asarray(stacked)
        if array.ndim == 0:
            raise ValueError("ordered_sum requires a leading reduction axis.")
        if self.mode == "fast":
            return jnp.sum(array, axis=0)
        initial = jnp.zeros(array.shape[1:], dtype=array.dtype)
        if self.mode == "deterministic":
            return jax.lax.fori_loop(
                0, array.shape[0], lambda index, total: total + array[index], initial
            )

        def compensated(
            index: int | Array, carry: tuple[Array, Array]
        ) -> tuple[Array, Array]:
            total, correction = carry
            increment = array[index] - correction
            updated = total + increment
            return updated, (updated - total) - increment

        total, _ = jax.lax.fori_loop(
            0, array.shape[0], compensated, (initial, jnp.zeros_like(initial))
        )
        return total

    def owner_sum(self, value: Array, axis_name: str, /) -> Array:
        """Replicated owner-ordered sum inside a mapped owner region."""
        provider = JaxCollectiveProvider(axis_name)
        if self.mode == "fast":
            return provider.sum(value)
        return self.ordered_sum(provider.all_gather(jnp.asarray(value), tiled=False))


def _require_complete_stencil(
    cell: PeriodicCell, stencil: PeriodicImageStencil, radius: float, /
) -> None:
    """Refuse an image stencil that is not the complete box covering ``radius``."""
    extents = stencil.extents
    if len(extents) != cell.rank or not np.array_equal(
        np.asarray(stencil.shifts), _complete_image_shifts(extents)
    ):
        raise ValueError(
            "The image stencil shifts are not the complete box of its extents."
        )
    if any(
        extent != 0
        for extent, periodic in zip(extents, cell.periodic_axes, strict=True)
        if not periodic
    ):
        raise ValueError("The image stencil translates along a nonperiodic axis.")
    if any(
        excursion < 1.0
        for excursion, periodic in zip(
            stencil.fractional_excursion, cell.periodic_axes, strict=True
        )
        if periodic
    ):
        raise ValueError(
            "The image stencil must admit wrapped endpoints (fractional_excursion >= 1)."
        )
    try:
        required = cell.image_stencil(
            radius, maximum_image_count=stencil.image_count
        ).extents
    except ValueError as error:
        raise ValueError(
            "The image stencil does not cover cutoff + skin in its cell."
        ) from error
    if any(have < need for have, need in zip(extents, required, strict=True)):
        raise ValueError("The image stencil does not cover cutoff + skin in its cell.")


@final
class AtomisticDomainRolePlan(StrictModule, NonTrainableState):
    """One neighborhood role of a domain: requirement, stencil, tiling, capacities.

    ``requirement`` is the scientific requirement resolved from the model or
    term (``requirements.neighborhood(role)``): it fixes the role, cutoff and
    declared nearest-neighbor ``selection``. Candidate edges cover
    ``cutoff + skin``; a declared selection is applied owner-locally on the
    complete candidate set of every receiver at the current geometry with the
    single-device canonical order. The domain topology of every role holds
    every directed image route of the owned receivers; half-counting pair
    roles (minimum-image pairs, image-pair sums) count each pair once from its
    two directed routes in the consuming domain term. Interaction-site roles
    require an identity physical-atom mapping admitted by the term. Receiver
    groups and complete all-pairs relations are not domain roles.

    Capacities: ``alias_capacity`` image aliases per (sender, destination)
    owner pair, ``edge_capacity`` candidate edges per owner and
    ``halo_capacity`` deduplicated source columns per (requester, source)
    owner pair of the halo strategy.

    Every field is declared layout: the requirement (role, resource-defining
    cutoff, selection), the reference-cell stencil, the streamed tiling, the
    Verlet skin (it sizes the stencil and the reuse certificate), the
    capacities and the ``term_key`` name slot; ``role_id`` is their
    content-free configuration fingerprint.
    """

    requirement: AtomisticNeighborhoodRequirement
    stencil: PeriodicImageStencil
    streaming: StreamedRelationPlan
    term_key: str = eqx.field(static=True)
    skin: float = eqx.field(static=True)
    alias_capacity: int = eqx.field(static=True)
    edge_capacity: int = eqx.field(static=True)
    halo_capacity: int = eqx.field(static=True)
    role_id: str = eqx.field(static=True)

    def __init__(
        self,
        requirement: AtomisticNeighborhoodRequirement,
        stencil: PeriodicImageStencil,
        /,
        *,
        streaming: StreamedRelationPlan,
        skin: float,
        edge_capacity: int,
        alias_capacity: int,
        halo_capacity: int,
        term_key: str = "model",
    ) -> None:
        if not isinstance(requirement, AtomisticNeighborhoodRequirement):
            raise TypeError("requirement must be an AtomisticNeighborhoodRequirement.")
        if not isinstance(stencil, PeriodicImageStencil):
            raise TypeError("stencil must be a PeriodicImageStencil.")
        if not isinstance(streaming, StreamedRelationPlan):
            raise TypeError("streaming must be a StreamedRelationPlan.")
        require_domain_requirement(requirement)
        key = str(term_key).strip()
        if not key:
            raise ValueError("term_key must be non-empty.")
        cutoff = requirement.bounded_cutoff()
        skin_ = _length("skin", skin, positive=False)
        if stencil.radius < cutoff + skin_:
            raise ValueError("The image stencil radius must cover cutoff + skin.")
        self.requirement = requirement
        self.stencil = stencil
        self.streaming = streaming
        self.term_key = key
        self.skin = skin_
        self.alias_capacity = _integer("alias_capacity", alias_capacity, minimum=1)
        self.edge_capacity = _integer("edge_capacity", edge_capacity, minimum=1)
        self.halo_capacity = _integer("halo_capacity", halo_capacity, minimum=1)
        self.role_id = canonical_fingerprint(
            {
                "kind": "atomistic-domain-role",
                "requirement": requirement.requirement_id,
                "term_key": key,
                "stencil": stencil.stencil_id,
                "streaming": streaming.plan_id,
                "skin": skin_,
                "alias_capacity": self.alias_capacity,
                "edge_capacity": self.edge_capacity,
                "halo_capacity": self.halo_capacity,
            }
        )

    @property
    def role(self) -> AtomisticNeighborhoodRole:
        return self.requirement.role

    @property
    def cutoff(self) -> float:
        return self.requirement.bounded_cutoff()

    @property
    def selection(self) -> ImageNeighborSelectionPolicy | None:
        return self.requirement.selection

    @property
    def search_radius(self) -> float:
        return self.cutoff + self.skin

    @property
    def relation_owner_id(self) -> str:
        return f"atomistic-domain-role:{self.role_id}"


@final
class AtomisticDomainPlan(StrictModule, NonTrainableState):
    """Static contract of one distributed periodic atomistic domain.

    The ownership address is the unit fractional box of the partition cell,
    so layout points are wrapped fractional coordinates of the CURRENT cell.
    ``partition.cell`` is the reference cell: every role stencil is
    enumerated for it, and a dynamic cell (``dynamic_cell=True``) is certified
    against it at every evaluation and rebuild. ``migration_capacity`` bounds
    migrating atoms per owner pair and ``message_capacity_bytes`` one
    interaction's halo feature message per owner. ``terms`` are canonical
    classical, Ewald and polarization terms executed owner-locally on their own
    roles (``term_key`` = term name); the layered model, if any, uses the
    ``(DIRECTED_IMAGE_GRAPH, "model")`` role and ``reciprocal`` its global
    payload epilogue.

    ``plan_identity`` is the exact canonical plan digest (it binds the domain
    terms' scientific identities) carried as dynamic `IdentityWords`;
    ``plan_id`` decodes it on the host. Every static field is declared layout,
    so all material and parameter revisions of one layout share an executable.
    """

    ownership: DistributedOwnershipPlan
    partition: FractionalOwnerPartition
    roles: tuple[AtomisticDomainRolePlan, ...]
    reduction: AtomisticDomainReductionPolicy
    graph_partition: AtomisticGraphPartitionPlan | None
    reciprocal: AtomisticReciprocalDistributionPlan | None
    terms: tuple[AbstractAtomisticDomainTerm, ...]
    strategy: AtomisticDomainStrategy = eqx.field(static=True)
    migration_capacity: int = eqx.field(static=True)
    message_capacity_bytes: int = eqx.field(static=True)
    dynamic_cell: bool = eqx.field(static=True)
    plan_identity: IdentityWords

    # Ownership, role requirements, reciprocal work and transport bounds jointly
    # define one domain. Keep the explicit fail-first admission order and canonical
    # identity assembly together rather than inferring compatibility from shapes.
    def __init__(
        self,
        ownership: DistributedOwnershipPlan,
        partition: FractionalOwnerPartition,
        roles: tuple[AtomisticDomainRolePlan, ...],
        /,
        *,
        migration_capacity: int,
        message_capacity_bytes: int,
        strategy: AtomisticDomainStrategy = AtomisticDomainStrategy.HALO,
        reduction: AtomisticDomainReductionPolicy | None = None,
        graph_partition: AtomisticGraphPartitionPlan | None = None,
        reciprocal: AtomisticReciprocalDistributionPlan | None = None,
        dynamic_cell: bool = False,
        terms: tuple[AbstractAtomisticDomainTerm, ...] = (),
    ) -> None:
        if not isinstance(ownership, DistributedOwnershipPlan):
            raise TypeError("ownership must be a DistributedOwnershipPlan.")
        if not isinstance(partition, FractionalOwnerPartition):
            raise TypeError("partition must be a FractionalOwnerPartition.")
        strategy_ = parse(strategy, AtomisticDomainStrategy, "strategy")
        reduction_ = AtomisticDomainReductionPolicy() if reduction is None else reduction
        if not isinstance(reduction_, AtomisticDomainReductionPolicy):
            raise TypeError("reduction must be AtomisticDomainReductionPolicy or None.")
        if not isinstance(dynamic_cell, (bool, np.bool_)):
            raise TypeError("dynamic_cell must be a bool.")
        role_plans = tuple(roles)
        if not role_plans or any(
            not isinstance(role, AtomisticDomainRolePlan) for role in role_plans
        ):
            raise TypeError("roles must be a nonempty tuple of AtomisticDomainRolePlan.")
        keys = [(role.role, role.term_key) for role in role_plans]
        if len(set(keys)) != len(keys):
            raise ValueError("Every (role, term_key) must be declared once.")
        cell = partition.cell
        if cell.ambient_dimension != _AMBIENT_DIMENSION:
            raise ValueError("Domain execution requires a three-dimensional cell.")
        if partition.owner_count != ownership.owner_count:
            raise ValueError(
                "The fractional partition must declare one region per owner."
            )
        address = ownership.address_plan
        if (
            address.dimension != cell.rank
            or any(value != 0.0 for value in address.lower)
            or any(value != 1.0 for value in address.upper)
            or address.periodic_axes != cell.periodic_axes
        ):
            raise ValueError(
                "Domain ownership must address the unit fractional box of the cell "
                "with its periodic axes."
            )
        for role in role_plans:
            if role.stencil.cell_id != cell.cell_id:
                raise ValueError(
                    f"The image stencil of {role.requirement.label} belongs to another "
                    "PeriodicCell."
                )
            _require_complete_stencil(cell, role.stencil, role.search_radius)
        if strategy_ is AtomisticDomainStrategy.GRAPH_PARTITION:
            if not isinstance(graph_partition, AtomisticGraphPartitionPlan):
                raise ValueError(
                    "The graph-partition strategy requires an AtomisticGraphPartitionPlan."
                )
            graph_partition.require_compatible(ownership, partition, role_plans)
        elif graph_partition is not None:
            raise ValueError(
                "Only the graph-partition strategy declares a partition plan."
            )
        if reciprocal is not None and not isinstance(
            reciprocal, AtomisticReciprocalDistributionPlan
        ):
            raise TypeError(
                "reciprocal must be an AtomisticReciprocalDistributionPlan or None."
            )
        term_plans = tuple(terms)
        if any(not isinstance(term, AbstractAtomisticDomainTerm) for term in term_plans):
            raise TypeError("terms must contain AbstractAtomisticDomainTerm values.")
        term_keys = [term.term_key for term in term_plans]
        if len(set(term_keys)) != len(term_keys) or "model" in term_keys:
            raise ValueError("Domain term keys must be distinct and differ from 'model'.")
        if len({term.system.prepared_id for term in term_plans}) > 1:
            raise ValueError("Every domain term must belong to one prepared system.")
        for term in term_plans:
            for requirement in term.requirements():
                index = next(
                    (
                        position
                        for position, role in enumerate(role_plans)
                        if role.role is requirement.role
                        and role.term_key == term.term_key
                    ),
                    None,
                )
                if (
                    index is None
                    or role_plans[index].requirement.requirement_id
                    != requirement.requirement_id
                ):
                    raise ValueError(
                        f"Term {term.term_key!r} needs a {requirement.label} role plan "
                        f"with term_key={term.term_key!r} and its exact requirement."
                    )
            if term.system.cell is None or term.system.cell.cell_id != cell.cell_id:
                raise ValueError(
                    f"Term {term.term_key!r} belongs to a system of another cell."
                )
        self.ownership = ownership
        self.partition = partition
        self.roles = role_plans
        self.reduction = reduction_
        self.graph_partition = graph_partition
        self.reciprocal = reciprocal
        self.terms = term_plans
        self.strategy = strategy_
        self.migration_capacity = _integer(
            "migration_capacity", migration_capacity, minimum=1
        )
        self.message_capacity_bytes = _integer(
            "message_capacity_bytes", message_capacity_bytes, minimum=1
        )
        self.dynamic_cell = bool(dynamic_cell)
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-domain-plan",
                "ownership": ownership.plan_id,
                "partition": partition.partition_id,
                "roles": [role.role_id for role in role_plans],
                "reduction": reduction_.policy_id,
                "strategy": strategy_.value,
                "graph_partition": (
                    None if graph_partition is None else graph_partition.plan_id
                ),
                "reciprocal": None if reciprocal is None else reciprocal.plan_id,
                "terms": [term.term_id for term in term_plans],
                "migration_capacity": self.migration_capacity,
                "message_capacity_bytes": self.message_capacity_bytes,
                "dynamic_cell": self.dynamic_cell,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical domain-plan digest."""
        return self.plan_identity.fingerprint()

    def role_index(
        self, role: AtomisticNeighborhoodRole, term_key: str = "model", /
    ) -> int:
        kind = parse(role, AtomisticNeighborhoodRole, "role")
        for index, plan in enumerate(self.roles):
            if plan.role is kind and plan.term_key == term_key:
                return index
        raise ValueError(f"The domain declares no {kind.value} role for {term_key!r}.")

    def role(
        self, role: AtomisticNeighborhoodRole, term_key: str = "model", /
    ) -> AtomisticDomainRolePlan:
        return self.roles[self.role_index(role, term_key)]

    @property
    def system(self) -> PreparedAtomisticSystem | None:
        """The prepared system shared by every domain term (``None`` without terms)."""
        return None if not self.terms else self.terms[0].system

    def term_role_indices(self, term: AbstractAtomisticDomainTerm, /) -> tuple[int, ...]:
        """Role indices of ``term``'s requirements, in requirement order."""
        indices = []
        for requirement in term.requirements():
            index = self.role_index(requirement.role, term.term_key)
            if self.roles[index].requirement.requirement_id != requirement.requirement_id:
                raise ValueError(
                    f"The {requirement.label} role of term {term.term_key!r} carries "
                    "another requirement."
                )
            indices.append(index)
        return tuple(indices)


__all__ = [
    "AtomisticDomainPlan",
    "AtomisticDomainReductionMode",
    "AtomisticDomainReductionPolicy",
    "AtomisticDomainRolePlan",
]
