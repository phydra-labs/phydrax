#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Atomic cross-owner rebind of one accepted numerical composition.

A composition is the set of owner artifacts (topologies, discretizations,
interface routes, transfers, preconditioners, observations, prepared graphs,
worksets) and owner state (physical, exchange, budget, model, optimizer,
history, statistics, RNG) bound at one accepted boundary. Every entry carries
explicit structure, revision, and semantics identities and the identities of
the entries it was prepared against.

A rebind stages every change on the host before anything is published: numeric
refreshes keep their exact PyTree layout, topology reprepares replace derived
artifacts whole, and state crosses only through explicit owner transports
(physical remap or ownership migration) with evidence. Construction validates
that every source entry has exactly one disposition and that the candidate
composition has no stale binding. Commit takes one explicit accepted-boundary
decision; on refusal the original composition object is returned unchanged.

This module owns only that shared invariant. It never imports a numerical owner,
never executes a transport, and never infers or zero-fills state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, unique_identifiers
from ..typing import parse
from ._transaction import commit_candidate, TransactionalCandidate


CompositionRole: TypeAlias = Literal[
    "topology",
    "discretization",
    "interface-route",
    "transfer",
    "preconditioner",
    "observation",
    "prepared-graph",
    "workset",
    "physical-state",
    "exchange-state",
    "budget",
    "model-parameter",
    "model-state",
    "optimizer-state",
    "history",
    "statistics",
    "rng",
]
CompositionFacet: TypeAlias = Literal["structure", "revision", "semantics"]
CompositionTransportKind: TypeAlias = Literal["physical-remap", "ownership-migration"]


def _carries_state(role: CompositionRole, /) -> bool:
    """Whether entries of `role` hold state that may never be dropped or re-created.

    Derived artifacts are rebuilt by their owners from staged dependencies and
    may be invalidated. State crosses a rebind only by retention or by an
    explicit owner transport.
    """

    match role:
        case (
            "topology"
            | "discretization"
            | "interface-route"
            | "transfer"
            | "preconditioner"
            | "observation"
            | "prepared-graph"
            | "workset"
        ):
            return False
        case (
            "physical-state"
            | "exchange-state"
            | "budget"
            | "model-parameter"
            | "model-state"
            | "optimizer-state"
            | "history"
            | "statistics"
            | "rng"
        ):
            return True
        case _:
            assert_never(role)


@final
class CompositionDependency(StrictModule, NonTrainableState):
    """One identity facet of another entry that a dependent was prepared against."""

    entry_id: str = eqx.field(static=True)
    facet: CompositionFacet = eqx.field(static=True)
    bound_id: str = eqx.field(static=True)

    def __init__(self, entry_id: str, facet: CompositionFacet, bound_id: str, /) -> None:
        self.entry_id = canonical_identifier(entry_id, "Dependency entry_id")
        self.facet = parse(facet, CompositionFacet, "facet")
        self.bound_id = canonical_identifier(bound_id, "Dependency bound_id")


@final
class CompositionEntry(StrictModule):
    """One owner artifact or state payload with explicit identities.

    `structure_id` names the static structure (topology, partition, layout,
    symbolic sparsity, routes); `revision_id` names this exact artifact or state
    revision; `semantics_id` names its scientific meaning. Equal values, shapes,
    or names never substitute for these identities.
    """

    value: Any
    dependencies: tuple[CompositionDependency, ...]
    entry_id: str = eqx.field(static=True)
    role: CompositionRole = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    revision_id: str = eqx.field(static=True)
    semantics_id: str = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        value: Any,
        /,
        *,
        entry_id: str,
        role: CompositionRole,
        owner_id: str,
        structure_id: str,
        revision_id: str,
        semantics_id: str,
        dependencies: Sequence[CompositionDependency] = (),
    ) -> None:
        identifier = canonical_identifier(entry_id, "entry_id")
        role_ = parse(role, CompositionRole, "role")
        identities = tuple(
            canonical_identifier(item, name)
            for item, name in (
                (owner_id, "owner_id"),
                (structure_id, "structure_id"),
                (revision_id, "revision_id"),
                (semantics_id, "semantics_id"),
            )
        )
        bindings = tuple(dependencies)
        if any(not isinstance(item, CompositionDependency) for item in bindings):
            raise TypeError("dependencies must contain CompositionDependency values.")
        keys = [(item.entry_id, item.facet) for item in bindings]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Entry {identifier!r} binds one dependency facet twice.")
        if any(item.entry_id == identifier for item in bindings):
            raise ValueError(f"Entry {identifier!r} cannot depend on itself.")
        ordered = tuple(sorted(bindings, key=lambda item: (item.entry_id, item.facet)))
        self.value = value
        self.dependencies = ordered
        self.entry_id = identifier
        self.role = role_
        self.owner_id, self.structure_id, self.revision_id, self.semantics_id = identities
        self.record_id = canonical_fingerprint(
            {
                "kind": "composition-entry",
                "entry": identifier,
                "role": role_,
                "owner": identities[0],
                "structure": identities[1],
                "revision": identities[2],
                "semantics": identities[3],
                "dependencies": [
                    [item.entry_id, item.facet, item.bound_id] for item in ordered
                ],
            }
        )

    def identity(self, facet: CompositionFacet, /) -> str:
        """The identity a dependent binds for `facet`."""
        match parse(facet, CompositionFacet, "facet"):
            case "structure":
                return self.structure_id
            case "revision":
                return self.revision_id
            case "semantics":
                return self.semantics_id
            case unknown:
                assert_never(unknown)

    def binding(self, facet: CompositionFacet, /) -> CompositionDependency:
        """The dependency record of a consumer prepared against this entry."""
        return CompositionDependency(self.entry_id, facet, self.identity(facet))


def _binding_failures(entries: Mapping[str, CompositionEntry], /) -> list[str]:
    failures: list[str] = []
    for entry_id in sorted(entries):
        for dependency in entries[entry_id].dependencies:
            target = entries.get(dependency.entry_id)
            if target is None:
                failures.append(
                    f"{entry_id!r} depends on absent entry {dependency.entry_id!r}"
                )
                continue
            if target.role == "model-parameter" and dependency.facet != "semantics":
                failures.append(
                    f"{entry_id!r} binds model parameter {dependency.entry_id!r} "
                    f"by {dependency.facet}; parameters bind only by semantics"
                )
            elif target.identity(dependency.facet) != dependency.bound_id:
                failures.append(
                    f"{entry_id!r} is stale against the {dependency.facet} of "
                    f"{dependency.entry_id!r}"
                )
    return failures


@final
class Composition(StrictModule):
    """Accepted owner artifacts and state bound at one accepted boundary.

    Every dependency resolves to an entry of the same composition with the
    identity it was prepared against, so no published artifact is stale.
    `composition_id` names this exact boundary revision; `structure_id` names
    only the roles, structures, and meanings of the entries, so it stays fixed
    while state advances and changes exactly when a rebind changes structure
    (the topology identity a checkpoint of this composition binds).
    """

    entries: tuple[CompositionEntry, ...]
    entry_ids: tuple[str, ...] = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    composition_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self, entries: Sequence[CompositionEntry], /, *, boundary_id: str
    ) -> None:
        values = tuple(entries)
        if not values or any(not isinstance(item, CompositionEntry) for item in values):
            raise TypeError("A composition needs CompositionEntry values.")
        identifiers = unique_identifiers(
            [item.entry_id for item in values], "Composition entry IDs"
        )
        by_id = dict(zip(identifiers, values, strict=True))
        failures = _binding_failures(by_id)
        if failures:
            raise ValueError(
                "Composition has unresolved or stale bindings: " + "; ".join(failures)
            )
        ordered = tuple(by_id[identifier] for identifier in sorted(by_id))
        boundary = canonical_identifier(boundary_id, "boundary_id")
        self.entries = ordered
        self.entry_ids = tuple(item.entry_id for item in ordered)
        self.boundary_id = boundary
        self.composition_id = canonical_fingerprint(
            {
                "kind": "composition",
                "boundary": boundary,
                "entries": [item.record_id for item in ordered],
            }
        )
        self.structure_id = canonical_fingerprint(
            {
                "kind": "composition-structure",
                "entries": [
                    [item.entry_id, item.role, item.structure_id, item.semantics_id]
                    for item in ordered
                ],
            }
        )

    def entry(self, entry_id: str, /) -> CompositionEntry:
        if entry_id not in self.entry_ids:
            raise ValueError(f"Composition has no entry {entry_id!r}.")
        return self.entries[self.entry_ids.index(entry_id)]

    def value(self, entry_id: str, /) -> Any:
        return self.entry(entry_id).value

    def dependents(self, entry_id: str, /) -> tuple[str, ...]:
        """Entries prepared against any facet of `entry_id`."""
        return tuple(
            item.entry_id
            for item in self.entries
            if any(binding.entry_id == entry_id for binding in item.dependencies)
        )


def _content(value: ArrayLike | None, name: str, /) -> Array | None:
    if value is None:
        return None
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        raise TypeError(f"{name} must be a real or complex inexact array.")
    return array


@final
class CompositionTransport(StrictModule):
    """Staged evidence of one explicit owner transport of state.

    `source_entry_ids` are consumed from the source composition and must have the
    structure identities `source_structure_ids` the owner route was prepared from;
    `targets` are the staged entries the route produced. A conservative transport
    reports the conserved content per component before and after with the owner's
    tolerance; ownership migration must always report it, since moving rows
    between owners may neither create nor destroy content.
    """

    targets: tuple[CompositionEntry, ...]
    successful: Array
    source_content: Array | None
    target_content: Array | None
    content_tolerance: Array | None
    kind: CompositionTransportKind = eqx.field(static=True)
    source_entry_ids: tuple[str, ...] = eqx.field(static=True)
    source_structure_ids: tuple[str, ...] = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: CompositionTransportKind,
        source_entry_ids: Sequence[str],
        targets: Sequence[CompositionEntry],
        /,
        *,
        source_structure_ids: Sequence[str],
        route_id: str,
        successful: ArrayLike,
        source_content: ArrayLike | None = None,
        target_content: ArrayLike | None = None,
        content_tolerance: ArrayLike | None = None,
    ) -> None:
        kind_ = parse(kind, CompositionTransportKind, "kind")
        sources = unique_identifiers(source_entry_ids, "Transport source entry IDs")
        structures = tuple(
            canonical_identifier(item, "source_structure_ids")
            for item in source_structure_ids
        )
        if len(structures) != len(sources):
            raise ValueError("One source structure identity is required per source.")
        staged = tuple(targets)
        if not staged or any(not isinstance(item, CompositionEntry) for item in staged):
            raise TypeError("Transport targets must be CompositionEntry values.")
        unique_identifiers([item.entry_id for item in staged], "Transport target IDs")
        accepted = jnp.asarray(successful)
        if accepted.shape != () or accepted.dtype != jnp.bool_:
            raise ValueError("Transport successful must be a scalar boolean.")
        contents = (
            _content(source_content, "source_content"),
            _content(target_content, "target_content"),
            _content(content_tolerance, "content_tolerance"),
        )
        present = tuple(item is not None for item in contents)
        if any(present) and not all(present):
            raise ValueError(
                "Conserved content needs source, target, and tolerance together."
            )
        if (
            all(present)
            and len({item.shape for item in contents if item is not None}) != 1
        ):
            raise ValueError("Conserved content arrays must share one component shape.")
        if kind_ == "ownership-migration" and not all(present):
            raise ValueError(
                "Ownership migration must report the content it moves; created rows "
                "are never physical content."
            )
        route = canonical_identifier(route_id, "route_id")
        self.targets = staged
        self.successful = accepted
        self.source_content, self.target_content, self.content_tolerance = contents
        self.kind = kind_
        self.source_entry_ids = sources
        self.source_structure_ids = structures
        self.route_id = route
        self.transport_id = canonical_fingerprint(
            {
                "kind": "composition-transport",
                "transport": kind_,
                "route": route,
                "sources": list(zip(sources, structures, strict=True)),
                "targets": [item.record_id for item in staged],
                "conservative": all(present),
            }
        )

    @property
    def conservative(self) -> bool:
        return self.source_content is not None


def _same_layout(source: Any, proposed: Any, /) -> bool:
    source_arrays, source_static = eqx.partition(source, eqx.is_array)
    proposed_arrays, proposed_static = eqx.partition(proposed, eqx.is_array)
    if jax.tree_util.tree_structure(source_arrays) != jax.tree_util.tree_structure(
        proposed_arrays
    ):
        return False
    if not eqx.tree_equal(source_static, proposed_static):
        return False
    return all(
        left.shape == right.shape and left.dtype == right.dtype
        for left, right in zip(
            jax.tree.leaves(source_arrays), jax.tree.leaves(proposed_arrays), strict=True
        )
    )


def _check_refresh(source: CompositionEntry, staged: CompositionEntry, /) -> None:
    if (
        staged.role != source.role
        or staged.owner_id != source.owner_id
        or staged.structure_id != source.structure_id
        or staged.semantics_id != source.semantics_id
    ):
        raise ValueError(
            f"Numeric refresh of {source.entry_id!r} must keep its role, owner, "
            "structure, and semantics; a structural change is a topology reprepare."
        )
    if staged.revision_id == source.revision_id:
        raise ValueError(
            f"Numeric refresh of {source.entry_id!r} must name a new revision."
        )
    if not _same_layout(source.value, staged.value):
        raise ValueError(
            f"Numeric refresh of {source.entry_id!r} changed its PyTree layout; "
            "incompatible prepared objects are reprepared, never selected."
        )
    # A refresh advances values on the prepared structure it already binds; only
    # bound revisions may advance with it. Rebinding onto another structure or
    # meaning (renumbering, repartition, mesh motion) is a reprepare of derived
    # artifacts or an explicit transport of state, even when layouts coincide.
    source_bindings = {
        (item.entry_id, item.facet): item.bound_id for item in source.dependencies
    }
    staged_bindings = {
        (item.entry_id, item.facet): item.bound_id for item in staged.dependencies
    }
    if set(source_bindings) != set(staged_bindings) or any(
        facet != "revision" and staged_bindings[(entry_id, facet)] != bound
        for (entry_id, facet), bound in source_bindings.items()
    ):
        raise ValueError(
            f"Numeric refresh of {source.entry_id!r} must keep its dependencies and "
            "their structure and semantics bindings; only bound revisions may "
            "advance. Rebinding onto another structure is a reprepare of derived "
            "artifacts or an explicit transport of state."
        )


def _check_reprepare(
    source: CompositionEntry | None, staged: CompositionEntry, /
) -> None:
    if _carries_state(staged.role):
        raise ValueError(
            f"State entry {staged.entry_id!r} ({staged.role}) cannot be reprepared; "
            "state crosses a rebind only through an explicit owner transport."
        )
    if source is not None and (
        staged.role != source.role or staged.semantics_id != source.semantics_id
    ):
        raise ValueError(
            f"Reprepared {staged.entry_id!r} must keep its role and semantics."
        )


def _check_transport(source: Composition, transport: CompositionTransport, /) -> None:
    for entry_id, structure_id in zip(
        transport.source_entry_ids, transport.source_structure_ids, strict=True
    ):
        entry = source.entry(entry_id)
        if not _carries_state(entry.role):
            raise ValueError(
                f"Transport {transport.route_id!r} consumes derived artifact "
                f"{entry_id!r}; derived artifacts are reprepared by their owners."
            )
        if entry.structure_id != structure_id:
            raise ValueError(
                f"Transport {transport.route_id!r} was prepared from another "
                f"structure than {entry_id!r} holds."
            )
    members = (
        *(source.entry(item) for item in transport.source_entry_ids),
        *transport.targets,
    )
    if len({(item.role, item.semantics_id) for item in members}) != 1:
        raise ValueError(
            f"Transport {transport.route_id!r} must keep one role and scientific "
            "meaning across its sources and targets."
        )
    if members[0].role == "budget" and not transport.conservative:
        raise ValueError(
            f"Transport {transport.route_id!r} replaces a conservation ledger; a "
            "budget crosses a rebind only with conserved-content evidence."
        )


def _check_invalidation(entry: CompositionEntry, /) -> None:
    if _carries_state(entry.role):
        raise ValueError(
            f"State entry {entry.entry_id!r} ({entry.role}) cannot be invalidated; "
            "unknown state needs an explicit owner transport or the rebind is refused."
        )


def _dispositions(
    source: Composition,
    retained: tuple[str, ...],
    refreshed: tuple[CompositionEntry, ...],
    reprepared: tuple[CompositionEntry, ...],
    transports: tuple[CompositionTransport, ...],
    invalidated: tuple[str, ...],
    /,
) -> None:
    """Every source entry takes exactly one explicit disposition."""
    claims: dict[str, int] = dict.fromkeys(source.entry_ids, 0)
    groups = (
        retained,
        tuple(item.entry_id for item in refreshed),
        tuple(item.entry_id for item in reprepared if item.entry_id in claims),
        tuple(item for transport in transports for item in transport.source_entry_ids),
        invalidated,
    )
    for group in groups:
        for entry_id in group:
            if entry_id not in claims:
                raise ValueError(f"Rebind names unknown source entry {entry_id!r}.")
            claims[entry_id] += 1
    missing = sorted(entry_id for entry_id, count in claims.items() if count == 0)
    repeated = sorted(entry_id for entry_id, count in claims.items() if count > 1)
    if missing:
        raise ValueError(
            "Source entries lack an explicit rebind disposition: " + ", ".join(missing)
        )
    if repeated:
        raise ValueError(
            "Source entries take more than one rebind disposition: " + ", ".join(repeated)
        )


def _orphaned_parameters(
    source: Composition,
    target: Composition,
    transports: tuple[CompositionTransport, ...],
    /,
) -> list[str]:
    """Candidate parameters whose source consumers were all replaced without rebinding.

    A parameter reaches the candidate by retention, refresh, or transport; each
    origin that had consumers in the source must keep at least one consumer.
    """
    produced = {
        target_entry.entry_id: route.source_entry_ids
        for route in transports
        for target_entry in route.targets
    }
    orphaned: list[str] = []
    for entry in target.entries:
        if entry.role != "model-parameter" or target.dependents(entry.entry_id):
            continue
        origins = set(produced.get(entry.entry_id, ()))
        if entry.entry_id in source.entry_ids:
            origins.add(entry.entry_id)
        if any(source.dependents(origin) for origin in sorted(origins)):
            orphaned.append(entry.entry_id)
    return orphaned


@final
class CompositionRebind(StrictModule):
    """Validated, unpublished candidate for one cross-owner rebind.

    Construction is host preparation only: it checks dispositions, role rules,
    transport source identities, refresh layouts, and every binding of the
    candidate composition. Owners stage their target artifacts and transports
    before construction; nothing is published until `commit_composition_rebind`.
    """

    source: Composition
    candidate: Composition
    transports: tuple[CompositionTransport, ...]
    retained: tuple[str, ...] = eqx.field(static=True)
    refreshed: tuple[str, ...] = eqx.field(static=True)
    reprepared: tuple[str, ...] = eqx.field(static=True)
    invalidated: tuple[str, ...] = eqx.field(static=True)
    rebind_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: Composition,
        /,
        *,
        retain: Sequence[str] = (),
        refresh: Sequence[CompositionEntry] = (),
        reprepare: Sequence[CompositionEntry] = (),
        transports: Sequence[CompositionTransport] = (),
        invalidate: Sequence[str] = (),
    ) -> None:
        if not isinstance(source, Composition):
            raise TypeError("source must be a Composition.")
        retained = unique_identifiers(retain, "retain", allow_empty=True, sort=True)
        invalidated = unique_identifiers(
            invalidate, "invalidate", allow_empty=True, sort=True
        )
        refreshed = tuple(sorted(refresh, key=lambda item: item.entry_id))
        reprepared = tuple(sorted(reprepare, key=lambda item: item.entry_id))
        routes = tuple(transports)
        if any(
            not isinstance(item, CompositionEntry) for item in (*refreshed, *reprepared)
        ):
            raise TypeError("Refreshed and reprepared values must be CompositionEntry.")
        if any(not isinstance(item, CompositionTransport) for item in routes):
            raise TypeError("transports must contain CompositionTransport values.")
        _dispositions(source, retained, refreshed, reprepared, routes, invalidated)
        for staged in refreshed:
            _check_refresh(source.entry(staged.entry_id), staged)
        for staged in reprepared:
            known = staged.entry_id in source.entry_ids
            _check_reprepare(source.entry(staged.entry_id) if known else None, staged)
        for route in routes:
            _check_transport(source, route)
        for entry_id in invalidated:
            _check_invalidation(source.entry(entry_id))
        staged_entries = (
            *(source.entry(item) for item in retained),
            *refreshed,
            *reprepared,
            *(item for route in routes for item in route.targets),
        )
        candidate = Composition(staged_entries, boundary_id=source.boundary_id)
        orphaned = _orphaned_parameters(source, candidate, routes)
        if orphaned:
            raise ValueError(
                "Model parameters survive only through an explicit same-semantics "
                "binding of a consumer: " + ", ".join(orphaned)
            )
        self.source = source
        self.candidate = candidate
        self.transports = routes
        self.retained = retained
        self.refreshed = tuple(item.entry_id for item in refreshed)
        self.reprepared = tuple(item.entry_id for item in reprepared)
        self.invalidated = invalidated
        self.rebind_id = canonical_fingerprint(
            {
                "kind": "composition-rebind",
                "source": source.composition_id,
                "candidate": candidate.composition_id,
                "transports": [item.transport_id for item in routes],
                "invalidated": list(invalidated),
            }
        )

    def transported(self, kind: CompositionTransportKind, /) -> tuple[str, ...]:
        """Target entry IDs produced by transports of `kind`."""
        kind_ = parse(kind, CompositionTransportKind, "kind")
        return tuple(
            sorted(
                target.entry_id
                for route in self.transports
                if route.kind == kind_
                for target in route.targets
            )
        )

    @property
    def consumed(self) -> tuple[str, ...]:
        """Source state entries a transport consumed without a same-ID target."""
        return tuple(
            sorted(
                entry_id
                for route in self.transports
                for entry_id in route.source_entry_ids
                if entry_id not in self.candidate.entry_ids
            )
        )


@final
class CompositionRebindReceipt(StrictModule):
    """Published (or refused) composition with its complete rebind account.

    `source_structure_id` and `composition.structure_id` are the topology
    identities a restart relation across a published rebind binds.
    """

    composition: Composition
    transports: tuple[CompositionTransport, ...]
    published: bool = eqx.field(static=True)
    boundary_accepted: bool = eqx.field(static=True)
    transport_accepted: tuple[bool, ...] = eqx.field(static=True)
    retained: tuple[str, ...] = eqx.field(static=True)
    refreshed: tuple[str, ...] = eqx.field(static=True)
    reprepared: tuple[str, ...] = eqx.field(static=True)
    remapped: tuple[str, ...] = eqx.field(static=True)
    migrated: tuple[str, ...] = eqx.field(static=True)
    consumed: tuple[str, ...] = eqx.field(static=True)
    invalidated: tuple[str, ...] = eqx.field(static=True)
    source_composition_id: str = eqx.field(static=True)
    source_structure_id: str = eqx.field(static=True)
    candidate_composition_id: str = eqx.field(static=True)
    receipt_id: str = eqx.field(static=True)


def _transport_accepted(transport: CompositionTransport, /) -> bool:
    """Host decision on one transport's owner status and conservation claim."""
    if not bool(np.asarray(transport.successful)):
        return False
    if not transport.conservative:
        return True
    source = np.asarray(transport.source_content)
    target = np.asarray(transport.target_content)
    tolerance = np.asarray(transport.content_tolerance)
    finite = np.all(np.isfinite(source)) and np.all(np.isfinite(target))
    return bool(finite and np.all(np.abs(target - source) <= tolerance))


def _published(rebind: CompositionRebind, /) -> Composition:
    """Candidate composition with refreshed payloads committed transactionally.

    A refresh keeps its exact layout (checked at staging), so its array payload
    passes through the fixed-structure transaction; everything else is swapped
    whole because old and new prepared objects need not be congruent.
    """
    entries: list[CompositionEntry] = []
    for staged in rebind.candidate.entries:
        if staged.entry_id not in rebind.refreshed:
            entries.append(staged)
            continue
        proposed, static = eqx.partition(staged.value, eqx.is_array)
        current, _ = eqx.partition(rebind.source.value(staged.entry_id), eqx.is_array)
        committed = commit_candidate(
            TransactionalCandidate(current, proposed, None, True, staged.entry_id)
        )
        entries.append(
            CompositionEntry(
                eqx.combine(committed.state, static),
                entry_id=staged.entry_id,
                role=staged.role,
                owner_id=staged.owner_id,
                structure_id=staged.structure_id,
                revision_id=staged.revision_id,
                semantics_id=staged.semantics_id,
                dependencies=staged.dependencies,
            )
        )
    return Composition(entries, boundary_id=rebind.candidate.boundary_id)


def commit_composition_rebind(
    rebind: CompositionRebind, /, *, accepted_boundary: bool
) -> CompositionRebindReceipt:
    """Publish the staged composition only at one explicitly accepted boundary.

    `accepted_boundary` is the caller's explicit host decision that the source
    composition sits at an accepted boundary (for example an accepted coupling
    window). The candidate is published only if that holds and every transport
    reports owner success and, when conservative, content agreement within its
    owner tolerance. Otherwise the receipt carries the original composition
    object, so every old owner artifact and state remains usable.
    """

    if not isinstance(rebind, CompositionRebind):
        raise TypeError("rebind must be a CompositionRebind.")
    if not isinstance(accepted_boundary, bool):
        raise TypeError("accepted_boundary must be an explicit host bool decision.")
    evidence = tuple(_transport_accepted(item) for item in rebind.transports)
    published = accepted_boundary and all(evidence)
    composition = _published(rebind) if published else rebind.source
    remapped = rebind.transported("physical-remap")
    migrated = rebind.transported("ownership-migration")
    receipt_id = canonical_fingerprint(
        {
            "kind": "composition-rebind-receipt",
            "rebind": rebind.rebind_id,
            "published": published,
            "boundary_accepted": accepted_boundary,
            "transport_accepted": list(evidence),
            "composition": composition.composition_id,
        }
    )
    return CompositionRebindReceipt(
        composition,
        rebind.transports,
        published,
        accepted_boundary,
        evidence,
        rebind.retained,
        rebind.refreshed,
        rebind.reprepared,
        remapped,
        migrated,
        rebind.consumed,
        rebind.invalidated,
        rebind.source.composition_id,
        rebind.source.structure_id,
        rebind.candidate.composition_id,
        receipt_id,
    )


__all__ = [
    "Composition",
    "CompositionDependency",
    "CompositionEntry",
    "CompositionFacet",
    "CompositionRebind",
    "CompositionRebindReceipt",
    "CompositionRole",
    "CompositionTransport",
    "CompositionTransportKind",
    "commit_composition_rebind",
]
