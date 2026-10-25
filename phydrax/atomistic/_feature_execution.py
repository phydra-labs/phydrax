#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owner-region layer schedule of learned atomistic message passing.

This module is the strategy-neutral executor of one owner region (``R`` owned
receiver rows, inside ``ownership.map`` with named axis ``axis_name``). It owns
only the domain schedule of a layered potential: which node payload crosses
the owner boundary before each interaction (an `AtomisticColumnExchange`), the
order of interactions and declared system-reduction stages, the readout and
optional global (reciprocal) epilogue, and the explicit reverse sweep.

Columns
    Each owner holds ``C = exchange.column_count`` source columns. Owner-local
    edges carry ``(receiver row, source column, n)`` with displacement
    ``d = x_receiver - x_source + n @ H`` from the gathered coordinates, so a
    periodic image alias reuses its physical atom's column and repeated images
    keep their multiplicity as distinct edges. `HaloColumnExchange` binds the
    canonical `DistributedHaloPlan` (owned rows followed by deduplicated halo
    columns); the graph-partition route supplies its own replicated exchange.

System stages
    A model may declare `AtomisticSystemStage` reductions. At a stage every
    owner forms ``system_partials`` of its owned receivers, `owner_system_sum`
    combines them into one replicated ``system_context`` in a declared order,
    and ``apply_system_context`` consumes it (for example AIMNet2 charge
    equilibration or UMA composition conditioning). The transpose of a
    replicated consumption sums every owner's local cotangent once with the
    same owner sum and returns it once to the owned partial producers; nothing
    is scaled by the owner count.

Global epilogue
    ``global_payload`` of a nonlocal model (MACELES latent charges) feeds an
    `OwnerGlobalTerm` injected by the runtime (the distributed reciprocal
    term). A replicated ``system_energy`` is added exactly once, on axis index
    zero. ``global_energy`` of a result is this owner's share of both.

Reverse sweep
    `owner_layer_gradients` retains only inter-layer owned states, replays each
    interaction and stage in reverse from them, applies the model's ordinary
    JAX local VJPs with respect to the parameter lane, node states and layer
    context, returns source-column cotangents once through the exchange
    transpose and context cotangents once through the owner sum, and routes
    the edge-displacement cotangent to receivers and (through the transpose)
    sources. Everything is ordinary differentiable JAX in both modes: forces
    may themselves be differentiated in reverse mode (parameter gradients of
    force losses) and in forward mode (``jax.linearize`` Hessian actions
    ``H v = -DF v``). Exchanges and owner sums are linear collectives whose
    tangents are the same collectives and whose JAX transposes return every
    contribution exactly once (see `owner_system_sum`).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import final, Literal, NamedTuple, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree

from .._numerics._compensated import compensated_sum
from .._strict import StrictModule
from .._trainable import combine_parameters, NonTrainableState, partition_parameters
from ..backends.distributed import JaxCollectiveProvider
from ..discretization.spatial._distributed_relations import DistributedHaloPlan
from ..ein import contract
from ..sparse import RelationAccumulation, RelationExecutionState
from ..sparse._streamed import PreparedStreamedRelation, StreamedRelationEvidence
from ..typing import PRNGKey
from ._conditioning import AtomisticConditioning, AtomisticObservable


type NodeState = PyTree[Array]
"""PyTree of node arrays whose leaves all lead with the node (row/column) axis."""

type OwnerReductionMode = Literal["fast", "deterministic", "compensated"]
"""``fast``: native ``psum``; ``deterministic``: owner-ordered sequential sum of
an all-gather; ``compensated``: owner-ordered compensated (Sum2) sum."""

_REDUCTION_MODES: tuple[OwnerReductionMode, ...] = (
    "fast",
    "deterministic",
    "compensated",
)


class AtomisticExchangeCapacityError(ValueError):
    """A column exchange message exceeds its declared byte capacity.

    Raised at trace time before any communication; it is a resource refusal,
    distinct from a scientific or topology failure.
    """


def _inexact(value: Array) -> bool:
    return jnp.issubdtype(value.dtype, jnp.inexact)


def _zero_cotangent(primal: Array) -> Array | np.ndarray:
    if _inexact(primal):
        return jnp.zeros_like(primal)
    return np.zeros(primal.shape, dtype=jax.dtypes.float0)


def _add_cotangents(first: PyTree[Array], second: PyTree[Array]) -> PyTree[Array]:
    def add(left: Array, right: Array) -> Array:
        if left.dtype == jax.dtypes.float0:
            return left
        return left + right

    return jax.tree.map(add, first, second)


def _row_bytes(tree: NodeState, /) -> int:
    return sum(
        (leaf.size // leaf.shape[0]) * leaf.dtype.itemsize
        for leaf in jax.tree.leaves(tree)
    )


class AtomisticColumnExchange(Protocol):
    """Owned rows to the receiver-visible source columns of one owner.

    ``gather`` is differentiable and its VJP is ``transpose``; ``transpose``
    is the exact adjoint that returns every column contribution to its owned
    row exactly once (integer primal leaves receive ``float0`` cotangents).
    ``message_bytes`` is the static per-exchange message charge.
    """

    @property
    def column_count(self) -> int: ...

    def gather(self, owned: NodeState, /) -> NodeState: ...

    def transpose(self, columns: NodeState, owned: NodeState, /) -> NodeState: ...

    def message_bytes(self, owned: NodeState, /) -> int: ...


@final
class HaloColumnExchange(StrictModule):
    """One owner's halo columns: owned rows then deduplicated halo columns.

    Construct INSIDE a mapped owner region with this owner's ``send_slots`` and
    ``send_valid`` (``owner_count * halo_capacity`` entries each, flat or
    ``(owner_count, halo_capacity)``). One message is ``owner_count *
    halo_capacity`` padded rows of the payload; a payload whose static byte
    count exceeds ``message_capacity_bytes`` is refused at trace time with
    `AtomisticExchangeCapacityError`.

    Both directions are the halo owner's linear JAX maps (a slot gather plus
    ``all_to_all``; ``all_to_all`` plus requester-ordered slot additions), so
    tangents propagate in forward mode as the same exchange and reverse mode
    transposes them exactly (``all_to_all`` is its own inverse transpose). The
    explicit reverse sweep calls ``transpose`` directly, which returns halo
    contributions in ascending requesting-owner order.
    """

    halo: DistributedHaloPlan
    send_slots: Array
    send_valid: Array
    message_capacity_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        halo: DistributedHaloPlan,
        send_slots: Array,
        send_valid: Array,
        message_capacity_bytes: int,
        /,
    ) -> None:
        if not isinstance(halo, DistributedHaloPlan):
            raise TypeError("halo must be a DistributedHaloPlan.")
        entries = halo.ownership.owner_count * halo.halo_capacity
        slots = jnp.asarray(send_slots)
        valid = jnp.asarray(send_valid)
        if slots.size != entries or valid.size != entries or slots.shape != valid.shape:
            raise ValueError(
                "send_slots and send_valid must hold owner_count * halo_capacity "
                "entries of this owner."
            )
        if not jnp.issubdtype(slots.dtype, jnp.integer):
            raise TypeError("send_slots must be integer slots.")
        if valid.dtype != jnp.bool_:
            raise TypeError("send_valid must be boolean.")
        if (
            isinstance(message_capacity_bytes, bool)
            or not isinstance(message_capacity_bytes, int)
            or message_capacity_bytes <= 0
        ):
            raise ValueError("message_capacity_bytes must be a positive integer.")
        self.halo = halo
        self.send_slots = slots
        self.send_valid = valid
        self.message_capacity_bytes = message_capacity_bytes

    @property
    def column_count(self) -> int:
        return self.halo.column_count

    def message_bytes(self, owned: NodeState, /) -> int:
        rows = self.halo.ownership.owner_count * self.halo.halo_capacity
        return rows * _row_bytes(owned)

    def _require_rows(self, tree: NodeState, rows: int, name: str, /) -> None:
        for leaf in jax.tree.leaves(tree):
            if leaf.ndim == 0 or leaf.shape[0] != rows:
                raise ValueError(f"Every {name} leaf must lead with {rows} rows.")

    def gather(self, owned: NodeState, /) -> NodeState:
        self._require_rows(owned, self.halo.ownership.local_capacity, "owned")
        message = self.message_bytes(owned)
        if message > self.message_capacity_bytes:
            raise AtomisticExchangeCapacityError(
                f"One halo exchange needs {message} bytes, exceeding "
                f"message_capacity_bytes={self.message_capacity_bytes}."
            )

        return jax.tree.map(
            lambda value: self.halo.local_gather(value, self.send_slots, self.send_valid),
            owned,
        )

    def transpose(self, columns: NodeState, owned: NodeState, /) -> NodeState:
        def leaf(cotangent: Array, primal: Array) -> Array | np.ndarray:
            if not _inexact(primal):
                return _zero_cotangent(primal)
            if cotangent.shape[0] != self.column_count:
                raise ValueError("Column cotangents must lead with column_count rows.")
            return self.halo.local_transpose(
                cotangent.astype(primal.dtype), self.send_slots, self.send_valid
            )

        return jax.tree.map(leaf, columns, owned)


class OwnerLayerTopology(NamedTuple):
    """One owner's prepared edges inside a mapped owner region.

    ``valid`` is structural edge validity (including any per-evaluation
    selection mask); ``receivers`` index owned rows and ``columns`` index the
    exchange's source columns; ``shifts`` are integer image vectors ``n``.
    ``route_identity`` holds the canonical per-edge semantic addresses in the
    exact `AtomisticGraphInputs.route_identity_fields` order (case identity
    words, source particle id, receiver particle id, material image shift
    components), independent of rank, layout and route storage; ``None`` when
    the runtime does not bind them (keyed models then refuse).
    """

    relation: PreparedStreamedRelation
    receiver_reduction: RelationExecutionState
    source_reduction: RelationExecutionState
    receivers: Array
    columns: Array
    shifts: Array
    valid: Array
    route_identity: tuple[Array, ...] | None = None


class OwnerEdgeGeometry(NamedTuple):
    """Owner-local edge displacements and their cutoff activity.

    ``coincident`` is whether an active edge has zero length (a scientific
    geometry failure, never a capacity failure).
    """

    displacements: Array
    vectors: Array
    active: Array
    coincident: Array


@final
class AtomisticSystemStage(StrictModule, NonTrainableState):
    """One declared system reduction of a layered model.

    ``after_interaction`` is ``-1`` for a prologue before ``init_features``,
    ``k`` after interaction ``k`` (``interaction_count - 1``: before the
    readout). Stages with equal positions run in declaration order.
    """

    name: str = eqx.field(static=True)
    after_interaction: int = eqx.field(static=True)

    def __init__(self, name: str, after_interaction: int) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("A system stage needs a non-empty name.")
        if (
            isinstance(after_interaction, bool)
            or not isinstance(after_interaction, int)
            or after_interaction < -1
        ):
            raise ValueError("after_interaction must be an integer >= -1.")
        self.name = name.strip()
        self.after_interaction = after_interaction


class OwnerConditioning(NamedTuple):
    """Conditioning of one owner region's ``layer_context``.

    ``conditioning`` is the system's typed conditioning (or ``None``);
    ``atom_cases`` is the case index of every owned row (zeros for one system);
    ``key`` is the evaluation key of keyed (stochastic-gauge) models.
    """

    conditioning: AtomisticConditioning | None
    atom_cases: Array
    key: PRNGKey | None = None


class AtomisticLayerUpdate(Protocol):
    """Updated receiver states, per-receiver layer energies, streamed evidence."""

    @property
    def features(self) -> NodeState: ...

    @property
    def energy(self) -> Array: ...

    @property
    def evidence(self) -> StreamedRelationEvidence: ...


class AtomisticLayeredModel(Protocol):
    """Interaction-layer contract of a learned message-passing potential.

    Every method is pure ordinary JAX, padding safe, owner local (no hidden
    collective) and makes no global-node assumption: receivers ``R`` and source
    columns ``C`` may differ, and the relation may be rectangular. ``index``
    is a static interaction index and ``stage`` a static index into
    ``system_stages``. ``context`` is the model's layer context PyTree; it may
    carry parameter-derived leaves and is differentiated like node states.
    ``species`` are the model's executor species (`species_indices`).
    """

    @property
    def interaction_count(self) -> int: ...

    @property
    def system_stages(self) -> tuple[AtomisticSystemStage, ...]: ...

    def species_indices(self, species_ids: Array, atom_mask: Array, /) -> Array: ...

    def layer_context(
        self,
        conditioning: AtomisticConditioning | None,
        atom_cases: Array,
        species: Array,
        atom_mask: Array,
        /,
        *,
        key: PRNGKey | None = None,
    ) -> PyTree[Array]: ...

    def admitted(
        self, species: Array, atom_mask: Array, context: PyTree[Array], /
    ) -> Array: ...

    def system_partials(
        self,
        stage: int,
        features: NodeState | None,
        species: Array,
        atom_mask: Array,
        context: PyTree[Array],
        /,
    ) -> PyTree[Array]: ...

    def apply_system_context(
        self,
        stage: int,
        features: NodeState | None,
        system_context: PyTree[Array],
        species: Array,
        atom_mask: Array,
        context: PyTree[Array],
        /,
    ) -> tuple[NodeState | None, PyTree[Array]]: ...

    def init_features(
        self, species: Array, atom_mask: Array, context: PyTree[Array], /
    ) -> NodeState: ...

    def prepare_sources(
        self,
        index: int,
        features: NodeState,
        species: Array,
        atom_mask: Array,
        context: PyTree[Array],
        /,
    ) -> NodeState: ...

    def update_receivers(
        self,
        index: int,
        relation: PreparedStreamedRelation,
        vectors: Array,
        sources: NodeState,
        features: NodeState,
        species: Array,
        atom_mask: Array,
        context: PyTree[Array],
        /,
        *,
        edge_active: Array | None = None,
        route_identity: tuple[Array, ...] | None = None,
    ) -> AtomisticLayerUpdate: ...

    def atom_energies(
        self,
        species: Array,
        atom_mask: Array,
        layer_energies: Array,
        features: NodeState,
        context: PyTree[Array],
        /,
    ) -> Array: ...

    def global_payload(
        self,
        features: NodeState,
        species: Array,
        atom_mask: Array,
        context: PyTree[Array],
        /,
    ) -> PyTree[Array] | None: ...

    def system_energy(self, context: PyTree[Array], /) -> Array | None: ...

    def owner_fields(
        self,
        features: NodeState,
        species: Array,
        atom_mask: Array,
        context: PyTree[Array],
        /,
    ) -> Mapping[AtomisticObservable, Array]: ...


class OwnerGlobalTerm(Protocol):
    """Runtime-injected global epilogue (for example distributed reciprocal).

    Returns this owner's energy share (summed once over owners) and its
    success; it is differentiable in positions, payload and cell, including
    its own collectives.
    """

    def __call__(
        self,
        positions: Array,
        payload: PyTree[Array],
        atom_mask: Array,
        cell_vectors: Array,
        /,
    ) -> tuple[Array, Array]: ...


class OwnerFieldTerm(Protocol):
    """Runtime-injected terms consuming the model's predicted per-atom fields.

    ``fields`` are the model's `owner_fields` of the owned rows (for example
    predicted charges). Returns this owner's energy share (summed once over
    owners), its success and an auxiliary per-term record; it is
    differentiable in positions, fields and cell, including its collectives,
    so its cotangent returns through the layered reverse sweep exactly once.
    """

    def __call__(
        self,
        positions: Array,
        fields: Mapping[AtomisticObservable, Array],
        atom_mask: Array,
        cell_vectors: Array,
        /,
    ) -> tuple[Array, Array, PyTree[Array]]: ...


class OwnerLayerForward(NamedTuple):
    """Owner contributions of one forward sweep.

    ``atom_energies`` are this owner's rows (zero on inactive rows);
    ``global_energy`` is this owner's share of the global-term, system and
    field-term energies; ``system_contexts`` are the replicated forward system
    contexts in stage order. ``relation_successful`` (streamed evidence) and
    ``model_successful`` (model admission, no coincident active edge, global
    and field term success) are owner local. ``global_share_declared`` is the
    structural (trace-time) fact that the model declares a global payload or a
    system energy, so its atom energies are not a complete decomposition;
    ``field_aux`` is the field term's auxiliary output (``None`` without one).
    """

    atom_energies: Array
    global_energy: Array
    system_contexts: tuple[PyTree[Array], ...]
    relation_successful: Array
    model_successful: Array
    global_share_declared: Array
    field_aux: PyTree[Array] | None


class OwnerLayerGradients(NamedTuple):
    """Owner contributions of one explicit layered reverse sweep.

    ``gradient`` is ``dE/dx`` of the owned rows after every column and global
    contribution has returned exactly once. ``strain_gradient`` is this
    owner's partial of ``dE/dstrain`` (edge partial plus global-term position
    and cell partials); the shared cell cotangent is the owner sum of these
    partials. ``parameter_gradient`` is this owner's PARAMETER-lane partial
    (``None`` unless requested); the owner sum is the global gradient.
    ``global_share_declared`` and ``field_aux`` are as in `OwnerLayerForward`.
    """

    atom_energies: Array
    global_energy: Array
    gradient: Array
    strain_gradient: Array
    parameter_gradient: PyTree[Array] | None
    system_contexts: tuple[PyTree[Array], ...]
    relation_successful: Array
    model_successful: Array
    global_share_declared: Array
    field_aux: PyTree[Array] | None


def _require_mode(mode: str, /) -> OwnerReductionMode:
    """The declared reduction mode equal to ``mode``."""
    for declared in _REDUCTION_MODES:
        if mode == declared:
            return declared
    raise ValueError(f"reduction_mode must be one of {_REDUCTION_MODES}.")


def _ordered_owner_sum(
    partials: PyTree[Array], axis_name: str, mode: OwnerReductionMode, /
) -> PyTree[Array]:
    provider = JaxCollectiveProvider(axis_name)

    def leaf(value: Array) -> Array:
        if mode == "fast" or not _inexact(value):
            return provider.sum(value)
        gathered = provider.all_gather(value, tiled=False)
        if mode == "compensated":
            return compensated_sum(gathered, axis=0).astype(value.dtype)
        return jax.lax.fori_loop(
            0,
            gathered.shape[0],
            lambda index, total: total + gathered[index],
            jnp.zeros(value.shape, dtype=value.dtype),
        )

    return jax.tree.map(leaf, partials)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _owner_system_sum(
    partials: PyTree[Array], axis_name: str, mode: OwnerReductionMode, /
) -> PyTree[Array]:
    return _ordered_owner_sum(partials, axis_name, mode)


@_owner_system_sum.defjvp
def _owner_system_sum_jvp(
    axis_name: str,
    mode: OwnerReductionMode,
    primals: tuple[PyTree[Array]],
    tangents: tuple[PyTree[Array]],
) -> tuple[PyTree[Array], PyTree[Array]]:
    """The tangent is the same declared owner sum of the tangents (linear)."""
    (partials,) = primals
    (tangent,) = tangents
    linear = jax.tree.map(
        lambda primal, value: value if _inexact(primal) else None, partials, tangent
    )
    summed = _ordered_owner_sum(linear, axis_name, mode)
    output = jax.tree.map(
        lambda primal, value: _zero_cotangent(primal) if value is None else value,
        partials,
        summed,
    )
    return _ordered_owner_sum(partials, axis_name, mode), output


def owner_system_sum(
    partials: PyTree[Array], axis_name: str, mode: OwnerReductionMode
) -> PyTree[Array]:
    """Replicated owner sum of per-owner partials in a declared order.

    ``fast`` uses the native ``psum``; ``deterministic`` and ``compensated``
    all-gather the partials and sum them in ascending owner order (plain or
    Sum2-compensated), so every owner holds bitwise-identical values. Integer
    leaves are summed exactly in every mode; boolean leaves are refused.

    The tangent of the sum is the same declared owner sum of the tangents, so
    forward mode (``jvp``/``linearize``, Hessian actions) runs through it. Its
    reverse mode is JAX's transpose of that linear map: ``psum`` transposes to
    ``psum`` and the ordered all-gather sum to ``psum_scatter`` of the
    broadcast cotangent. Both deliver, for per-owner local surrogates
    ``L = sum_o L_o``, the complete ``dL/dS = sum_o dL_o/dS`` to every owner's
    partial producer exactly once, never scaled by the owner count. This
    holds in lane-reference regions (named ``vmap``) and in ``shard_map``
    regions without varying-manual-axes checking (``check_vma=False``, as
    ``DistributedOwnershipPlan.map`` runs them), where ``psum`` binds the
    primitive whose transpose is ``psum``.
    """
    if not isinstance(axis_name, str) or not axis_name:
        raise ValueError("axis_name must be a non-empty string.")
    mode_ = _require_mode(mode)
    for leaf in jax.tree.leaves(partials):
        if not (_inexact(leaf) or jnp.issubdtype(leaf.dtype, jnp.integer)):
            raise TypeError(
                "System partials must be inexact or integer arrays (use int32 counts "
                "instead of booleans)."
            )
    return _owner_system_sum(partials, axis_name, mode_)


def owner_edge_geometry(
    exchange: AtomisticColumnExchange,
    topology: OwnerLayerTopology,
    positions: Array,
    cell_vectors: Array,
    cutoff: float,
) -> OwnerEdgeGeometry:
    """Gather source coordinates and form ``d = x_r - x_s + n @ H`` per edge.

    Padded edges evaluate a fixed nonzero in-domain sentinel displacement so
    directional and radial functions stay finite; their activity is false and
    their recorded displacement is zero, so padding never reaches the strain
    partial.
    """
    columns = exchange.gather(positions)
    translations = contract(
        "ea,ad->ed", topology.shifts.astype(positions.dtype), cell_vectors
    )
    raw = positions[topology.receivers] - columns[topology.columns] + translations
    sentinel = (
        jnp.zeros((positions.shape[-1],), dtype=positions.dtype).at[0].set(0.5 * cutoff)
    )
    valid = topology.valid[:, None]
    vectors = jnp.where(valid, raw, sentinel)
    squared = jnp.sum(vectors * vectors, axis=-1)
    active = topology.valid & (squared < cutoff * cutoff)
    coincident = jnp.any(active & (squared <= 0.0))
    return OwnerEdgeGeometry(jnp.where(valid, raw, 0.0), vectors, active, coincident)


class _Operation(NamedTuple):
    kind: Literal["stage", "init", "interaction"]
    index: int


def _operations(model: AtomisticLayeredModel, /) -> tuple[_Operation, ...]:
    """Static execution order of stages, initialization and interactions."""
    count = model.interaction_count
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("A layered model needs at least one interaction.")
    stages = tuple(model.system_stages)
    positions = []
    for stage in stages:
        if not isinstance(stage, AtomisticSystemStage):
            raise TypeError("system_stages must be AtomisticSystemStage values.")
        if stage.after_interaction > count - 1:
            raise ValueError(
                f"Stage {stage.name!r} follows interaction {stage.after_interaction} "
                f"of a {count}-interaction model."
            )
        positions.append(stage.after_interaction)
    if positions != sorted(positions):
        raise ValueError("system_stages must be ordered by after_interaction.")
    operations = [
        _Operation("stage", index)
        for index, stage in enumerate(stages)
        if stage.after_interaction == -1
    ]
    operations.append(_Operation("init", -1))
    for interaction in range(count):
        operations.append(_Operation("interaction", interaction))
        operations.extend(
            _Operation("stage", index)
            for index, stage in enumerate(stages)
            if stage.after_interaction == interaction
        )
    return tuple(operations)


class _Region(NamedTuple):
    """Static operands of one owner region."""

    exchange: AtomisticColumnExchange
    topology: OwnerLayerTopology
    geometry: OwnerEdgeGeometry
    species: Array
    atom_mask: Array
    axis_name: str
    mode: OwnerReductionMode


class _ForwardSweep(NamedTuple):
    inputs: tuple[tuple[NodeState | None, PyTree[Array]], ...]
    system_contexts: tuple[PyTree[Array], ...]
    features: NodeState
    context: PyTree[Array]
    layer_energies: Array
    relation_successful: Array


def _forward(
    model: AtomisticLayeredModel,
    region: _Region,
    operations: tuple[_Operation, ...],
    context: PyTree[Array],
    /,
) -> _ForwardSweep:
    """Run every operation; retain only the inter-operation owned states."""
    species, mask = region.species, region.atom_mask
    features: NodeState | None = None
    inputs: list[tuple[NodeState | None, PyTree[Array]]] = []
    contexts: list[PyTree[Array]] = []
    energies: list[Array] = []
    successful = jnp.asarray(True)
    for operation in operations:
        inputs.append((features, context))
        match operation.kind:
            case "stage":
                partial = model.system_partials(
                    operation.index, features, species, mask, context
                )
                system = owner_system_sum(partial, region.axis_name, region.mode)
                contexts.append(system)
                updated, context = model.apply_system_context(
                    operation.index, features, system, species, mask, context
                )
                if (features is None) != (updated is None):
                    raise ValueError(
                        "A prologue stage has no node states and later stages must "
                        "return them."
                    )
                features = updated
            case "init":
                features = model.init_features(species, mask, context)
            case "interaction":
                sources = model.prepare_sources(
                    operation.index, features, species, mask, context
                )
                update = model.update_receivers(
                    operation.index,
                    region.topology.relation,
                    region.geometry.vectors,
                    region.exchange.gather(sources),
                    features,
                    species,
                    mask,
                    context,
                    edge_active=region.geometry.active,
                    route_identity=region.topology.route_identity,
                )
                features = update.features
                energies.append(update.energy)
                successful = successful & update.evidence.successful
    return _ForwardSweep(
        tuple(inputs),
        tuple(contexts),
        features,
        context,
        jnp.stack(energies),
        successful,
    )


class _Readout(NamedTuple):
    atom_energies: Array
    global_energy: Array
    global_successful: Array
    global_share_declared: Array
    field_aux: PyTree[Array] | None


def _readout(
    model: AtomisticLayeredModel,
    region: _Region,
    layer_energies: Array,
    features: NodeState,
    context: PyTree[Array],
    positions: Array,
    cell_vectors: Array,
    global_term: OwnerGlobalTerm | None,
    field_term: OwnerFieldTerm | None,
    /,
) -> _Readout:
    species, mask = region.species, region.atom_mask
    atom = model.atom_energies(species, mask, layer_energies, features, context)
    atom = jnp.where(mask, atom, jnp.zeros((), dtype=atom.dtype))
    payload = model.global_payload(features, species, mask, context)
    if payload is not None and global_term is None:
        raise ValueError(
            "The model is nonlocal (it declares a global payload); the runtime must "
            "inject its global term."
        )
    if payload is None and global_term is not None:
        raise ValueError("A global term was supplied for a model without payload.")
    share = jnp.zeros((), dtype=atom.dtype)
    successful = jnp.asarray(True)
    if payload is not None and global_term is not None:
        energy, successful = global_term(positions, payload, mask, cell_vectors)
        share = share + jnp.asarray(energy).astype(atom.dtype)
    system = model.system_energy(context)
    if system is not None:
        first = jax.lax.axis_index(region.axis_name) == 0
        share = share + jnp.where(
            first, jnp.asarray(system).astype(atom.dtype), jnp.zeros((), atom.dtype)
        )
    aux = None
    if field_term is not None:
        fields = model.owner_fields(features, species, mask, context)
        energy, field_successful, aux = field_term(positions, fields, mask, cell_vectors)
        share = share + jnp.asarray(energy).astype(atom.dtype)
        successful = jnp.asarray(successful, dtype=jnp.bool_) & jnp.asarray(
            field_successful, dtype=jnp.bool_
        )
    declared = jnp.asarray(payload is not None or system is not None)
    return _Readout(atom, share, jnp.asarray(successful, dtype=jnp.bool_), declared, aux)


def _region(
    exchange: AtomisticColumnExchange,
    topology: OwnerLayerTopology,
    positions: Array,
    species: Array,
    atom_mask: Array,
    cell_vectors: Array,
    cutoff: float,
    axis_name: str,
    reduction_mode: str,
    /,
) -> _Region:
    mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
    if positions.ndim != 2 or mask.shape != positions.shape[:1]:
        raise ValueError("positions and atom_mask must cover the owned rows.")
    if species.shape != mask.shape:
        raise ValueError("species must hold one entry per owned row.")
    if not isinstance(axis_name, str) or not axis_name:
        raise ValueError("axis_name must be a non-empty string.")
    geometry = owner_edge_geometry(exchange, topology, positions, cell_vectors, cutoff)
    return _Region(
        exchange,
        topology,
        geometry,
        species,
        mask,
        axis_name,
        _require_mode(reduction_mode),
    )


def owner_atom_energies(
    model: AtomisticLayeredModel,
    exchange: AtomisticColumnExchange,
    topology: OwnerLayerTopology,
    positions: Array,
    species: Array,
    atom_mask: Array,
    cell_vectors: Array,
    cutoff: float,
    *,
    context: OwnerConditioning,
    reduction_mode: OwnerReductionMode,
    axis_name: str,
    global_term: OwnerGlobalTerm | None = None,
    field_term: OwnerFieldTerm | None = None,
) -> OwnerLayerForward:
    """Owned atom energies, global share and system contexts of one owner.

    Ordinary differentiable JAX: the exchange and owner sums carry their
    exact adjoints, so reverse-mode derivatives of this forward are exact.
    """
    if not isinstance(context, OwnerConditioning):
        raise TypeError("context must be an OwnerConditioning.")
    operations = _operations(model)
    region = _region(
        exchange,
        topology,
        positions,
        species,
        atom_mask,
        cell_vectors,
        cutoff,
        axis_name,
        reduction_mode,
    )
    layer_context = model.layer_context(
        context.conditioning,
        context.atom_cases,
        region.species,
        region.atom_mask,
        key=context.key,
    )
    sweep = _forward(model, region, operations, layer_context)
    readout = _readout(
        model,
        region,
        sweep.layer_energies,
        sweep.features,
        sweep.context,
        positions,
        cell_vectors,
        global_term,
        field_term,
    )
    admitted = model.admitted(region.species, region.atom_mask, layer_context)
    return OwnerLayerForward(
        readout.atom_energies,
        readout.global_energy,
        sweep.system_contexts,
        sweep.relation_successful,
        jnp.asarray(admitted, dtype=jnp.bool_)
        & ~region.geometry.coincident
        & readout.global_successful,
        readout.global_share_declared,
        readout.field_aux,
    )


type _Bind = Callable[[PyTree[Array]], AtomisticLayeredModel]


def _add_parameters(total: PyTree[Array], *terms: PyTree[Array]) -> PyTree[Array]:
    for term in terms:
        total = jax.tree.map(jnp.add, total, term)
    return total


def owner_layer_gradients(
    model: AtomisticLayeredModel,
    exchange: AtomisticColumnExchange,
    topology: OwnerLayerTopology,
    positions: Array,
    species: Array,
    atom_mask: Array,
    cell_vectors: Array,
    cutoff: float,
    *,
    context: OwnerConditioning,
    reduction_mode: OwnerReductionMode,
    axis_name: str,
    accumulation: RelationAccumulation,
    parameters: bool,
    global_term: OwnerGlobalTerm | None = None,
    field_term: OwnerFieldTerm | None = None,
) -> OwnerLayerGradients:
    """Owned energies plus the explicit reverse sweep of one owner.

    The seed is ``dE/d atom_energy = 1`` on every active owned atom and
    ``dE/d global_energy = 1``, so the sum over owners of every partial is the
    derivative of the global energy. The forward retains only the owned states
    between operations; each interaction and stage is recomputed from them in
    reverse. A stage's context cotangent is the owner sum of the local
    cotangents, applied once to the owned partial producers.
    """
    if not isinstance(context, OwnerConditioning):
        raise TypeError("context must be an OwnerConditioning.")
    if not isinstance(parameters, bool):
        raise TypeError("parameters must be a bool.")
    operations = _operations(model)
    region = _region(
        exchange,
        topology,
        positions,
        species,
        atom_mask,
        cell_vectors,
        cutoff,
        axis_name,
        reduction_mode,
    )
    species_, mask = region.species, region.atom_mask
    geometry = region.geometry
    lanes = partition_parameters(model)
    lane = lanes[0]

    def bind(values: PyTree[Array]) -> AtomisticLayeredModel:
        return combine_parameters(values, lanes[1], lanes[2])

    layer_context, context_vjp = jax.vjp(
        lambda values: bind(values).layer_context(
            context.conditioning, context.atom_cases, species_, mask, key=context.key
        ),
        lane,
    )
    sweep = _forward(model, region, operations, layer_context)
    successful = sweep.relation_successful

    def readout(
        values: PyTree[Array],
        layer_energies: Array,
        features: NodeState,
        state: PyTree[Array],
        coordinates: Array,
        cell: Array,
    ) -> tuple[Array, _Readout]:
        result = _readout(
            bind(values),
            region,
            layer_energies,
            features,
            state,
            coordinates,
            cell,
            global_term,
            field_term,
        )
        return jnp.sum(result.atom_energies) + result.global_energy, result

    _, readout_vjp, readout_values = jax.vjp(
        readout,
        lane,
        sweep.layer_energies,
        sweep.features,
        sweep.context,
        positions,
        cell_vectors,
        has_aux=True,
    )
    (
        parameter_cotangent,
        energy_cotangent,
        feature_cotangent,
        context_cotangent,
        global_positions,
        global_cell,
    ) = readout_vjp(jnp.ones((), dtype=readout_values.atom_energies.dtype))
    vector_cotangent = jnp.zeros_like(geometry.vectors)
    stage_contexts = list(sweep.system_contexts)
    for operation, (features, state) in zip(
        reversed(operations), reversed(sweep.inputs), strict=True
    ):
        match operation.kind:
            case "stage":
                system = stage_contexts.pop()
                cotangents = _stage_reverse(
                    bind,
                    region,
                    operation.index,
                    lane,
                    features,
                    system,
                    state,
                    (feature_cotangent, context_cotangent),
                )
                parameter_cotangent = _add_parameters(
                    parameter_cotangent, cotangents.parameters
                )
                feature_cotangent = cotangents.features
                context_cotangent = cotangents.context
            case "init":
                _, init_vjp = jax.vjp(
                    lambda values, current: bind(values).init_features(
                        species_, mask, current
                    ),
                    lane,
                    state,
                )
                initial_parameters, initial_context = init_vjp(feature_cotangent)
                parameter_cotangent = _add_parameters(
                    parameter_cotangent, initial_parameters
                )
                context_cotangent = _add_cotangents(context_cotangent, initial_context)
                feature_cotangent = None
            case "interaction":
                cotangents = _interaction_reverse(
                    bind,
                    region,
                    operation.index,
                    lane,
                    features,
                    state,
                    feature_cotangent,
                    context_cotangent,
                    energy_cotangent[operation.index],
                )
                parameter_cotangent = _add_parameters(
                    parameter_cotangent, cotangents.parameters
                )
                feature_cotangent = cotangents.features
                context_cotangent = cotangents.context
                vector_cotangent = vector_cotangent + cotangents.vectors
                successful = successful & cotangents.successful
    (context_parameters,) = context_vjp(context_cotangent)
    parameter_cotangent = _add_parameters(parameter_cotangent, context_parameters)
    edge_cotangent = jnp.where(topology.valid[:, None], vector_cotangent, 0.0)
    gradient, reduction_successful = _edge_position_gradient(
        exchange, topology, edge_cotangent, positions, accumulation=accumulation
    )
    direct = jnp.where(mask[:, None], global_positions, 0.0)
    gradient = gradient + direct
    strain_gradient = (
        contract("ea,eb->ab", edge_cotangent, geometry.displacements)
        + contract("ia,ib->ab", direct, positions)
        + contract("ka,kb->ab", global_cell, cell_vectors)
    )
    admitted = model.admitted(species_, mask, layer_context)
    return OwnerLayerGradients(
        atom_energies=readout_values.atom_energies,
        global_energy=readout_values.global_energy,
        gradient=jnp.where(mask[:, None], gradient, 0.0),
        strain_gradient=strain_gradient,
        parameter_gradient=parameter_cotangent if parameters else None,
        system_contexts=sweep.system_contexts,
        relation_successful=successful & reduction_successful,
        model_successful=jnp.asarray(admitted, dtype=jnp.bool_)
        & ~geometry.coincident
        & readout_values.global_successful,
        global_share_declared=readout_values.global_share_declared,
        field_aux=readout_values.field_aux,
    )


class _StageCotangents(NamedTuple):
    parameters: PyTree[Array]
    features: NodeState | None
    context: PyTree[Array]


def _stage_reverse(
    bind: _Bind,
    region: _Region,
    stage: int,
    lane: PyTree[Array],
    features: NodeState | None,
    system: PyTree[Array],
    state: PyTree[Array],
    output_cotangent: tuple[NodeState | None, PyTree[Array]],
    /,
) -> _StageCotangents:
    """Replay one system stage; transpose its context consumption exactly once."""
    species, mask = region.species, region.atom_mask

    def apply(
        values: PyTree[Array],
        current: NodeState | None,
        reduced: PyTree[Array],
        carried: PyTree[Array],
    ) -> tuple[NodeState | None, PyTree[Array]]:
        return bind(values).apply_system_context(
            stage, current, reduced, species, mask, carried
        )

    def partials(
        values: PyTree[Array], current: NodeState | None, carried: PyTree[Array]
    ) -> PyTree[Array]:
        return bind(values).system_partials(stage, current, species, mask, carried)

    _, apply_vjp = jax.vjp(apply, lane, features, system, state)
    applied_parameters, applied_features, local_system, applied_context = apply_vjp(
        output_cotangent
    )
    _, partial_vjp = jax.vjp(partials, lane, features, state)
    # Exactly-once transpose of the replicated consumption: every owner's local
    # cotangent is summed once and returned once to the owned partial producers.
    reduced = owner_system_sum(
        jax.tree.map(
            lambda primal, value: value if _inexact(primal) else None,
            system,
            local_system,
        ),
        region.axis_name,
        region.mode,
    )
    reduced = jax.tree.map(
        lambda primal, value: _zero_cotangent(primal) if value is None else value,
        system,
        reduced,
    )
    partial_parameters, partial_features, partial_context = partial_vjp(reduced)
    return _StageCotangents(
        _add_parameters(applied_parameters, partial_parameters),
        None if features is None else _add_cotangents(applied_features, partial_features),
        _add_cotangents(applied_context, partial_context),
    )


class _InteractionCotangents(NamedTuple):
    parameters: PyTree[Array]
    features: NodeState
    context: PyTree[Array]
    vectors: Array
    successful: Array


def _interaction_reverse(
    bind: _Bind,
    region: _Region,
    index: int,
    lane: PyTree[Array],
    features: NodeState,
    state: PyTree[Array],
    feature_cotangent: NodeState,
    context_cotangent: PyTree[Array],
    energy_cotangent: Array,
    /,
) -> _InteractionCotangents:
    """Replay one interaction from its retained state and pull back once."""
    species, mask = region.species, region.atom_mask
    geometry = region.geometry

    def source(
        values: PyTree[Array], current: NodeState, carried: PyTree[Array]
    ) -> NodeState:
        return bind(values).prepare_sources(index, current, species, mask, carried)

    def receiver(
        values: PyTree[Array],
        vectors: Array,
        columns: NodeState,
        current: NodeState,
        carried: PyTree[Array],
    ) -> tuple[tuple[NodeState, Array], Array]:
        update = bind(values).update_receivers(
            index,
            region.topology.relation,
            vectors,
            columns,
            current,
            species,
            mask,
            carried,
            edge_active=geometry.active,
            route_identity=region.topology.route_identity,
        )
        return (update.features, update.energy), update.evidence.successful

    sources, source_vjp = jax.vjp(source, lane, features, state)
    columns = region.exchange.gather(sources)
    outputs, receiver_vjp, replayed = jax.vjp(
        receiver, lane, geometry.vectors, columns, features, state, has_aux=True
    )
    (
        receiver_parameters,
        vectors_cotangent,
        columns_cotangent,
        direct_features,
        receiver_context,
    ) = receiver_vjp((feature_cotangent, energy_cotangent.astype(outputs[1].dtype)))
    owned = region.exchange.transpose(columns_cotangent, sources)
    source_parameters, source_features, source_context = source_vjp(owned)
    return _InteractionCotangents(
        _add_parameters(receiver_parameters, source_parameters),
        _add_cotangents(direct_features, source_features),
        _add_cotangents(
            _add_cotangents(context_cotangent, receiver_context), source_context
        ),
        vectors_cotangent,
        replayed,
    )


def _edge_position_gradient(
    exchange: AtomisticColumnExchange,
    topology: OwnerLayerTopology,
    edge_cotangent: Array,
    positions: Array,
    *,
    accumulation: RelationAccumulation,
) -> tuple[Array, Array]:
    """Route ``dE/dd`` to receivers and, through the exchange transpose, sources.

    The topology owns canonical target-grouped reductions in both directions;
    their prepared route order honors the caller's accumulation policy instead
    of unordered duplicate-index scatter additions.
    """
    receivers, receiver_evidence = topology.receiver_reduction.reduce(
        edge_cotangent, accumulation=accumulation
    )
    columns, source_evidence = topology.source_reduction.reduce(
        -edge_cotangent, accumulation=accumulation
    )
    gradient = receivers + exchange.transpose(columns, positions)
    return gradient, receiver_evidence.successful & source_evidence.successful


__all__ = [
    "AtomisticColumnExchange",
    "AtomisticExchangeCapacityError",
    "AtomisticLayeredModel",
    "AtomisticLayerUpdate",
    "AtomisticSystemStage",
    "HaloColumnExchange",
    "NodeState",
    "OwnerConditioning",
    "OwnerEdgeGeometry",
    "OwnerFieldTerm",
    "OwnerGlobalTerm",
    "OwnerLayerForward",
    "OwnerLayerGradients",
    "OwnerLayerTopology",
    "OwnerReductionMode",
    "owner_atom_energies",
    "owner_edge_geometry",
    "owner_layer_gradients",
    "owner_system_sum",
]
