#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Data-parallel and data×domain execution of the shared atomistic training kernel.

`AtomisticTrainingDistributionPlan` binds one `ExecutionGroup` (or a one-device
lane reference of it) to a named data axis, optionally composed with an
`AtomisticDomainPlan` on an explicit second (owner) mesh axis. Data shards are
group-local: `ExecutionGroupRank` maps the calling process to its contiguous
data shards, never to the whole process world, and every shard carries the
immutable global epoch/draw identities of `epoch_shard`. Semantic randomness is
addressed by those identities and the kernel cursors, never by rank.

`AtomisticDataParallelReduction` is the training-kernel reduction:

* ``reduce`` sums each objective's local additive numerator and support (and
  its capacity/scientific failure counts) over the data axis BEFORE the kernel
  normalizes, so unequal shard supports are never averaged; its reverse pass is
  the identity, so every rank differentiates only its own numerator. A shard
  whose support is zero contributes an exact zero numerator and support and is
  inert.
* ``bind_parameters`` is the identity forward; its reverse pass sums the
  parameter cotangent over the owner axis (data×domain) and then the data axis,
  so every rank holds exactly ``Σ_r ∂n_r/∂θ`` once (no owner/rank-count scaling)
  and parameters/optimizer state stay replicated.
* ``map_step`` executes a kernel step or validation evaluation in the group's
  mapped region: replicated state, data-sharded batch, replicated outputs.

Arithmetic contract. ``"fast"`` uses the device ``psum`` (summation tree chosen
by the collective implementation); ``"deterministic"`` all-gathers shard totals
and adds them in shard order; ``"compensated"`` adds the gathered totals with
Kahan compensation. Ordered modes are run-to-run reproducible for one fixed
layout. An ordered sum of already-reduced shard totals is NOT partition
invariant: changing the shard count or placement changes the summation tree, so
results agree with a single-process reduction only to a roundoff bound
(``|Δ| ≲ (S - 1) ε Σ|shard totals|`` per reduction for S shards). Bitwise
partition invariance would require a canonical per-atom contribution order
through every primitive; it is not offered.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast, final, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import PyTree

from .._data_plane import (
    epoch_shard,
    ExecutionGroupRank,
    IndexEpochPlan,
    make_global_array_from_process_local_data,
    ProcessLocalBatch,
)
from .._execution_runtime import ExecutionGroup
from .._identity import IdentityWords
from .._jax_config import x64_enabled
from .._strict import StrictModule
from .._trainable import ExplicitFreeze, NonTrainableState
from .._training_kernel import (
    enforce_rejection_budget,
    KernelObjective,
    PreparedTrainingKernel,
    TrainingAttemptEvidence,
    TrainingKernelState,
    TrainingKeys,
)
from .._training_objective import _ObjectiveContribution
from ..typing import parse
from ._domain_plan import (
    AtomisticDomainPlan,
    AtomisticDomainReductionMode,
    AtomisticDomainReductionPolicy,
)
from ._domain_runtime import (
    AtomisticDomainOwnedRows,
    AtomisticDomainOwnerView,
    AtomisticDomainReplicatedInputs,
    AtomisticDomainState,
    AtomisticDomainTermInputs,
    AtomisticDomainTermRows,
    owner_parameter_partials,
    owner_predictions,
    PreparedAtomisticDomainRuntime,
)
from ._training_objectives import (
    AbstractAtomisticObjective,
    AbstractAtomisticObjectiveReduction,
    AtomisticEnergyObjective,
    AtomisticForceObjective,
    AtomisticLabelLevel,
    AtomisticObjectiveTerm,
    AtomisticStressObjective,
    energy_objective_term,
    force_objective_term,
    stress_objective_term,
)


_T = TypeVar("_T")
_Shard = TypeVar("_Shard")


# ------------------------------------------------------------ axis reductions


def _axis_total(value: Array, axis: str, mode: AtomisticDomainReductionMode, /) -> Array:
    """Sum of one rank-local value over a named mapped axis (replicated)."""

    if mode == "fast":
        return jax.lax.psum(value, axis)
    stacked = jax.lax.all_gather(value, axis, axis=0, tiled=False)
    return AtomisticDomainReductionPolicy(mode).ordered_sum(stacked)


def _hierarchical_total(
    value: Array, axes: tuple[str, ...], mode: AtomisticDomainReductionMode, /
) -> Array:
    for axis in axes:
        value = _axis_total(value, axis, mode)
    return value


def _replicated_sum_impl(
    value: Array, axes: tuple[str, ...], mode: AtomisticDomainReductionMode
) -> Array:
    return _hierarchical_total(value, axes, mode)


def _replicated_sum_fwd(
    value: Array, axes: tuple[str, ...], mode: AtomisticDomainReductionMode
) -> tuple[Array, None]:
    return _hierarchical_total(value, axes, mode), None


def _replicated_sum_bwd(
    axes: tuple[str, ...],
    mode: AtomisticDomainReductionMode,
    residual: None,
    cotangent: Array,
) -> tuple[Array]:
    del axes, mode, residual
    # Each rank's local value enters the replicated total exactly once: its
    # cotangent is the (replicated) cotangent of the total, never scaled.
    return (cotangent,)


_replicated_sum = jax.custom_vjp(_replicated_sum_impl, nondiff_argnums=(1, 2))
_replicated_sum.defvjp(_replicated_sum_fwd, _replicated_sum_bwd)


def _bound_core(
    parameters: PyTree[Any], axes: tuple[str, ...], mode: AtomisticDomainReductionMode
) -> PyTree[Any]:
    return parameters


def _bound_fwd(
    parameters: PyTree[Any], axes: tuple[str, ...], mode: AtomisticDomainReductionMode
) -> tuple[PyTree[Any], None]:
    return parameters, None


def _bound_bwd(
    axes: tuple[str, ...],
    mode: AtomisticDomainReductionMode,
    residual: None,
    cotangent: PyTree[Any],
) -> tuple[PyTree[Any]]:
    del residual
    return (jax.tree.map(lambda leaf: _hierarchical_total(leaf, axes, mode), cotangent),)


_bound = jax.custom_vjp(_bound_core, nondiff_argnums=(1, 2))
_bound.defvjp(_bound_fwd, _bound_bwd)


def _integer_total(value: Array, axis: str, /) -> Array:
    # Integer counts are exact under any summation order.
    return jax.lax.psum(value, axis)


# ---------------------------------------------------------------- region map


def _is_spec(value: object) -> bool:
    return isinstance(value, PartitionSpec)


def _spec_axes(spec: PartitionSpec, data: str, owner: str | None, /) -> tuple[bool, bool]:
    """Whether a leaf is data-mapped (dim 0) and owner-mapped (dim 1)."""

    entries = tuple(spec)
    data_mapped = bool(entries) and entries[0] == data
    owner_mapped = owner is not None and len(entries) > 1 and entries[1] == owner
    allowed = {0} if data_mapped else set()
    if owner_mapped:
        allowed.add(1)
    for position, entry in enumerate(entries):
        if entry is not None and position not in allowed:
            raise ValueError(
                f"Training regions map only the leading data dimension by {data!r} "
                f"and the next dimension by the owner axis; got {spec}."
            )
    if owner_mapped and not data_mapped:
        raise ValueError(f"An owner-mapped leaf must also be data-mapped; got {spec}.")
    return data_mapped, owner_mapped


def _structure_word(tree: PyTree[Any], /) -> int:
    """Non-negative 31-bit digest of a pytree's structure, statics, shapes, dtypes.

    Object addresses in static reprs are process-local and are removed, so
    equal layouts digest equally on every process.
    """

    leaves, treedef = jax.tree_util.tree_flatten(tree)
    text = repr(
        (
            str(treedef),
            [
                (tuple(leaf.shape), str(leaf.dtype)) if eqx.is_array(leaf) else repr(leaf)
                for leaf in leaves
            ],
        )
    )
    text = re.sub(r" at 0x[0-9a-fA-F]+", "", text)
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _guard_first_inexact(
    tree: PyTree[Any], predicate: Array, message: str, /
) -> PyTree[Any]:
    """Attach a runtime refusal to the first floating leaf of ``tree``."""

    leaves, treedef = jax.tree_util.tree_flatten(tree)
    for index, leaf in enumerate(leaves):
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            leaves[index] = eqx.error_if(leaf, predicate, message)
            return jax.tree_util.tree_unflatten(treedef, leaves)
    raise TypeError("A data-parallel shard must carry at least one floating array.")


def _squeeze(spec: PartitionSpec, data: str, owner: str | None, leaf: Array) -> Array:
    data_mapped, _ = _spec_axes(spec, data, owner)
    return leaf[0] if data_mapped else leaf


def _slice_range(index: slice, size: int, /) -> tuple[int, int]:
    start = 0 if index.start is None else int(index.start)
    return start, size if index.stop is None else int(index.stop)


def _process_local_rows(
    leaf: Array | np.ndarray, /
) -> tuple[np.ndarray, list[tuple[int, int]] | None]:
    """This process's rows of one shard leaf, without any cross-process gather.

    A fully addressable leaf is whole and a fully replicated leaf is this
    process's replica (no held row ranges); otherwise the leaf is blocked on
    its leading axis and its addressable blocks are concatenated once each in
    global row order, with the held ``(start, stop)`` row ranges.
    """

    if not isinstance(leaf, jax.Array) or leaf.is_fully_addressable:
        return np.asarray(leaf), None
    if leaf.is_fully_replicated:
        return np.asarray(leaf.addressable_data(0)), None
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for shard in leaf.addressable_shards:
        if any(index != slice(None) for index in shard.index[1:]):
            raise ValueError(
                "A process-local data shard leaf may be blocked only on its leading axis."
            )
        blocks.setdefault(
            _slice_range(shard.index[0], leaf.shape[0]), np.asarray(shard.data)
        )
    held = sorted(blocks)
    return np.concatenate([blocks[rows] for rows in held], axis=0), held


# ---------------------------------------------------------------- plan


@final
class AtomisticTrainingDistributionPlan(StrictModule, NonTrainableState):
    """Execution group, named data axis and optional domain composition.

    Device execution: the group's mesh is ``(data_axis,)`` for data
    parallelism, or ``(data_axis, owner_axis)`` for data×domain where
    ``owner_axis`` is the domain plan's ownership axis and its size is the
    domain owner count; every data row executes one domain system. Lane
    reference (``data_lanes``): one-device group; data shards (and, with a
    lane-reference domain ownership, owner regions) execute as named vmap lanes
    with identical collectives. Lanes are the numerical oracle of the device
    route, not distributed hardware evidence.

    ``collective_schedule`` is the qualified, ordered collective program of one
    objective evaluation; it enters ``plan_id`` and through the reduction
    identity the training continuation identity. A process outside the group
    cannot build the plan, so non-members never enter its collectives.

    ``plan_identity`` is the exact canonical plan digest (it binds the domain
    plan's scientific identity) carried as dynamic `IdentityWords`; ``plan_id``
    decodes it on the host. Every static field is declared layout.
    """

    execution_group: ExecutionGroup = eqx.field(static=True)
    data_axis: str = eqx.field(static=True)
    data_shards: int = eqx.field(static=True)
    lane_reference: bool = eqx.field(static=True)
    reduction: AtomisticDomainReductionPolicy
    domain: AtomisticDomainPlan | None
    owner_axis: str | None = eqx.field(static=True)
    owner_count: int = eqx.field(static=True)
    process_rank: ExecutionGroupRank | None = eqx.field(static=True)
    local_shards: tuple[int, ...] = eqx.field(static=True)
    collective_schedule: tuple[str, ...] = eqx.field(static=True)
    plan_identity: IdentityWords

    def __init__(
        self,
        execution_group: ExecutionGroup,
        /,
        *,
        data_axis: str = "data",
        reduction: AtomisticDomainReductionPolicy | None = None,
        data_lanes: int | None = None,
        domain: AtomisticDomainPlan | None = None,
    ) -> None:
        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("execution_group must be a live ExecutionGroup.")
        axis = str(data_axis).strip()
        if not axis:
            raise ValueError("data_axis must be non-empty.")
        policy = (
            AtomisticDomainReductionPolicy("deterministic")
            if reduction is None
            else reduction
        )
        if not isinstance(policy, AtomisticDomainReductionPolicy):
            raise TypeError("reduction must be an AtomisticDomainReductionPolicy.")
        if domain is not None and not isinstance(domain, AtomisticDomainPlan):
            raise TypeError("domain must be an AtomisticDomainPlan or None.")
        if not execution_group.is_member:
            raise ValueError(
                "The calling process is not a member of the training execution group; "
                "non-members must not enter its collectives."
            )
        mesh_axes = tuple(execution_group.mesh.axis_names)
        owner_axis = None if domain is None else str(domain.ownership.axis_name)
        owner_count = 1 if domain is None else int(domain.ownership.owner_count)
        if owner_axis == axis:
            raise ValueError("The data axis and the domain owner axis must differ.")
        if data_lanes is None:
            expected = (axis,) if domain is None else (axis, owner_axis)
            if mesh_axes != expected:
                raise ValueError(
                    f"Device data parallelism needs the group mesh axes {expected}; "
                    f"the group declares {mesh_axes}."
                )
            if domain is not None:
                if domain.ownership.lane_reference:
                    raise ValueError(
                        "A device data axis composes only with device domain ownership; "
                        "use data_lanes with a lane-reference domain."
                    )
                size = int(execution_group.mesh.shape[owner_axis])
                if size != owner_count:
                    raise ValueError(
                        f"The owner mesh axis has {size} devices but the domain plan "
                        f"declares {owner_count} owners."
                    )
            rank = ExecutionGroupRank.current(execution_group, axis=axis)
            shards = rank.shard_count
            local = rank.shards
            lanes = False
        else:
            if len(execution_group.devices) != 1:
                raise ValueError(
                    "Lane-reference data shards execute on a one-device group."
                )
            if isinstance(data_lanes, bool) or int(data_lanes) <= 0:
                raise ValueError("data_lanes must be a positive integer.")
            if domain is not None:
                if not domain.ownership.lane_reference:
                    raise ValueError(
                        "Lane-reference data shards compose only with a lane-reference "
                        "domain ownership."
                    )
                if domain.ownership.execution_group.devices != execution_group.devices:
                    raise ValueError(
                        "The lane-reference domain must run on the training group's device."
                    )
            rank = None
            shards = int(data_lanes)
            local = tuple(range(shards))
            lanes = True
        schedule = (
            # Lanes share one trace; ranks of a device group cross-check their
            # traced shard structure before any data collective.
            *(() if lanes else (f"structure-agreement:pmin+pmax:int32:{axis}",)),
            f"objective-numerator-support:{policy.mode}:{axis}",
            f"objective-failure-counts:psum:int32:{axis}",
            *(
                ()
                if owner_axis is None
                else (f"parameter-cotangent:{policy.mode}:{owner_axis}",)
            ),
            f"parameter-cotangent:{policy.mode}:{axis}",
        )
        self.execution_group = execution_group
        self.data_axis = axis
        self.data_shards = shards
        self.lane_reference = lanes
        self.reduction = policy
        self.domain = domain
        self.owner_axis = owner_axis
        self.owner_count = owner_count
        self.process_rank = rank
        self.local_shards = local
        self.collective_schedule = schedule
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-training-distribution",
                "group_id": execution_group.spec.group_id,
                "mesh_axes": [
                    [name, int(size)] for name, size in execution_group.mesh.shape.items()
                ],
                "data_axis": axis,
                "data_shards": shards,
                "lane_reference": lanes,
                "reduction": policy.policy_id,
                "domain": None if domain is None else domain.plan_id,
                "owner_axis": owner_axis,
                "owner_count": owner_count,
                "layout": None if rank is None else rank.layout_id,
                "schedule": list(schedule),
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical training-distribution plan digest."""
        return self.plan_identity.fingerprint()

    # Placement ----------------------------------------------------------------

    @property
    def parameter_axes(self) -> tuple[str, ...]:
        """Axes the parameter cotangent is summed over, innermost first."""

        if self.owner_axis is None:
            return (self.data_axis,)
        return (self.owner_axis, self.data_axis)

    @property
    def single_controller(self) -> bool:
        """Whether this process holds every data shard (lanes or one process)."""

        return self.process_rank is None or self.process_rank.rank_count == 1

    def epoch_shards(
        self, global_plan: IndexEpochPlan, batch_index: int, /
    ) -> tuple[ProcessLocalBatch, ...]:
        """Global draw positions and validity of this process's data shards.

        Positions are immutable global epoch identities; a lane reference and
        a device group with the same shard count see identical shards.
        """

        if not isinstance(global_plan, IndexEpochPlan):
            raise TypeError("global_plan must be an IndexEpochPlan.")
        return tuple(
            epoch_shard(global_plan, self.data_shards, batch_index, shard)
            for shard in self.local_shards
        )

    # Shard batches -------------------------------------------------------------

    def _shard_specs(self, arrays: PyTree[Any], shard: Any, /) -> PyTree[Any]:
        """Stacked region specs of one shard's array tree."""

        data = self.data_axis
        if isinstance(shard, AtomisticDomainTrainingShard):
            if self.owner_axis is None:
                raise TypeError("Domain training shards need a data×domain plan.")
            return shard.region_specs(arrays, data, self.owner_axis)
        if self.owner_axis is not None:
            raise TypeError(
                "A data×domain plan executes AtomisticDomainTrainingShard values."
            )
        return jax.tree.map(lambda _: PartitionSpec(data), arrays)

    def assemble(self, shards: Sequence[_Shard], /) -> _Shard:
        """Stack this process's data shards on a leading data axis and place them.

        Every shard must share one tree structure including static fields
        (layout-bound identities), shapes and dtypes: the mapped region traces
        one program for every rank. Single-controller execution stacks all
        ``data_shards`` shards (device placement on the group mesh, or one
        device for lanes); multi-process execution assembles each process's
        contiguous shards into global arrays without host gathering: every
        process supplies only its addressable rows of each shard leaf (the
        owner blocks it holds of an owner-blocked leaf, its replica of a
        replicated leaf), so one domain plan whose owners span processes serves
        every data row.
        """

        values = tuple(shards)
        if len(values) != len(self.local_shards):
            raise ValueError(
                f"This process holds {len(self.local_shards)} data shards; "
                f"{len(values)} were supplied."
            )
        reference = jax.tree_util.tree_structure(values[0])
        arrays_list = []
        static = None
        for position, value in enumerate(values):
            if jax.tree_util.tree_structure(value) != reference:
                raise ValueError(
                    f"Data shard {self.local_shards[position]} differs in structure or "
                    "static identity from the first shard; data-parallel shards must be "
                    "layout-bound (identical statics) to share one compiled program."
                )
            arrays, rest = eqx.partition(value, eqx.is_array)
            if static is None:
                static = rest
            arrays_list.append(arrays)
        shapes = [
            [(tuple(leaf.shape), str(leaf.dtype)) for leaf in jax.tree.leaves(a)]
            for a in arrays_list
        ]
        if any(item != shapes[0] for item in shapes[1:]):
            raise ValueError("Data shards must share leaf shapes and dtypes.")
        specs = self._shard_specs(arrays_list[0], values[0])
        if self.lane_reference:
            stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *arrays_list)
            placed = jax.device_put(stacked, self.execution_group.devices[0])
        elif self.single_controller:
            stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *arrays_list)
            placed = jax.tree.map(
                lambda spec, leaf: jax.device_put(
                    leaf, NamedSharding(self.execution_group.mesh, spec)
                ),
                specs,
                stacked,
                is_leaf=_is_spec,
            )
        else:
            placed = jax.tree.map(
                lambda spec, *leaves: self._process_global(spec, leaves),
                specs,
                *arrays_list,
                is_leaf=_is_spec,
            )
        return cast(_Shard, eqx.combine(placed, static))

    def _process_global(self, spec: PartitionSpec, leaves: tuple[Array, ...], /) -> Array:
        """Global data-stacked array of one leaf from this process's rows only."""

        key_impl = None
        if jax.dtypes.issubdtype(leaves[0].dtype, jax.dtypes.prng_key):
            key_impl = jax.random.key_impl(leaves[0])
            leaves = tuple(jax.random.key_data(leaf) for leaf in leaves)
        global_shape = (self.data_shards,) + tuple(leaves[0].shape)
        sharding = NamedSharding(self.execution_group.mesh, spec)
        _, owner_mapped = _spec_axes(spec, self.data_axis, self.owner_axis)
        expected = None
        if owner_mapped:
            spans = set()
            for index in sharding.addressable_devices_indices_map(global_shape).values():
                if index is None:
                    raise ValueError(
                        "An addressable training device has no declared array index."
                    )
                spans.add(_slice_range(index[1], global_shape[1]))
            expected = sorted(spans)
        local = []
        for leaf in leaves:
            rows, held = _process_local_rows(leaf)
            if expected is not None and held is not None and held != expected:
                raise ValueError(
                    "The owner rows of the shard do not match the training mesh owner "
                    f"columns of this process: held {held}, expected {expected}."
                )
            local.append(rows)
        placed = make_global_array_from_process_local_data(
            sharding, np.stack(local), global_shape=global_shape
        )
        if key_impl is None:
            return placed
        return jax.random.wrap_key_data(placed, impl=key_impl)

    # Mapped region ---------------------------------------------------------------

    def map(
        self, function: Callable[[Any, Any], _T], batch: Any, /
    ) -> Callable[[Any, Any], _T]:
        """Bind ``function(replicated, shard)`` to one data shard per rank.

        ``batch`` is a template of the assembled batch (only its structure,
        static fields and the domain view's partition specs are read). The
        first argument is replicated; the batch leaves are data-sharded (and
        owner-sharded for domain rows); every output must be a replicated
        collective result.
        """

        _, static = eqx.partition(batch, eqx.is_array)
        data, owner = self.data_axis, self.owner_axis

        def region_specs(arrays: PyTree[Any]) -> PyTree[Any]:
            return self._shard_specs(arrays, batch)

        if self.lane_reference:
            return self._lane_region(function, static, region_specs)

        mesh = self.execution_group.mesh

        def mapped(replicated: Any, assembled: Any) -> _T:
            arrays, _ = eqx.partition(assembled, eqx.is_array)
            specs = region_specs(arrays)

            def body(local_replicated: Any, local_arrays: Any) -> _T:
                local = jax.tree.map(
                    lambda spec, leaf: _squeeze(spec, data, owner, leaf),
                    specs,
                    local_arrays,
                    is_leaf=_is_spec,
                )
                word = jnp.asarray(_structure_word(eqx.combine(local, static)), jnp.int32)
                disagree = jax.lax.pmin(word, data) != jax.lax.pmax(word, data)
                # Ranks of one group must trace one program: a structural or
                # static mismatch (non-layout-bound statics) is refused before
                # the step consumes the shard.
                local = _guard_first_inexact(
                    local,
                    disagree,
                    "Data-parallel ranks traced different shard structures; shard "
                    "statics must be layout-bound.",
                )
                return function(local_replicated, eqx.combine(local, static))

            return jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(PartitionSpec(), specs),
                out_specs=PartitionSpec(),
                check_vma=False,
            )(replicated, arrays)

        return mapped

    def _lane_region(
        self,
        function: Callable[[Any, Any], _T],
        static: Any,
        region_specs: Callable[[PyTree[Any]], PyTree[Any]],
        /,
    ) -> Callable[[Any, Any], _T]:
        data, owner = self.data_axis, self.owner_axis
        shards, owners = self.data_shards, self.owner_count

        def mapped(replicated: Any, assembled: Any) -> _T:
            arrays, _ = eqx.partition(assembled, eqx.is_array)
            specs = region_specs(arrays)

            def split(spec: PartitionSpec, leaf: Array) -> Array:
                data_mapped, owner_mapped = _spec_axes(spec, data, owner)
                if data_mapped and leaf.shape[0] != shards:
                    raise ValueError(
                        f"Data-mapped leaves need a leading axis of {shards} shards; "
                        f"got {leaf.shape}."
                    )
                if not owner_mapped:
                    return leaf
                if leaf.shape[1] % owners:
                    raise ValueError(
                        f"Owner-blocked leaves need an owner axis divisible by {owners}; "
                        f"got {leaf.shape}."
                    )
                return leaf.reshape(
                    (leaf.shape[0], owners, leaf.shape[1] // owners) + leaf.shape[2:]
                )

            lanes = jax.tree.map(split, specs, arrays, is_leaf=_is_spec)
            outer_axes = jax.tree.map(
                lambda spec: 0 if _spec_axes(spec, data, owner)[0] else None,
                specs,
                is_leaf=_is_spec,
            )
            inner_axes = jax.tree.map(
                lambda spec: 0 if _spec_axes(spec, data, owner)[1] else None,
                specs,
                is_leaf=_is_spec,
            )

            def shard_body(local_replicated: Any, local_arrays: Any) -> _T:
                return function(local_replicated, eqx.combine(local_arrays, static))

            if owner is None:
                region = shard_body
            else:
                region = jax.vmap(
                    shard_body,
                    in_axes=(None, inner_axes),
                    out_axes=None,
                    axis_name=owner,
                    axis_size=owners,
                )
            return jax.vmap(
                region,
                in_axes=(None, outer_axes),
                out_axes=None,
                axis_name=data,
                axis_size=shards,
            )(replicated, lanes)

        return mapped


# ---------------------------------------------------------------- reduction


@final
class AtomisticDataParallelReduction(AbstractAtomisticObjectiveReduction):
    """Data-axis objective reduction and parameter binding of the training kernel.

    ``reduce`` and ``bind_parameters`` must run inside ``map_step``'s region.
    The identity binds the plan (group, axes, layout, arithmetic mode and
    collective schedule) and enters the training continuation identity;
    ``reduction_identity`` carries its exact digest as dynamic `IdentityWords`
    (``reduction_id`` decodes it on the host).
    """

    plan: AtomisticTrainingDistributionPlan
    reduction_identity: IdentityWords

    def __init__(self, plan: AtomisticTrainingDistributionPlan, /) -> None:
        if not isinstance(plan, AtomisticTrainingDistributionPlan):
            raise TypeError("plan must be an AtomisticTrainingDistributionPlan.")
        self.plan = plan
        self.reduction_identity = IdentityWords.of(
            {"kind": "atomistic-data-parallel-reduction", "plan": plan.plan_id}
        )

    @property
    def reduction_id(self) -> str:
        """Host decoding of the canonical reduction digest."""
        return self.reduction_identity.fingerprint()

    @property
    def local_shards(self) -> int:
        return len(self.plan.local_shards)

    def bind_parameters(self, parameters: PyTree[Any], /) -> PyTree[Any]:
        return _bound(parameters, self.plan.parameter_axes, self.plan.reduction.mode)

    def reduce(self, term: AtomisticObjectiveTerm, /) -> AtomisticObjectiveTerm:
        if not isinstance(term, AtomisticObjectiveTerm):
            raise TypeError("reduce takes an AtomisticObjectiveTerm.")
        axes = (self.plan.data_axis,)
        mode = self.plan.reduction.mode
        return AtomisticObjectiveTerm(
            numerator=_replicated_sum(term.numerator, axes, mode),
            support=_replicated_sum(term.support, axes, mode),
            capacity_failures=_integer_total(term.capacity_failures, self.plan.data_axis),
            scientific_failures=_integer_total(
                term.scientific_failures, self.plan.data_axis
            ),
        )

    def assemble(self, batches: tuple[Any, ...], /) -> Any:
        return self.plan.assemble(batches)

    def map_step(self, step: Callable[..., _T], /) -> Callable[..., _T]:
        plan = self.plan

        def mapped(replicated: Any, batch: Any) -> _T:
            return plan.map(step, batch)(replicated, batch)

        return mapped


# ------------------------------------------------------------- domain shards


def _zero_cotangent(leaf: Any) -> Any:
    if leaf is None:
        return None
    if jnp.issubdtype(jnp.result_type(leaf), jnp.inexact):
        return jnp.zeros_like(leaf)
    return np.zeros(np.shape(leaf), dtype=jax.dtypes.float0)


@final
class AtomisticDomainTrainingShard(StrictModule):
    """One periodic system of a data×domain training row with its labels.

    ``view`` is the domain runtime's owner view of the system state
    (owner-blocked rows plus replicated leaves; its ``replicated.admitted``
    carries the reuse certificate of the state). Structure-level labels have
    shape ``(1, *trailing)`` and availability ``(1,)``; atom-level labels are
    owner-blocked rows ``(owners * capacity, *trailing)`` aligned with the
    state's layout (``state.layout.distribute`` for logical-order ingress) and
    availability ``(owners * capacity,)``. ``draw_index``/``case_epoch`` are
    the semantic draw identity of the row.
    """

    view: AtomisticDomainOwnerView[
        AtomisticDomainOwnedRows,
        AtomisticDomainReplicatedInputs,
        AtomisticDomainTermRows,
        AtomisticDomainTermInputs,
    ]
    case_valid: Array
    case_weight: Array
    draw_index: Array
    case_epoch: Array
    labels: tuple[Array, ...]
    available: tuple[Array, ...]
    label_slots: tuple[str, ...] = eqx.field(static=True)
    label_levels: tuple[AtomisticLabelLevel, ...] = eqx.field(static=True)

    def __init__(
        self,
        view: AtomisticDomainOwnerView[
            AtomisticDomainOwnedRows,
            AtomisticDomainReplicatedInputs,
            AtomisticDomainTermRows,
            AtomisticDomainTermInputs,
        ],
        /,
        *,
        labels: Mapping[str, tuple[Array, Array]],
        label_levels: Mapping[str, AtomisticLabelLevel],
        case_valid: bool | Array = True,
        case_weight: float | Array = 1.0,
        draw_index: int | Array = 0,
        case_epoch: int | Array = 0,
    ) -> None:
        if not isinstance(view, AtomisticDomainOwnerView):
            raise TypeError("view must be an AtomisticDomainOwnerView.")
        slots = tuple(labels)
        if set(slots) != set(label_levels):
            raise ValueError("labels and label_levels must name the same slots.")
        levels = tuple(
            parse(label_levels[slot], AtomisticLabelLevel, f"label_levels[{slot!r}]")
            for slot in slots
        )
        values: list[Array] = []
        availability: list[Array] = []
        for slot, level in zip(slots, levels, strict=True):
            value, available = (jnp.asarray(item) for item in labels[slot])
            if available.dtype != jnp.bool_:
                raise TypeError(f"Label {slot!r} availability must be boolean.")
            if level == "structure" and (value.ndim < 1 or value.shape[0] != 1):
                raise ValueError(f"Structure label {slot!r} must have shape (1, ...).")
            if available.shape != value.shape[:1]:
                raise ValueError(f"Label {slot!r} availability must match its rows.")
            values.append(value)
            availability.append(available)
        self.view = view
        self.case_valid = jnp.reshape(jnp.asarray(case_valid, dtype=jnp.bool_), (1,))
        self.case_weight = jnp.reshape(
            jnp.asarray(case_weight, dtype=jnp.float64 if x64_enabled() else jnp.float32),
            (1,),
        )
        self.draw_index = jnp.reshape(jnp.asarray(draw_index, dtype=jnp.uint32), (1,))
        self.case_epoch = jnp.reshape(jnp.asarray(case_epoch, dtype=jnp.uint32), (1,))
        self.labels = tuple(values)
        self.available = tuple(availability)
        self.label_slots = slots
        self.label_levels = levels

    @classmethod
    def from_domain_state(
        cls,
        runtime: PreparedAtomisticDomainRuntime,
        state: AtomisticDomainState,
        /,
        *,
        labels: Mapping[str, tuple[Array, Array]],
        label_levels: Mapping[str, AtomisticLabelLevel],
        case_valid: bool | Array = True,
        case_weight: float | Array = 1.0,
        draw_index: int | Array = 0,
        case_epoch: int | Array = 0,
    ) -> AtomisticDomainTrainingShard:
        """Owner view (with its reuse certificate) and labels of one system."""

        return cls(
            runtime.owner_view(state),
            labels=labels,
            label_levels=label_levels,
            case_valid=case_valid,
            case_weight=case_weight,
            draw_index=draw_index,
            case_epoch=case_epoch,
        )

    def label(self, slot: str, /) -> tuple[Array, Array, AtomisticLabelLevel]:
        if slot not in self.label_slots:
            raise KeyError(f"The domain training shard carries no label slot {slot!r}.")
        index = self.label_slots.index(slot)
        return self.labels[index], self.available[index], self.label_levels[index]

    def region_specs(self, arrays: PyTree[Any], data: str, owner: str, /) -> PyTree[Any]:
        """Stacked (data-leading) region specs aligned with ``arrays``.

        ``arrays`` is ``eqx.partition(shard, eqx.is_array)[0]`` of this shard
        (unstacked or data-stacked: only its structure is read).
        """

        view_specs = self.view.partition_specs()

        def prefixed(spec: PartitionSpec, subtree: Any) -> Any:
            entries = tuple(spec)
            if entries and (
                entries[0] != owner or any(e is not None for e in entries[1:])
            ):
                raise ValueError(f"Owner view specs are P() or P({owner!r}); got {spec}.")
            return jax.tree.map(lambda _: PartitionSpec(data, *entries), subtree)

        stacked_view = jax.tree.map(prefixed, view_specs, arrays.view, is_leaf=_is_spec)
        specs = jax.tree.map(lambda _: PartitionSpec(data), arrays)
        specs = eqx.tree_at(lambda value: value.view, specs, stacked_view)
        if not self.label_levels:
            return specs
        label_specs = tuple(
            PartitionSpec(data, owner) if level == "atom" else PartitionSpec(data)
            for level in self.label_levels
        )
        specs = eqx.tree_at(lambda value: value.labels, specs, label_specs)
        return eqx.tree_at(lambda value: value.available, specs, label_specs)


def _owner_atom_sum(
    owner: str, mode: AtomisticDomainReductionMode
) -> Callable[[Array], Array]:
    def atom_sum(value: Array, /) -> Array:
        # Per-case sum over this owner's atom rows, then the exactly-once
        # owner-axis total (identity reverse: owned rows return their own share).
        return _replicated_sum(jnp.sum(value, axis=1), (owner,), mode)

    return atom_sum


def _domain_term(
    runtime: PreparedAtomisticDomainRuntime,
    objective: AbstractAtomisticObjective,
    owner: str,
    mode: AtomisticDomainReductionMode,
    parameters: PyTree[Any],
    model_state: PyTree[Any],
    shard: AtomisticDomainTrainingShard,
    /,
    *,
    scale: float,
    dtype: str,
) -> AtomisticObjectiveTerm:
    """One owner region's objective term on its domain row.

    Predictions come from the domain runtime; the scientific objective
    arithmetic is the objectives' own (`energy/force/stress_objective_term`)
    with atom-axis sums replaced by owner-axis totals. The reverse pass returns
    this owner's exactly-once parameter share from
    `owner_parameter_partials`, seeded by the loss cotangents of the replicated
    energy/stress and the owned forces.
    """

    forces = isinstance(objective, AtomisticForceObjective)
    stress = isinstance(objective, AtomisticStressObjective)
    if not isinstance(
        objective,
        (AtomisticEnergyObjective, AtomisticForceObjective, AtomisticStressObjective),
    ):
        raise TypeError(
            "Data×domain training supports energy, force and stress objectives; "
            f"{type(objective).__name__} has no domain prediction route."
        )
    atom_sum = _owner_atom_sum(owner, mode)
    (slot,) = objective.label_slots
    values, available, level = shard.label(slot)
    expected = "atom" if forces else "structure"
    if level != expected:
        raise ValueError(
            f"Label {slot!r} must be {expected}-level for {objective.name!r}."
        )
    arrays, static = eqx.partition(
        (runtime, objective, model_state, shard, values, available), eqx.is_array
    )

    def predict(parameters_: PyTree[Any], arrays_: PyTree[Any]) -> tuple[Any, Any]:
        """Domain predictions, the scalar-loss function of the predicted
        quantity this objective consumes, and that quantity."""

        runtime_, objective_, model_state_, shard_, values_, available_ = eqx.combine(
            arrays_, static
        )
        predictions = owner_predictions(
            runtime_,
            parameters_,
            shard_.view,
            forces=forces,
            stress=stress,
            model_state=model_state_,
        )
        # `successful` already conjoins the view's admission (reuse certificate,
        # owner epochs, topology evidence). A failed admission means the domain
        # topology must be rebuilt: classified as a resource/capacity failure,
        # distinct from scientific failure.
        admitted = shard_.view.replicated.admitted
        successful = jnp.reshape(predictions.successful, (1,))
        overflow = jnp.reshape(predictions.capacity_failure | ~admitted, (1,))
        atom_mask = predictions.atom_mask[None, :]

        def term_of(quantity: Array, /) -> AtomisticObjectiveTerm:
            if forces:
                return force_objective_term(
                    objective_,
                    quantity[None],
                    successful,
                    overflow,
                    labels=values_[None],
                    available=available_[None],
                    case_valid=shard_.case_valid,
                    case_weight=shard_.case_weight,
                    atom_mask=atom_mask,
                    scale=scale,
                    dtype=dtype,
                    atom_sum=atom_sum,
                )
            if stress:
                return stress_objective_term(
                    objective_,
                    quantity[None],
                    successful,
                    overflow,
                    labels=values_,
                    available=available_,
                    requested=available_,
                    case_valid=shard_.case_valid,
                    case_weight=shard_.case_weight,
                    scale=scale,
                    dtype=dtype,
                )
            return energy_objective_term(
                objective_,
                jnp.reshape(quantity, (1,)),
                successful,
                overflow,
                labels=values_,
                available=available_,
                case_valid=shard_.case_valid,
                case_weight=shard_.case_weight,
                species=shard_.view.owned.species[None, :],
                atom_mask=atom_mask,
                scale=scale,
                dtype=dtype,
                atom_sum=atom_sum,
            )

        if forces:
            quantity = predictions.forces
        elif stress:
            quantity = predictions.stress
        else:
            quantity = predictions.energy
        if quantity is None:
            raise RuntimeError(
                f"The domain runtime did not return {objective.name!r} inputs."
            )
        return term_of, (quantity, jnp.zeros_like(predictions.energy))

    def numerator_term_impl(parameters_: PyTree[Any], arrays_: PyTree[Any]) -> Any:
        term_of, (quantity, _) = predict(parameters_, arrays_)
        return term_of(quantity)

    def forward(parameters_: PyTree[Any], arrays_: PyTree[Any]) -> tuple[Any, Any]:
        term_of, (quantity, energy_zero) = predict(parameters_, arrays_)
        term = term_of(quantity)
        # Loss cotangent of the consumed prediction (replicated energy/stress,
        # or this owner's force rows; owner totals reverse as identities).
        gradient = jax.grad(lambda value: term_of(value).numerator)(quantity)
        return term, (parameters_, arrays_, gradient, energy_zero)

    def backward(residual: Any, cotangent: AtomisticObjectiveTerm) -> tuple[Any, Any]:
        parameters_, arrays_, gradient, energy_zero = residual
        runtime_, _, model_state_, shard_, _, _ = eqx.combine(arrays_, static)
        seed = cotangent.numerator * gradient
        share = owner_parameter_partials(
            runtime_,
            parameters_,
            shard_.view,
            energy_cotangent=energy_zero if (forces or stress) else seed,
            force_cotangent=seed if forces else None,
            stress_cotangent=seed if stress else None,
            model_state=model_state_,
        )
        return share, jax.tree.map(_zero_cotangent, arrays_)

    numerator_term = jax.custom_vjp(numerator_term_impl)
    numerator_term.defvjp(forward, backward)
    return numerator_term(parameters, arrays)


@final
class AtomisticDomainKernelObjective(StrictModule, ExplicitFreeze):
    """Kernel objective callable of one objective on data×domain rows.

    Holds the domain runtime's training binding (no PARAMETER or MODEL_STATE
    arrays). The schedule weight multiplies the local numerator before the
    data-axis reduction, as in the batch route.
    """

    runtime: PreparedAtomisticDomainRuntime
    objective: AbstractAtomisticObjective
    reduction: AtomisticDataParallelReduction
    scale: float = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    def __call__(
        self,
        parameters: Any,
        model_state: Any,
        fixed: Any,
        shard: AtomisticDomainTrainingShard,
        keys: TrainingKeys,
    ) -> tuple[_ObjectiveContribution, Any, AtomisticObjectiveTerm]:
        del fixed
        plan = self.reduction.plan
        bound = self.reduction.bind_parameters(parameters)
        term = _domain_term(
            self.runtime,
            self.objective,
            cast(str, plan.owner_axis),
            plan.reduction.mode,
            bound,
            model_state,
            shard,
            scale=self.scale,
            dtype=self.dtype,
        )
        weight = self.objective.schedule(keys.accepted_cursor, dtype=self.dtype)
        term = self.reduction.reduce(
            dataclasses.replace(term, numerator=term.numerator * weight)
        )
        return _ObjectiveContribution(term.numerator, term.support), model_state, term


def atomistic_domain_training_objectives(
    runtime: PreparedAtomisticDomainRuntime,
    objectives: Sequence[AbstractAtomisticObjective],
    reduction: AtomisticDataParallelReduction,
    /,
    *,
    scales: Sequence[float],
    dtype: str = "float64",
) -> tuple[KernelObjective, ...]:
    """Kernel objectives of energy/force/stress objectives on domain rows.

    ``scales`` are the objectives' physical residual scales (the batch
    route's fitted or declared normalization); the trained tree of the kernel
    is ``runtime.model``.
    """

    if not isinstance(runtime, PreparedAtomisticDomainRuntime):
        raise TypeError("runtime must be a PreparedAtomisticDomainRuntime.")
    if not isinstance(reduction, AtomisticDataParallelReduction):
        raise TypeError("reduction must be an AtomisticDataParallelReduction.")
    plan = reduction.plan
    if plan.domain is None or plan.domain.plan_id != runtime.plan.plan_id:
        raise ValueError(
            "The reduction's data×domain plan must bind this runtime's plan."
        )
    items = tuple(objectives)
    scale_values = tuple(float(value) for value in scales)
    if len(scale_values) != len(items):
        raise ValueError("Every objective needs one scale.")
    if any(not np.isfinite(value) or value <= 0.0 for value in scale_values):
        raise ValueError("Objective scales must be finite and positive.")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'.")
    binding = runtime.training_binding()
    return tuple(
        KernelObjective(
            objective_id=f"atomistic-domain-{objective.name}",
            kind=objective.kind,
            route=objective.route,
            fn=AtomisticDomainKernelObjective(
                binding, objective, reduction, scale=scale, dtype=dtype
            ),
        )
        for objective, scale in zip(items, scale_values, strict=True)
    )


@eqx.filter_jit
def _mapped_attempt(
    kernel: PreparedTrainingKernel,
    reduction: AtomisticDataParallelReduction,
    state: TrainingKernelState,
    batch: Any,
) -> tuple[TrainingKernelState, TrainingAttemptEvidence]:
    return reduction.map_step(kernel.attempt)(state, batch)


def run_distributed_training_attempt(
    kernel: PreparedTrainingKernel,
    reduction: AtomisticDataParallelReduction,
    state: TrainingKernelState,
    batch: Any,
    /,
) -> tuple[TrainingKernelState, TrainingAttemptEvidence]:
    """One compiled kernel attempt in the group's mapped region.

    ``batch`` is ``reduction.assemble(...)`` of this process's shards. The
    single host decision reads the replicated outcome and rejection counters
    once, after the step returns, and enforces the kernel's rejection budget.
    """

    if not isinstance(kernel, PreparedTrainingKernel):
        raise TypeError("kernel must be a PreparedTrainingKernel.")
    if not isinstance(reduction, AtomisticDataParallelReduction):
        raise TypeError("reduction must be an AtomisticDataParallelReduction.")
    if not isinstance(state, TrainingKernelState):
        raise TypeError("state must be a TrainingKernelState.")
    next_state, evidence = _mapped_attempt(kernel, reduction, state, batch)
    outcome, consecutive, attempt = jax.device_get(
        (evidence.outcome, next_state.consecutive_rejections, next_state.attempt_cursor)
    )
    enforce_rejection_budget(
        kernel, outcome, consecutive, attempt, state=next_state, evidence=evidence
    )
    return next_state, evidence


__all__ = (
    "AtomisticDataParallelReduction",
    "AtomisticDomainKernelObjective",
    "AtomisticDomainTrainingShard",
    "AtomisticTrainingDistributionPlan",
    "atomistic_domain_training_objectives",
    "run_distributed_training_attempt",
)
