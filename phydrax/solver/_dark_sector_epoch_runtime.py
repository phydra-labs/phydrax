#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-residency epochs with durable, semantically unbounded continuation."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint, canonical_json
from .._identity import ExecutableSignature
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..lifecycle._chunk_repository import RepositoryConflictError
from ..lifecycle._event_graph_repository import (
    checkpoint_content_id,
    deterministic_commit_owner,
    EpochCommitReceipt,
    EventGraphEpochManifest,
    EventGraphRepository,
    GlobalEntity,
    GlobalEvent,
    GlobalEventEdge,
    GlobalWorkItem,
    RunTip,
    WorkLease,
)


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_POOL_NAMES = ("packet", "event", "product", "radiation", "work", "frontier")
CONSERVATION_COMPONENTS = (
    "energy",
    "momentum-x",
    "momentum-y",
    "momentum-z",
    "electric-charge",
    "dark-charge",
    "baryon-number",
    "lepton-number",
)
EpochStatus = Literal[
    "ready",
    "complete",
    "backpressured",
    "conservation-violated",
    "capacity-refused",
    "failed",
]
_STATUS_CODES = {
    "ready": 0,
    "complete": 1,
    "backpressured": 2,
    "conservation-violated": 3,
    "capacity-refused": 4,
    "failed": 5,
}


def _digest(value: str, role: str, /) -> str:
    result = str(value)
    if _DIGEST.fullmatch(result) is None:
        raise ValueError(f"{role} must be a lowercase SHA-256 digest.")
    return result


def _identifier(value: str, role: str, /) -> str:
    result = str(value).strip()
    if not result or len(result) > 256:
        raise ValueError(f"{role} must be a non-empty bounded identifier.")
    return result


def _positive(value: int, role: str, /) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{role} must be a positive integer.")
    return value


def _nonnegative(value: int, role: str, /) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{role} must be a non-negative integer.")
    return value


def encode_content_id(content_id: str, /) -> np.ndarray:
    """Encode a SHA-256 identity as eight stable big-endian uint32 words."""

    identity = _digest(content_id, "content_id")
    return np.asarray(
        [int(identity[index : index + 8], 16) for index in range(0, 64, 8)],
        dtype=np.uint32,
    )


def decode_content_id(words: ArrayLike, /) -> str:
    """Decode one eight-word device identity without depending on x64 support."""

    value = np.asarray(words, dtype=np.uint32)
    if value.shape != (8,):
        raise ValueError("A device content ID must have shape (8,).")
    return "".join(f"{int(item):08x}" for item in value)


def encode_content_ids(content_ids: Sequence[str], capacity: int, /) -> np.ndarray:
    capacity_ = _positive(capacity, "content ID capacity")
    identities = tuple(content_ids)
    if len(identities) > capacity_:
        raise ValueError("Content IDs exceed fixed resident capacity.")
    result = np.zeros((capacity_, 8), dtype=np.uint32)
    for index, identity in enumerate(identities):
        result[index] = encode_content_id(identity)
    return result


class DarkSectorEpochPlan(StrictModule, NonTrainableState):
    """Static shape, revision, and conservation contract for one compiled epoch."""

    packet_capacity: int = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    product_capacity: int = eqx.field(static=True)
    radiation_capacity: int = eqx.field(static=True)
    work_capacity: int = eqx.field(static=True)
    frontier_capacity: int = eqx.field(static=True)
    packet_width: int = eqx.field(static=True)
    event_width: int = eqx.field(static=True)
    product_width: int = eqx.field(static=True)
    radiation_width: int = eqx.field(static=True)
    work_width: int = eqx.field(static=True)
    frontier_width: int = eqx.field(static=True)
    species_revision_id: str = eqx.field(static=True)
    topology_revision_id: str = eqx.field(static=True)
    shard_count: int = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    conservation_atol: float = eqx.field(static=True)
    conservation_rtol: float = eqx.field(static=True)
    capacity_revision_id: str = eqx.field(static=True)
    executable_signature: ExecutableSignature = eqx.field(static=True)
    compile_signature_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        packet_capacity: int,
        event_capacity: int,
        product_capacity: int,
        radiation_capacity: int,
        work_capacity: int,
        frontier_capacity: int,
        packet_width: int,
        event_width: int,
        product_width: int,
        radiation_width: int,
        work_width: int,
        frontier_width: int,
        species_revision_id: str,
        topology_revision_id: str,
        shard_count: int = 1,
        precision_id: str = "float64",
        backend_id: str = "jax",
        conservation_atol: float = 1.0e-10,
        conservation_rtol: float = 1.0e-10,
    ):
        capacities = tuple(
            _positive(value, f"{name}_capacity")
            for name, value in zip(
                _POOL_NAMES,
                (
                    packet_capacity,
                    event_capacity,
                    product_capacity,
                    radiation_capacity,
                    work_capacity,
                    frontier_capacity,
                ),
                strict=True,
            )
        )
        widths = tuple(
            _positive(value, f"{name}_width")
            for name, value in zip(
                _POOL_NAMES,
                (
                    packet_width,
                    event_width,
                    product_width,
                    radiation_width,
                    work_width,
                    frontier_width,
                ),
                strict=True,
            )
        )
        if widths[4] != widths[5]:
            raise ValueError(
                "work_width and frontier_width must match for exact deferred replay."
            )
        species = _digest(species_revision_id, "species_revision_id")
        topology = _digest(topology_revision_id, "topology_revision_id")
        shards = _positive(shard_count, "shard_count")
        precision = _identifier(precision_id, "precision_id")
        if precision not in ("float32", "float64"):
            raise ValueError("precision_id must be 'float32' or 'float64'.")
        backend = _identifier(backend_id, "backend_id")
        atol = float(conservation_atol)
        rtol = float(conservation_rtol)
        if not all(math.isfinite(value) and value >= 0.0 for value in (atol, rtol)):
            raise ValueError("Conservation tolerances must be finite and non-negative.")
        capacity_revision = canonical_fingerprint(
            {
                "kind": "dark-sector-capacity-revision",
                "capacities": dict(zip(_POOL_NAMES, capacities, strict=True)),
                "widths": dict(zip(_POOL_NAMES, widths, strict=True)),
                "shard_count": shards,
            }
        )
        shapes = {
            **{
                f"{name}_ids": (capacity, 8)
                for name, capacity in zip(_POOL_NAMES, capacities, strict=True)
            },
            **{
                f"{name}_values": (capacity, width)
                for name, capacity, width in zip(
                    _POOL_NAMES, capacities, widths, strict=True
                )
            },
            "conservation": (len(CONSERVATION_COMPONENTS),),
        }
        signature = ExecutableSignature(
            shapes=shapes,
            dtypes={
                "content_ids": np.dtype(np.uint32),
                "mask": np.dtype(bool),
                "status": np.dtype(np.int8),
                "values": np.dtype(precision),
            },
            topology_ids={"dark-sector": topology, "species": species},
            capacities={
                **dict(zip(_POOL_NAMES, capacities, strict=True)),
                "shards": shards,
            },
            algorithm_facts={
                "runtime": "finite-resident-durable-epoch",
                "capacity_revision_id": capacity_revision,
                "conservation_components": CONSERVATION_COMPONENTS,
                "conservation_atol": atol,
                "conservation_rtol": rtol,
            },
            backend_facts={"backend_id": backend, "precision_id": precision},
        )
        for name, value in zip(_POOL_NAMES, capacities, strict=True):
            setattr(self, f"{name}_capacity", value)
        for name, value in zip(_POOL_NAMES, widths, strict=True):
            setattr(self, f"{name}_width", value)
        self.species_revision_id = species
        self.topology_revision_id = topology
        self.shard_count = shards
        self.precision_id = precision
        self.backend_id = backend
        self.conservation_atol = atol
        self.conservation_rtol = rtol
        self.capacity_revision_id = capacity_revision
        self.executable_signature = signature
        self.compile_signature_id = signature.signature_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dark-sector-epoch-plan",
                "capacity_revision_id": capacity_revision,
                "compile_signature_id": signature.signature_id,
                "species_revision_id": species,
                "topology_revision_id": topology,
                "conservation_atol": atol,
                "conservation_rtol": rtol,
            }
        )

    @property
    def value_dtype(self) -> np.dtype[Any]:
        return np.dtype(self.precision_id)

    def capacity(self, pool_name: str, /) -> int:
        _pool_name(pool_name)
        return int(object.__getattribute__(self, f"{pool_name}_capacity"))

    def width(self, pool_name: str, /) -> int:
        _pool_name(pool_name)
        return int(object.__getattribute__(self, f"{pool_name}_width"))


class DarkSectorEpochState(StrictModule):
    """Fixed-capacity device state; semantic growth occurs only through epochs."""

    plan: DarkSectorEpochPlan = eqx.field(static=True)
    epoch_sequence: int = eqx.field(static=True)
    parent_epoch_manifest_id: str | None = eqx.field(static=True)
    packet_ids: Array
    packet_values: Array
    packet_mask: Array
    packet_status: Array
    event_ids: Array
    event_values: Array
    event_mask: Array
    event_status: Array
    product_ids: Array
    product_values: Array
    product_mask: Array
    product_status: Array
    radiation_ids: Array
    radiation_values: Array
    radiation_mask: Array
    radiation_status: Array
    work_ids: Array
    work_values: Array
    work_mask: Array
    work_status: Array
    frontier_ids: Array
    frontier_values: Array
    frontier_mask: Array
    frontier_status: Array
    conservation_in: Array
    conservation_out: Array
    status: Array

    def __init__(
        self,
        plan: DarkSectorEpochPlan,
        /,
        *,
        epoch_sequence: int,
        parent_epoch_manifest_id: str | None,
        packet_ids: ArrayLike,
        packet_values: ArrayLike,
        packet_mask: ArrayLike,
        packet_status: ArrayLike,
        event_ids: ArrayLike,
        event_values: ArrayLike,
        event_mask: ArrayLike,
        event_status: ArrayLike,
        product_ids: ArrayLike,
        product_values: ArrayLike,
        product_mask: ArrayLike,
        product_status: ArrayLike,
        radiation_ids: ArrayLike,
        radiation_values: ArrayLike,
        radiation_mask: ArrayLike,
        radiation_status: ArrayLike,
        work_ids: ArrayLike,
        work_values: ArrayLike,
        work_mask: ArrayLike,
        work_status: ArrayLike,
        frontier_ids: ArrayLike,
        frontier_values: ArrayLike,
        frontier_mask: ArrayLike,
        frontier_status: ArrayLike,
        conservation_in: ArrayLike,
        conservation_out: ArrayLike,
        status: ArrayLike,
    ):
        if not isinstance(plan, DarkSectorEpochPlan):
            raise TypeError("plan must be DarkSectorEpochPlan.")
        epoch = _nonnegative(epoch_sequence, "epoch_sequence")
        parent = (
            None
            if parent_epoch_manifest_id is None
            else _digest(parent_epoch_manifest_id, "parent_epoch_manifest_id")
        )
        if (epoch == 0) != (parent is None):
            raise ValueError("Only epoch zero may omit parent_epoch_manifest_id.")
        self.plan = plan
        self.epoch_sequence = epoch
        self.parent_epoch_manifest_id = parent
        supplied = {
            "packet": (packet_ids, packet_values, packet_mask, packet_status),
            "event": (event_ids, event_values, event_mask, event_status),
            "product": (product_ids, product_values, product_mask, product_status),
            "radiation": (
                radiation_ids,
                radiation_values,
                radiation_mask,
                radiation_status,
            ),
            "work": (work_ids, work_values, work_mask, work_status),
            "frontier": (
                frontier_ids,
                frontier_values,
                frontier_mask,
                frontier_status,
            ),
        }
        for name, (ids, values, mask, pool_status) in supplied.items():
            capacity = plan.capacity(name)
            width = plan.width(name)
            ids_ = jnp.asarray(ids, dtype=jnp.uint32)
            values_ = jnp.asarray(values, dtype=jnp.dtype(plan.precision_id))
            mask_ = jnp.asarray(mask, dtype=bool)
            status_ = jnp.asarray(pool_status, dtype=jnp.int8)
            if ids_.shape != (capacity, 8):
                raise ValueError(f"{name}_ids must have shape ({capacity}, 8).")
            if values_.shape != (capacity, width):
                raise ValueError(f"{name}_values must have shape ({capacity}, {width}).")
            if mask_.shape != (capacity,) or status_.shape != (capacity,):
                raise ValueError(f"{name} mask/status must match pool capacity.")
            setattr(self, f"{name}_ids", ids_)
            setattr(self, f"{name}_values", values_)
            setattr(self, f"{name}_mask", mask_)
            setattr(self, f"{name}_status", status_)
        conservation_in_ = jnp.asarray(
            conservation_in, dtype=jnp.dtype(plan.precision_id)
        )
        conservation_out_ = jnp.asarray(
            conservation_out, dtype=jnp.dtype(plan.precision_id)
        )
        expected = (len(CONSERVATION_COMPONENTS),)
        if conservation_in_.shape != expected or conservation_out_.shape != expected:
            raise ValueError(f"Conservation ledgers must have shape {expected}.")
        status_ = jnp.asarray(status, dtype=jnp.int8)
        if status_.shape != ():
            raise ValueError("Epoch status must be scalar.")
        self.conservation_in = conservation_in_
        self.conservation_out = conservation_out_
        self.status = status_

    @property
    def conservation_residual(self) -> Array:
        return self.conservation_out - self.conservation_in

    @property
    def resident_counts(self) -> Array:
        return jnp.stack(
            tuple(
                jnp.sum(object.__getattribute__(self, f"{name}_mask"), dtype=jnp.int32)
                for name in _POOL_NAMES
            )
        )


class DarkSectorEpochResult(StrictModule):
    """Conservation-complete epoch proposal with rollback and capacity evidence."""

    state: DarkSectorEpochState
    complete: Array
    backpressured: Array
    rolled_back: Array
    conservation_ok: Array
    resident_high_water: Array
    evidence_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        state: DarkSectorEpochState,
        /,
        *,
        complete: ArrayLike,
        backpressured: ArrayLike,
        rolled_back: ArrayLike,
        evidence_ids: Sequence[str] = (),
    ):
        if not isinstance(state, DarkSectorEpochState):
            raise TypeError("state must be DarkSectorEpochState.")
        complete_ = _scalar_bool(complete, "complete")
        backpressured_ = _scalar_bool(backpressured, "backpressured")
        rolled_back_ = _scalar_bool(rolled_back, "rolled_back")
        scale = jnp.maximum(
            jnp.maximum(
                jnp.abs(state.conservation_in),
                jnp.abs(state.conservation_out),
            ),
            1.0,
        )
        tolerance = state.plan.conservation_atol + state.plan.conservation_rtol * scale
        conservation_ok = (
            jnp.all(jnp.isfinite(state.conservation_in))
            & jnp.all(jnp.isfinite(state.conservation_out))
            & jnp.all(jnp.abs(state.conservation_residual) <= tolerance)
        )
        evidence = tuple(
            sorted(_identifier(value, "epoch evidence ID") for value in evidence_ids)
        )
        if len(set(evidence)) != len(evidence):
            raise ValueError("Epoch evidence IDs must be unique.")
        self.state = state
        self.complete = complete_
        self.backpressured = backpressured_
        self.rolled_back = rolled_back_
        self.conservation_ok = conservation_ok
        self.resident_high_water = state.resident_counts
        self.evidence_ids = evidence


class DarkSectorWorkAdmission(StrictModule):
    state: DarkSectorEpochState
    accepted_count: Array
    deferred_count: Array
    backpressured: Array
    refused: Array


@dataclass(frozen=True, slots=True)
class DarkSectorResumePoint:
    state: DarkSectorEpochState
    tip: RunTip | None
    source_manifest: EventGraphEpochManifest | None
    exact_compile_replay: bool
    repartitioned: bool
    resume_id: str

    def __init__(
        self,
        state: DarkSectorEpochState,
        tip: RunTip | None,
        source_manifest: EventGraphEpochManifest | None,
        /,
        *,
        exact_compile_replay: bool,
        repartitioned: bool,
    ):
        source_id = None if source_manifest is None else source_manifest.epoch_manifest_id
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "tip", tip)
        object.__setattr__(self, "source_manifest", source_manifest)
        object.__setattr__(self, "exact_compile_replay", bool(exact_compile_replay))
        object.__setattr__(self, "repartitioned", bool(repartitioned))
        object.__setattr__(
            self,
            "resume_id",
            canonical_fingerprint(
                {
                    "kind": "dark-sector-resume-point",
                    "source_manifest_id": source_id,
                    "destination_plan_id": state.plan.plan_id,
                    "epoch_sequence": state.epoch_sequence,
                    "exact_compile_replay": bool(exact_compile_replay),
                    "repartitioned": bool(repartitioned),
                }
            ),
        )


def empty_dark_sector_epoch_state(
    plan: DarkSectorEpochPlan,
    /,
    *,
    epoch_sequence: int,
    parent_epoch_manifest_id: str | None = None,
) -> DarkSectorEpochState:
    """Create a zero-resident epoch with fixed shapes from ``plan``."""

    if not isinstance(plan, DarkSectorEpochPlan):
        raise TypeError("plan must be DarkSectorEpochPlan.")
    pools: dict[str, Array] = {}
    for name in _POOL_NAMES:
        capacity = plan.capacity(name)
        pools[f"{name}_ids"] = jnp.zeros((capacity, 8), dtype=jnp.uint32)
        pools[f"{name}_values"] = jnp.zeros(
            (capacity, plan.width(name)), dtype=jnp.dtype(plan.precision_id)
        )
        pools[f"{name}_mask"] = jnp.zeros((capacity,), dtype=bool)
        pools[f"{name}_status"] = jnp.zeros((capacity,), dtype=jnp.int8)
    return DarkSectorEpochState(
        plan,
        epoch_sequence=epoch_sequence,
        parent_epoch_manifest_id=parent_epoch_manifest_id,
        **pools,
        conservation_in=jnp.zeros(
            (len(CONSERVATION_COMPONENTS),), dtype=jnp.dtype(plan.precision_id)
        ),
        conservation_out=jnp.zeros(
            (len(CONSERVATION_COMPONENTS),), dtype=jnp.dtype(plan.precision_id)
        ),
        status=jnp.asarray(_STATUS_CODES["ready"], dtype=jnp.int8),
    )


def replace_dark_sector_pool(
    state: DarkSectorEpochState,
    pool_name: str,
    /,
    *,
    ids: ArrayLike,
    values: ArrayLike,
    mask: ArrayLike,
    status: ArrayLike,
) -> DarkSectorEpochState:
    """Replace one complete fixed pool without changing epoch identity."""

    name = _pool_name(pool_name)
    fields = _state_fields(state)
    fields[f"{name}_ids"] = ids
    fields[f"{name}_values"] = values
    fields[f"{name}_mask"] = mask
    fields[f"{name}_status"] = status
    return DarkSectorEpochState(
        state.plan,
        epoch_sequence=state.epoch_sequence,
        parent_epoch_manifest_id=state.parent_epoch_manifest_id,
        **fields,
    )


def replace_dark_sector_conservation(
    state: DarkSectorEpochState,
    conservation_in: ArrayLike,
    conservation_out: ArrayLike,
    /,
) -> DarkSectorEpochState:
    fields = _state_fields(state)
    fields["conservation_in"] = conservation_in
    fields["conservation_out"] = conservation_out
    return DarkSectorEpochState(
        state.plan,
        epoch_sequence=state.epoch_sequence,
        parent_epoch_manifest_id=state.parent_epoch_manifest_id,
        **fields,
    )


def admit_dark_sector_work(
    state: DarkSectorEpochState,
    work_ids: Sequence[str] | ArrayLike,
    work_values: ArrayLike,
    /,
    *,
    work_status: ArrayLike | None = None,
) -> DarkSectorWorkAdmission:
    """Fill work then durable frontier, transactionally refusing excess."""

    if not isinstance(state, DarkSectorEpochState):
        raise TypeError("state must be DarkSectorEpochState.")
    values = np.asarray(work_values, dtype=state.plan.value_dtype)
    if values.ndim != 2 or values.shape[1] != state.plan.work_width:
        raise ValueError(f"work_values must have shape (count, {state.plan.work_width}).")
    count = values.shape[0]
    ids = _normalized_id_rows(work_ids, count)
    statuses = (
        np.zeros((count,), dtype=np.int8)
        if work_status is None
        else np.asarray(work_status, dtype=np.int8)
    )
    if statuses.shape != (count,):
        raise ValueError("work_status must align with admitted work.")
    work_mask = np.asarray(state.work_mask, dtype=bool).copy()
    frontier_mask = np.asarray(state.frontier_mask, dtype=bool).copy()
    work_free = np.flatnonzero(~work_mask)
    frontier_free = np.flatnonzero(~frontier_mask)
    available = work_free.size + frontier_free.size
    if count > available:
        return DarkSectorWorkAdmission(
            state,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(count, dtype=jnp.int32),
            jnp.asarray(True),
            jnp.asarray(True),
        )
    direct = min(count, work_free.size)
    deferred = count - direct
    work_ids_host = np.asarray(state.work_ids, dtype=np.uint32).copy()
    work_values_host = np.asarray(state.work_values).copy()
    work_status_host = np.asarray(state.work_status, dtype=np.int8).copy()
    frontier_ids_host = np.asarray(state.frontier_ids, dtype=np.uint32).copy()
    frontier_values_host = np.asarray(state.frontier_values).copy()
    frontier_status_host = np.asarray(state.frontier_status, dtype=np.int8).copy()
    if direct:
        targets = work_free[:direct]
        work_ids_host[targets] = ids[:direct]
        work_values_host[targets] = values[:direct]
        work_status_host[targets] = statuses[:direct]
        work_mask[targets] = True
    if deferred:
        targets = frontier_free[:deferred]
        frontier_ids_host[targets] = ids[direct:]
        frontier_values_host[targets] = values[direct:]
        frontier_status_host[targets] = statuses[direct:]
        frontier_mask[targets] = True
    proposed = replace_dark_sector_pool(
        state,
        "work",
        ids=work_ids_host,
        values=work_values_host,
        mask=work_mask,
        status=work_status_host,
    )
    proposed = replace_dark_sector_pool(
        proposed,
        "frontier",
        ids=frontier_ids_host,
        values=frontier_values_host,
        mask=frontier_mask,
        status=frontier_status_host,
    )
    return DarkSectorWorkAdmission(
        proposed,
        jnp.asarray(count, dtype=jnp.int32),
        jnp.asarray(deferred, dtype=jnp.int32),
        jnp.asarray(deferred > 0),
        jnp.asarray(False),
    )


def finalize_dark_sector_epoch(
    previous_state: DarkSectorEpochState,
    proposed_state: DarkSectorEpochState,
    /,
    *,
    complete: ArrayLike,
    backpressured: ArrayLike,
    evidence_ids: Sequence[str] = (),
) -> DarkSectorEpochResult:
    """Atomically accept the proposal or return the exact previous state."""

    if previous_state.plan.plan_id != proposed_state.plan.plan_id:
        raise ValueError("Epoch rollback states must use the same plan.")
    if (
        previous_state.epoch_sequence != proposed_state.epoch_sequence
        or previous_state.parent_epoch_manifest_id
        != proposed_state.parent_epoch_manifest_id
    ):
        raise ValueError("Epoch rollback states must share one durable epoch identity.")
    scale = jnp.maximum(
        jnp.maximum(
            jnp.abs(proposed_state.conservation_in),
            jnp.abs(proposed_state.conservation_out),
        ),
        1.0,
    )
    conservation_ok = (
        jnp.all(jnp.isfinite(proposed_state.conservation_in))
        & jnp.all(jnp.isfinite(proposed_state.conservation_out))
        & jnp.all(
            jnp.abs(proposed_state.conservation_residual)
            <= proposed_state.plan.conservation_atol
            + proposed_state.plan.conservation_rtol * scale
        )
    )
    pools_ok = jnp.asarray(True)
    for name in _POOL_NAMES:
        mask = object.__getattribute__(proposed_state, f"{name}_mask")
        status = object.__getattribute__(proposed_state, f"{name}_status")
        values = object.__getattribute__(proposed_state, f"{name}_values")
        pools_ok = pools_ok & ~jnp.any(mask & (status < 0))
        pools_ok = pools_ok & jnp.all(
            jnp.where(mask[:, None], jnp.isfinite(values), True)
        )
    rolled_back = ~(conservation_ok & pools_ok)
    selected = _select_state(rolled_back, previous_state, proposed_state)
    complete_ = _scalar_bool(complete, "complete") & ~rolled_back
    backpressured_ = _scalar_bool(backpressured, "backpressured") & ~rolled_back
    status = jnp.where(
        rolled_back,
        _STATUS_CODES["conservation-violated"],
        jnp.where(
            backpressured_,
            _STATUS_CODES["backpressured"],
            jnp.where(complete_, _STATUS_CODES["complete"], _STATUS_CODES["ready"]),
        ),
    ).astype(jnp.int8)
    selected = _replace_epoch_status(selected, status)
    return DarkSectorEpochResult(
        selected,
        complete=complete_,
        backpressured=backpressured_,
        rolled_back=rolled_back,
        evidence_ids=evidence_ids,
    )


def dark_sector_epoch_checkpoint(state: DarkSectorEpochState, /) -> bytes:
    """Encode exact resident arrays and restart identities canonically."""

    arrays = {
        name: _array_record(object.__getattribute__(state, name))
        for name in _array_field_names()
    }
    record = {
        "kind": "dark-sector-epoch-checkpoint",
        "epoch_sequence": state.epoch_sequence,
        "parent_epoch_manifest_id": state.parent_epoch_manifest_id,
        "plan_id": state.plan.plan_id,
        "compile_signature_id": state.plan.compile_signature_id,
        "capacity_revision_id": state.plan.capacity_revision_id,
        "species_revision_id": state.plan.species_revision_id,
        "topology_revision_id": state.plan.topology_revision_id,
        "arrays": arrays,
    }
    return (canonical_json(record) + "\n").encode("utf-8")


class DarkSectorRunCoordinator:
    """Host transaction owner for epoch CAS, leases, resume, and repartitioning."""

    def __init__(
        self,
        graph_repository: EventGraphRepository,
        plan: DarkSectorEpochPlan,
        run_id: str,
        /,
        *,
        worker_id: str,
        eligible_worker_ids: Sequence[str] | None = None,
    ):
        if not isinstance(graph_repository, EventGraphRepository):
            raise TypeError("graph_repository must be EventGraphRepository.")
        if not isinstance(plan, DarkSectorEpochPlan):
            raise TypeError("plan must be DarkSectorEpochPlan.")
        self.graph_repository = graph_repository
        self.plan = plan
        self.run_id = _identifier(run_id, "run_id")
        self.worker_id = _identifier(worker_id, "worker_id")
        workers = (
            (self.worker_id,)
            if eligible_worker_ids is None
            else tuple(
                sorted(
                    _identifier(value, "eligible worker ID")
                    for value in eligible_worker_ids
                )
            )
        )
        if self.worker_id not in workers or len(set(workers)) != len(workers):
            raise ValueError("Eligible workers must be unique and include worker_id.")
        self.eligible_worker_ids = workers

    def resume(self) -> DarkSectorResumePoint:
        """Resume only the committed run tip; unreachable staged epochs are ignored."""

        tip = self.graph_repository.run_tip(self.run_id)
        if tip is None:
            state = empty_dark_sector_epoch_state(
                self.plan, epoch_sequence=0, parent_epoch_manifest_id=None
            )
            return DarkSectorResumePoint(
                state,
                None,
                None,
                exact_compile_replay=True,
                repartitioned=False,
            )
        persisted = self.graph_repository.load_epoch(tip.epoch_manifest_id)
        manifest = persisted.manifest
        if (
            manifest.run_id != self.run_id
            or manifest.epoch_sequence != tip.epoch_sequence
        ):
            raise RepositoryConflictError("Run tip and epoch manifest disagree.")
        record = _checkpoint_record(persisted.checkpoint_payload)
        arrays = _checkpoint_arrays(record)
        frontier_ids = arrays["frontier_ids"]
        frontier_values = arrays["frontier_values"]
        frontier_mask = arrays["frontier_mask"].astype(bool, copy=False)
        frontier_status = arrays["frontier_status"]
        active = np.flatnonzero(frontier_mask)
        count = int(active.size)
        if frontier_values.shape[1] != self.plan.work_width:
            raise ValueError(
                "Deferred frontier width is incompatible with destination plan."
            )
        if count > self.plan.work_capacity + self.plan.frontier_capacity:
            raise ValueError(
                "Destination capacities cannot losslessly admit the durable frontier."
            )
        state = empty_dark_sector_epoch_state(
            self.plan,
            epoch_sequence=manifest.epoch_sequence + 1,
            parent_epoch_manifest_id=manifest.epoch_manifest_id,
        )
        direct = min(count, self.plan.work_capacity)
        deferred = count - direct
        if direct:
            ids = np.zeros((self.plan.work_capacity, 8), dtype=np.uint32)
            values = np.zeros(
                (self.plan.work_capacity, self.plan.work_width),
                dtype=self.plan.value_dtype,
            )
            mask = np.zeros((self.plan.work_capacity,), dtype=bool)
            status = np.zeros((self.plan.work_capacity,), dtype=np.int8)
            ids[:direct] = frontier_ids[active[:direct]]
            values[:direct] = frontier_values[active[:direct]]
            status[:direct] = frontier_status[active[:direct]]
            mask[:direct] = True
            state = replace_dark_sector_pool(
                state, "work", ids=ids, values=values, mask=mask, status=status
            )
        if deferred:
            ids = np.zeros((self.plan.frontier_capacity, 8), dtype=np.uint32)
            values = np.zeros(
                (self.plan.frontier_capacity, self.plan.frontier_width),
                dtype=self.plan.value_dtype,
            )
            mask = np.zeros((self.plan.frontier_capacity,), dtype=bool)
            status = np.zeros((self.plan.frontier_capacity,), dtype=np.int8)
            ids[:deferred] = frontier_ids[active[direct:]]
            values[:deferred] = frontier_values[active[direct:]]
            status[:deferred] = frontier_status[active[direct:]]
            mask[:deferred] = True
            state = replace_dark_sector_pool(
                state, "frontier", ids=ids, values=values, mask=mask, status=status
            )
        exact = manifest.compile_signature_id == self.plan.compile_signature_id
        return DarkSectorResumePoint(
            state,
            tip,
            manifest,
            exact_compile_replay=exact,
            repartitioned=not exact or count > 0,
        )

    def claim_work(
        self,
        work: GlobalWorkItem,
        /,
        *,
        issued_at: int,
        expires_at: int,
    ) -> WorkLease:
        return self.graph_repository.acquire_work_lease(
            work,
            self.worker_id,
            eligible_worker_ids=self.eligible_worker_ids,
            issued_at=issued_at,
            expires_at=expires_at,
        )

    def commit_epoch(
        self,
        result: DarkSectorEpochResult,
        /,
        *,
        entities: Sequence[GlobalEntity] = (),
        events: Sequence[GlobalEvent] = (),
        edges: Sequence[GlobalEventEdge] = (),
        work_items: Sequence[GlobalWorkItem] = (),
        matrix_element_revision_id: str,
        evidence_ids: Sequence[str] = (),
        committed_at: int | None = None,
    ) -> EpochCommitReceipt:
        """Publish graph records, immutable checkpoint, epoch slot, and CAS tip."""

        if not isinstance(result, DarkSectorEpochResult):
            raise TypeError("result must be DarkSectorEpochResult.")
        if result.state.plan.plan_id != self.plan.plan_id:
            raise ValueError("Epoch result was produced by a different plan.")
        if not bool(np.asarray(result.complete)):
            raise ValueError("Only complete finite epochs may be committed.")
        if bool(np.asarray(result.rolled_back)):
            raise ValueError("Rolled-back epoch proposals cannot be committed.")
        if not bool(np.asarray(result.conservation_ok)):
            raise ValueError("Epoch conservation ledger is incomplete.")
        state = result.state
        checkpoint = dark_sector_epoch_checkpoint(state)
        deferred_ids = _active_content_ids(state.frontier_ids, state.frontier_mask)
        matrix_revision = _digest(
            matrix_element_revision_id, "matrix_element_revision_id"
        )
        requested_entities = tuple(sorted(entity.entity_id for entity in entities))
        requested_events = tuple(sorted(event.event_id for event in events))
        requested_edges = tuple(sorted(edge.edge_id for edge in edges))
        requested_work = tuple(
            sorted({*(work.work_id for work in work_items), *deferred_ids})
        )
        requested_evidence = tuple(sorted((*result.evidence_ids, *tuple(evidence_ids))))

        def matches_request(
            existing: EventGraphEpochManifest, persisted_checkpoint: bytes
        ) -> bool:
            return (
                persisted_checkpoint == checkpoint
                and existing.parent_manifest_id == state.parent_epoch_manifest_id
                and existing.plan_id == self.plan.plan_id
                and existing.matrix_element_revision_id == matrix_revision
                and existing.entity_ids == requested_entities
                and existing.event_ids == requested_events
                and existing.edge_ids == requested_edges
                and existing.work_ids == requested_work
                and existing.deferred_work_ids == deferred_ids
                and existing.evidence_ids == requested_evidence
            )

        tip = self.graph_repository.run_tip(self.run_id)
        if tip is not None and tip.epoch_sequence == state.epoch_sequence:
            persisted = self.graph_repository.load_epoch(tip.epoch_manifest_id)
            existing = persisted.manifest
            if not matches_request(existing, persisted.checkpoint_payload):
                raise RepositoryConflictError(
                    "Committed epoch sequence contains different immutable content."
                )
            return self.graph_repository.append_epoch(
                existing,
                checkpoint,
                writer_id=self.worker_id,
                committed_at=committed_at,
            )
        expected_sequence = 0 if tip is None else tip.epoch_sequence + 1
        expected_parent = None if tip is None else tip.epoch_manifest_id
        if (
            state.epoch_sequence != expected_sequence
            or state.parent_epoch_manifest_id != expected_parent
        ):
            raise RepositoryConflictError(
                "Epoch result does not continue the committed run tip."
            )
        semantic_epoch_id = canonical_fingerprint(
            {
                "kind": "dark-sector-epoch-owner-key",
                "run_id": self.run_id,
                "epoch_sequence": state.epoch_sequence,
                "parent_manifest_id": expected_parent,
            }
        )
        owner = deterministic_commit_owner(semantic_epoch_id, self.eligible_worker_ids)
        if self.worker_id != owner:
            raise RepositoryConflictError(
                f"Worker {self.worker_id!r} is not epoch commit owner {owner!r}."
            )
        slotted = self.graph_repository.epoch_slot(self.run_id, state.epoch_sequence)
        if slotted is not None:
            persisted = self.graph_repository.load_epoch(slotted.epoch_manifest_id)
            if not matches_request(slotted, persisted.checkpoint_payload):
                raise RepositoryConflictError(
                    "Staged epoch slot contains different immutable content."
                )
            return self.graph_repository.append_epoch(
                slotted,
                checkpoint,
                writer_id=self.worker_id,
                committed_at=committed_at,
            )
        for entity in entities:
            self.graph_repository.put_entity(
                entity, self.worker_id, committed_at=committed_at
            )
        for event in events:
            self.graph_repository.put_event(
                event, self.worker_id, committed_at=committed_at
            )
        for edge in edges:
            self.graph_repository.put_edge(
                edge, self.worker_id, committed_at=committed_at
            )
        for work in work_items:
            self.graph_repository.put_work(
                work, self.worker_id, committed_at=committed_at
            )
        work_by_id = {work.work_id: work for work in work_items}
        for identity in deferred_ids:
            if identity not in work_by_id:
                work_by_id[identity] = self.graph_repository.get_work(identity)
        manifest = EventGraphEpochManifest(
            self.run_id,
            state.epoch_sequence,
            state.parent_epoch_manifest_id,
            self.plan.plan_id,
            self.plan.compile_signature_id,
            self.plan.capacity_revision_id,
            self.plan.species_revision_id,
            self.plan.topology_revision_id,
            entity_ids=requested_entities,
            event_ids=requested_events,
            edge_ids=requested_edges,
            work_ids=tuple(work_by_id),
            deferred_work_ids=deferred_ids,
            matrix_element_revision_id=matrix_revision,
            checkpoint_id=checkpoint_content_id(checkpoint),
            commit_owner_id=owner,
            conservation_status="conserved",
            evidence_ids=(*result.evidence_ids, *tuple(evidence_ids)),
        )
        return self.graph_repository.append_epoch(
            manifest,
            checkpoint,
            writer_id=self.worker_id,
            committed_at=committed_at,
        )


def _pool_name(value: str, /) -> str:
    if value not in _POOL_NAMES:
        raise ValueError(f"pool_name must be one of {_POOL_NAMES!r}.")
    return value


def _scalar_bool(value: ArrayLike, role: str, /) -> Array:
    result = jnp.asarray(value, dtype=bool)
    if result.shape != ():
        raise ValueError(f"{role} must be scalar.")
    return result


def _state_fields(state: DarkSectorEpochState, /) -> dict[str, ArrayLike]:
    fields: dict[str, ArrayLike] = {}
    for name in _POOL_NAMES:
        for suffix in ("ids", "values", "mask", "status"):
            field = f"{name}_{suffix}"
            fields[field] = object.__getattribute__(state, field)
    fields["conservation_in"] = state.conservation_in
    fields["conservation_out"] = state.conservation_out
    fields["status"] = state.status
    return fields


def _replace_epoch_status(
    state: DarkSectorEpochState, status: ArrayLike, /
) -> DarkSectorEpochState:
    fields = _state_fields(state)
    fields["status"] = status
    return DarkSectorEpochState(
        state.plan,
        epoch_sequence=state.epoch_sequence,
        parent_epoch_manifest_id=state.parent_epoch_manifest_id,
        **fields,
    )


def _select_state(
    predicate: Array,
    selected_when_true: DarkSectorEpochState,
    selected_when_false: DarkSectorEpochState,
    /,
) -> DarkSectorEpochState:
    true_fields = _state_fields(selected_when_true)
    false_fields = _state_fields(selected_when_false)
    fields = {
        name: jnp.where(predicate, true_fields[name], false_fields[name])
        for name in true_fields
    }
    return DarkSectorEpochState(
        selected_when_true.plan,
        epoch_sequence=selected_when_true.epoch_sequence,
        parent_epoch_manifest_id=selected_when_true.parent_epoch_manifest_id,
        **fields,
    )


def _normalized_id_rows(
    identities: Sequence[str] | ArrayLike, count: int, /
) -> np.ndarray:
    if isinstance(identities, Sequence) and not isinstance(
        identities, (str, bytes, np.ndarray)
    ):
        if not all(isinstance(item, str) for item in identities):
            raise TypeError("String work IDs must contain only strings.")
        if len(identities) != count:
            raise ValueError("work_ids must align with work_values.")
        if count:
            return np.stack(tuple(encode_content_id(item) for item in identities), axis=0)
        return np.zeros((0, 8), dtype=np.uint32)
    result = np.asarray(identities, dtype=np.uint32)
    if result.shape != (count, 8):
        raise ValueError("Encoded work_ids must have shape (count, 8).")
    return result


def _array_field_names() -> tuple[str, ...]:
    return (
        *tuple(
            f"{name}_{suffix}"
            for name in _POOL_NAMES
            for suffix in ("ids", "values", "mask", "status")
        ),
        "conservation_in",
        "conservation_out",
        "status",
    )


def _array_record(value: ArrayLike, /) -> dict[str, object]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Checkpoint arrays cannot use object dtype.")
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    payload = canonical.tobytes(order="C")
    return {
        "shape": list(canonical.shape),
        "dtype": canonical.dtype.str,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "base64": base64.b64encode(payload).decode("ascii"),
    }


def _checkpoint_record(payload: bytes, /) -> Mapping[str, object]:
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Dark-sector checkpoint is not valid JSON.") from error
    if (
        not isinstance(record, Mapping)
        or record.get("kind") != "dark-sector-epoch-checkpoint"
    ):
        raise ValueError("Dark-sector checkpoint has the wrong record kind.")
    if (canonical_json(record) + "\n").encode("utf-8") != payload:
        raise ValueError("Dark-sector checkpoint is not canonically encoded.")
    return record


def _checkpoint_arrays(record: Mapping[str, object], /) -> dict[str, np.ndarray]:
    raw_arrays = record.get("arrays")
    if not isinstance(raw_arrays, Mapping) or set(raw_arrays) != set(
        _array_field_names()
    ):
        raise ValueError("Dark-sector checkpoint array inventory is incomplete.")
    result: dict[str, np.ndarray] = {}
    for name in _array_field_names():
        raw = raw_arrays[name]
        if not isinstance(raw, Mapping):
            raise TypeError("Dark-sector checkpoint array record is malformed.")
        shape_value = raw.get("shape")
        dtype_value = raw.get("dtype")
        digest_value = raw.get("sha256")
        payload_value = raw.get("base64")
        if (
            not isinstance(shape_value, Sequence)
            or isinstance(shape_value, (str, bytes))
            or not all(type(item) is int and item >= 0 for item in shape_value)
            or not isinstance(dtype_value, str)
            or not isinstance(digest_value, str)
            or not isinstance(payload_value, str)
        ):
            raise TypeError("Dark-sector checkpoint array metadata is malformed.")
        try:
            data = base64.b64decode(payload_value, validate=True)
        except ValueError as error:
            raise ValueError(
                "Dark-sector checkpoint array payload is invalid."
            ) from error
        if hashlib.sha256(data).hexdigest() != _digest(digest_value, "array sha256"):
            raise ValueError("Dark-sector checkpoint array digest is invalid.")
        dtype = np.dtype(dtype_value)
        if dtype.hasobject:
            raise TypeError("Dark-sector checkpoint cannot restore object arrays.")
        shape = tuple(shape_value)
        expected = math.prod(shape) * dtype.itemsize
        if len(data) != expected:
            raise ValueError("Dark-sector checkpoint array byte size is invalid.")
        array = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        array.setflags(write=False)
        result[name] = array
    return result


def _active_content_ids(ids: ArrayLike, mask: ArrayLike, /) -> tuple[str, ...]:
    ids_host = np.asarray(ids, dtype=np.uint32)
    mask_host = np.asarray(mask, dtype=bool)
    if ids_host.shape != (mask_host.size, 8) or mask_host.shape != (ids_host.shape[0],):
        raise ValueError("Content ID masks do not align.")
    identities = tuple(
        decode_content_id(ids_host[index]) for index in np.flatnonzero(mask_host)
    )
    if any(identity == "0" * 64 for identity in identities):
        raise ValueError("Active resident slots require nonzero content identities.")
    if len(set(identities)) != len(identities):
        raise ValueError("Active resident content identities must be unique.")
    return identities


__all__ = [
    "CONSERVATION_COMPONENTS",
    "DarkSectorEpochPlan",
    "DarkSectorEpochResult",
    "DarkSectorEpochState",
    "DarkSectorResumePoint",
    "DarkSectorRunCoordinator",
    "DarkSectorWorkAdmission",
    "EpochStatus",
    "admit_dark_sector_work",
    "dark_sector_epoch_checkpoint",
    "decode_content_id",
    "empty_dark_sector_epoch_state",
    "encode_content_id",
    "encode_content_ids",
    "finalize_dark_sector_epoch",
    "replace_dark_sector_conservation",
    "replace_dark_sector_pool",
]
