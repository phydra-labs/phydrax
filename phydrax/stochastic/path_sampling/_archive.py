#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical pickle-free host archives for exact path-sampling restart lineage."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._array_archive import (
    array_collection_digest,
    read_array_archive,
    write_array_archive,
)
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import path_trajectory_id, PathBuffer, PathLineageLog
from ._moves import PathProposalEvaluation
from ._samplers import (
    initialize_retis,
    initialize_tis,
    initialize_tps,
    PreparedRETIS,
    PreparedTIS,
    PreparedTPS,
    RETISState,
    TISState,
    TPSState,
)
from ._targets import path_log_target


_EVALUATION_FIELDS = (
    "target_log_ratio",
    "selector_log_ratio",
    "modifier_log_ratio",
    "propagation_log_ratio",
    "length_log_ratio",
    "exchange_log_ratio",
    "log_acceptance_ratio",
    "target_valid",
    "selector_valid",
    "modifier_valid",
    "propagation_valid",
    "length_valid",
    "exchange_valid",
    "proposal_valid",
    "propagation_status",
)


class TPSRestart(StrictModule, NonTrainableState):
    """Restored state plus verified plan, preparation, and trajectory identities."""

    state: TPSState
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)
    archive_id: str = eqx.field(static=True)


def _state_arrays(state: TPSState, /) -> dict[str, np.ndarray]:
    arrays = {
        "path/positions": np.asarray(state.path.positions),
        "path/times": np.asarray(state.path.times),
        "path/length": np.asarray(state.path.length),
        "path/mask": np.asarray(state.path.mask),
        "path/direction": np.asarray(state.path.direction),
        "path/lineage": np.asarray(state.path.lineage),
        "state/log_target": np.asarray(state.log_target),
        "state/step_index": np.asarray(state.step_index),
        "state/accepted_count": np.asarray(state.accepted_count),
        "state/rejected_count": np.asarray(state.rejected_count),
        "state/trajectory_serial": np.asarray(state.trajectory_serial),
        "state/proposal_serial": np.asarray(state.proposal_serial),
        "lineage/parent": np.asarray(state.lineage.parent),
        "lineage/candidate": np.asarray(state.lineage.candidate),
        "lineage/committed": np.asarray(state.lineage.committed),
        "lineage/accepted": np.asarray(state.lineage.accepted),
        "lineage/mask": np.asarray(state.lineage.mask),
        "lineage/count": np.asarray(state.lineage.count),
        "lineage/overflowed": np.asarray(state.lineage.overflowed),
    }
    for name in _EVALUATION_FIELDS:
        arrays[f"evaluation/{name}"] = np.asarray(
            object.__getattribute__(state.last_evaluation, name)
        )
    return arrays


def _validate_tps_state(prepared: PreparedTPS, state: TPSState, /) -> None:
    reference = prepared.initial_path
    if (
        state.path.positions.shape != reference.positions.shape
        or state.path.positions.dtype != reference.positions.dtype
        or state.path.times.shape != reference.times.shape
        or state.path.times.dtype != reference.times.dtype
        or state.path.mask.shape != reference.mask.shape
        or state.path.mask.dtype != reference.mask.dtype
        or state.path.lineage.shape != reference.lineage.shape
        or state.path.lineage.dtype != reference.lineage.dtype
        or state.lineage.capacity != prepared.plan.lineage_capacity
        or state.prepared_id != prepared.prepared_id
    ):
        raise ValueError(
            "Restart state does not match prepared path shapes, dtypes, or capacities."
        )
    counters = (
        state.step_index,
        state.accepted_count,
        state.rejected_count,
        state.trajectory_serial,
        state.proposal_serial,
    )
    expected_lineage_count = jnp.minimum(
        state.step_index,
        jnp.asarray(state.lineage.capacity, dtype=state.step_index.dtype),
    ).astype(jnp.int32)
    previous_committed = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.uint32),
            state.lineage.committed[:-1],
        )
    )
    continuity = jnp.all(
        jnp.where(
            state.lineage.mask,
            state.lineage.parent == previous_committed,
            True,
        )
    )
    serial_order = jnp.all(
        jnp.where(
            state.lineage.mask[1:],
            state.lineage.candidate[1:] > state.lineage.candidate[:-1],
            True,
        )
    )
    last_index = jnp.maximum(state.lineage.count - 1, 0)
    current_matches = jnp.where(
        state.lineage.count > 0,
        state.trajectory_serial == state.lineage.committed[last_index],
        state.trajectory_serial == 0,
    )
    proposal_matches = jnp.where(
        state.lineage.count > 0,
        jnp.where(
            state.lineage.overflowed,
            state.proposal_serial >= state.lineage.candidate[last_index],
            state.proposal_serial == state.lineage.candidate[last_index],
        ),
        state.proposal_serial == 0,
    )
    overflow_matches = state.lineage.overflowed == (
        state.step_index > state.lineage.capacity
    )
    if any(jnp.asarray(value).shape != () for value in counters) or not bool(
        state.lineage.valid()
        & (state.lineage.count == expected_lineage_count)
        & continuity
        & serial_order
        & current_matches
        & proposal_matches
        & overflow_matches
        & (state.accepted_count + state.rejected_count == state.step_index)
        & (state.proposal_serial >= state.trajectory_serial)
    ):
        raise ValueError("Restart counters or proposal lineage are inconsistent.")
    target = path_log_target(prepared.plan.ensemble, prepared.plan.action, state.path)
    stored = jnp.asarray(state.log_target)
    if (
        stored.shape != ()
        or jnp.iscomplexobj(stored)
        or stored.dtype != target.dtype
        or not bool(state.path.valid())
        or not bool(jnp.isfinite(target) & jnp.isfinite(stored))
    ):
        raise ValueError("Restart path target or canonical invariants are invalid.")
    tolerance = 32.0 * jnp.finfo(target.dtype).eps * jnp.maximum(jnp.abs(target), 1.0)
    if not bool(jnp.abs(stored - target) <= tolerance):
        raise ValueError(
            "Restart state log_target does not match its prepared path target."
        )


def write_tps_restart(
    path: str | Path,
    prepared: PreparedTPS,
    state: TPSState,
    /,
) -> Path:
    """Atomically archive every dynamic state leaf without pickle or object arrays."""

    if not isinstance(prepared, PreparedTPS) or not isinstance(state, TPSState):
        raise TypeError("write_tps_restart requires PreparedTPS and TPSState.")
    _validate_tps_state(prepared, state)
    _require_arrays(
        _state_arrays(state),
        _state_arrays(initialize_tps(prepared)),
    )
    trajectory_id = path_trajectory_id(state.path)
    arrays = _state_arrays(state)
    payload_digest = array_collection_digest(arrays)
    metadata = {
        "kind": "transition-path-sampling-restart",
        "plan_id": prepared.plan.plan_id,
        "prepared_id": prepared.prepared_id,
        "initial_trajectory_id": prepared.initial_trajectory_id,
        "trajectory_id": trajectory_id,
        "payload_digest": payload_digest,
        "capacity": state.path.capacity,
        "event_shape": list(state.path.event_shape),
        "lineage_capacity": state.lineage.capacity,
    }
    metadata["archive_id"] = canonical_fingerprint(metadata)
    return write_array_archive(path, manifest=metadata, arrays=arrays)


def _require_arrays(
    arrays: dict[str, np.ndarray],
    template: dict[str, np.ndarray],
    /,
) -> None:
    if set(arrays) != set(template):
        raise ValueError("TPS restart array inventory changed.")
    for name, expected in template.items():
        value = arrays[name]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"TPS restart array {name!r} changed shape or dtype.")


def read_tps_restart(path: str | Path, prepared: PreparedTPS, /) -> TPSRestart:
    """Restore exact state only against the identical prepared runtime."""

    if not isinstance(prepared, PreparedTPS):
        raise TypeError("prepared must be PreparedTPS.")
    manifest, arrays = read_array_archive(path)
    required = {
        "kind",
        "plan_id",
        "prepared_id",
        "initial_trajectory_id",
        "trajectory_id",
        "payload_digest",
        "capacity",
        "event_shape",
        "lineage_capacity",
        "archive_id",
        "arrays",
    }
    if (
        set(manifest) != required
        or manifest["kind"] != "transition-path-sampling-restart"
    ):
        raise ValueError("TPS restart manifest is not canonical.")
    if (
        manifest["plan_id"] != prepared.plan.plan_id
        or manifest["prepared_id"] != prepared.prepared_id
        or manifest["initial_trajectory_id"] != prepared.initial_trajectory_id
    ):
        raise ValueError("TPS restart identity does not match the prepared runtime.")
    template = _state_arrays(initialize_tps(prepared))
    _require_arrays(arrays, template)
    if array_collection_digest(arrays) != manifest["payload_digest"]:
        raise ValueError("TPS restart payload digest changed.")
    metadata = {
        name: value
        for name, value in manifest.items()
        if name not in ("arrays", "archive_id")
    }
    if canonical_fingerprint(metadata) != manifest["archive_id"]:
        raise ValueError("TPS restart archive identity changed.")
    path_state = PathBuffer(
        jnp.asarray(arrays["path/positions"]),
        jnp.asarray(arrays["path/times"]),
        jnp.asarray(arrays["path/length"]),
        jnp.asarray(arrays["path/mask"]),
        jnp.asarray(arrays["path/direction"]),
        jnp.asarray(arrays["path/lineage"]),
    )
    lineage = PathLineageLog(
        jnp.asarray(arrays["lineage/parent"]),
        jnp.asarray(arrays["lineage/candidate"]),
        jnp.asarray(arrays["lineage/committed"]),
        jnp.asarray(arrays["lineage/accepted"]),
        jnp.asarray(arrays["lineage/mask"]),
        jnp.asarray(arrays["lineage/count"]),
        jnp.asarray(arrays["lineage/overflowed"]),
    )
    evaluation = PathProposalEvaluation(
        *(jnp.asarray(arrays[f"evaluation/{name}"]) for name in _EVALUATION_FIELDS)
    )
    state = TPSState(
        path_state,
        jnp.asarray(arrays["state/log_target"]),
        jnp.asarray(arrays["state/step_index"]),
        jnp.asarray(arrays["state/accepted_count"]),
        jnp.asarray(arrays["state/rejected_count"]),
        jnp.asarray(arrays["state/trajectory_serial"]),
        jnp.asarray(arrays["state/proposal_serial"]),
        lineage,
        evaluation,
        prepared.prepared_id,
    )
    _validate_tps_state(prepared, state)
    trajectory_id = path_trajectory_id(path_state)
    if trajectory_id != manifest["trajectory_id"] or not bool(path_state.valid()):
        raise ValueError("TPS restart trajectory identity or invariants changed.")
    return TPSRestart(
        state,
        prepared.plan.plan_id,
        prepared.prepared_id,
        trajectory_id,
        manifest["archive_id"],
    )


class TISRestart(StrictModule, NonTrainableState):
    """Exact restored TIS replicas and cross-replica trajectory identities."""

    state: TISState
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    trajectory_ids: tuple[str, ...] = eqx.field(static=True)
    archive_id: str = eqx.field(static=True)


class RETISRestart(StrictModule, NonTrainableState):
    """Exact restored RETIS replicas, exchanges, and proposal lineages."""

    state: RETISState
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    trajectory_ids: tuple[str, ...] = eqx.field(static=True)
    archive_id: str = eqx.field(static=True)


def _replica_arrays(
    replicas: tuple[TPSState, ...],
    counters: dict[str, np.ndarray],
    /,
) -> dict[str, np.ndarray]:
    arrays = dict(counters)
    for index, state in enumerate(replicas):
        prefix = f"replicas/{index:06d}/"
        arrays.update(
            {prefix + name: value for name, value in _state_arrays(state).items()}
        )
    return arrays


def _restore_tps_state(
    arrays: dict[str, np.ndarray],
    prefix: str,
    prepared_id: str,
    /,
) -> TPSState:
    def value(name: str):
        return jnp.asarray(arrays[prefix + name])

    path = PathBuffer(
        value("path/positions"),
        value("path/times"),
        value("path/length"),
        value("path/mask"),
        value("path/direction"),
        value("path/lineage"),
    )
    lineage = PathLineageLog(
        value("lineage/parent"),
        value("lineage/candidate"),
        value("lineage/committed"),
        value("lineage/accepted"),
        value("lineage/mask"),
        value("lineage/count"),
        value("lineage/overflowed"),
    )
    evaluation = PathProposalEvaluation(
        *(value(f"evaluation/{name}") for name in _EVALUATION_FIELDS)
    )
    return TPSState(
        path,
        value("state/log_target"),
        value("state/step_index"),
        value("state/accepted_count"),
        value("state/rejected_count"),
        value("state/trajectory_serial"),
        value("state/proposal_serial"),
        lineage,
        evaluation,
        prepared_id,
    )


def _write_replica_restart(
    path: str | Path,
    *,
    kind: str,
    plan_id: str,
    prepared_id: str,
    initial_trajectory_ids: tuple[str, ...],
    replicas: tuple[TPSState, ...],
    counters: dict[str, np.ndarray],
) -> Path:
    trajectory_ids = tuple(path_trajectory_id(state.path) for state in replicas)
    arrays = _replica_arrays(replicas, counters)
    metadata = {
        "kind": kind,
        "plan_id": plan_id,
        "prepared_id": prepared_id,
        "initial_trajectory_ids": list(initial_trajectory_ids),
        "trajectory_ids": list(trajectory_ids),
        "replica_count": len(replicas),
        "payload_digest": array_collection_digest(arrays),
    }
    metadata["archive_id"] = canonical_fingerprint(metadata)
    return write_array_archive(path, manifest=metadata, arrays=arrays)


def _read_replica_restart(
    path: str | Path,
    *,
    kind: str,
    plan_id: str,
    prepared_id: str,
    initial_trajectory_ids: tuple[str, ...],
    template_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    manifest, arrays = read_array_archive(path)
    required = {
        "kind",
        "plan_id",
        "prepared_id",
        "initial_trajectory_ids",
        "trajectory_ids",
        "replica_count",
        "payload_digest",
        "archive_id",
        "arrays",
    }
    if set(manifest) != required or manifest["kind"] != kind:
        raise ValueError("Replica path restart manifest is not canonical.")
    if (
        manifest["plan_id"] != plan_id
        or manifest["prepared_id"] != prepared_id
        or manifest["initial_trajectory_ids"] != list(initial_trajectory_ids)
        or manifest["replica_count"] != len(initial_trajectory_ids)
    ):
        raise ValueError("Replica path restart identity changed.")
    _require_arrays(arrays, template_arrays)
    if array_collection_digest(arrays) != manifest["payload_digest"]:
        raise ValueError("Replica path restart payload digest changed.")
    metadata = {
        name: value
        for name, value in manifest.items()
        if name not in ("arrays", "archive_id")
    }
    if canonical_fingerprint(metadata) != manifest["archive_id"]:
        raise ValueError("Replica path restart archive identity changed.")
    return manifest, arrays


def _validate_tis_counters(state: TISState, /) -> None:
    replica_steps = sum(
        (replica.step_index for replica in state.replicas),
        jnp.asarray(0, jnp.uint32),
    )
    if state.step_index.shape != () or not bool(state.step_index == replica_steps):
        raise ValueError("TIS global and replica step counters are inconsistent.")


def _validate_retis_counters(state: RETISState, /) -> None:
    replica_steps = sum(
        (replica.step_index for replica in state.replicas),
        jnp.asarray(0, jnp.uint32),
    )
    scalars = (
        state.step_index,
        state.exchange_count,
        state.accepted_exchange_count,
    )
    if any(value.shape != () for value in scalars) or not bool(
        (replica_steps == state.step_index + state.exchange_count)
        & (state.accepted_exchange_count <= state.exchange_count)
        & (state.exchange_count <= state.step_index)
    ):
        raise ValueError("RETIS global, replica, and exchange counters are inconsistent.")


def write_tis_restart(
    path: str | Path,
    prepared: PreparedTIS,
    state: TISState,
    /,
) -> Path:
    """Archive every TIS replica and rejected-move lineage exactly."""

    if not isinstance(prepared, PreparedTIS) or not isinstance(state, TISState):
        raise TypeError("write_tis_restart requires PreparedTIS and TISState.")
    if len(state.replicas) != len(prepared.replicas):
        raise ValueError("TIS restart replica count changed.")
    for replica_prepared, replica_state in zip(
        prepared.replicas, state.replicas, strict=True
    ):
        _validate_tps_state(replica_prepared, replica_state)
        _require_arrays(
            _state_arrays(replica_state),
            _state_arrays(initialize_tps(replica_prepared)),
        )
    _validate_tis_counters(state)
    return _write_replica_restart(
        path,
        kind="transition-interface-sampling-restart",
        plan_id=prepared.plan.plan_id,
        prepared_id=prepared.prepared_id,
        initial_trajectory_ids=tuple(
            replica.initial_trajectory_id for replica in prepared.replicas
        ),
        replicas=state.replicas,
        counters={"state/step_index": np.asarray(state.step_index)},
    )


def read_tis_restart(path: str | Path, prepared: PreparedTIS, /) -> TISRestart:
    """Restore TIS only against the identical interface prepared runtime."""

    if not isinstance(prepared, PreparedTIS):
        raise TypeError("prepared must be PreparedTIS.")
    initial_ids = tuple(replica.initial_trajectory_id for replica in prepared.replicas)
    template_state = initialize_tis(prepared)
    template = _replica_arrays(
        template_state.replicas,
        {"state/step_index": np.asarray(template_state.step_index)},
    )
    manifest, arrays = _read_replica_restart(
        path,
        kind="transition-interface-sampling-restart",
        plan_id=prepared.plan.plan_id,
        prepared_id=prepared.prepared_id,
        initial_trajectory_ids=initial_ids,
        template_arrays=template,
    )
    replicas = tuple(
        _restore_tps_state(
            arrays,
            f"replicas/{index:06d}/",
            prepared.replicas[index].prepared_id,
        )
        for index in range(len(prepared.replicas))
    )
    for replica_prepared, replica_state in zip(prepared.replicas, replicas, strict=True):
        _validate_tps_state(replica_prepared, replica_state)
    trajectory_ids = tuple(path_trajectory_id(state.path) for state in replicas)
    if list(trajectory_ids) != manifest["trajectory_ids"] or any(
        not bool(state.path.valid()) for state in replicas
    ):
        raise ValueError("TIS restart trajectory identity or invariants changed.")
    state = TISState(replicas, jnp.asarray(arrays["state/step_index"]))
    _validate_tis_counters(state)
    return TISRestart(
        state,
        prepared.plan.plan_id,
        prepared.prepared_id,
        trajectory_ids,
        str(manifest["archive_id"]),
    )


def write_retis_restart(
    path: str | Path,
    prepared: PreparedRETIS,
    state: RETISState,
    /,
) -> Path:
    """Archive minus/interface replicas, exchanges, and rejected lineages."""

    if not isinstance(prepared, PreparedRETIS) or not isinstance(state, RETISState):
        raise TypeError("write_retis_restart requires PreparedRETIS and RETISState.")
    if len(state.replicas) != len(prepared.replicas):
        raise ValueError("RETIS restart replica count changed.")
    for replica_prepared, replica_state in zip(
        prepared.replicas, state.replicas, strict=True
    ):
        _validate_tps_state(replica_prepared, replica_state)
        _require_arrays(
            _state_arrays(replica_state),
            _state_arrays(initialize_tps(replica_prepared)),
        )
    _validate_retis_counters(state)
    return _write_replica_restart(
        path,
        kind="replica-exchange-transition-interface-sampling-restart",
        plan_id=prepared.plan.plan_id,
        prepared_id=prepared.prepared_id,
        initial_trajectory_ids=tuple(
            replica.initial_trajectory_id for replica in prepared.replicas
        ),
        replicas=state.replicas,
        counters={
            "state/step_index": np.asarray(state.step_index),
            "state/exchange_count": np.asarray(state.exchange_count),
            "state/accepted_exchange_count": np.asarray(state.accepted_exchange_count),
        },
    )


def read_retis_restart(
    path: str | Path,
    prepared: PreparedRETIS,
    /,
) -> RETISRestart:
    """Restore RETIS only against the identical replica prepared runtime."""

    if not isinstance(prepared, PreparedRETIS):
        raise TypeError("prepared must be PreparedRETIS.")
    initial_ids = tuple(replica.initial_trajectory_id for replica in prepared.replicas)
    template_state = initialize_retis(prepared)
    template = _replica_arrays(
        template_state.replicas,
        {
            "state/step_index": np.asarray(template_state.step_index),
            "state/exchange_count": np.asarray(template_state.exchange_count),
            "state/accepted_exchange_count": np.asarray(
                template_state.accepted_exchange_count
            ),
        },
    )
    manifest, arrays = _read_replica_restart(
        path,
        kind="replica-exchange-transition-interface-sampling-restart",
        plan_id=prepared.plan.plan_id,
        prepared_id=prepared.prepared_id,
        initial_trajectory_ids=initial_ids,
        template_arrays=template,
    )
    replicas = tuple(
        _restore_tps_state(
            arrays,
            f"replicas/{index:06d}/",
            prepared.replicas[index].prepared_id,
        )
        for index in range(len(prepared.replicas))
    )
    for replica_prepared, replica_state in zip(prepared.replicas, replicas, strict=True):
        _validate_tps_state(replica_prepared, replica_state)
    trajectory_ids = tuple(path_trajectory_id(state.path) for state in replicas)
    if list(trajectory_ids) != manifest["trajectory_ids"] or any(
        not bool(state.path.valid()) for state in replicas
    ):
        raise ValueError("RETIS restart trajectory identity or invariants changed.")
    state = RETISState(
        replicas,
        jnp.asarray(arrays["state/step_index"]),
        jnp.asarray(arrays["state/exchange_count"]),
        jnp.asarray(arrays["state/accepted_exchange_count"]),
    )
    _validate_retis_counters(state)
    return RETISRestart(
        state,
        prepared.plan.plan_id,
        prepared.prepared_id,
        trajectory_ids,
        str(manifest["archive_id"]),
    )


__all__ = [
    "read_retis_restart",
    "read_tis_restart",
    "read_tps_restart",
    "RETISRestart",
    "TISRestart",
    "TPSRestart",
    "write_retis_restart",
    "write_tis_restart",
    "write_tps_restart",
]
