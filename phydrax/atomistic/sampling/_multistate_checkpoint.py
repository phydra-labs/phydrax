#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._units import AtomisticUnitSystem
from ._multistate import (
    _identity_token,
    AtomisticMultistateSegmentResult,
    AtomisticMultistateState,
    PreparedAtomisticMultistate,
)


_MULTISTATE_CHECKPOINT_FORMAT = "phydrax-atomistic-multistate-segment"


class AtomisticMultistateCheckpointPlan(StrictModule, NonTrainableState):
    runtime: PreparedAtomisticMultistate
    segment_capacity: int = eqx.field(static=True)
    scope_id: str | None = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: PreparedAtomisticMultistate,
        segment_capacity: int,
        /,
        *,
        scope_id: str | None = None,
    ):
        if not isinstance(runtime, PreparedAtomisticMultistate):
            raise TypeError("runtime must be PreparedAtomisticMultistate.")
        capacity = int(segment_capacity)
        if capacity <= 0:
            raise ValueError("segment_capacity must be positive.")
        if scope_id is not None and (
            not isinstance(scope_id, str) or not scope_id or scope_id != scope_id.strip()
        ):
            raise ValueError("scope_id must be a canonical nonempty string or None.")
        self.runtime = runtime
        self.segment_capacity = capacity
        self.scope_id = scope_id
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "atomistic-multistate-checkpoint-plan",
                "runtime": runtime.prepared_id,
                "multistate": runtime.plan.plan_id,
                "dynamics": runtime.dynamics.prepared_id,
                "thermodynamic": runtime.thermodynamic.table_id,
                "units": runtime.dynamics.system.plan.units.unit_system_id,
                "segment_capacity": capacity,
                "scope_id": scope_id,
            }
        )


class AtomisticMultistateCheckpoint(StrictModule):
    segment: AtomisticMultistateSegmentResult
    units: AtomisticUnitSystem
    payload_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    @property
    def state(self) -> AtomisticMultistateState:
        return self.segment.successor_state


def _state_with_continuation(
    state: AtomisticMultistateState,
    continuation_id: str,
    /,
) -> AtomisticMultistateState:
    return AtomisticMultistateState(
        dynamics=state.dynamics,
        state_at_replica=state.state_at_replica,
        reduced_potential_cache=state.reduced_potential_cache,
        reduced_potential_valid=state.reduced_potential_valid,
        cache_iteration=state.cache_iteration,
        iteration_index=state.iteration_index,
        draw_index=state.draw_index,
        exchange_action_counter=state.exchange_action_counter,
        exchange_parity=state.exchange_parity,
        barostat_action_counter=state.barostat_action_counter,
        sams_action_counter=state.sams_action_counter,
        sams=state.sams,
        root_key=state.root_key,
        segment_index=state.segment_index,
        continuation_token=_identity_token(continuation_id),
        plan_id=state.plan_id,
    )


def _segment_template(
    plan: AtomisticMultistateCheckpointPlan,
    state: AtomisticMultistateState,
    /,
    *,
    segment_id: str,
    predecessor_id: str,
) -> AtomisticMultistateSegmentResult:
    runtime = plan.runtime
    capacity = plan.segment_capacity
    replicas = runtime.plan.replica_count
    states = runtime.plan.state_count
    pair_count = max(replicas - 1, 0)
    dtype = state.dynamics.kinematics.positions.dtype
    boolean = jnp.zeros((capacity,), dtype=jnp.bool_)
    replica_boolean = jnp.zeros((capacity, replicas), dtype=jnp.bool_)
    pair_boolean = jnp.zeros((capacity, pair_count), dtype=jnp.bool_)
    return AtomisticMultistateSegmentResult(
        successor_state=_state_with_continuation(state, segment_id),
        reduced_potentials=jnp.zeros((capacity, states, replicas), dtype=dtype),
        coverage=jnp.zeros((capacity, states, replicas), dtype=jnp.bool_),
        sample_active=replica_boolean,
        origin_state=jnp.full((capacity, replicas), -1, dtype=jnp.int32),
        state_at_replica=jnp.full((capacity, replicas), -1, dtype=jnp.int32),
        chain_index=jnp.full((capacity, replicas), -1, dtype=jnp.int32),
        draw_index=jnp.full((capacity, replicas), -1, dtype=jnp.int64),
        repeat_index=jnp.full((capacity, replicas), -1, dtype=jnp.int32),
        dependence_group_index=jnp.full((capacity, replicas), -1, dtype=jnp.int32),
        pair_indices=jnp.full((capacity, pair_count, 2), -1, dtype=jnp.int32),
        exchange_attempted=pair_boolean,
        exchange_accepted=pair_boolean,
        exchange_log_acceptance=jnp.zeros((capacity, pair_count), dtype=dtype),
        sams_attempted=replica_boolean,
        sams_changed=replica_boolean,
        sams_adapting=replica_boolean,
        dynamics_accepted=replica_boolean,
        barostat_attempted=replica_boolean,
        barostat_accepted=replica_boolean,
        iteration_valid=boolean,
        count=jnp.zeros((), dtype=jnp.int64),
        start_watermark=jnp.zeros((), dtype=jnp.int64),
        stop_watermark=jnp.zeros((), dtype=jnp.int64),
        successful=jnp.asarray(False),
        inverse_temperatures=runtime.thermodynamic.beta,
        units=runtime.dynamics.system.plan.units,
        run_id=runtime.plan.run_id,
        measure_id=runtime.thermodynamic.phase_space_measure_id,
        state_ids=runtime.thermodynamic.state_ids,
        potential_ids=runtime.thermodynamic.potential_ids,
        bias_ids=runtime.thermodynamic.bias_ids,
        producer_id=runtime.prepared_id,
        unit_id="1",
        reduced_convention_id=runtime.thermodynamic.reduced_convention_id,
        qualification_id=runtime.plan.qualification.qualification_id,
        sampling_exact=runtime.plan.qualification.sampling_exact,
        sampling_bias_bound=runtime.plan.qualification.sampling_bias_bound,
        segment_id=segment_id,
        predecessor_id=predecessor_id,
        runtime_id=runtime.prepared_id,
    )


def _validate_segment(
    plan: AtomisticMultistateCheckpointPlan,
    result: AtomisticMultistateSegmentResult,
    /,
) -> None:
    runtime = plan.runtime
    capacity = plan.segment_capacity
    replicas = runtime.plan.replica_count
    states = runtime.plan.state_count
    pair_count = max(replicas - 1, 0)
    shaped_arrays = (
        (result.reduced_potentials, (capacity, states, replicas)),
        (result.coverage, (capacity, states, replicas)),
        (result.sample_active, (capacity, replicas)),
        (result.origin_state, (capacity, replicas)),
        (result.state_at_replica, (capacity, replicas)),
        (result.chain_index, (capacity, replicas)),
        (result.draw_index, (capacity, replicas)),
        (result.repeat_index, (capacity, replicas)),
        (result.dependence_group_index, (capacity, replicas)),
        (result.pair_indices, (capacity, pair_count, 2)),
        (result.exchange_attempted, (capacity, pair_count)),
        (result.exchange_accepted, (capacity, pair_count)),
        (result.exchange_log_acceptance, (capacity, pair_count)),
        (result.sams_attempted, (capacity, replicas)),
        (result.sams_changed, (capacity, replicas)),
        (result.sams_adapting, (capacity, replicas)),
        (result.dynamics_accepted, (capacity, replicas)),
        (result.barostat_attempted, (capacity, replicas)),
        (result.barostat_accepted, (capacity, replicas)),
        (result.iteration_valid, (capacity,)),
        (result.inverse_temperatures, (states,)),
    )
    if any(value.shape != shape for value, shape in shaped_arrays):
        raise ValueError("Multistate segment arrays do not match checkpoint capacities.")
    valid = np.asarray(result.iteration_valid, dtype=np.bool_)
    prefix = np.arange(capacity) < int(result.count)
    sample = np.asarray(result.sample_active, dtype=np.bool_)
    coverage = np.asarray(result.coverage, dtype=np.bool_)
    inactive_sample = ~sample
    inactive_iteration = ~valid
    inactive_pair = inactive_iteration[:, None]
    if (
        result.runtime_id != runtime.prepared_id
        or result.run_id != runtime.plan.run_id
        or result.measure_id != runtime.thermodynamic.phase_space_measure_id
        or result.state_ids != runtime.thermodynamic.state_ids
        or result.potential_ids != runtime.thermodynamic.potential_ids
        or result.bias_ids != runtime.thermodynamic.bias_ids
        or result.producer_id != runtime.prepared_id
        or result.unit_id != "1"
        or result.reduced_convention_id != runtime.thermodynamic.reduced_convention_id
        or result.qualification_id != runtime.plan.qualification.qualification_id
        or result.sampling_exact != runtime.plan.qualification.sampling_exact
        or result.sampling_bias_bound != runtime.plan.qualification.sampling_bias_bound
        or not np.all(np.isfinite(np.asarray(result.inverse_temperatures)))
        or np.any(np.asarray(result.inverse_temperatures) <= 0.0)
        or not np.array_equal(
            np.asarray(result.inverse_temperatures),
            np.asarray(runtime.thermodynamic.beta),
        )
        or result.units.unit_system_id
        != runtime.dynamics.system.plan.units.unit_system_id
        or result.successor_state.plan_id != runtime.prepared_id
        or not np.array_equal(
            np.asarray(result.successor_state.continuation_token),
            np.asarray(_identity_token(result.segment_id)),
        )
        or not np.array_equal(valid, prefix)
        or int(result.stop_watermark) - int(result.start_watermark) != int(result.count)
        or int(result.successor_state.iteration_index) != int(result.stop_watermark)
        or np.any(sample & ~valid[:, None])
        or not np.all(sample == sample[:, :1])
        or bool(result.successful) != (int(result.count) == capacity)
        or np.any(sample & np.asarray(result.sams_adapting))
        or not np.array_equal(
            coverage,
            sample[:, None, :] * np.ones((1, states, 1), dtype=np.bool_),
        )
        or np.any(~np.isfinite(np.asarray(result.reduced_potentials)[coverage]))
        or np.any(np.asarray(result.reduced_potentials)[~coverage] != 0.0)
        or np.any(np.asarray(result.origin_state)[inactive_sample] != -1)
        or np.any(np.asarray(result.state_at_replica)[inactive_sample] != -1)
        or np.any(np.asarray(result.origin_state)[sample] < 0)
        or np.any(np.asarray(result.origin_state)[sample] >= states)
        or np.any(np.asarray(result.state_at_replica)[sample] < 0)
        or np.any(np.asarray(result.state_at_replica)[sample] >= states)
        or np.any(np.asarray(result.chain_index)[sample] < 0)
        or np.any(np.asarray(result.draw_index)[sample] < 0)
        or np.any(np.asarray(result.repeat_index)[sample] < 0)
        or np.any(np.asarray(result.dependence_group_index)[sample] < 0)
        or np.any(np.asarray(result.chain_index)[inactive_sample] != -1)
        or np.any(np.asarray(result.draw_index)[inactive_sample] != -1)
        or np.any(np.asarray(result.repeat_index)[inactive_sample] != -1)
        or np.any(np.asarray(result.dependence_group_index)[inactive_sample] != -1)
        or np.any(np.asarray(result.pair_indices)[inactive_iteration] != -1)
        or np.any(np.asarray(result.exchange_attempted) & inactive_pair)
        or np.any(np.asarray(result.exchange_accepted) & inactive_pair)
        or np.any(
            np.where(
                inactive_pair,
                np.asarray(result.exchange_log_acceptance),
                0.0,
            )
            != 0.0
        )
        or np.any(np.asarray(result.sams_attempted) & inactive_iteration[:, None])
        or np.any(np.asarray(result.sams_changed) & inactive_iteration[:, None])
        or np.any(np.asarray(result.sams_adapting) & inactive_iteration[:, None])
        or np.any(np.asarray(result.dynamics_accepted) & inactive_iteration[:, None])
        or np.any(np.asarray(result.barostat_attempted) & inactive_iteration[:, None])
        or np.any(np.asarray(result.barostat_accepted) & inactive_iteration[:, None])
    ):
        raise ValueError("Multistate segment watermark, identity, or padding is invalid.")


def write_atomistic_multistate_checkpoint(
    path: str | Path,
    plan: AtomisticMultistateCheckpointPlan,
    result: AtomisticMultistateSegmentResult,
    /,
) -> AtomisticMultistateCheckpoint:
    if not isinstance(plan, AtomisticMultistateCheckpointPlan):
        raise TypeError("plan must be AtomisticMultistateCheckpointPlan.")
    if not isinstance(result, AtomisticMultistateSegmentResult):
        raise TypeError("result must be AtomisticMultistateSegmentResult.")
    _validate_segment(plan, result)
    arrays: dict[str, object] = {}
    specification = pack_array_tree("segment", result, arrays)
    segment_index = int(result.successor_state.segment_index) - 1
    expected_segment_id = canonical_fingerprint(
        {
            "kind": "atomistic-multistate-segment",
            "runtime": plan.runtime.prepared_id,
            "capacity": plan.segment_capacity,
            "start_iteration": int(result.start_watermark),
            "segment_index": segment_index,
            "predecessor": result.predecessor_id,
        }
    )
    if result.segment_id != expected_segment_id:
        raise ValueError("Segment identity does not authenticate its watermark.")
    payload_id = canonical_fingerprint(
        {
            "kind": "atomistic-multistate-checkpoint-payload",
            "checkpoint": plan.checkpoint_id,
            "segment": result.segment_id,
            "predecessor": result.predecessor_id,
            "start": int(result.start_watermark),
            "stop": int(result.stop_watermark),
            "count": int(result.count),
            "segment_index": segment_index,
            "result": specification,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    manifest = {
        "format": _MULTISTATE_CHECKPOINT_FORMAT,
        "kind": "atomistic-multistate-segment-and-checkpoint",
        "checkpoint_id": plan.checkpoint_id,
        "runtime_id": plan.runtime.prepared_id,
        "multistate_plan_id": plan.runtime.plan.plan_id,
        "dynamics_id": plan.runtime.dynamics.prepared_id,
        "thermodynamic_table_id": plan.runtime.thermodynamic.table_id,
        "control_layout_id": plan.runtime.thermodynamic.control_layout_id,
        "unit_system": plan.runtime.dynamics.system.plan.units.to_dict(),
        "state_ids": list(plan.runtime.thermodynamic.state_ids),
        "potential_ids": list(plan.runtime.thermodynamic.potential_ids),
        "bias_ids": list(plan.runtime.thermodynamic.bias_ids),
        "producer_id": plan.runtime.prepared_id,
        "unit_id": "1",
        "reduced_convention_id": plan.runtime.thermodynamic.reduced_convention_id,
        "qualification_id": plan.runtime.plan.qualification.qualification_id,
        "sampling_exact": plan.runtime.plan.qualification.sampling_exact,
        "sampling_bias_bound": plan.runtime.plan.qualification.sampling_bias_bound,
        "replica_ids": np.asarray(plan.runtime.plan.replica_ids).tolist(),
        "segment_capacity": plan.segment_capacity,
        "segment_index": segment_index,
        "segment_id": result.segment_id,
        "predecessor_id": result.predecessor_id,
        "start_watermark": int(result.start_watermark),
        "stop_watermark": int(result.stop_watermark),
        "count": int(result.count),
        "result": specification,
        "payload_id": payload_id,
        **({} if plan.scope_id is None else {"scope_id": plan.scope_id}),
    }
    write_array_archive(path, manifest=manifest, arrays=arrays)
    return AtomisticMultistateCheckpoint(
        result,
        plan.runtime.dynamics.system.plan.units,
        payload_id,
        plan.checkpoint_id,
        result.segment_id,
    )


def read_atomistic_multistate_checkpoint(
    path: str | Path,
    plan: AtomisticMultistateCheckpointPlan,
    template_state: AtomisticMultistateState,
    /,
) -> AtomisticMultistateCheckpoint:
    if not isinstance(plan, AtomisticMultistateCheckpointPlan):
        raise TypeError("plan must be AtomisticMultistateCheckpointPlan.")
    if not isinstance(template_state, AtomisticMultistateState):
        raise TypeError("template_state must be AtomisticMultistateState.")
    if template_state.plan_id != plan.runtime.prepared_id:
        raise ValueError("Checkpoint template belongs to another multistate runtime.")
    manifest, arrays = read_array_archive(path)
    expected_fields = {
        "format",
        "kind",
        "checkpoint_id",
        "runtime_id",
        "multistate_plan_id",
        "dynamics_id",
        "thermodynamic_table_id",
        "control_layout_id",
        "unit_system",
        "state_ids",
        "potential_ids",
        "bias_ids",
        "producer_id",
        "unit_id",
        "reduced_convention_id",
        "qualification_id",
        "sampling_exact",
        "sampling_bias_bound",
        "replica_ids",
        "segment_capacity",
        "segment_index",
        "segment_id",
        "predecessor_id",
        "start_watermark",
        "stop_watermark",
        "count",
        "result",
        "payload_id",
        "arrays",
    }
    if plan.scope_id is not None:
        expected_fields.add("scope_id")
    if set(manifest) != expected_fields:
        raise ValueError("Multistate checkpoint manifest is not canonical.")
    identities = {
        "format": _MULTISTATE_CHECKPOINT_FORMAT,
        "kind": "atomistic-multistate-segment-and-checkpoint",
        "checkpoint_id": plan.checkpoint_id,
        "runtime_id": plan.runtime.prepared_id,
        "multistate_plan_id": plan.runtime.plan.plan_id,
        "dynamics_id": plan.runtime.dynamics.prepared_id,
        "thermodynamic_table_id": plan.runtime.thermodynamic.table_id,
        "control_layout_id": plan.runtime.thermodynamic.control_layout_id,
        "segment_capacity": plan.segment_capacity,
        "state_ids": list(plan.runtime.thermodynamic.state_ids),
        "potential_ids": list(plan.runtime.thermodynamic.potential_ids),
        "bias_ids": list(plan.runtime.thermodynamic.bias_ids),
        "producer_id": plan.runtime.prepared_id,
        "unit_id": "1",
        "reduced_convention_id": plan.runtime.thermodynamic.reduced_convention_id,
        "qualification_id": plan.runtime.plan.qualification.qualification_id,
        "sampling_exact": plan.runtime.plan.qualification.sampling_exact,
        "sampling_bias_bound": plan.runtime.plan.qualification.sampling_bias_bound,
        "replica_ids": np.asarray(plan.runtime.plan.replica_ids).tolist(),
    }
    if plan.scope_id is not None:
        identities["scope_id"] = plan.scope_id
    if any(manifest[name] != value for name, value in identities.items()):
        raise ValueError("Multistate checkpoint identity does not match the runtime.")
    units = AtomisticUnitSystem.from_dict(manifest["unit_system"])
    if units.unit_system_id != plan.runtime.dynamics.system.plan.units.unit_system_id:
        raise ValueError("Multistate checkpoint unit descriptor is incompatible.")
    start = int(manifest["start_watermark"])
    stop = int(manifest["stop_watermark"])
    count = int(manifest["count"])
    segment_index = int(manifest["segment_index"])
    predecessor = str(manifest["predecessor_id"])
    segment_id = str(manifest["segment_id"])
    if (
        start < 0
        or stop < start
        or count != stop - start
        or count > plan.segment_capacity
        or segment_index < 0
        or not predecessor
    ):
        raise ValueError("Multistate checkpoint watermark is invalid.")
    expected_segment_id = canonical_fingerprint(
        {
            "kind": "atomistic-multistate-segment",
            "runtime": plan.runtime.prepared_id,
            "capacity": plan.segment_capacity,
            "start_iteration": start,
            "segment_index": segment_index,
            "predecessor": predecessor,
        }
    )
    if segment_id != expected_segment_id:
        raise ValueError("Multistate segment identity is corrupt.")
    state_template = _state_with_continuation(template_state, segment_id)
    result_template = _segment_template(
        plan,
        state_template,
        segment_id=segment_id,
        predecessor_id=predecessor,
    )
    result = unpack_array_tree(manifest["result"], arrays, result_template)
    if not isinstance(result, AtomisticMultistateSegmentResult):
        raise TypeError("Checkpoint did not reconstruct a multistate segment.")
    _validate_segment(plan, result)
    payload_id = canonical_fingerprint(
        {
            "kind": "atomistic-multistate-checkpoint-payload",
            "checkpoint": plan.checkpoint_id,
            "segment": segment_id,
            "predecessor": predecessor,
            "start": start,
            "stop": stop,
            "count": count,
            "segment_index": segment_index,
            "result": manifest["result"],
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    if manifest["payload_id"] != payload_id:
        raise ValueError("Multistate checkpoint payload identity is corrupt.")
    if (
        int(result.start_watermark) != start
        or int(result.stop_watermark) != stop
        or int(result.count) != count
        or int(result.successor_state.segment_index) != segment_index + 1
    ):
        raise ValueError("Multistate checkpoint payload and manifest watermarks differ.")
    return AtomisticMultistateCheckpoint(
        result,
        units,
        payload_id,
        plan.checkpoint_id,
        segment_id,
    )


__all__ = [
    "AtomisticMultistateCheckpoint",
    "AtomisticMultistateCheckpointPlan",
    "read_atomistic_multistate_checkpoint",
    "write_atomistic_multistate_checkpoint",
]
