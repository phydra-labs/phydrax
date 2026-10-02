#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Committed projector checkpoints and explicit resource-only state transports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from .._array_archive import array_payload_digest, ArrayArchiveLimits
from .._fingerprint import canonical_fingerprint, canonical_json
from .._identity import strict_module_payload
from ..lifecycle import create, LifecycleArchive, ResultManifest, ResultRevision
from ..typing import validate
from ..units import ONE
from ._projector_monte_carlo import validate_projector_state
from ._projector_monte_carlo_contracts import (
    PreparedProjectorMonteCarlo,
    ProjectorMonteCarloHistory,
    ProjectorMonteCarloResult,
    ProjectorMonteCarloState,
)
from ._projector_monte_carlo_estimators import ProjectorMonteCarloAnalysis
from ._runtime_lifecycle import (
    _default_runtime_id,
    _runtime_checkpoint_manifest,
    _unpack_state_tree,
    read_runtime_checkpoint,
    restore_runtime_checkpoint_arrays,
    RuntimeCheckpointEncodingPlan,
    RuntimeCheckpointEnvelope,
    RuntimeRestartRelation,
    write_runtime_checkpoint,
)


_PRECISION_ID = "projector-complex128-float64"


def _history_payload(history: ProjectorMonteCarloHistory, /) -> dict[str, Array]:
    return {
        "history.applied_shifts": history.applied_shifts,
        "history.populations": history.populations,
        "history.projected_numerator": history.projected_numerator,
        "history.projected_denominator": history.projected_denominator,
        "history.pair_numerators": history.pair_numerators,
        "history.pair_denominators": history.pair_denominators,
        "history.pre_annihilation_norm": history.pre_annihilation_norm,
        "history.post_annihilation_norm": history.post_annihilation_norm,
        "history.valid": history.valid,
        "history.count": history.count,
    }


def _state_payload(state: ProjectorMonteCarloState, /) -> dict[str, Array]:
    # History is in the transported state, not a separate observer tree whose
    # fixed shape would prevent the native restorer from enlarging its capacity.
    return {
        "support_keys": state.support_keys,
        "coefficients": state.coefficients,
        "active": state.active,
        "shifts": state.shifts,
        "populations": state.populations,
        "step": state.step,
        "root_key_data": jr.key_data(state.root_key),
        **_history_payload(state.history),
    }


def _checkpoint_ids(
    prepared: PreparedProjectorMonteCarlo,
    state: ProjectorMonteCarloState,
    /,
) -> tuple[str, str]:
    storage = canonical_fingerprint(
        {
            "kind": "projector-storage-binding",
            "domain": state.domain_id,
            "codec": state.codec_id,
            "resources": prepared.plan.plan_id,
        }
    )
    method = canonical_fingerprint(
        {
            "kind": "projector-checkpoint-method",
            "scientific": prepared.scientific_id,
            "random_implementation": str(jr.key_impl(state.root_key)),
        }
    )
    return storage, method


def _checkpoint_envelope(
    prepared: PreparedProjectorMonteCarlo,
    state: ProjectorMonteCarloState,
    /,
) -> RuntimeCheckpointEnvelope:
    validate_projector_state(prepared, state)
    storage, method = _checkpoint_ids(prepared, state)
    payload = _state_payload(state)
    payload_bytes = sum(value.nbytes for value in payload.values())
    if payload_bytes > prepared.plan.maximum_retained_bytes:
        raise ValueError("Projector checkpoint exceeds the admitted retained-byte limit.")
    return RuntimeCheckpointEnvelope(
        payload,
        time=state.step.astype(jnp.float64) * prepared.problem.dt,
        step_index=state.step,
        schedule_cursor=state.history.count,
        mesh_id=storage,
        method_id=method,
        precision_id=_PRECISION_ID,
        topology_epoch_id=state.domain_id,
    )


def _validated_payload(
    value: object,
    template: Mapping[str, Array],
    /,
) -> dict[str, Array]:
    if not isinstance(value, Mapping) or set(value) != set(template):
        raise ValueError(
            "Restored projector payload fields do not match the exact template."
        )
    payload: dict[str, Array] = {}
    for name, expected in template.items():
        array = value[name]
        if not isinstance(array, (Array, np.ndarray)):
            raise TypeError(
                "Restored projector payloads must contain native numerical arrays."
            )
        if array.shape != expected.shape or array.dtype != expected.dtype:
            raise ValueError(f"Restored projector array {name!r} changed shape or dtype.")
        payload[name] = jnp.asarray(array)
    return payload


def _restore_state(
    payload: Mapping[str, Array],
    template: ProjectorMonteCarloState,
    /,
) -> ProjectorMonteCarloState:
    history_template = template.history
    history = ProjectorMonteCarloHistory(
        applied_shifts=payload["history.applied_shifts"],
        populations=payload["history.populations"],
        projected_numerator=payload["history.projected_numerator"],
        projected_denominator=payload["history.projected_denominator"],
        pair_numerators=payload["history.pair_numerators"],
        pair_denominators=payload["history.pair_denominators"],
        pre_annihilation_norm=payload["history.pre_annihilation_norm"],
        post_annihilation_norm=payload["history.post_annihilation_norm"],
        valid=payload["history.valid"],
        count=payload["history.count"],
        scientific_id=history_template.scientific_id,
        domain_id=history_template.domain_id,
        operator_id=history_template.operator_id,
        guide_id=history_template.guide_id,
        metric_id=history_template.metric_id,
    )
    state = ProjectorMonteCarloState(
        support_keys=payload["support_keys"],
        coefficients=payload["coefficients"],
        active=payload["active"],
        shifts=payload["shifts"],
        populations=payload["populations"],
        step=payload["step"],
        root_key=jr.wrap_key_data(
            payload["root_key_data"], impl=str(jr.key_impl(template.root_key))
        ),
        history=history,
        scientific_id=template.scientific_id,
        prepared_id=template.prepared_id,
        plan_id=template.plan_id,
        domain_id=template.domain_id,
        codec_id=template.codec_id,
        operator_id=template.operator_id,
        guide_id=template.guide_id,
        metric_id=template.metric_id,
    )
    validate(state)
    return state


def write_projector_monte_carlo_checkpoint(
    path: str | PathLike[str],
    prepared: PreparedProjectorMonteCarlo,
    state: ProjectorMonteCarloState,
    /,
) -> Path:
    """Write only a complete committed state through the native checkpoint owner."""
    return write_runtime_checkpoint(path, _checkpoint_envelope(prepared, state))


def read_projector_monte_carlo_checkpoint(
    path: str | PathLike[str],
    prepared: PreparedProjectorMonteCarlo,
    template: ProjectorMonteCarloState,
    /,
) -> ProjectorMonteCarloState:
    """Restore matching scientific/storage bindings and the recorded typed key.

    The template fixes structure and key implementation, not the archived key
    value. A different root stream in an otherwise matching template is not a
    request to replace the persisted random stream.
    """
    validate_projector_state(prepared, template)
    storage, method = _checkpoint_ids(prepared, template)
    payload_template = _state_payload(template)
    envelope = read_runtime_checkpoint(
        path,
        state_template=payload_template,
        mesh_id=storage,
        method_id=method,
        precision_id=_PRECISION_ID,
        topology_epoch_id=template.domain_id,
    )
    payload = _validated_payload(envelope.state, payload_template)
    state = _restore_state(payload, template)
    if not bool(jnp.array_equal(state.step, envelope.step_index)) or not bool(
        jnp.array_equal(state.history.count, envelope.schedule_cursor)
    ):
        raise ValueError(
            "Projector checkpoint cursors disagree with the native envelope."
        )
    expected_time = state.step.astype(jnp.float64) * prepared.problem.dt
    if not bool(jnp.array_equal(expected_time, envelope.time)):
        raise ValueError("Projector checkpoint time does not match its logical step.")
    validate_projector_state(prepared, state)
    return state


def _resource_compatibility(
    source: PreparedProjectorMonteCarlo,
    target: PreparedProjectorMonteCarlo,
    /,
) -> None:
    if source.scientific_id != target.scientific_id:
        raise ValueError("Resource replay cannot change projector scientific bindings.")
    old, new = source.plan, target.plan
    if old.replicas != new.replicas or old.policy_id != new.policy_id:
        raise ValueError("Resource replay cannot change replicas or numerical policies.")
    limits = (
        (old.support_capacity, new.support_capacity),
        (old.group_capacity, new.group_capacity),
        (old.event_capacity, new.event_capacity),
        (old.attempt_capacity, new.attempt_capacity),
        (old.source_capacity, new.source_capacity),
        (old.history_capacity, new.history_capacity),
        (old.maximum_retained_bytes, new.maximum_retained_bytes),
        (old.maximum_workspace_bytes, new.maximum_workspace_bytes),
    )
    if any(current < previous for previous, current in limits):
        raise ValueError("Resource replay admits only monotone capacity enlargement.")
    if old.plan_id == new.plan_id:
        raise ValueError("Unchanged resources use ordinary continuation, not transport.")


def _destination_template(
    payload: Mapping[str, Array],
    source: PreparedProjectorMonteCarlo,
    target: PreparedProjectorMonteCarlo,
    /,
) -> dict[str, Array]:
    template: dict[str, Array] = {}
    for name, value in payload.items():
        shape = list(value.shape)
        if name in ("support_keys", "coefficients", "active"):
            shape[1] = target.plan.support_capacity
        elif name == "history.valid":
            shape[0] = target.plan.history_capacity
        elif name.startswith("history.") and name != "history.count":
            shape[1] = target.plan.history_capacity
        if any(new < old for new, old in zip(shape, value.shape, strict=True)):
            raise ValueError("Resource transport would truncate a state or history axis.")
        template[name] = jnp.zeros(tuple(shape), dtype=value.dtype)
    peak_bytes = sum(value.nbytes for value in payload.values()) + sum(
        value.nbytes for value in template.values()
    )
    if peak_bytes > target.plan.maximum_retained_bytes:
        raise ValueError("Resource transport exceeds the admitted snapshot/state bytes.")
    if source.plan.replicas != target.plan.replicas:
        raise ValueError("Resource transport cannot alter replica lineage.")
    return template


def _resource_restorer(
    source_template: Mapping[str, Array],
    /,
) -> Callable[
    [
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Array],
        RuntimeCheckpointEncodingPlan,
    ],
    dict[str, Array],
]:
    # The callback is the owning native archive interchange boundary. No
    # numerical/provider type is widened to Any outside that boundary.
    def restore(
        source_arrays: Mapping[str, Any],
        specification: Mapping[str, Any],
        destination_template: Mapping[str, Array],
        encoding: RuntimeCheckpointEncodingPlan,
        /,
    ) -> dict[str, Array]:
        unpacked = _unpack_state_tree(
            specification, source_arrays, source_template, encoding
        )
        payload = _validated_payload(unpacked, source_template)
        restored: dict[str, Array] = {}
        if set(payload) != set(destination_template):
            raise ValueError("Resource replay changed projector payload fields.")
        for name, value in payload.items():
            expected = destination_template[name]
            if value.dtype != expected.dtype or value.ndim != expected.ndim:
                raise ValueError("Resource replay changed a payload dtype or rank.")
            widths = tuple(
                (0, new - old)
                for old, new in zip(value.shape, expected.shape, strict=True)
            )
            if any(width[1] < 0 for width in widths):
                raise ValueError("Resource replay cannot discard state/history values.")
            restored[name] = (
                value if value.shape == expected.shape else jnp.pad(value, widths)
            )
        return restored

    return restore


def transport_projector_monte_carlo_resources(
    source: PreparedProjectorMonteCarlo,
    target: PreparedProjectorMonteCarlo,
    state: ProjectorMonteCarloState,
    /,
) -> tuple[ProjectorMonteCarloState, RuntimeRestartRelation]:
    """Enlarge storage through an explicit native restart relation, without draws."""
    validate_projector_state(source, state)
    _resource_compatibility(source, target)
    source_envelope = _checkpoint_envelope(source, state)
    source_payload = _state_payload(state)
    destination = _destination_template(source_payload, source, target)
    source_storage, source_method = _checkpoint_ids(source, state)
    target_storage = canonical_fingerprint(
        {
            "kind": "projector-storage-binding",
            "domain": state.domain_id,
            "codec": state.codec_id,
            "resources": target.plan.plan_id,
        }
    )
    relation = RuntimeRestartRelation(
        source_storage,
        target_storage,
        classification="bitwise",
        relation_id=canonical_fingerprint(
            {
                "kind": "projector-resource-only-restart",
                "source": source.plan.plan_id,
                "target": target.plan.plan_id,
                "scientific": source.scientific_id,
            }
        ),
        restorer=_resource_restorer(source_payload),
    )
    envelope, _ = restore_runtime_checkpoint_arrays(
        _runtime_checkpoint_manifest(source_envelope),
        source_envelope.archive_arrays,
        state_template=destination,
        target_mesh_id=target_storage,
        target_method_id=source_method,
        target_precision_id=_PRECISION_ID,
        target_topology_epoch_id=state.domain_id,
        target_runtime_id=_default_runtime_id(
            target_storage, source_method, _PRECISION_ID, state.domain_id, None
        ),
        restart_relation=relation,
    )
    payload = _validated_payload(envelope.state, destination)
    rebound_template = ProjectorMonteCarloState(
        support_keys=destination["support_keys"],
        coefficients=destination["coefficients"],
        active=destination["active"],
        shifts=destination["shifts"],
        populations=destination["populations"],
        step=destination["step"],
        root_key=state.root_key,
        history=state.history,
        scientific_id=target.scientific_id,
        prepared_id=target.prepared_id,
        plan_id=target.plan.plan_id,
        domain_id=state.domain_id,
        codec_id=state.codec_id,
        operator_id=state.operator_id,
        guide_id=state.guide_id,
        metric_id=state.metric_id,
    )
    restored = _restore_state(payload, rebound_template)
    validate_projector_state(target, restored)
    return restored, relation


def _analysis_payload(
    prepared: PreparedProjectorMonteCarlo,
    state: ProjectorMonteCarloState,
    analysis: ProjectorMonteCarloAnalysis,
    /,
) -> dict[str, Array]:
    binding = analysis.systematic
    identities = (
        (binding.scientific_id, state.scientific_id),
        (binding.domain_id, state.domain_id),
        (binding.operator_id, state.operator_id),
        (binding.guide_id, state.guide_id),
        (binding.metric_id, state.metric_id),
    )
    if any(actual != expected for actual, expected in identities):
        raise ValueError("Archived analysis does not match the physical projector state.")
    if analysis.observable_units != prepared.observable_units:
        raise ValueError("Archived analysis observable units changed.")
    payload: dict[str, Array] = {}
    for path, value in jax.tree_util.tree_flatten_with_path(analysis)[0]:
        if not isinstance(value, Array):
            raise TypeError("Projector analysis dynamic payloads must be JAX arrays.")
        payload[f"analysis/{jax.tree_util.keystr(path)}"] = value
    return payload


def write_projector_monte_carlo_result(
    path: str | PathLike[str],
    prepared: PreparedProjectorMonteCarlo,
    result: ProjectorMonteCarloResult,
    /,
    *,
    run_id: str,
    analysis: ProjectorMonteCarloAnalysis | None = None,
) -> LifecycleArchive:
    """Persist raw records and optional statistics using canonical lifecycle data."""
    validate_projector_state(prepared, result.state)
    payload = {
        **_state_payload(result.state),
        "propagation_status": result.status,
        "requested_steps": result.requested_steps,
        "accepted_steps": result.accepted_steps,
    }
    interpretation: dict[str, object] = {}
    if analysis is not None:
        payload.update(_analysis_payload(prepared, result.state, analysis))
        interpretation = {
            "identity": strict_module_payload(analysis),
            "history_depths": tuple(item.history_depth for item in analysis.reweighted),
            "history_horizons": tuple(
                item.history_horizon for item in analysis.reweighted
            ),
            "observable_ids": analysis.observable_ids,
        }
    retained_bytes = sum(value.nbytes for value in payload.values())
    if retained_bytes > prepared.plan.maximum_retained_bytes:
        raise ValueError("Projector result archive exceeds admitted retained bytes.")
    digests = {
        name: array_payload_digest(value) for name, value in sorted(payload.items())
    }
    result_id = canonical_fingerprint(
        {
            "kind": "native-projector-result",
            "scientific": prepared.scientific_id,
            "prepared": prepared.prepared_id,
            "payloads": tuple(digests.items()),
            "analysis_interpretation": interpretation,
        }
    )
    fields = (
        ("represented-coefficients", "coefficients", ONE.unit_id),
        ("represented-population", "populations", ONE.unit_id),
        ("applied-shift", "history.applied_shifts", prepared.problem.energy_unit.unit_id),
        (
            "projected-numerator",
            "history.projected_numerator",
            prepared.problem.energy_unit.unit_id,
        ),
        ("projected-denominator", "history.projected_denominator", ONE.unit_id),
        ("replica-denominator", "history.pair_denominators", ONE.unit_id),
    )
    semantics = {
        "scientific_id": prepared.scientific_id,
        "domain_id": result.state.domain_id,
        "operator_id": result.state.operator_id,
        "guide_id": result.state.guide_id,
        "metric_id": result.state.metric_id,
        "key_implementation": str(jr.key_impl(result.state.root_key)),
        "ordered_replica_pairs": canonical_json(prepared.pair_ids),
        "observable_unit_ids": canonical_json(
            tuple(unit.unit_id for unit in prepared.observable_units)
        ),
        "completion": "operational-not-scientific-release",
        "joint_covariance_units": "numerator-quantity-and-dimensionless-denominator;full-joint-matrix",
    }
    if analysis is not None:
        semantics["analysis_interpretation"] = canonical_json(interpretation)
        semantics["systematic_assumptions"] = canonical_json(
            analysis.systematic.assumptions
        )
        semantics["finite_history_claim"] = analysis.systematic.finite_history_claim
        semantics["asymptotic_claim"] = analysis.systematic.asymptotic_claim
    manifest = ResultManifest(
        result_id,
        run_id,
        fields,
        digests,
        evidence_ids=(prepared.scientific_id,),
        sampled_semantics=semantics,
    )
    budget = prepared.plan.maximum_retained_bytes
    limits = ArrayArchiveLimits(
        max_container_bytes=budget,
        max_aggregate_bytes=budget,
        max_member_bytes=budget,
        max_manifest_bytes=min(1_048_576, budget),
        max_members=len(payload) + 1,
        max_central_directory_bytes=min(1_048_576, budget),
        max_axis_length=budget,
        max_array_elements=budget,
        max_total_array_elements=budget,
    )
    return create(
        Path(path), manifest=ResultRevision(manifest), arrays=payload, limits=limits
    )


__all__ = [
    "read_projector_monte_carlo_checkpoint",
    "transport_projector_monte_carlo_resources",
    "write_projector_monte_carlo_checkpoint",
    "write_projector_monte_carlo_result",
]
