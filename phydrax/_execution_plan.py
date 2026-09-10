#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Logical decomposition, placement, provider binding, and plan resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ._execution_resources import (
    DeterminismScope,
    DistributionMode,
    ExecutionGroupSpec,
    ExecutionPolicy,
    RecoveryPolicy,
    ResourceInventory,
)
from ._fingerprint import canonical_fingerprint


def _identifier(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _identifiers(values: Sequence[str], name: str) -> tuple[str, ...]:
    normalized = tuple(_identifier(value, name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} values must be unique")
    return normalized


class LogicalAxisKind(str, Enum):
    """Scientific coupling class of a logical execution axis."""

    INDEPENDENT = "independent"
    ADDITIVE = "additive"
    SPATIAL = "spatial"
    MODEL = "model"
    POPULATION = "population"
    SEQUENCE = "sequence"


class PlacementKind(str, Enum):
    """Ownership state of a value under one execution plan."""

    REPLICATED = "replicated"
    PARTITIONED = "partitioned"
    PARTIAL = "partial"
    RANK_LOCAL = "rank_local"


@dataclass(frozen=True, slots=True)
class LogicalAxis:
    name: str
    size: int
    kind: LogicalAxisKind
    minimum_local_size: int = 1
    allow_padding: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "logical axis"))
        object.__setattr__(self, "kind", LogicalAxisKind(self.kind))
        if self.size <= 0:
            raise ValueError("logical axis size must be positive")
        if self.minimum_local_size <= 0:
            raise ValueError("minimum_local_size must be positive")
        if self.minimum_local_size > self.size:
            raise ValueError("minimum_local_size cannot exceed axis size")

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "size": self.size,
            "kind": self.kind.value,
            "minimum_local_size": self.minimum_local_size,
            "allow_padding": self.allow_padding,
        }


@dataclass(frozen=True, slots=True)
class AxisBinding:
    logical_axis: str
    mesh_axes: tuple[str, ...]

    def __init__(self, logical_axis: str, mesh_axes: Sequence[str]) -> None:
        object.__setattr__(
            self, "logical_axis", _identifier(logical_axis, "logical_axis")
        )
        object.__setattr__(self, "mesh_axes", _identifiers(mesh_axes, "mesh axis"))

    def to_payload(self) -> dict[str, object]:
        return {
            "logical_axis": self.logical_axis,
            "mesh_axes": list(self.mesh_axes),
        }


@dataclass(frozen=True, slots=True)
class ValuePlacement:
    """Serializable semantic placement of one input, state, or output value."""

    name: str
    kind: PlacementKind
    logical_axes: tuple[str, ...] = ()
    mesh_axes: tuple[str | None, ...] = ()
    global_shape: tuple[int, ...] | None = None
    dtype: str | None = None
    authoritative_process: int | None = None

    def __init__(
        self,
        name: str,
        kind: PlacementKind,
        *,
        logical_axes: Sequence[str] = (),
        mesh_axes: Sequence[str | None] = (),
        global_shape: Sequence[int] | None = None,
        dtype: str | None = None,
        authoritative_process: int | None = None,
    ) -> None:
        logical = _identifiers(logical_axes, "logical axis")
        mesh = tuple(
            None if axis is None else _identifier(axis, "mesh axis") for axis in mesh_axes
        )
        shape = (
            None if global_shape is None else tuple(int(size) for size in global_shape)
        )
        if shape is not None and any(size < 0 for size in shape):
            raise ValueError("global_shape dimensions must be non-negative")
        if shape is not None and mesh and len(shape) != len(mesh):
            raise ValueError("mesh_axes must have one entry per global dimension")
        if authoritative_process is not None and authoritative_process < 0:
            raise ValueError("authoritative_process must be non-negative")
        object.__setattr__(self, "name", _identifier(name, "value placement name"))
        object.__setattr__(self, "kind", PlacementKind(kind))
        object.__setattr__(self, "logical_axes", logical)
        object.__setattr__(self, "mesh_axes", mesh)
        object.__setattr__(self, "global_shape", shape)
        object.__setattr__(
            self,
            "dtype",
            None if dtype is None else _identifier(dtype, "dtype"),
        )
        object.__setattr__(self, "authoritative_process", authoritative_process)

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "logical_axes": list(self.logical_axes),
            "mesh_axes": list(self.mesh_axes),
            "global_shape": None
            if self.global_shape is None
            else list(self.global_shape),
            "dtype": self.dtype,
            "authoritative_process": self.authoritative_process,
        }


@dataclass(frozen=True, slots=True)
class ProviderBinding:
    role: str
    provider_id: str
    capability_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _identifier(self.role, "provider role"))
        object.__setattr__(
            self, "provider_id", _identifier(self.provider_id, "provider_id")
        )
        object.__setattr__(
            self,
            "capability_id",
            _identifier(self.capability_id, "capability_id"),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "role": self.role,
            "provider_id": self.provider_id,
            "capability_id": self.capability_id,
        }


@dataclass(frozen=True, slots=True)
class ExecutionRequirements:
    """Scientific and transformation requirements declared by one owner."""

    owner_id: str
    logical_axes: tuple[LogicalAxis, ...] = ()
    operations: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()
    transformations: tuple[str, ...] = ()
    requires_process_local_input: bool = False
    allows_distributed_output: bool = True

    def __init__(
        self,
        owner_id: str,
        *,
        logical_axes: Sequence[LogicalAxis] = (),
        operations: Sequence[str] = (),
        dtypes: Sequence[str] = (),
        transformations: Sequence[str] = (),
        requires_process_local_input: bool = False,
        allows_distributed_output: bool = True,
    ) -> None:
        axes = tuple(logical_axes)
        if len({axis.name for axis in axes}) != len(axes):
            raise ValueError("logical axis names must be unique")
        object.__setattr__(self, "owner_id", _identifier(owner_id, "owner_id"))
        object.__setattr__(self, "logical_axes", axes)
        object.__setattr__(self, "operations", _identifiers(operations, "operation"))
        object.__setattr__(self, "dtypes", _identifiers(dtypes, "dtype"))
        object.__setattr__(
            self,
            "transformations",
            _identifiers(transformations, "transformation"),
        )
        object.__setattr__(
            self,
            "requires_process_local_input",
            bool(requires_process_local_input),
        )
        object.__setattr__(
            self,
            "allows_distributed_output",
            bool(allows_distributed_output),
        )

    @property
    def requirements_id(self) -> str:
        return canonical_fingerprint(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "owner_id": self.owner_id,
            "logical_axes": [axis.to_payload() for axis in self.logical_axes],
            "operations": list(self.operations),
            "dtypes": list(self.dtypes),
            "transformations": list(self.transformations),
            "requires_process_local_input": self.requires_process_local_input,
            "allows_distributed_output": self.allows_distributed_output,
        }


@dataclass(frozen=True, slots=True)
class ExecutionCandidate:
    """One complete owner-approved decomposition and provider strategy."""

    name: str
    requirements_id: str
    group: ExecutionGroupSpec
    axis_bindings: tuple[AxisBinding, ...] = ()
    value_placements: tuple[ValuePlacement, ...] = ()
    providers: tuple[ProviderBinding, ...] = ()
    priority: int = 0
    estimated_memory_bytes: int = 0
    estimated_communication_bytes: int = 0
    rejection_reasons: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        requirements_id: str,
        group: ExecutionGroupSpec,
        *,
        axis_bindings: Sequence[AxisBinding] = (),
        value_placements: Sequence[ValuePlacement] = (),
        providers: Sequence[ProviderBinding] = (),
        priority: int = 0,
        estimated_memory_bytes: int = 0,
        estimated_communication_bytes: int = 0,
        rejection_reasons: Sequence[str] = (),
    ) -> None:
        bindings = tuple(axis_bindings)
        placements = tuple(value_placements)
        providers_ = tuple(providers)
        if len({binding.logical_axis for binding in bindings}) != len(bindings):
            raise ValueError("logical axes may be bound only once")
        if len({placement.name for placement in placements}) != len(placements):
            raise ValueError("value placement names must be unique")
        if len({provider.role for provider in providers_}) != len(providers_):
            raise ValueError("provider roles must be unique")
        if estimated_memory_bytes < 0 or estimated_communication_bytes < 0:
            raise ValueError("execution estimates must be non-negative")
        object.__setattr__(self, "name", _identifier(name, "candidate name"))
        object.__setattr__(
            self,
            "requirements_id",
            _identifier(requirements_id, "requirements_id"),
        )
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "axis_bindings", bindings)
        object.__setattr__(self, "value_placements", placements)
        object.__setattr__(self, "providers", providers_)
        object.__setattr__(self, "priority", int(priority))
        object.__setattr__(self, "estimated_memory_bytes", estimated_memory_bytes)
        object.__setattr__(
            self,
            "estimated_communication_bytes",
            estimated_communication_bytes,
        )
        object.__setattr__(
            self,
            "rejection_reasons",
            tuple(
                str(reason).strip() for reason in rejection_reasons if str(reason).strip()
            ),
        )

    @property
    def candidate_id(self) -> str:
        return canonical_fingerprint(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "requirements_id": self.requirements_id,
            "group": self.group.to_payload(),
            "axis_bindings": [binding.to_payload() for binding in self.axis_bindings],
            "value_placements": [
                placement.to_payload() for placement in self.value_placements
            ],
            "providers": [provider.to_payload() for provider in self.providers],
            "priority": self.priority,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "estimated_communication_bytes": self.estimated_communication_bytes,
        }


def _group_from_payload(value: Mapping[str, Any]) -> ExecutionGroupSpec:
    return ExecutionGroupSpec(
        value["group_id"],
        value["process_indices"],
        tuple(tuple(key) for key in value["device_keys"]),
        mesh_axes=tuple(tuple(axis) for axis in value.get("mesh_axes", ())),
        parent_group_id=value.get("parent_group_id"),
    )


def _axis_binding_from_payload(value: Mapping[str, Any]) -> AxisBinding:
    return AxisBinding(value["logical_axis"], value["mesh_axes"])


def _value_placement_from_payload(value: Mapping[str, Any]) -> ValuePlacement:
    return ValuePlacement(
        value["name"],
        PlacementKind(value["kind"]),
        logical_axes=value.get("logical_axes", ()),
        mesh_axes=value.get("mesh_axes", ()),
        global_shape=value.get("global_shape"),
        dtype=value.get("dtype"),
        authoritative_process=value.get("authoritative_process"),
    )


def _provider_binding_from_payload(value: Mapping[str, Any]) -> ProviderBinding:
    return ProviderBinding(
        value["role"],
        value["provider_id"],
        value["capability_id"],
    )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Immutable backend, placement, numerical-policy, and allocation identity."""

    execution_plan_id: str
    backend: str
    precision_policy_id: str
    solver_policy_id: str
    device_mesh_id: str | None = None
    reduction_policy_id: str | None = None
    cache_key: str | None = None
    policy_id: str | None = None
    requirements_id: str | None = None
    inventory_id: str | None = None
    group: ExecutionGroupSpec | None = None
    axis_bindings: tuple[AxisBinding, ...] = ()
    value_placements: tuple[ValuePlacement, ...] = ()
    providers: tuple[ProviderBinding, ...] = ()
    determinism: DeterminismScope = DeterminismScope.LOGICAL
    recovery: RecoveryPolicy = RecoveryPolicy.FAIL_FAST
    topology_epoch: int = 0
    decision_evidence: tuple[str, ...] = ()
    plan_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        identifiers = (
            ("execution_plan_id", self.execution_plan_id),
            ("backend", self.backend),
            ("precision_policy_id", self.precision_policy_id),
            ("solver_policy_id", self.solver_policy_id),
        )
        for name, value in identifiers:
            object.__setattr__(self, name, _identifier(value, name))
        for name in (
            "device_mesh_id",
            "reduction_policy_id",
            "cache_key",
            "policy_id",
            "requirements_id",
            "inventory_id",
        ):
            value = object.__getattribute__(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name))
        if self.topology_epoch < 0:
            raise ValueError("topology_epoch must be non-negative")
        payload = self.to_payload(include_fingerprint=False)
        object.__setattr__(self, "plan_fingerprint", canonical_fingerprint(payload))

    def to_payload(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": "execution-plan",
            "execution_plan_id": self.execution_plan_id,
            "backend": self.backend,
            "precision_policy_id": self.precision_policy_id,
            "solver_policy_id": self.solver_policy_id,
            "device_mesh_id": self.device_mesh_id,
            "reduction_policy_id": self.reduction_policy_id,
            "cache_key": self.cache_key,
        }
        extended = (
            self.policy_id is not None
            or self.requirements_id is not None
            or self.inventory_id is not None
            or self.group is not None
            or bool(self.axis_bindings)
            or bool(self.value_placements)
            or bool(self.providers)
            or self.determinism is not DeterminismScope.LOGICAL
            or self.recovery is not RecoveryPolicy.FAIL_FAST
            or self.topology_epoch != 0
            or bool(self.decision_evidence)
        )
        if extended:
            payload.update(
                {
                    "policy_id": self.policy_id,
                    "requirements_id": self.requirements_id,
                    "inventory_id": self.inventory_id,
                    "group": None if self.group is None else self.group.to_payload(),
                    "axis_bindings": [
                        binding.to_payload() for binding in self.axis_bindings
                    ],
                    "value_placements": [
                        placement.to_payload() for placement in self.value_placements
                    ],
                    "providers": [provider.to_payload() for provider in self.providers],
                    "determinism": self.determinism.value,
                    "recovery": self.recovery.value,
                    "topology_epoch": self.topology_epoch,
                    "decision_evidence": list(self.decision_evidence),
                }
            )
        if include_fingerprint:
            payload["plan_fingerprint"] = self.plan_fingerprint
        return payload

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> ExecutionPlan:
        group_payload = value.get("group")
        group = None if group_payload is None else _group_from_payload(group_payload)
        return cls(
            value["execution_plan_id"],
            value["backend"],
            value["precision_policy_id"],
            value["solver_policy_id"],
            device_mesh_id=value.get("device_mesh_id"),
            reduction_policy_id=value.get("reduction_policy_id"),
            cache_key=value.get("cache_key"),
            policy_id=value.get("policy_id"),
            requirements_id=value.get("requirements_id"),
            inventory_id=value.get("inventory_id"),
            group=group,
            axis_bindings=tuple(
                _axis_binding_from_payload(item)
                for item in value.get("axis_bindings", ())
            ),
            value_placements=tuple(
                _value_placement_from_payload(item)
                for item in value.get("value_placements", ())
            ),
            providers=tuple(
                _provider_binding_from_payload(item)
                for item in value.get("providers", ())
            ),
            determinism=DeterminismScope(
                value.get("determinism", DeterminismScope.LOGICAL.value)
            ),
            recovery=RecoveryPolicy(
                value.get("recovery", RecoveryPolicy.FAIL_FAST.value)
            ),
            topology_epoch=int(value.get("topology_epoch", 0)),
            decision_evidence=tuple(value.get("decision_evidence", ())),
        )


class ExecutionAdmissionError(RuntimeError):
    """No owner-approved execution candidate satisfies policy and resources."""


def resolve_execution_plan(
    policy: ExecutionPolicy,
    inventory: ResourceInventory,
    candidates: Sequence[ExecutionCandidate],
    *,
    precision_policy_id: str,
    solver_policy_id: str,
    reduction_policy_id: str | None = None,
) -> ExecutionPlan:
    """Choose one complete candidate before execution begins."""

    candidates_ = tuple(candidates)
    if not candidates_:
        raise ExecutionAdmissionError("the execution owner supplied no candidates")

    inventory_keys = {device.key for device in inventory.devices}
    eligible: list[ExecutionCandidate] = []
    rejected: list[str] = []
    for candidate in candidates_:
        reasons = list(candidate.rejection_reasons)
        if not set(candidate.group.device_keys).issubset(inventory_keys):
            reasons.append("candidate group references unavailable devices")
        if any(
            process >= inventory.process_count
            for process in candidate.group.process_indices
        ):
            reasons.append("candidate group references unavailable processes")
        if policy.distribution is DistributionMode.SINGLE and (
            candidate.group.device_count != 1 or len(candidate.group.process_indices) != 1
        ):
            reasons.append("single execution policy requires one process and device")
        if policy.distribution is DistributionMode.DISTRIBUTED and (
            candidate.group.device_count == 1
            and len(candidate.group.process_indices) == 1
        ):
            reasons.append("distributed policy requires a multi-resource candidate")
        provider_ids = {provider.provider_id for provider in candidate.providers}
        if policy.strict_providers and not provider_ids.intersection(
            policy.provider_preferences
        ):
            reasons.append("candidate does not use a required provider")
        request = policy.resources
        if request is not None:
            if len(candidate.group.process_indices) < request.process_count:
                reasons.append("candidate has fewer processes than requested")
            candidate_platforms = {
                device.platform
                for device in inventory.devices
                if device.key in candidate.group.device_keys
            }
            if (
                request.accelerator_platform is not None
                and request.accelerator_platform not in candidate_platforms
            ):
                reasons.append("candidate lacks the requested accelerator platform")
            accelerator_count = sum(
                device.platform != "cpu"
                for device in inventory.devices
                if device.key in candidate.group.device_keys
            )
            if accelerator_count < request.gpu_count:
                reasons.append("candidate has fewer accelerators than requested")
            if candidate.estimated_memory_bytes > request.memory_bytes:
                reasons.append("candidate memory estimate exceeds the request")
        if reasons:
            rejected.append(f"{candidate.name}: {'; '.join(reasons)}")
        else:
            eligible.append(candidate)

    if not eligible:
        details = " | ".join(rejected)
        raise ExecutionAdmissionError(f"no execution candidate admitted: {details}")

    def candidate_score(candidate: ExecutionCandidate) -> tuple[int, int, int, int]:
        provider_ids = tuple(provider.provider_id for provider in candidate.providers)
        preference = len(policy.provider_preferences)
        for index, provider in enumerate(policy.provider_preferences):
            if provider in provider_ids:
                preference = index
                break
        return (
            preference,
            candidate.priority,
            candidate.estimated_communication_bytes,
            candidate.estimated_memory_bytes,
        )

    selected = min(eligible, key=candidate_score)
    providers = selected.providers
    backend = providers[0].provider_id if providers else "jax-native"
    return ExecutionPlan(
        selected.candidate_id,
        backend,
        _identifier(precision_policy_id, "precision_policy_id"),
        _identifier(solver_policy_id, "solver_policy_id"),
        device_mesh_id=selected.group.group_id,
        reduction_policy_id=reduction_policy_id,
        policy_id=policy.policy_id,
        requirements_id=selected.requirements_id,
        inventory_id=inventory.inventory_id,
        group=selected.group,
        axis_bindings=selected.axis_bindings,
        value_placements=selected.value_placements,
        providers=providers,
        determinism=policy.determinism,
        recovery=policy.recovery,
        decision_evidence=(f"selected {selected.name}", *rejected),
    )


__all__ = (
    "AxisBinding",
    "ExecutionAdmissionError",
    "ExecutionCandidate",
    "ExecutionPlan",
    "ExecutionRequirements",
    "LogicalAxis",
    "LogicalAxisKind",
    "PlacementKind",
    "ProviderBinding",
    "ValuePlacement",
    "resolve_execution_plan",
)
