#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...backends._types import BackendUnavailableError


RoboticsOperation: TypeAlias = Literal[
    "forward-kinematics",
    "smooth-dynamics",
    "contact",
    "step",
    "sensors",
    "model-batching",
    "jit",
    "vmap",
    "jvp",
    "vjp",
]
RoboticsDifferentiability: TypeAlias = Literal[
    "none", "conditional", "guaranteed"
]
RoboticsProjectionKind: TypeAlias = Literal[
    "qpos", "qvel", "control", "observation"
]
ObservationFreshness: TypeAlias = Literal[
    "state-current", "pre-step", "post-step-refreshed"
]

ROBOTICS_OPERATIONS: tuple[RoboticsOperation, ...] = (
    "forward-kinematics",
    "smooth-dynamics",
    "contact",
    "step",
    "sensors",
    "model-batching",
    "jit",
    "vmap",
    "jvp",
    "vjp",
)
_DIFFERENTIABILITY_RANK = {"none": 0, "conditional": 1, "guaranteed": 2}


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _normalized_values(
    values: Sequence[str],
    name: str,
    /,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    normalized = tuple(str(value).strip().lower() for value in values)
    if (not allow_empty and not normalized) or any(not value for value in normalized):
        raise ValueError(f"{name} must contain non-empty values.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique values.")
    return normalized


def _operation(value: str, /) -> RoboticsOperation:
    operation = str(value)
    if operation not in ROBOTICS_OPERATIONS:
        raise ValueError(f"Unknown robotics backend operation {operation!r}.")
    return operation  # type: ignore[return-value]


class RoboticsOperationStatus(IntEnum):
    """Canonical status codes for one robotics backend operation."""

    SUCCESS = 0
    UNAVAILABLE = 1
    UNSUPPORTED = 2
    NONFINITE = 3
    INVALID_STATE = 4
    PROVIDER_FAILURE = 5


class RoboticsOperationCapability(StrictModule, NonTrainableState):
    """Immutable support declaration for exactly one backend operation."""

    operation: RoboticsOperation = eqx.field(static=True)
    supported: bool = eqx.field(static=True)
    implementation: str = eqx.field(static=True)
    devices: tuple[str, ...] = eqx.field(static=True)
    dtypes: tuple[str, ...] = eqx.field(static=True)
    differentiability: RoboticsDifferentiability = eqx.field(static=True)
    solvers: tuple[str, ...] = eqx.field(static=True)
    contact_features: tuple[str, ...] = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    def __init__(
        self,
        operation: RoboticsOperation,
        /,
        *,
        supported: bool,
        implementation: str,
        devices: Sequence[str] = (),
        dtypes: Sequence[str] = (),
        differentiability: RoboticsDifferentiability = "none",
        solvers: Sequence[str] = (),
        contact_features: Sequence[str] = (),
        reason: str = "",
    ):
        operation_ = _operation(operation)
        implementation_ = _identifier(implementation, "implementation")
        devices_ = _normalized_values(devices, "devices")
        dtypes_ = tuple(
            np.dtype(dtype).name if str(dtype).lower() != "bfloat16" else "bfloat16"
            for dtype in _normalized_values(dtypes, "dtypes")
        )
        if len(set(dtypes_)) != len(dtypes_):
            raise ValueError("dtypes must contain unique canonical values.")
        if differentiability not in _DIFFERENTIABILITY_RANK:
            raise ValueError("Unknown robotics differentiability level.")
        solvers_ = _normalized_values(solvers, "solvers")
        contact_features_ = _normalized_values(
            contact_features, "contact_features"
        )
        reason_ = str(reason).strip()
        if supported and (not devices_ or not dtypes_):
            raise ValueError("Supported operations must declare devices and dtypes.")
        if not supported and not reason_:
            raise ValueError("Unsupported operations must provide a reason.")
        if not supported and differentiability != "none":
            raise ValueError("Unsupported operations cannot claim differentiability.")
        self.operation = operation_
        self.supported = bool(supported)
        self.implementation = implementation_
        self.devices = devices_
        self.dtypes = dtypes_
        self.differentiability = differentiability
        self.solvers = solvers_
        self.contact_features = contact_features_
        self.reason = reason_

    def rejection_reason(self, requirement: RoboticsOperationRequirement, /) -> str | None:
        """Return the first unmet condition, or ``None`` when accepted."""
        if not isinstance(requirement, RoboticsOperationRequirement):
            raise TypeError("requirement must be RoboticsOperationRequirement.")
        if requirement.operation != self.operation:
            raise ValueError("Capability and requirement operations must match.")
        if not self.supported:
            return self.reason
        if requirement.device is not None and requirement.device not in self.devices:
            return f"device {requirement.device!r} is not supported"
        if requirement.dtype is not None and requirement.dtype not in self.dtypes:
            return f"dtype {requirement.dtype!r} is not supported"
        if (
            _DIFFERENTIABILITY_RANK[self.differentiability]
            < _DIFFERENTIABILITY_RANK[requirement.minimum_differentiability]
        ):
            return (
                f"requires {requirement.minimum_differentiability} differentiability; "
                f"only {self.differentiability} is declared"
            )
        if requirement.solver is not None and requirement.solver not in self.solvers:
            return f"solver {requirement.solver!r} is not in the closed support set"
        if (
            requirement.contact_feature is not None
            and requirement.contact_feature not in self.contact_features
        ):
            return (
                f"contact feature {requirement.contact_feature!r} is not in the "
                "closed support set"
            )
        return None

    def supports(self, requirement: RoboticsOperationRequirement, /) -> bool:
        """Return whether this exact per-operation requirement is declared."""
        return self.rejection_reason(requirement) is None


class RoboticsOperationRequirement(StrictModule, NonTrainableState):
    """Consumer requirements negotiated against one operation declaration."""

    operation: RoboticsOperation = eqx.field(static=True)
    device: str | None = eqx.field(static=True)
    dtype: str | None = eqx.field(static=True)
    minimum_differentiability: RoboticsDifferentiability = eqx.field(static=True)
    solver: str | None = eqx.field(static=True)
    contact_feature: str | None = eqx.field(static=True)

    def __init__(
        self,
        operation: RoboticsOperation,
        /,
        *,
        device: str | None = None,
        dtype: Any | None = None,
        minimum_differentiability: RoboticsDifferentiability = "none",
        solver: str | None = None,
        contact_feature: str | None = None,
    ):
        if minimum_differentiability not in _DIFFERENTIABILITY_RANK:
            raise ValueError("Unknown minimum differentiability level.")
        device_ = None if device is None else _identifier(device, "device").lower()
        if dtype is None:
            dtype_ = None
        elif str(dtype).lower() == "bfloat16":
            dtype_ = "bfloat16"
        else:
            dtype_ = np.dtype(dtype).name
        solver_ = None if solver is None else _identifier(solver, "solver").lower()
        contact_ = (
            None
            if contact_feature is None
            else _identifier(contact_feature, "contact_feature").lower()
        )
        self.operation = _operation(operation)
        self.device = device_
        self.dtype = dtype_
        self.minimum_differentiability = minimum_differentiability
        self.solver = solver_
        self.contact_feature = contact_


class RoboticsRequirementRejection(StrictModule, NonTrainableState):
    """Typed evidence for one rejected operation requirement."""

    requirement: RoboticsOperationRequirement
    reason: str = eqx.field(static=True)

    def __init__(self, requirement: RoboticsOperationRequirement, reason: str, /):
        if not isinstance(requirement, RoboticsOperationRequirement):
            raise TypeError("requirement must be RoboticsOperationRequirement.")
        self.requirement = requirement
        self.reason = _identifier(reason, "reason")


class RoboticsCapabilityNegotiation(StrictModule, NonTrainableState):
    """Complete accepted/rejected result for one profile negotiation."""

    backend: str = eqx.field(static=True)
    requirements: tuple[RoboticsOperationRequirement, ...]
    rejections: tuple[RoboticsRequirementRejection, ...]
    status: RoboticsOperationStatus = eqx.field(static=True)

    def __init__(
        self,
        *,
        backend: str,
        requirements: Sequence[RoboticsOperationRequirement],
        rejections: Sequence[RoboticsRequirementRejection],
    ):
        requirements_ = tuple(requirements)
        rejections_ = tuple(rejections)
        if any(
            not isinstance(requirement, RoboticsOperationRequirement)
            for requirement in requirements_
        ):
            raise TypeError("requirements must contain operation requirements.")
        if any(
            not isinstance(rejection, RoboticsRequirementRejection)
            for rejection in rejections_
        ):
            raise TypeError("rejections must contain requirement rejection evidence.")
        self.backend = _identifier(backend, "backend")
        self.requirements = requirements_
        self.rejections = rejections_
        self.status = (
            RoboticsOperationStatus.SUCCESS
            if not rejections_
            else RoboticsOperationStatus.UNSUPPORTED
        )

    @property
    def accepted(self) -> bool:
        return self.status == RoboticsOperationStatus.SUCCESS

    def require(self, /) -> None:
        """Raise the shared backend error when any requirement was rejected."""
        if self.rejections:
            rejection = self.rejections[0]
            raise BackendUnavailableError(
                self.backend,
                rejection.requirement.operation,
                "robotics operation capability negotiation",
                rejection.reason,
            )


class RoboticsBackendProfile(StrictModule, NonTrainableState):
    """Immutable collection of backend capabilities, one entry per operation."""

    backend: str = eqx.field(static=True)
    implementation: str = eqx.field(static=True)
    operations: tuple[RoboticsOperationCapability, ...]

    def __init__(
        self,
        *,
        backend: str,
        implementation: str,
        operations: Sequence[RoboticsOperationCapability],
    ):
        operations_ = tuple(operations)
        if not operations_ or any(
            not isinstance(operation, RoboticsOperationCapability)
            for operation in operations_
        ):
            raise TypeError("operations must contain operation capabilities.")
        names = tuple(operation.operation for operation in operations_)
        if len(set(names)) != len(names):
            raise ValueError("A backend profile may declare each operation only once.")
        self.backend = _identifier(backend, "backend")
        self.implementation = _identifier(implementation, "implementation")
        self.operations = operations_

    def capability(self, operation: RoboticsOperation, /) -> RoboticsOperationCapability:
        operation_ = _operation(operation)
        for capability in self.operations:
            if capability.operation == operation_:
                return capability
        raise BackendUnavailableError(
            self.backend,
            operation_,
            "robotics backend operation declaration",
            "the profile does not declare this operation",
        )

    def negotiate(
        self,
        requirements: Sequence[RoboticsOperationRequirement],
        /,
    ) -> RoboticsCapabilityNegotiation:
        """Evaluate all requirements without silently weakening any of them."""
        requirements_ = tuple(requirements)
        rejections: list[RoboticsRequirementRejection] = []
        for requirement in requirements_:
            if not isinstance(requirement, RoboticsOperationRequirement):
                raise TypeError("requirements must contain operation requirements.")
            capability = next(
                (
                    declared
                    for declared in self.operations
                    if declared.operation == requirement.operation
                ),
                None,
            )
            if capability is None:
                rejections.append(
                    RoboticsRequirementRejection(
                        requirement, "the profile does not declare this operation"
                    )
                )
                continue
            reason = capability.rejection_reason(requirement)
            if reason is not None:
                rejections.append(RoboticsRequirementRejection(requirement, reason))
        return RoboticsCapabilityNegotiation(
            backend=self.backend,
            requirements=requirements_,
            rejections=rejections,
        )

    def require(
        self,
        requirements: Sequence[RoboticsOperationRequirement],
        /,
    ) -> RoboticsCapabilityNegotiation:
        """Negotiate and raise explicitly rather than returning partial support."""
        negotiation = self.negotiate(requirements)
        negotiation.require()
        return negotiation


class RoboticsIndexEntry(StrictModule, NonTrainableState):
    """Stable half-open range assigned to one canonical robotics name."""

    name: str = eqx.field(static=True)
    start: int = eqx.field(static=True)
    stop: int = eqx.field(static=True)

    def __init__(self, name: str, start: int, stop: int, /):
        start_ = int(start)
        stop_ = int(stop)
        if start_ < 0 or stop_ <= start_:
            raise ValueError("Index ranges must be non-empty and non-negative.")
        self.name = _identifier(name, "name")
        self.start = start_
        self.stop = stop_

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop))

class RoboticsProjectionProvenance(StrictModule, NonTrainableState):
    """Immutable origin of one prepared-model projection layout."""

    model: str = eqx.field(static=True)
    compiler: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    asset: str = eqx.field(static=True)
    unit_system: str = eqx.field(static=True)
    frame_convention: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        model: str,
        compiler: str,
        provider: str,
        asset: str,
        unit_system: str,
        frame_convention: str,
    ):
        self.model = _identifier(model, "model")
        self.compiler = _identifier(compiler, "compiler")
        self.provider = _identifier(provider, "provider")
        self.asset = _identifier(asset, "asset")
        self.unit_system = _identifier(unit_system, "unit_system")
        self.frame_convention = _identifier(frame_convention, "frame_convention")



class RoboticsProjectionMap(StrictModule, NonTrainableState):
    """Canonical, immutable name-to-range map for one flat projection."""

    kind: RoboticsProjectionKind = eqx.field(static=True)
    size: int = eqx.field(static=True)
    entries: tuple[RoboticsIndexEntry, ...]
    provenance: RoboticsProjectionProvenance

    def __init__(
        self,
        kind: RoboticsProjectionKind,
        size: int,
        entries: Sequence[RoboticsIndexEntry],
        provenance: RoboticsProjectionProvenance,
        /,
    ):
        if kind not in ("qpos", "qvel", "control", "observation"):
            raise ValueError(f"Unknown robotics projection kind {kind!r}.")
        size_ = int(size)
        entries_ = tuple(entries)
        if size_ < 0:
            raise ValueError("Projection size must be non-negative.")
        if any(not isinstance(entry, RoboticsIndexEntry) for entry in entries_):
            raise TypeError("entries must contain RoboticsIndexEntry values.")
        if len({entry.name for entry in entries_}) != len(entries_):
            raise ValueError("Projection entry names must be unique.")
        cursor = 0
        for entry in entries_:
            if entry.start != cursor:
                raise ValueError("Projection entries must form contiguous ordered ranges.")
            cursor = entry.stop
        if cursor != size_:
            raise ValueError("Projection entries must cover the complete projection.")
        if not isinstance(provenance, RoboticsProjectionProvenance):
            raise TypeError("provenance must be RoboticsProjectionProvenance.")
        self.kind = kind
        self.size = size_
        self.entries = entries_
        self.provenance = provenance

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)

    @property
    def name_to_range(self) -> tuple[tuple[str, tuple[int, int]], ...]:
        return tuple((entry.name, (entry.start, entry.stop)) for entry in self.entries)

    def entry(self, name: str, /) -> RoboticsIndexEntry:
        name_ = str(name)
        for entry in self.entries:
            if entry.name == name_:
                return entry
        raise KeyError(name_)


class RoboticsProjection(StrictModule, NonTrainableState):
    """Typed flat projection with epoch-derived observation freshness."""

    values: Any
    index_map: RoboticsProjectionMap
    state_epoch: Array | None
    sample_epoch: Array | None

    def __init__(
        self,
        values: Any,
        index_map: RoboticsProjectionMap,
        /,
        *,
        state_epoch: Any | None = None,
        sample_epoch: Any | None = None,
    ):
        if not isinstance(index_map, RoboticsProjectionMap):
            raise TypeError("index_map must be RoboticsProjectionMap.")
        shape = jnp.shape(values)
        if not shape or shape[-1] != index_map.size:
            raise ValueError(
                f"{index_map.kind} projection must end in axis size {index_map.size}; "
                f"got shape {shape}."
            )
        if index_map.kind == "observation":
            if state_epoch is None or sample_epoch is None:
                raise ValueError(
                    "Observation projections require state and sample epochs."
                )
            state_epoch_ = jnp.asarray(state_epoch, dtype=jnp.int32)
            sample_epoch_ = jnp.asarray(sample_epoch, dtype=jnp.int32)
            if state_epoch_.shape != shape[:-1] or sample_epoch_.shape != shape[:-1]:
                raise ValueError(
                    "Observation epochs must have exactly the projection case axes."
                )
            state_epoch_ = eqx.error_if(
                state_epoch_,
                (state_epoch_ < 0) | (sample_epoch_ < 0),
                "Observation epochs must be non-negative.",
            )
            sample_epoch_ = eqx.error_if(
                sample_epoch_,
                sample_epoch_ > state_epoch_,
                "An observation sample cannot be newer than its state.",
            )
        else:
            if state_epoch is not None or sample_epoch is not None:
                raise ValueError("Only observation projections bind state epochs.")
            state_epoch_ = None
            sample_epoch_ = None
        self.values = values
        self.index_map = index_map
        self.state_epoch = state_epoch_
        self.sample_epoch = sample_epoch_

    @property
    def provenance(self) -> RoboticsProjectionProvenance:
        return self.index_map.provenance

    @property
    def freshness(self) -> Array:
        """Return casewise freshness derived only from bound state epochs."""
        if self.state_epoch is None or self.sample_epoch is None:
            return jnp.asarray(True)
        return self.sample_epoch == self.state_epoch


class RoboticsOperationEvidence(StrictModule, NonTrainableState):
    """Typed casewise numerical status and execution facts."""

    status: Array
    finite: Array
    backend: str = eqx.field(static=True)
    operation: RoboticsOperation = eqx.field(static=True)
    implementation: str = eqx.field(static=True)
    device: str = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    detail: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        status: RoboticsOperationStatus | Array | int,
        finite: Any,
        backend: str,
        operation: RoboticsOperation,
        implementation: str,
        device: str,
        dtype: Any,
        detail: str,
    ):
        status_ = jnp.asarray(status, dtype=jnp.int32)
        finite_ = jnp.asarray(finite, dtype=jnp.bool_)
        if status_.shape != finite_.shape:
            raise ValueError(
                "Robotics operation status and finite evidence must have equal "
                "case axes."
            )
        status_ = eqx.error_if(
            status_,
            jnp.any(
                (status_ < int(RoboticsOperationStatus.SUCCESS))
                | (status_ > int(RoboticsOperationStatus.PROVIDER_FAILURE))
            ),
            "Unknown robotics operation status.",
        )
        self.status = status_
        self.finite = finite_
        self.backend = _identifier(backend, "backend")
        self.operation = _operation(operation)
        self.implementation = _identifier(implementation, "implementation")
        self.device = _identifier(device, "device").lower()
        self.dtype = np.dtype(dtype).name
        self.detail = _identifier(detail, "detail")

    @property
    def successful(self) -> Array:
        return (self.status == int(RoboticsOperationStatus.SUCCESS)) & self.finite


__all__ = [
    "ObservationFreshness",
    "ROBOTICS_OPERATIONS",
    "RoboticsBackendProfile",
    "RoboticsCapabilityNegotiation",
    "RoboticsDifferentiability",
    "RoboticsIndexEntry",
    "RoboticsOperation",
    "RoboticsOperationCapability",
    "RoboticsOperationEvidence",
    "RoboticsOperationRequirement",
    "RoboticsOperationStatus",
    "RoboticsProjection",
    "RoboticsProjectionKind",
    "RoboticsProjectionMap",
    "RoboticsProjectionProvenance",
    "RoboticsRequirementRejection",
]
