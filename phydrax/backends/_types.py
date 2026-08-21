#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule


BackendExecution: TypeAlias = Literal["host", "device"]


class BackendUnavailableError(RuntimeError):
    """An explicitly selected optional backend cannot provide a capability."""

    backend: str
    capability: str
    requirement: str

    def __init__(
        self,
        backend: str,
        capability: str,
        requirement: str,
        reason: str,
        /,
    ):
        backend_ = str(backend)
        capability_ = str(capability)
        requirement_ = str(requirement)
        reason_ = str(reason)
        if any(not value for value in (backend_, capability_, requirement_, reason_)):
            raise ValueError("Backend availability error fields must be non-empty.")
        self.backend = backend_
        self.capability = capability_
        self.requirement = requirement_
        super().__init__(
            f"Backend {backend_!r} cannot provide {capability_!r}: "
            f"requirement {requirement_!r}; {reason_}"
        )


class BackendCapabilities(StrictModule):
    """Immutable, dependency-free description of an optional backend boundary."""

    backend: str = eqx.field(static=True)
    problem_kinds: tuple[str, ...] = eqx.field(static=True)
    execution: BackendExecution = eqx.field(static=True)
    host_only: bool = eqx.field(static=True)
    supports_matrix_free: bool = eqx.field(static=True)
    supports_assembled: bool = eqx.field(static=True)
    coordinate_dtypes: tuple[str, ...] = eqx.field(static=True)
    supports_plan_prepare_solve_refresh: bool = eqx.field(static=True)
    requires_explicit_release: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        backend: str,
        problem_kinds: tuple[str, ...],
        execution: BackendExecution,
        host_only: bool,
        supports_matrix_free: bool,
        supports_assembled: bool,
        coordinate_dtypes: tuple[str, ...],
        supports_plan_prepare_solve_refresh: bool = True,
        requires_explicit_release: bool = False,
    ):
        backend_ = str(backend)
        kinds = tuple(str(kind) for kind in problem_kinds)
        dtypes = tuple(str(dtype) for dtype in coordinate_dtypes)
        if not backend_ or not kinds or any(not value for value in kinds):
            raise ValueError("Backend name and problem kinds must be non-empty.")
        if not dtypes or any(not value for value in dtypes):
            raise ValueError("Backend coordinate dtypes must be non-empty.")
        if execution not in ("host", "device"):
            raise ValueError("Backend execution must be 'host' or 'device'.")
        if execution == "device" and bool(host_only):
            raise ValueError("A device backend cannot be declared host-only.")
        self.backend = backend_
        self.problem_kinds = kinds
        self.execution = execution
        self.host_only = bool(host_only)
        self.supports_matrix_free = bool(supports_matrix_free)
        self.supports_assembled = bool(supports_assembled)
        self.coordinate_dtypes = dtypes
        self.supports_plan_prepare_solve_refresh = bool(
            supports_plan_prepare_solve_refresh
        )
        self.requires_explicit_release = bool(requires_explicit_release)

    def supports(self, problem_kind: str, /) -> bool:
        """Return whether the declared backend accepts ``problem_kind``."""
        return str(problem_kind) in self.problem_kinds


class BackendAvailability(StrictModule):
    """Deterministic availability evidence without importing at package import time."""

    capabilities: BackendCapabilities
    backend: str = eqx.field(static=True)
    available: bool = eqx.field(static=True)
    requirement: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    versions: tuple[tuple[str, str], ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        capabilities: BackendCapabilities,
        available: bool,
        requirement: str,
        reason: str,
        versions: tuple[tuple[str, str], ...] = (),
    ):
        if not isinstance(capabilities, BackendCapabilities):
            raise TypeError("capabilities must be BackendCapabilities.")
        requirement_ = str(requirement)
        reason_ = str(reason)
        versions_ = tuple((str(name), str(version)) for name, version in versions)
        if not requirement_ or not reason_:
            raise ValueError(
                "Backend requirement and availability reason must be non-empty."
            )
        if any(not name or not version for name, version in versions_):
            raise ValueError("Backend version evidence must contain non-empty pairs.")
        self.capabilities = capabilities
        self.backend = capabilities.backend
        self.available = bool(available)
        self.requirement = requirement_
        self.reason = reason_
        self.versions = versions_

    def require(self, capability: str, /) -> None:
        """Raise a precise error unless this backend provides ``capability``."""
        capability_ = str(capability)
        if not self.capabilities.supports(capability_):
            raise BackendUnavailableError(
                self.backend,
                capability_,
                self.requirement,
                "the backend does not declare this capability",
            )
        if not self.available:
            raise BackendUnavailableError(
                self.backend,
                capability_,
                self.requirement,
                self.reason,
            )


class BackendTransferEvidence(StrictModule):
    """Observable data movement performed by one external backend operation."""

    host_to_device_bytes: Array
    device_to_host_bytes: Array
    synchronization_count: Array

    def __init__(
        self,
        *,
        host_to_device_bytes: Any = 0,
        device_to_host_bytes: Any = 0,
        synchronization_count: Any = 0,
    ):
        host_to_device = jnp.asarray(host_to_device_bytes, dtype=jnp.int64)
        device_to_host = jnp.asarray(device_to_host_bytes, dtype=jnp.int64)
        synchronizations = jnp.asarray(synchronization_count, dtype=jnp.int32)
        if any(
            value.shape != ()
            for value in (host_to_device, device_to_host, synchronizations)
        ):
            raise ValueError("Backend transfer evidence values must be scalars.")
        host_to_device = eqx.error_if(
            host_to_device,
            host_to_device < 0,
            "host_to_device_bytes must be non-negative.",
        )
        device_to_host = eqx.error_if(
            device_to_host,
            device_to_host < 0,
            "device_to_host_bytes must be non-negative.",
        )
        synchronizations = eqx.error_if(
            synchronizations,
            synchronizations < 0,
            "synchronization_count must be non-negative.",
        )
        self.host_to_device_bytes = host_to_device
        self.device_to_host_bytes = device_to_host
        self.synchronization_count = synchronizations


class AbstractExternalBackend(StrictModule):
    """Inspection boundary implemented by every optional external provider."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError

    @abc.abstractmethod
    def availability(self, /) -> BackendAvailability:
        raise NotImplementedError


__all__ = [
    "AbstractExternalBackend",
    "BackendAvailability",
    "BackendCapabilities",
    "BackendExecution",
    "BackendTransferEvidence",
    "BackendUnavailableError",
]
