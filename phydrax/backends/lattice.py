#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Granular lattice-kernel capabilities and the functional native JAX provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._availability import probe_backend
from ._types import (
    BackendAvailability,
    BackendCapabilities,
    BackendDistributionCapabilities,
)


_GAUGE_ACTION = "gauge.wilson_action"
_GAUGE_FORCE = "gauge.wilson_force"
_WILSON_DSLASH = "fermion.wilson_dslash"
_CLOVER_DSLASH = "fermion.clover_dslash"
_BLOCK_DSLASH = "fermion.block_dslash"
_PROJECTED_HALO = "fermion.projected_halo"


class LatticeKernelCapabilityError(RuntimeError):
    provider_id: str
    capability: str

    def __init__(self, provider_id: str, capability: str, reason: str, /) -> None:
        provider = str(provider_id).strip()
        requested = str(capability).strip()
        reason_ = str(reason).strip()
        if not provider or not requested or not reason_:
            raise ValueError("Lattice capability error fields must be non-empty.")
        self.provider_id = provider
        self.capability = requested
        super().__init__(
            f"Lattice provider {provider!r} cannot provide {requested!r}: {reason_}"
        )


class LatticeKernelCapabilities(StrictModule, NonTrainableState):
    provider_id: str = eqx.field(static=True)
    operations: tuple[str, ...] = eqx.field(static=True)
    dtypes: tuple[str, ...] = eqx.field(static=True)
    execution_scopes: tuple[str, ...] = eqx.field(static=True)
    transformations: tuple[str, ...] = eqx.field(static=True)
    native_complex: bool = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        operations: Sequence[str],
        dtypes: Sequence[str],
        execution_scopes: Sequence[str],
        transformations: Sequence[str],
        /,
        *,
        native_complex: bool,
    ) -> None:
        provider = str(provider_id).strip()
        operations_ = tuple(str(value).strip() for value in operations)
        dtypes_ = tuple(str(value).strip() for value in dtypes)
        scopes = tuple(str(value).strip() for value in execution_scopes)
        transforms = tuple(str(value).strip() for value in transformations)
        if (
            not provider
            or not operations_
            or not dtypes_
            or not scopes
            or any(not value for value in (*operations_, *dtypes_, *scopes, *transforms))
            or len(set(operations_)) != len(operations_)
        ):
            raise ValueError(
                "Lattice kernel capability declarations must be non-empty and unique."
            )
        self.provider_id = provider
        self.operations = operations_
        self.dtypes = dtypes_
        self.execution_scopes = scopes
        self.transformations = transforms
        self.native_complex = bool(native_complex)
        self.capabilities_id = canonical_fingerprint(
            {
                "kind": "lattice-kernel-capabilities",
                "provider": provider,
                "operations": operations_,
                "dtypes": dtypes_,
                "execution_scopes": scopes,
                "transformations": transforms,
                "native_complex": bool(native_complex),
            }
        )

    def supports(self, operation: str, /) -> bool:
        return str(operation) in self.operations

    def require(self, operation: str, dtype: object | None = None, /) -> None:
        requested = str(operation)
        if requested not in self.operations:
            raise LatticeKernelCapabilityError(
                self.provider_id,
                requested,
                "the operation is absent from the provider capability declaration",
            )
        if dtype is not None and np.dtype(dtype).name not in self.dtypes:
            raise LatticeKernelCapabilityError(
                self.provider_id,
                requested,
                f"dtype {np.dtype(dtype).name!r} is outside the declared precision envelope",
            )


class LatticeProviderStatus(StrictModule, NonTrainableState):
    capabilities: LatticeKernelCapabilities
    available: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    versions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    status_id: str = eqx.field(static=True)

    def __init__(
        self,
        capabilities: LatticeKernelCapabilities,
        /,
        *,
        available: bool,
        reason: str,
        versions: Sequence[tuple[str, str]] = (),
    ) -> None:
        if not isinstance(capabilities, LatticeKernelCapabilities):
            raise TypeError("capabilities must be LatticeKernelCapabilities.")
        reason_ = str(reason).strip()
        versions_ = tuple(
            (str(name).strip(), str(version).strip()) for name, version in versions
        )
        if not reason_ or any(not name or not version for name, version in versions_):
            raise ValueError("Provider status evidence must be non-empty.")
        self.capabilities = capabilities
        self.available = bool(available)
        self.reason = reason_
        self.versions = versions_
        self.status_id = canonical_fingerprint(
            {
                "kind": "lattice-provider-status",
                "capabilities": capabilities.capabilities_id,
                "available": bool(available),
                "reason": reason_,
                "versions": versions_,
            }
        )

    @property
    def provider_id(self) -> str:
        return self.capabilities.provider_id

    def require(self, operation: str, dtype: object | None = None, /) -> None:
        self.capabilities.require(operation, dtype)
        if not self.available:
            raise LatticeKernelCapabilityError(
                self.provider_id,
                str(operation),
                self.reason,
            )


@runtime_checkable
class LatticeKernelProvider(Protocol):
    provider_id: str
    capabilities: LatticeKernelCapabilities

    def gauge_action(
        self, plaquettes: ArrayLike, beta: ArrayLike, owned_mask: ArrayLike, /
    ) -> Array: ...

    def gauge_force(
        self,
        links: ArrayLike,
        staples: ArrayLike,
        beta: ArrayLike,
        owned_mask: ArrayLike,
        /,
    ) -> Array: ...

    def wilson_dslash(
        self, operator: Any, spinor: ArrayLike, /, *, adjoint: bool = False
    ) -> Array: ...

    def clover_dslash(
        self, operator: Any, spinor: ArrayLike, /, *, adjoint: bool = False
    ) -> Array: ...

    def block_dslash(
        self,
        operator: Any,
        right_hand_sides: ArrayLike,
        /,
        *,
        adjoint: bool = False,
    ) -> Array: ...


NATIVE_JAX_LATTICE_CAPABILITIES = LatticeKernelCapabilities(
    "jax-native-lattice",
    (
        _GAUGE_ACTION,
        _GAUGE_FORCE,
        _WILSON_DSLASH,
        _CLOVER_DSLASH,
        _BLOCK_DSLASH,
        _PROJECTED_HALO,
    ),
    ("complex64", "complex128"),
    ("single_device", "multi_device", "multi_host"),
    ("jit", "vmap", "jvp", "vjp", "rematerialization"),
    native_complex=True,
)


class NativeJaxLatticeProvider:
    """Allocation-transparent JAX kernels over canonical site/link arrays."""

    __slots__ = ()

    provider_id = "jax-native-lattice"
    capabilities = NATIVE_JAX_LATTICE_CAPABILITIES

    def status(self, /) -> LatticeProviderStatus:
        return LatticeProviderStatus(
            self.capabilities,
            available=True,
            reason="native JAX lattice kernels are part of Phydrax",
            versions=(("jax", jax.__version__),),
        )

    def gauge_action(
        self, plaquettes: ArrayLike, beta: ArrayLike, owned_mask: ArrayLike, /
    ) -> Array:
        values = jnp.asarray(plaquettes)
        mask = jnp.asarray(owned_mask, dtype=jnp.bool_)
        if values.ndim < 3 or values.shape[-1] != values.shape[-2]:
            raise ValueError("Plaquettes must end in square color-matrix axes.")
        if mask.shape != values.shape[:-2]:
            raise ValueError("owned_mask must match the plaquette batch shape.")
        self.capabilities.require(_GAUGE_ACTION, values.dtype)
        colors = values.shape[-1]
        density = 1.0 - jnp.real(jnp.trace(values, axis1=-2, axis2=-1)) / colors
        return jnp.asarray(beta, dtype=density.dtype) * jnp.sum(
            jnp.where(mask, density, 0)
        )

    def gauge_force(
        self,
        links: ArrayLike,
        staples: ArrayLike,
        beta: ArrayLike,
        owned_mask: ArrayLike,
        /,
    ) -> Array:
        links_ = jnp.asarray(links)
        staples_ = jnp.asarray(staples)
        mask = jnp.asarray(owned_mask, dtype=jnp.bool_)
        if (
            links_.shape != staples_.shape
            or links_.ndim < 3
            or links_.shape[-1] != links_.shape[-2]
        ):
            raise ValueError("Links and staples must share square color-matrix axes.")
        if mask.shape != links_.shape[:-2]:
            raise ValueError("owned_mask must match the link batch shape.")
        self.capabilities.require(_GAUGE_FORCE, links_.dtype)
        product = contract(
            "...ij,...kj->...ik", links_, jnp.conj(staples_), backend="jax"
        )
        adjoint = jnp.conj(jnp.swapaxes(product, -1, -2))
        antihermitian = 0.5 * (product - adjoint)
        colors = links_.shape[-1]
        trace = jnp.trace(antihermitian, axis1=-2, axis2=-1) / colors
        identity = jnp.eye(colors, dtype=links_.dtype)
        force = (
            jnp.asarray(beta, dtype=links_.real.dtype)
            * (antihermitian - trace[..., None, None] * identity)
            / colors
        )
        expanded = mask.reshape(mask.shape + (1, 1))
        return jnp.where(expanded, force, 0)

    def wilson_dslash(
        self,
        operator: Any,
        spinor: ArrayLike,
        /,
        *,
        adjoint: bool = False,
    ) -> Array:
        values = jnp.asarray(spinor)
        self.capabilities.require(_WILSON_DSLASH, values.dtype)
        return operator.adjoint_mv(values) if adjoint else operator.mv(values)

    def clover_dslash(
        self,
        operator: Any,
        spinor: ArrayLike,
        /,
        *,
        adjoint: bool = False,
    ) -> Array:
        values = jnp.asarray(spinor)
        self.capabilities.require(_CLOVER_DSLASH, values.dtype)
        return operator.adjoint_mv(values) if adjoint else operator.mv(values)

    def block_dslash(
        self,
        operator: Any,
        right_hand_sides: ArrayLike,
        /,
        *,
        adjoint: bool = False,
    ) -> Array:
        values = jnp.asarray(right_hand_sides)
        if values.ndim < 2 or values.shape[-1] < 1:
            raise ValueError("Block right-hand sides require a non-empty final RHS axis.")
        self.capabilities.require(_BLOCK_DSLASH, values.dtype)
        action = operator.adjoint_mv if adjoint else operator.mv
        return jax.vmap(action, in_axes=-1, out_axes=-1)(values)

    def project_spinors(self, spinors: ArrayLike, projector_basis: ArrayLike, /) -> Array:
        values = jnp.asarray(spinors)
        projectors = jnp.asarray(projector_basis)
        if (
            values.ndim < 2
            or projectors.ndim != 2
            or projectors.shape[1] != values.shape[-2]
        ):
            raise ValueError("Projectors must map the penultimate spin axis.")
        self.capabilities.require(_PROJECTED_HALO, values.dtype)
        return contract("hs,...sc->...hc", projectors, values, backend="jax")

    def reconstruct_spinors(
        self, payloads: ArrayLike, reconstructor_basis: ArrayLike, /
    ) -> Array:
        values = jnp.asarray(payloads)
        reconstructors = jnp.asarray(reconstructor_basis)
        if (
            values.ndim < 2
            or reconstructors.ndim != 2
            or reconstructors.shape[1] != values.shape[-2]
        ):
            raise ValueError(
                "Reconstructors must consume the penultimate half-spin axis."
            )
        self.capabilities.require(_PROJECTED_HALO, values.dtype)
        return contract("sh,...hc->...sc", reconstructors, values, backend="jax")


_QUDA_BACKEND = BackendCapabilities(
    backend="pyquda",
    problem_kinds=(
        _GAUGE_ACTION,
        _GAUGE_FORCE,
        _WILSON_DSLASH,
        _CLOVER_DSLASH,
        _BLOCK_DSLASH,
    ),
    execution="device",
    host_only=False,
    supports_matrix_free=True,
    supports_assembled=False,
    coordinate_dtypes=("complex64", "complex128"),
    distribution=BackendDistributionCapabilities(
        array_model="rank_local",
        scopes=("single_device", "multi_device", "multi_host"),
        platforms=("cuda",),
        collectives=("halo_exchange", "sum"),
        transformations=("operation_specific_jvp",),
        communicator_model="external",
        supports_process_local_input=True,
        supports_distributed_output=True,
        supports_topology_change_restart=False,
    ),
)

_GRID_BACKEND = BackendCapabilities(
    backend="gpt-grid",
    problem_kinds=(
        _GAUGE_ACTION,
        _GAUGE_FORCE,
        _WILSON_DSLASH,
        _CLOVER_DSLASH,
        _BLOCK_DSLASH,
    ),
    execution="host",
    host_only=True,
    supports_matrix_free=True,
    supports_assembled=False,
    coordinate_dtypes=("complex64", "complex128"),
    distribution=BackendDistributionCapabilities(
        array_model="rank_local",
        scopes=("single_device", "multi_process", "multi_host"),
        platforms=("cpu", "cuda"),
        collectives=("halo_exchange", "sum"),
        transformations=(),
        communicator_model="external",
        supports_process_local_input=True,
        supports_distributed_output=True,
        supports_topology_change_restart=True,
    ),
)

_QLAT_BACKEND = BackendCapabilities(
    backend="qlat",
    problem_kinds=(_GAUGE_ACTION, _GAUGE_FORCE, _WILSON_DSLASH),
    execution="host",
    host_only=True,
    supports_matrix_free=True,
    supports_assembled=False,
    coordinate_dtypes=("complex128",),
    distribution=BackendDistributionCapabilities(
        array_model="rank_local",
        scopes=("single_device", "multi_process"),
        platforms=("cpu",),
        collectives=("halo_exchange", "sum"),
        transformations=(),
        communicator_model="external",
        supports_process_local_input=True,
        supports_distributed_output=True,
        supports_topology_change_restart=False,
    ),
)


def pyquda_availability() -> BackendAvailability:
    return probe_backend(
        _QUDA_BACKEND,
        module="pyquda",
        requirement="install PyQUDA with an ABI-compatible QUDA and CUDA runtime",
        distributions=("pyquda",),
    )


def grid_availability() -> BackendAvailability:
    return probe_backend(
        _GRID_BACKEND,
        module="gpt",
        requirement="install the Grid Python GPT bindings and their native runtime",
        distributions=("gpt",),
    )


def qlat_availability() -> BackendAvailability:
    return probe_backend(
        _QLAT_BACKEND,
        module="qlat",
        requirement="install qlat and its native MPI/OpenMP runtime",
        distributions=("qlat",),
    )


def optional_lattice_provider_availability() -> tuple[BackendAvailability, ...]:
    """Return honest discovery evidence; no unavailable adapter is constructed."""

    return (pyquda_availability(), grid_availability(), qlat_availability())


__all__ = [
    "grid_availability",
    "LatticeKernelCapabilities",
    "LatticeKernelCapabilityError",
    "LatticeKernelProvider",
    "LatticeProviderStatus",
    "NATIVE_JAX_LATTICE_CAPABILITIES",
    "NativeJaxLatticeProvider",
    "optional_lattice_provider_availability",
    "pyquda_availability",
    "qlat_availability",
]
