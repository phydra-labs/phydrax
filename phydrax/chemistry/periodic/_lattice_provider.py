#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-only provider boundary for normalized lattice-dynamics artifacts."""

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._lattice_force_constants import SecondOrderForceConstants, ThirdOrderForceConstants


def _identifier(value: str, name: str) -> str:
    result = str(value)
    if not result or result != result.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return result


class LatticeDynamicsProviderCapabilities(StrictModule, NonTrainableState):
    """Exact provider outputs; absent capabilities may never be inferred."""

    second_order_force_constants: bool = eqx.field(static=True)
    third_order_force_constants: bool = eqx.field(static=True)
    born_effective_charges: bool = eqx.field(static=True)
    dielectric_tensor: bool = eqx.field(static=True)
    scalar_relativistic: bool = eqx.field(static=True)
    capability_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        second_order_force_constants: bool,
        third_order_force_constants: bool = False,
        born_effective_charges: bool = False,
        dielectric_tensor: bool = False,
        scalar_relativistic: bool = True,
    ):
        self.second_order_force_constants = bool(second_order_force_constants)
        self.third_order_force_constants = bool(third_order_force_constants)
        self.born_effective_charges = bool(born_effective_charges)
        self.dielectric_tensor = bool(dielectric_tensor)
        self.scalar_relativistic = bool(scalar_relativistic)
        if self.born_effective_charges != self.dielectric_tensor:
            raise ValueError(
                "Polar lattice capability requires Born charges and dielectric together."
            )
        self.capability_id = canonical_fingerprint(
            {
                "kind": "lattice-provider-capabilities",
                "ifc2": self.second_order_force_constants,
                "ifc3": self.third_order_force_constants,
                "born": self.born_effective_charges,
                "dielectric": self.dielectric_tensor,
                "scalar_relativistic": self.scalar_relativistic,
            }
        )


class LatticeDynamicsRequest(StrictModule, NonTrainableState):
    """Immutable bounded provider request with explicit structure and physics identity."""

    requested_orders: tuple[int, ...] = eqx.field(static=True)
    request_polar_tensors: bool = eqx.field(static=True)
    maximum_ifc2_routes: int = eqx.field(static=True)
    maximum_ifc3_routes: int = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    pseudopotential_id: str = eqx.field(static=True)
    spin_id: str = eqx.field(static=True)
    relativity_id: str = eqx.field(static=True)
    input_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        requested_orders: tuple[int, ...],
        request_polar_tensors: bool,
        maximum_ifc2_routes: int,
        maximum_ifc3_routes: int,
        system_id: str,
        cell_id: str,
        unit_system_id: str,
        method_id: str,
        basis_id: str,
        pseudopotential_id: str,
        spin_id: str,
        relativity_id: str,
        input_artifact_ids: tuple[str, ...],
    ):
        orders = tuple(int(value) for value in requested_orders)
        if (
            not orders
            or len(set(orders)) != len(orders)
            or any(value not in (2, 3) for value in orders)
        ):
            raise ValueError(
                "requested_orders must be a unique non-empty subset of (2,3)."
            )
        if int(maximum_ifc2_routes) < (1 if 2 in orders else 0) or int(
            maximum_ifc3_routes
        ) < (1 if 3 in orders else 0):
            raise ValueError("Requested IFC route capacities must be positive.")
        artifacts = tuple(
            _identifier(value, "input artifact ID") for value in input_artifact_ids
        )
        if not artifacts or len(set(artifacts)) != len(artifacts):
            raise ValueError("input_artifact_ids must be non-empty and unique.")
        self.requested_orders = orders
        self.request_polar_tensors = bool(request_polar_tensors)
        self.maximum_ifc2_routes = int(maximum_ifc2_routes)
        self.maximum_ifc3_routes = int(maximum_ifc3_routes)
        self.system_id = _identifier(system_id, "system_id")
        self.cell_id = _identifier(cell_id, "cell_id")
        self.unit_system_id = _identifier(unit_system_id, "unit_system_id")
        self.method_id = _identifier(method_id, "method_id")
        self.basis_id = _identifier(basis_id, "basis_id")
        self.pseudopotential_id = _identifier(pseudopotential_id, "pseudopotential_id")
        self.spin_id = _identifier(spin_id, "spin_id")
        self.relativity_id = _identifier(relativity_id, "relativity_id")
        self.input_artifact_ids = artifacts
        self.request_id = canonical_fingerprint(
            {
                "kind": "lattice-dynamics-request",
                "orders": list(orders),
                "polar": self.request_polar_tensors,
                "max_ifc2": self.maximum_ifc2_routes,
                "max_ifc3": self.maximum_ifc3_routes,
                "system": self.system_id,
                "cell": self.cell_id,
                "units": self.unit_system_id,
                "method": self.method_id,
                "basis": self.basis_id,
                "pseudopotential": self.pseudopotential_id,
                "spin": self.spin_id,
                "relativity": self.relativity_id,
                "inputs": list(artifacts),
            }
        )


class LatticeDynamicsArtifactSet(StrictModule, NonTrainableState):
    """Normalized provider outputs retaining provider and raw-residual provenance."""

    second_order: SecondOrderForceConstants | None
    third_order: ThirdOrderForceConstants | None
    born_effective_charges: Array | None
    dielectric_tensor: Array | None
    raw_residuals: tuple[tuple[str, float], ...] = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    provider_version: str = eqx.field(static=True)
    provider_build_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)
    input_digest: str = eqx.field(static=True)
    rights_id: str = eqx.field(static=True)
    first_principles_artifact_id: str | None = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        second_order: SecondOrderForceConstants | None,
        third_order: ThirdOrderForceConstants | None,
        born_effective_charges,
        dielectric_tensor,
        raw_residuals: Mapping[str, float],
        provider_id: str,
        provider_version: str,
        provider_build_id: str,
        request_id: str,
        input_digest: str,
        rights_id: str,
        first_principles_artifact_id: str | None = None,
    ):
        if second_order is not None and not isinstance(
            second_order, SecondOrderForceConstants
        ):
            raise TypeError("second_order must be SecondOrderForceConstants or None.")
        if third_order is not None and not isinstance(
            third_order, ThirdOrderForceConstants
        ):
            raise TypeError("third_order must be ThirdOrderForceConstants or None.")
        born = (
            None if born_effective_charges is None else np.asarray(born_effective_charges)
        )
        dielectric = None if dielectric_tensor is None else np.asarray(dielectric_tensor)
        if (born is None) != (dielectric is None):
            raise ValueError(
                "Provider polar data requires Born charges and dielectric together."
            )
        if born is not None:
            atoms = second_order.relation.source_size if second_order is not None else -1
            if born.shape != (atoms, 3, 3) or dielectric.shape != (3, 3):
                raise ValueError(
                    "Provider polar tensors do not match the IFC2 atom order."
                )
            if np.any(~np.isfinite(born)) or np.any(~np.isfinite(dielectric)):
                raise ValueError("Provider polar tensors must be finite.")
        residuals = tuple(
            sorted((str(name), float(value)) for name, value in raw_residuals.items())
        )
        if not residuals or any(
            not name or not np.isfinite(value) or value < 0.0 for name, value in residuals
        ):
            raise ValueError(
                "Provider raw_residuals must be finite, nonnegative, and non-empty."
            )
        self.second_order = second_order
        self.third_order = third_order
        self.born_effective_charges = None if born is None else jnp.asarray(born)
        self.dielectric_tensor = None if dielectric is None else jnp.asarray(dielectric)
        self.raw_residuals = residuals
        self.provider_id = _identifier(provider_id, "provider_id")
        self.provider_version = _identifier(provider_version, "provider_version")
        self.provider_build_id = _identifier(provider_build_id, "provider_build_id")
        self.request_id = _identifier(request_id, "request_id")
        self.input_digest = _identifier(input_digest, "input_digest")
        self.rights_id = _identifier(rights_id, "rights_id")
        self.first_principles_artifact_id = (
            None
            if first_principles_artifact_id is None
            else _identifier(first_principles_artifact_id, "first_principles_artifact_id")
        )
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "lattice-dynamics-artifact-set",
                "ifc2": None if second_order is None else second_order.ifc_id,
                "ifc3": None if third_order is None else third_order.ifc_id,
                "born": None
                if born is None
                else canonical_fingerprint({"values": born.tolist()}),
                "dielectric": None
                if dielectric is None
                else canonical_fingerprint({"values": dielectric.tolist()}),
                "raw_residuals": dict(residuals),
                "provider": self.provider_id,
                "version": self.provider_version,
                "build": self.provider_build_id,
                "request": self.request_id,
                "input": self.input_digest,
                "rights": self.rights_id,
                "first_principles": self.first_principles_artifact_id,
            }
        )

    def validate_request(
        self,
        request: LatticeDynamicsRequest,
        capabilities: LatticeDynamicsProviderCapabilities,
        /,
    ) -> None:
        if not isinstance(request, LatticeDynamicsRequest) or not isinstance(
            capabilities, LatticeDynamicsProviderCapabilities
        ):
            raise TypeError("Provider validation requires a request and capabilities.")
        if self.request_id != request.request_id:
            raise ValueError("Provider artifact does not bind the exact request.")
        if (2 in request.requested_orders) != (self.second_order is not None):
            raise ValueError("Provider IFC2 presence does not match the request.")
        if (3 in request.requested_orders) != (self.third_order is not None):
            raise ValueError("Provider IFC3 presence does not match the request.")
        if request.request_polar_tensors != (self.born_effective_charges is not None):
            raise ValueError("Provider polar tensor presence does not match the request.")
        if self.second_order is not None:
            if (
                not capabilities.second_order_force_constants
                or self.second_order.relation.capacity > request.maximum_ifc2_routes
            ):
                raise ValueError(
                    "Provider IFC2 capability or route capacity was exceeded."
                )
            if (
                self.second_order.system_id != request.system_id
                or self.second_order.cell.cell_id != request.cell_id
            ):
                raise ValueError(
                    "Provider IFC2 structure identity differs from the request."
                )
        if self.third_order is not None and (
            not capabilities.third_order_force_constants
            or int(self.third_order.atom_triplets.shape[0]) > request.maximum_ifc3_routes
            or self.third_order.system_id != request.system_id
        ):
            raise ValueError(
                "Provider IFC3 capability, structure, or route capacity is invalid."
            )
        if request.request_polar_tensors and not capabilities.born_effective_charges:
            raise ValueError("Provider did not declare polar tensor capability.")


class AbstractLatticeDynamicsProvider(abc.ABC):
    """Host-only lattice provider. Provider objects never enter numeric kernels."""

    provider_id: str
    provider_version: str
    provider_build_id: str
    capabilities: LatticeDynamicsProviderCapabilities

    @abc.abstractmethod
    def evaluate(self, request: LatticeDynamicsRequest, /) -> LatticeDynamicsArtifactSet:
        raise NotImplementedError


class CallableLatticeDynamicsProvider(AbstractLatticeDynamicsProvider):
    """Explicit adapter for a pinned host provider implementation."""

    def __init__(
        self,
        provider_id: str,
        provider_version: str,
        provider_build_id: str,
        capabilities: LatticeDynamicsProviderCapabilities,
        evaluator: Callable[[LatticeDynamicsRequest], LatticeDynamicsArtifactSet],
        /,
    ):
        if not isinstance(capabilities, LatticeDynamicsProviderCapabilities):
            raise TypeError("capabilities must be LatticeDynamicsProviderCapabilities.")
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        self.provider_id = _identifier(provider_id, "provider_id")
        self.provider_version = _identifier(provider_version, "provider_version")
        self.provider_build_id = _identifier(provider_build_id, "provider_build_id")
        self.capabilities = capabilities
        self._evaluator = evaluator

    def evaluate(self, request: LatticeDynamicsRequest, /) -> LatticeDynamicsArtifactSet:
        if not isinstance(request, LatticeDynamicsRequest):
            raise TypeError("request must be LatticeDynamicsRequest.")
        artifact = self._evaluator(request)
        if not isinstance(artifact, LatticeDynamicsArtifactSet):
            raise TypeError("Lattice provider returned a non-normalized artifact.")
        if (
            artifact.provider_id != self.provider_id
            or artifact.provider_version != self.provider_version
            or artifact.provider_build_id != self.provider_build_id
        ):
            raise ValueError("Lattice provider output identity differs from its adapter.")
        artifact.validate_request(request, self.capabilities)
        return artifact


__all__ = [
    "AbstractLatticeDynamicsProvider",
    "CallableLatticeDynamicsProvider",
    "LatticeDynamicsArtifactSet",
    "LatticeDynamicsProviderCapabilities",
    "LatticeDynamicsRequest",
]
