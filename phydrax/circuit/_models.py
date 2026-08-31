#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._ports import ElectricalWaveReference, ModalWaveReference, WavePort, WaveReference


class ScatteringResponse(StrictModule):
    """Frequency-batched scattering matrix with explicit output,input ordering."""

    matrix: Array
    references: tuple[WaveReference, ...]
    numeric_version: Array

    def __init__(
        self,
        matrix: ArrayLike,
        references: Sequence[WaveReference],
        numeric_version: ArrayLike = 0,
        /,
    ):
        value = jnp.asarray(matrix)
        if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
            raise ValueError(
                "Scattering matrix must end in equal output,input dimensions."
            )
        if not jnp.issubdtype(value.dtype, jnp.number):
            raise TypeError("Scattering matrix must be numeric.")
        value = value.astype(jnp.result_type(value, jnp.complex128))
        refs = tuple(references)
        if len(refs) != int(value.shape[-1]) or any(
            not isinstance(ref, (ElectricalWaveReference, ModalWaveReference))
            for ref in refs
        ):
            raise ValueError(
                "references must contain one typed reference per matrix port."
            )
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.ndim != 0:
            raise ValueError("numeric_version must be scalar.")
        version = eqx.error_if(
            version, version < 0, "numeric_version must be non-negative."
        )
        self.matrix = value
        self.references = refs
        self.numeric_version = version


class AbstractScatteringComponent(StrictModule):
    """Static port schema plus a differentiable scattering evaluation."""

    @property
    @abstractmethod
    def ports(self) -> tuple[WavePort, ...]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, angular_frequency: ArrayLike, /) -> ScatteringResponse:
        raise NotImplementedError


class MatrixScatteringComponent(AbstractScatteringComponent):
    """A fixed or already frequency-batched scattering matrix leaf."""

    matrix: Array
    _ports: tuple[WavePort, ...]
    numeric_version: Array
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        ports: Sequence[WavePort],
        /,
        *,
        numeric_version: ArrayLike = 0,
        component_id: str = "matrix-scattering-component",
    ):
        port_tuple = tuple(ports)
        response = ScatteringResponse(
            matrix,
            tuple(reference for port in port_tuple for reference in port.references),
            numeric_version,
        )
        if len({port.port_id for port in port_tuple}) != len(port_tuple):
            raise ValueError("Component port IDs must be unique.")
        identifier = str(component_id)
        if not identifier:
            raise ValueError("component_id must be non-empty.")
        self.matrix = response.matrix
        self._ports = port_tuple
        self.numeric_version = response.numeric_version
        self.component_id = identifier

    @property
    def ports(self) -> tuple[WavePort, ...]:
        return self._ports

    def evaluate(self, angular_frequency: ArrayLike, /) -> ScatteringResponse:
        omega = jnp.asarray(angular_frequency)
        if self.matrix.ndim == 2:
            matrix = jnp.broadcast_to(self.matrix, omega.shape + self.matrix.shape)
        elif self.matrix.shape[:-2] == omega.shape:
            matrix = self.matrix
        else:
            raise ValueError(
                "A batched fixed scattering matrix must have the angular-frequency batch shape."
            )
        return ScatteringResponse(
            matrix,
            tuple(reference for port in self._ports for reference in port.references),
            self.numeric_version,
        )


class ScatteringAudit(StrictModule):
    """Computed passivity and reciprocity evidence, never an author assertion."""

    minimum_passivity_eigenvalue: Array
    passivity_residual: Array
    reciprocity_residual: Array
    finite: Array
    passive: Array
    reciprocal: Array
    passivity_eligible: bool = eqx.field(static=True)
    reciprocity_eligible: bool = eqx.field(static=True)
    complete_matrix: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_passivity_eigenvalue: ArrayLike,
        passivity_residual: ArrayLike,
        reciprocity_residual: ArrayLike,
        finite: ArrayLike,
        passive: ArrayLike,
        reciprocal: ArrayLike,
        passivity_eligible: bool,
        reciprocity_eligible: bool,
        complete_matrix: bool,
    ):
        self.minimum_passivity_eigenvalue = jnp.asarray(minimum_passivity_eigenvalue)
        self.passivity_residual = jnp.asarray(passivity_residual)
        self.reciprocity_residual = jnp.asarray(reciprocity_residual)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.passive = jnp.asarray(passive, dtype=bool)
        self.reciprocal = jnp.asarray(reciprocal, dtype=bool)
        self.passivity_eligible = bool(passivity_eligible)
        self.reciprocity_eligible = bool(reciprocity_eligible)
        self.complete_matrix = bool(complete_matrix)


def _electrical_reciprocity_eligible(references: tuple[WaveReference, ...]) -> bool:
    if not all(
        isinstance(reference, ElectricalWaveReference) for reference in references
    ):
        return False
    return all(bool(jnp.all(jnp.imag(reference.z0) == 0.0)) for reference in references)


def audit_scattering(
    response: ScatteringResponse,
    /,
    *,
    tolerance: float = 1e-10,
    complete_matrix: bool = True,
) -> ScatteringAudit:
    """Audit power-wave passivity and only scientifically eligible reciprocity."""
    if not isinstance(response, ScatteringResponse):
        raise TypeError("response must be ScatteringResponse.")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    matrix = response.matrix
    count = matrix.shape[-1]
    identity = jnp.eye(count, dtype=matrix.dtype)
    defect = identity - jnp.swapaxes(jnp.conj(matrix), -1, -2) @ matrix
    eigenvalues = jnp.linalg.eigvalsh(defect)
    minimum = jnp.min(eigenvalues, axis=-1)
    passivity_residual = jnp.maximum(-minimum, 0.0)
    scale = jnp.maximum(jnp.linalg.norm(matrix, axis=(-2, -1)), 1.0)
    reciprocity_residual = (
        jnp.linalg.norm(matrix - jnp.swapaxes(matrix, -1, -2), axis=(-2, -1)) / scale
    )
    finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
    passivity_eligible = bool(complete_matrix)
    reciprocity_eligible = bool(complete_matrix) and _electrical_reciprocity_eligible(
        response.references
    )
    passive = (
        finite & (passivity_residual <= tolerance)
        if passivity_eligible
        else jnp.asarray(False)
    )
    reciprocal = (
        finite & (reciprocity_residual <= tolerance)
        if reciprocity_eligible
        else jnp.asarray(False)
    )
    return ScatteringAudit(
        minimum_passivity_eigenvalue=minimum,
        passivity_residual=passivity_residual,
        reciprocity_residual=reciprocity_residual,
        finite=finite,
        passive=passive,
        reciprocal=reciprocal,
        passivity_eligible=passivity_eligible,
        reciprocity_eligible=reciprocity_eligible,
        complete_matrix=complete_matrix,
    )


class CommonNodeJunction(AbstractScatteringComponent):
    """Ideal zero-delay common-voltage/KCL RF node for real positive references."""

    _ports: tuple[WavePort, ...]
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        references: Sequence[ElectricalWaveReference | ArrayLike],
        /,
        *,
        port_ids: Sequence[str] | None = None,
        component_id: str = "common-node-junction",
    ):
        refs = tuple(
            reference
            if isinstance(reference, ElectricalWaveReference)
            else ElectricalWaveReference(reference)
            for reference in references
        )
        if len(refs) < 2:
            raise ValueError("A common-node junction requires at least two ports.")
        if any(bool(jnp.any(jnp.imag(reference.z0) != 0.0)) for reference in refs):
            raise ValueError(
                "The RF common-node junction requires real reference impedances."
            )
        ids = (
            tuple(f"p{index + 1}" for index in range(len(refs)))
            if port_ids is None
            else tuple(str(port_id) for port_id in port_ids)
        )
        if (
            len(ids) != len(refs)
            or len(set(ids)) != len(ids)
            or any(not value for value in ids)
        ):
            raise ValueError("port_ids must be unique, non-empty, and match references.")
        identifier = str(component_id)
        if not identifier:
            raise ValueError("component_id must be non-empty.")
        self._ports = tuple(
            WavePort(port_id, reference)
            for port_id, reference in zip(ids, refs, strict=True)
        )
        self.component_id = identifier

    @property
    def ports(self) -> tuple[WavePort, ...]:
        return self._ports

    def evaluate(self, angular_frequency: ArrayLike, /) -> ScatteringResponse:
        omega = jnp.asarray(angular_frequency)
        values = []
        for port in self._ports:
            z0 = port.references[0].z0
            if z0.ndim == 0:
                z0 = jnp.broadcast_to(z0, omega.shape)
            elif z0.shape != omega.shape:
                raise ValueError(
                    "Junction reference batches must match angular_frequency."
                )
            values.append(z0)
        z0 = jnp.stack(values, axis=-1)
        weights = 1.0 / jnp.sqrt(jnp.real(z0))
        denominator = jnp.sum(weights**2, axis=-1)
        matrix = (
            2.0
            * weights[..., :, None]
            * weights[..., None, :]
            / denominator[..., None, None]
        )
        matrix = matrix - jnp.eye(len(self._ports), dtype=matrix.dtype)
        return ScatteringResponse(
            matrix, tuple(port.references[0] for port in self._ports), 0
        )


def rf_common_node_junction(
    references: Sequence[ElectricalWaveReference | ArrayLike],
    /,
    *,
    port_ids: Sequence[str] | None = None,
    component_id: str = "common-node-junction",
) -> CommonNodeJunction:
    """Construct the qualified ideal RF common-node component."""
    return CommonNodeJunction(references, port_ids=port_ids, component_id=component_id)


__all__ = [
    "AbstractScatteringComponent",
    "CommonNodeJunction",
    "MatrixScatteringComponent",
    "ScatteringAudit",
    "ScatteringResponse",
    "audit_scattering",
    "rf_common_node_junction",
]
