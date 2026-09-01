#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    solve,
)
from ._components import AdmittanceComponent, ImpedanceComponent
from ._mna import AbstractMNAComponent, MNAStamp
from ._models import AbstractScatteringComponent, ScatteringResponse
from ._ports import ElectricalWaveReference


class ConversionEvidence(StrictModule):
    """Original-equation residual and native solve status for one explicit conversion."""

    residual: Array
    relative_residual: Array
    linear_status: Array
    finite: Array
    source_representation: str = eqx.field(static=True)
    target_representation: str = eqx.field(static=True)


class NetworkConversion(StrictModule):
    matrix: Array
    evidence: ConversionEvidence
    references: tuple[ElectricalWaveReference, ...]


def _matrix(value: ArrayLike, name: str, /) -> Array:
    matrix = jnp.asarray(value)
    if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError(f"{name} must end in one square matrix.")
    return matrix.astype(jnp.result_type(matrix, jnp.complex128))


def _reference_arrays(
    references: Sequence[ElectricalWaveReference],
    count: int,
    batch: tuple[int, ...],
    /,
) -> tuple[tuple[ElectricalWaveReference, ...], Array, Array]:
    refs = tuple(references)
    if len(refs) != count or any(
        not isinstance(value, ElectricalWaveReference) for value in refs
    ):
        raise ValueError("One ElectricalWaveReference is required per matrix port.")
    values = []
    for reference in refs:
        z0 = reference.z0
        if z0.ndim == 0:
            z0 = jnp.broadcast_to(z0, batch)
        elif z0.shape != batch:
            raise ValueError("Reference impedance batch must match the matrix batch.")
        values.append(z0)
    z0 = jnp.stack(values, axis=-1)
    root = jnp.sqrt(jnp.real(z0))
    return refs, z0, root


def _right_solve(
    denominator: Array,
    numerator: Array,
    source: str,
    target: str,
    /,
) -> NetworkConversion:
    result = solve(
        LinearSystem(DenseLinearOperator(jnp.swapaxes(denominator, -1, -2))),
        jnp.swapaxes(numerator, -1, -2),
        policy=LinearSolvePolicy(DenseLU(), failure=FailurePolicy("status")),
    )
    matrix = jnp.swapaxes(jnp.asarray(result.value), -1, -2)
    defect = matrix @ denominator - numerator
    residual = jnp.linalg.norm(defect, axis=(-2, -1))
    scale = jnp.maximum(
        jnp.linalg.norm(numerator, axis=(-2, -1))
        + jnp.linalg.norm(matrix, axis=(-2, -1))
        * jnp.linalg.norm(denominator, axis=(-2, -1)),
        1.0,
    )
    evidence = ConversionEvidence(
        residual,
        residual / scale,
        jnp.asarray(result.status, dtype=jnp.int32),
        jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
        & jnp.all(result.status == int(LinearSolveStatus.SUCCESS)),
        source,
        target,
    )
    return NetworkConversion(matrix, evidence, ())


def admittance_to_scattering(
    admittance: ArrayLike,
    references: Sequence[ElectricalWaveReference],
    /,
) -> NetworkConversion:
    y = _matrix(admittance, "admittance")
    refs, z0, root = _reference_arrays(references, y.shape[-1], y.shape[:-2])
    identity = jnp.eye(y.shape[-1], dtype=y.dtype)
    a = (identity + z0[..., :, None] * y) / (2.0 * root[..., :, None])
    b = (identity - jnp.conj(z0)[..., :, None] * y) / (2.0 * root[..., :, None])
    converted = _right_solve(a, b, "Y", "S")
    return NetworkConversion(converted.matrix, converted.evidence, refs)


def impedance_to_scattering(
    impedance: ArrayLike,
    references: Sequence[ElectricalWaveReference],
    /,
) -> NetworkConversion:
    z = _matrix(impedance, "impedance")
    refs, z0, root = _reference_arrays(references, z.shape[-1], z.shape[:-2])
    identity = jnp.eye(z.shape[-1], dtype=z.dtype)
    f = identity / (2.0 * root[..., :, None])
    a = f @ (z + z0[..., :, None] * identity)
    b = f @ (z - jnp.conj(z0)[..., :, None] * identity)
    converted = _right_solve(a, b, "Z", "S")
    return NetworkConversion(converted.matrix, converted.evidence, refs)


def _scattering_voltage_current_maps(
    scattering: Array,
    references: Sequence[ElectricalWaveReference],
) -> tuple[tuple[ElectricalWaveReference, ...], Array, Array]:
    refs, z0, root = _reference_arrays(
        references, scattering.shape[-1], scattering.shape[:-2]
    )
    identity = jnp.eye(scattering.shape[-1], dtype=scattering.dtype)
    difference = identity - scattering
    current = difference / root[..., :, None]
    voltage = 2.0 * root[..., :, None] * identity - z0[..., :, None] * current
    return refs, voltage, current


def scattering_to_admittance(
    scattering: ArrayLike | ScatteringResponse,
    references: Sequence[ElectricalWaveReference] | None = None,
    /,
) -> NetworkConversion:
    if isinstance(scattering, ScatteringResponse):
        matrix = scattering.matrix
        refs = scattering.references if references is None else tuple(references)
    else:
        matrix = _matrix(scattering, "scattering")
        if references is None:
            raise ValueError("references are required for a raw scattering matrix.")
        refs = tuple(references)
    matrix = _matrix(matrix, "scattering")
    electrical, voltage, current = _scattering_voltage_current_maps(matrix, refs)
    converted = _right_solve(voltage, current, "S", "Y")
    return NetworkConversion(converted.matrix, converted.evidence, electrical)


def scattering_to_impedance(
    scattering: ArrayLike | ScatteringResponse,
    references: Sequence[ElectricalWaveReference] | None = None,
    /,
) -> NetworkConversion:
    if isinstance(scattering, ScatteringResponse):
        matrix = scattering.matrix
        refs = scattering.references if references is None else tuple(references)
    else:
        matrix = _matrix(scattering, "scattering")
        if references is None:
            raise ValueError("references are required for a raw scattering matrix.")
        refs = tuple(references)
    matrix = _matrix(matrix, "scattering")
    electrical, voltage, current = _scattering_voltage_current_maps(matrix, refs)
    converted = _right_solve(current, voltage, "S", "Z")
    return NetworkConversion(converted.matrix, converted.evidence, electrical)


def admittance_to_impedance(admittance: ArrayLike, /) -> NetworkConversion:
    y = _matrix(admittance, "admittance")
    identity = jnp.broadcast_to(jnp.eye(y.shape[-1], dtype=y.dtype), y.shape)
    return _right_solve(y, identity, "Y", "Z")


def impedance_to_admittance(impedance: ArrayLike, /) -> NetworkConversion:
    z = _matrix(impedance, "impedance")
    identity = jnp.broadcast_to(jnp.eye(z.shape[-1], dtype=z.dtype), z.shape)
    return _right_solve(z, identity, "Z", "Y")


def impedance_to_abcd(impedance: ArrayLike, /) -> NetworkConversion:
    z = _matrix(impedance, "impedance")
    if z.shape[-2:] != (2, 2):
        raise ValueError("ABCD conversion requires exactly two ports.")
    z21 = eqx.error_if(
        z[..., 1, 0], z[..., 1, 0] == 0.0, "Z21 is singular for ABCD conversion."
    )
    a = z[..., 0, 0] / z21
    b = z[..., 0, 0] * z[..., 1, 1] / z21 - z[..., 0, 1]
    c = 1.0 / z21
    d = z[..., 1, 1] / z21
    matrix = jnp.stack((jnp.stack((a, b), -1), jnp.stack((c, d), -1)), -2)
    reconstructed = abcd_to_impedance(matrix).matrix
    residual = jnp.linalg.norm(reconstructed - z, axis=(-2, -1))
    evidence = ConversionEvidence(
        residual,
        residual / jnp.maximum(jnp.linalg.norm(z, axis=(-2, -1)), 1.0),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.all(jnp.isfinite(matrix), axis=(-2, -1)),
        "Z",
        "ABCD",
    )
    return NetworkConversion(matrix, evidence, ())


def abcd_to_impedance(abcd: ArrayLike, /) -> NetworkConversion:
    matrix = _matrix(abcd, "abcd")
    if matrix.shape[-2:] != (2, 2):
        raise ValueError("ABCD conversion requires a 2x2 matrix.")
    a, b = matrix[..., 0, 0], matrix[..., 0, 1]
    c = eqx.error_if(
        matrix[..., 1, 0],
        matrix[..., 1, 0] == 0.0,
        "ABCD C is singular for Z conversion.",
    )
    d = matrix[..., 1, 1]
    z11 = a / c
    z12 = (a * d - b * c) / c
    z21 = 1.0 / c
    z22 = d / c
    z = jnp.stack((jnp.stack((z11, z12), -1), jnp.stack((z21, z22), -1)), -2)
    residual = jnp.zeros(z.shape[:-2], dtype=jnp.real(z).dtype)
    evidence = ConversionEvidence(
        residual,
        residual,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.all(jnp.isfinite(z), axis=(-2, -1)),
        "ABCD",
        "Z",
    )
    return NetworkConversion(z, evidence, ())


def scattering_to_abcd(
    scattering: ArrayLike | ScatteringResponse,
    references: Sequence[ElectricalWaveReference] | None = None,
    /,
) -> NetworkConversion:
    z = scattering_to_impedance(scattering, references)
    converted = impedance_to_abcd(z.matrix)
    return NetworkConversion(converted.matrix, converted.evidence, z.references)


def abcd_to_scattering(
    abcd: ArrayLike,
    references: Sequence[ElectricalWaveReference],
    /,
) -> NetworkConversion:
    z = abcd_to_impedance(abcd)
    return impedance_to_scattering(z.matrix, references)


def renormalize_scattering(
    response: ScatteringResponse,
    references: Sequence[ElectricalWaveReference],
    /,
) -> NetworkConversion:
    """Explicitly change electrical wave references through the physical Z map."""
    if not isinstance(response, ScatteringResponse):
        raise TypeError("response must be ScatteringResponse.")
    impedance = scattering_to_impedance(response)
    converted = impedance_to_scattering(impedance.matrix, references)
    residual = impedance.evidence.relative_residual + converted.evidence.relative_residual
    evidence = ConversionEvidence(
        impedance.evidence.residual + converted.evidence.residual,
        residual,
        jnp.maximum(impedance.evidence.linear_status, converted.evidence.linear_status),
        impedance.evidence.finite & converted.evidence.finite,
        "S",
        "renormalized-S",
    )
    return NetworkConversion(converted.matrix, evidence, converted.references)


class ScatteringMNAComponent(AbstractMNAComponent):
    """Exact Kurokawa scattering-to-MNA lowering with incident-wave auxiliaries."""

    component: AbstractScatteringComponent
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        component: AbstractScatteringComponent,
        /,
        *,
        component_id: str = "scattering-to-mna",
    ):
        if not isinstance(component, AbstractScatteringComponent):
            raise TypeError("component must be AbstractScatteringComponent.")
        if not all(
            isinstance(reference, ElectricalWaveReference)
            for port in component.ports
            for reference in port.references
        ):
            raise ValueError("Only electrical scattering channels can be lowered to MNA.")
        identifier = str(component_id)
        if not identifier:
            raise ValueError("component_id must be non-empty.")
        self.component = component
        self.component_id = identifier

    @property
    def terminal_count(self) -> int:
        return sum(port.size for port in self.component.ports)

    @property
    def auxiliary_count(self) -> int:
        return self.terminal_count

    def evaluate(self, angular_frequency: ArrayLike, /) -> MNAStamp:
        omega = jnp.asarray(angular_frequency)
        response = self.component.evaluate(omega)
        count = self.terminal_count
        refs, z0, root = _reference_arrays(response.references, count, omega.shape)
        identity = jnp.broadcast_to(
            jnp.eye(count, dtype=response.matrix.dtype), omega.shape + (count, count)
        )
        inverse_z0 = 1.0 / z0
        q = jnp.conj(z0) * inverse_z0
        y = -inverse_z0[..., :, None] * identity
        b = (2.0 * root * inverse_z0)[..., :, None] * identity
        c = (1.0 + q)[..., :, None] * identity
        d = -2.0 * (
            (q * root)[..., :, None] * identity + root[..., :, None] * response.matrix
        )
        return MNAStamp(y, b, c, d)


def admittance_to_mna(
    admittance: ArrayLike,
    /,
    *,
    component_id: str = "admittance-to-mna",
) -> AdmittanceComponent:
    return AdmittanceComponent(admittance, component_id=component_id)


def impedance_to_mna(
    impedance: ArrayLike,
    /,
    *,
    component_id: str = "impedance-to-mna",
) -> ImpedanceComponent:
    return ImpedanceComponent(impedance, component_id=component_id)


def scattering_to_mna(
    component: AbstractScatteringComponent,
    /,
    *,
    component_id: str = "scattering-to-mna",
) -> ScatteringMNAComponent:
    return ScatteringMNAComponent(component, component_id=component_id)


__all__ = [
    "ConversionEvidence",
    "NetworkConversion",
    "ScatteringMNAComponent",
    "abcd_to_impedance",
    "abcd_to_scattering",
    "admittance_to_impedance",
    "admittance_to_mna",
    "admittance_to_scattering",
    "impedance_to_abcd",
    "impedance_to_admittance",
    "impedance_to_mna",
    "impedance_to_scattering",
    "renormalize_scattering",
    "scattering_to_abcd",
    "scattering_to_admittance",
    "scattering_to_impedance",
    "scattering_to_mna",
]
