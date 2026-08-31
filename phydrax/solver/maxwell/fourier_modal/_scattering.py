#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ....discretization.spectral import LatticeHarmonicDiscretization
from ....linalg import ArraySpace, DenseLinearOperator
from ._boundary_cascade import BoundaryRelation
from ._contracts import HomogeneousMaxwellPort
from ._factorization import _dense_solve


class HomogeneousPortModes(StrictModule):
    electric_matrix: Array
    magnetic_matrix: Array
    longitudinal_wavevector: Array
    flux_weights: Array
    propagating: Array
    evanescent: Array
    grazing: Array
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    port_id: str = eqx.field(static=True)


class PortScatteringDiagnostics(StrictModule):
    conversion_residual: Array
    finite: Array
    power_normalized: Array


class MaxwellPortScatteringOperator(StrictModule):
    """Two-port scattering map with outputs ordered as right-forward, left-backward."""

    s11: DenseLinearOperator
    s12: DenseLinearOperator
    s21: DenseLinearOperator
    s22: DenseLinearOperator
    left_modes: HomogeneousPortModes
    right_modes: HomogeneousPortModes
    diagnostics: PortScatteringDiagnostics

    @property
    def block_size(self) -> int:
        return int(self.s11.matrix.shape[-1])

    def matrix(self) -> Array:
        return jnp.block(
            [[self.s11.matrix, self.s12.matrix], [self.s21.matrix, self.s22.matrix]]
        )


def _homogeneous_scalar(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.ndim == 0:
        return array
    if array.shape != (3, 3):
        raise ValueError(f"Homogeneous port {name} must be scalar or one 3x3 tensor.")
    diagonal = jnp.diag(array)
    reference = diagonal[0]
    off_diagonal = array - jnp.eye(3, dtype=array.dtype) * reference
    scale = jnp.maximum(jnp.max(jnp.abs(array)), 1.0)
    tolerance = 100 * jnp.finfo(array.real.dtype).eps * scale
    return eqx.error_if(
        reference,
        (jnp.max(jnp.abs(diagonal - reference)) > tolerance)
        | (jnp.max(jnp.abs(off_diagonal)) > tolerance),
        f"Homogeneous port {name} must be isotropic.",
    )


def prepare_homogeneous_port_modes(
    port: HomogeneousMaxwellPort,
    lattice: LatticeHarmonicDiscretization,
    angular_frequency: ArrayLike,
    bloch_wavevector: ArrayLike,
    /,
    *,
    grazing_tolerance: float = 1e-10,
) -> HomogeneousPortModes:
    epsilon = _homogeneous_scalar(port.material.permittivity, "permittivity")
    mu = _homogeneous_scalar(port.material.permeability, "permeability")
    dtype = jnp.result_type(epsilon, mu, jnp.complex64)
    epsilon = jnp.asarray(epsilon, dtype=dtype)
    mu = jnp.asarray(mu, dtype=dtype)
    omega = jnp.asarray(angular_frequency, dtype=dtype)
    wavevectors = lattice.in_plane_wavevectors(bloch_wavevector).astype(dtype)
    kx = wavevectors[:, 0]
    ky = wavevectors[:, 1]
    transverse = jnp.sqrt(jnp.real(kx) ** 2 + jnp.real(ky) ** 2)
    normal = transverse <= grazing_tolerance
    safe_transverse = jnp.where(normal, 1.0, transverse)
    radial_x = jnp.where(normal, 1.0, kx / safe_transverse)
    radial_y = jnp.where(normal, 0.0, ky / safe_transverse)
    tangent_x = jnp.where(normal, 0.0, -ky / safe_transverse)
    tangent_y = jnp.where(normal, 1.0, kx / safe_transverse)

    argument = omega**2 * epsilon * mu - kx**2 - ky**2
    kz = jnp.sqrt(argument + 0.0j)
    kz = jnp.where(jnp.imag(kz) < 0.0, -kz, kz)
    kz = jnp.where(
        (jnp.abs(jnp.imag(kz)) <= grazing_tolerance) & (jnp.real(kz) < 0.0),
        -kz,
        kz,
    )
    safe_kz = jnp.where(jnp.abs(kz) <= grazing_tolerance, 1.0 + 0.0j, kz)
    te_factor = kz / (omega * mu)
    tm_factor = omega * epsilon / safe_kz
    count = lattice.harmonic_count
    size = 2 * count
    columns_te = 2 * jnp.arange(count)
    columns_tm = columns_te + 1
    rows_x = jnp.arange(count)
    rows_y = rows_x + count
    electric = jnp.zeros((size, size), dtype=dtype)
    magnetic = jnp.zeros((size, size), dtype=dtype)
    electric = electric.at[rows_x, columns_te].set(tangent_x)
    electric = electric.at[rows_y, columns_te].set(tangent_y)
    electric = electric.at[rows_x, columns_tm].set(radial_x)
    electric = electric.at[rows_y, columns_tm].set(radial_y)
    magnetic = magnetic.at[rows_x, columns_te].set(-te_factor * radial_x)
    magnetic = magnetic.at[rows_y, columns_te].set(-te_factor * radial_y)
    magnetic = magnetic.at[rows_x, columns_tm].set(tm_factor * tangent_x)
    magnetic = magnetic.at[rows_y, columns_tm].set(tm_factor * tangent_y)

    power_te = 0.5 * jnp.real(te_factor)
    power_tm = 0.5 * jnp.real(tm_factor)
    raw_power = jnp.stack((power_te, power_tm), axis=-1).reshape((-1,))
    grazing = jnp.repeat(jnp.abs(kz) <= grazing_tolerance, 2)
    propagating = (jnp.abs(jnp.imag(kz)) <= grazing_tolerance) & (
        jnp.real(kz) > grazing_tolerance
    )
    propagating_modes = jnp.repeat(propagating, 2)
    evanescent = jnp.repeat(jnp.imag(kz) > grazing_tolerance, 2)
    normalization = jnp.where(
        propagating_modes,
        jnp.sqrt(jnp.maximum(jnp.abs(raw_power), grazing_tolerance)),
        1.0,
    )
    electric = electric / normalization[None, :]
    magnetic = magnetic / normalization[None, :]
    flux_weights = raw_power / normalization**2
    mode_ids = tuple(
        f"{mode_id}:{polarization}"
        for mode_id in lattice.plan.layout.mode_ids
        for polarization in ("te", "tm")
    )
    return HomogeneousPortModes(
        electric,
        magnetic,
        kz,
        flux_weights,
        propagating_modes,
        evanescent,
        grazing,
        mode_ids,
        port_id=port.port_id,
    )


def _relative_residual(matrix: Array, solution: Array, rhs: Array) -> Array:
    residual = matrix @ solution - rhs
    denominator = jnp.maximum(jnp.sqrt(jnp.sum(jnp.abs(rhs) ** 2)), 1.0)
    return jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2)) / denominator


def boundary_to_scattering(
    relation: BoundaryRelation,
    left_modes: HomogeneousPortModes,
    right_modes: HomogeneousPortModes,
    /,
) -> MaxwellPortScatteringOperator:
    wl = left_modes.electric_matrix
    vl = left_modes.magnetic_matrix
    wr = right_modes.electric_matrix
    vr = right_modes.magnetic_matrix
    size = relation.tangential_size
    if wl.shape != (size, size) or wr.shape != (size, size):
        raise ValueError("Port modes and boundary relation have incompatible sizes.")
    left_matrix = jnp.block(
        [
            [wr - relation.b @ vr, -relation.a @ wl],
            [-relation.d @ vr, -vl - relation.c @ wl],
        ]
    )
    right_matrix = jnp.block(
        [
            [relation.a @ wl, -wr - relation.b @ vr],
            [relation.c @ wl - vl, -relation.d @ vr],
        ]
    )
    scattering = _dense_solve(left_matrix, right_matrix)
    s11 = scattering[:size, :size]
    s12 = scattering[:size, size:]
    s21 = scattering[size:, :size]
    s22 = scattering[size:, size:]
    space = ArraySpace((size,), dtype=scattering.dtype)
    diagnostics = PortScatteringDiagnostics(
        _relative_residual(left_matrix, scattering, right_matrix),
        jnp.all(jnp.isfinite(scattering)),
        jnp.all(
            jnp.where(
                left_modes.propagating,
                jnp.abs(jnp.abs(left_modes.flux_weights) - 1.0) <= 1e-6,
                True,
            )
        )
        & jnp.all(
            jnp.where(
                right_modes.propagating,
                jnp.abs(jnp.abs(right_modes.flux_weights) - 1.0) <= 1e-6,
                True,
            )
        ),
    )
    return MaxwellPortScatteringOperator(
        DenseLinearOperator(s11, source=space, target=space),
        DenseLinearOperator(s12, source=space, target=space),
        DenseLinearOperator(s21, source=space, target=space),
        DenseLinearOperator(s22, source=space, target=space),
        left_modes,
        right_modes,
        diagnostics,
    )


def shift_scattering_reference_planes(
    scattering: MaxwellPortScatteringOperator,
    left_distance: ArrayLike,
    right_distance: ArrayLike,
    /,
) -> MaxwellPortScatteringOperator:
    left_kz = jnp.repeat(scattering.left_modes.longitudinal_wavevector, 2)
    right_kz = jnp.repeat(scattering.right_modes.longitudinal_wavevector, 2)
    left_phase = jnp.exp(1j * left_kz * jnp.asarray(left_distance))
    right_phase = jnp.exp(1j * right_kz * jnp.asarray(right_distance))

    def phase_block(output_phase: Array, matrix: Array, input_phase: Array) -> Array:
        return output_phase[:, None] * matrix * input_phase[None, :]

    s11 = phase_block(right_phase, scattering.s11.matrix, left_phase)
    s12 = phase_block(right_phase, scattering.s12.matrix, right_phase)
    s21 = phase_block(left_phase, scattering.s21.matrix, left_phase)
    s22 = phase_block(left_phase, scattering.s22.matrix, right_phase)
    space = scattering.s11.source
    return MaxwellPortScatteringOperator(
        DenseLinearOperator(s11, source=space, target=space),
        DenseLinearOperator(s12, source=space, target=space),
        DenseLinearOperator(s21, source=space, target=space),
        DenseLinearOperator(s22, source=space, target=space),
        scattering.left_modes,
        scattering.right_modes,
        scattering.diagnostics,
    )


def redheffer_star_product(
    left: MaxwellPortScatteringOperator,
    right: MaxwellPortScatteringOperator,
    /,
) -> MaxwellPortScatteringOperator:
    if left.block_size != right.block_size:
        raise ValueError("Scattering operators must have equal connected-port sizes.")
    size = left.block_size
    identity = jnp.eye(size, dtype=left.s11.matrix.dtype)
    first_system = identity - left.s12.matrix @ right.s21.matrix
    first_rhs = jnp.concatenate(
        (left.s11.matrix, left.s12.matrix @ right.s22.matrix), axis=1
    )
    first = _dense_solve(first_system, first_rhs)
    first_left = first[:, :size]
    first_right = first[:, size:]
    second_system = identity - right.s21.matrix @ left.s12.matrix
    second_rhs = jnp.concatenate(
        (right.s21.matrix @ left.s11.matrix, right.s22.matrix), axis=1
    )
    second = _dense_solve(second_system, second_rhs)
    second_left = second[:, :size]
    second_right = second[:, size:]
    s11 = right.s11.matrix @ first_left
    s12 = right.s12.matrix + right.s11.matrix @ first_right
    s21 = left.s21.matrix + left.s22.matrix @ second_left
    s22 = left.s22.matrix @ second_right
    space = left.s11.source
    diagnostics = PortScatteringDiagnostics(
        jnp.maximum(
            left.diagnostics.conversion_residual, right.diagnostics.conversion_residual
        ),
        left.diagnostics.finite & right.diagnostics.finite,
        left.diagnostics.power_normalized & right.diagnostics.power_normalized,
    )
    return MaxwellPortScatteringOperator(
        DenseLinearOperator(s11, source=space, target=space),
        DenseLinearOperator(s12, source=space, target=space),
        DenseLinearOperator(s21, source=space, target=space),
        DenseLinearOperator(s22, source=space, target=space),
        left.left_modes,
        right.right_modes,
        diagnostics,
    )


__all__ = [
    "HomogeneousPortModes",
    "MaxwellPortScatteringOperator",
    "PortScatteringDiagnostics",
    "boundary_to_scattering",
    "prepare_homogeneous_port_modes",
    "redheffer_star_product",
    "shift_scattering_reference_planes",
]
