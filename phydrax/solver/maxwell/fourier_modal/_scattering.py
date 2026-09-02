#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._strict import StrictModule
from ....discretization.spectral import LatticeHarmonicDiscretization
from ....linalg import ArraySpace, DenseLinearOperator, FailurePolicy
from ....linalg.eigen import (
    DenseSchurQZ,
    general_eigensolve,
    GeneralEigenproblem,
    GeneralEigenSelection,
    GeneralEigenSolvePolicy,
)
from ._boundary_cascade import BoundaryRelation
from ._contracts import (
    AbstractFourierModalPort,
    HomogeneousMaxwellPort,
    PeriodicMaxwellPort,
)
from ._factorization import _dense_solve, prepare_fourier_material
from ._layer import prepare_layer_operator


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


class PreparedPeriodicPortModes(StrictModule):
    """Separated incoming/outgoing invariant bases for a periodic exterior."""

    incoming_electric_matrix: Array
    incoming_magnetic_matrix: Array
    outgoing_electric_matrix: Array
    outgoing_magnetic_matrix: Array
    incoming_exponents: Array
    outgoing_exponents: Array
    outward_flux: Array
    propagating: Array
    evanescent: Array
    grazing: Array
    spectral_separation: Array
    conversion_residual: Array
    passivity_residual: Array
    reciprocity_residual: Array
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    port_id: str = eqx.field(static=True)

    @property
    def electric_matrix(self) -> Array:
        return self.outgoing_electric_matrix

    @property
    def magnetic_matrix(self) -> Array:
        return self.outgoing_magnetic_matrix

    @property
    def longitudinal_wavevector(self) -> Array:
        return -1j * self.outgoing_exponents

    @property
    def flux_weights(self) -> Array:
        return self.outward_flux


PreparedFourierModalPortModes = HomogeneousPortModes | PreparedPeriodicPortModes


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
    left_modes: PreparedFourierModalPortModes
    right_modes: PreparedFourierModalPortModes
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
    xi = _homogeneous_scalar(port.material.magnetoelectric_xi, "magnetoelectric_xi")
    zeta = _homogeneous_scalar(port.material.magnetoelectric_zeta, "magnetoelectric_zeta")
    epsilon = eqx.error_if(
        epsilon,
        (jnp.abs(xi) > grazing_tolerance) | (jnp.abs(zeta) > grazing_tolerance),
        "Bianisotropic exteriors require PeriodicMaxwellPort.",
    )
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


def _modal_flux(vectors: Array, harmonic_count: int, /) -> Array:
    ex = vectors[:harmonic_count]
    ey = vectors[harmonic_count : 2 * harmonic_count]
    hx = vectors[2 * harmonic_count : 3 * harmonic_count]
    hy = vectors[3 * harmonic_count :]
    return 0.5 * jnp.real(jnp.sum(jnp.conj(ex) * hy - jnp.conj(ey) * hx, axis=0))


def prepare_periodic_port_modes(
    port: PeriodicMaxwellPort,
    lattice: LatticeHarmonicDiscretization,
    angular_frequency: ArrayLike,
    bloch_wavevector: ArrayLike,
    /,
    *,
    outward_sign: int,
    separation_tolerance: float = 1.0e-8,
) -> PreparedPeriodicPortModes:
    """Prepare flux/decay-separated Schur-QZ exterior invariant bases."""

    if outward_sign not in (-1, 1):
        raise ValueError("outward_sign must be -1 for left or +1 for right.")
    material = prepare_fourier_material(port.material, lattice, port.factorization)
    layer = prepare_layer_operator(
        material, lattice, jnp.asarray(angular_frequency), jnp.asarray(bloch_wavevector)
    )
    matrix = (
        jax.lax.stop_gradient(layer.matrix)
        if port.mode_policy == "frozen"
        else layer.matrix
    )
    result = general_eigensolve(
        GeneralEigenproblem(DenseLinearOperator(matrix)),
        policy=GeneralEigenSolvePolicy(
            DenseSchurQZ(),
            selection=GeneralEigenSelection.all(),
            failure=FailurePolicy("status"),
        ),
    )
    exponents = result.eigenvalues
    vectors = result.right_eigenvector_coordinates
    count = lattice.harmonic_count
    size = 2 * count
    flux = _modal_flux(vectors, count)
    signed_flux = outward_sign * flux
    decay = jnp.real(outward_sign * exponents)
    evanescent_all = jnp.abs(jnp.real(exponents)) > separation_tolerance
    propagating_all = ~evanescent_all & (jnp.abs(signed_flux) > separation_tolerance)
    outgoing = (evanescent_all & (decay < -separation_tolerance)) | (
        propagating_all & (signed_flux > separation_tolerance)
    )
    exponents = eqx.error_if(
        exponents,
        jnp.sum(outgoing, dtype=jnp.int32) != size,
        "Periodic port Schur subspaces are not dimensionally separated.",
    )
    outgoing_indices = jnp.argsort(jnp.where(outgoing, 0, 1))[:size]
    incoming_indices = jnp.argsort(jnp.where(outgoing, 1, 0))[:size]
    outgoing_vectors = vectors[:, outgoing_indices]
    incoming_vectors = vectors[:, incoming_indices]
    outgoing_flux = signed_flux[outgoing_indices]
    propagating = propagating_all[outgoing_indices]
    evanescent = evanescent_all[outgoing_indices]
    normalization = jnp.where(
        propagating,
        jnp.sqrt(jnp.maximum(jnp.abs(outgoing_flux), separation_tolerance)),
        1.0,
    )
    outgoing_vectors = outgoing_vectors / normalization[None, :]
    outgoing_flux = outgoing_flux / normalization**2
    incoming_normalization = jnp.where(
        propagating_all[incoming_indices],
        jnp.sqrt(
            jnp.maximum(jnp.abs(signed_flux[incoming_indices]), separation_tolerance)
        ),
        1.0,
    )
    incoming_vectors = incoming_vectors / incoming_normalization[None, :]
    separation = jnp.min(
        jnp.abs(exponents[outgoing_indices, None] - exponents[incoming_indices][None, :])
    )
    separation = eqx.error_if(
        separation,
        (port.mode_policy == "spectral-subspace") & (separation <= separation_tolerance),
        "Periodic port spectral-subspace derivative lacks a gap certificate.",
    )
    return PreparedPeriodicPortModes(
        incoming_vectors[:size],
        incoming_vectors[size:],
        outgoing_vectors[:size],
        outgoing_vectors[size:],
        exponents[incoming_indices],
        exponents[outgoing_indices],
        outgoing_flux,
        propagating,
        evanescent,
        ~(propagating | evanescent),
        separation,
        jnp.max(result.diagnostics.right_relative_residuals),
        jnp.maximum(-layer.diagnostics.minimum_loss_diagonal, 0.0),
        layer.diagnostics.reciprocity_residual,
        tuple(f"{port.port_id}:mode:{index}" for index in range(size)),
        port.port_id,
    )


def prepare_fourier_modal_port_modes(
    port: AbstractFourierModalPort,
    lattice: LatticeHarmonicDiscretization,
    angular_frequency: ArrayLike,
    bloch_wavevector: ArrayLike,
    /,
    *,
    outward_sign: int,
) -> PreparedFourierModalPortModes:
    if isinstance(port, HomogeneousMaxwellPort):
        return prepare_homogeneous_port_modes(
            port, lattice, angular_frequency, bloch_wavevector
        )
    if isinstance(port, PeriodicMaxwellPort):
        return prepare_periodic_port_modes(
            port,
            lattice,
            angular_frequency,
            bloch_wavevector,
            outward_sign=outward_sign,
        )
    raise TypeError("Unknown Fourier-modal port type.")


def _port_bases(
    modes: PreparedFourierModalPortModes, side: str, /
) -> tuple[Array, Array, Array, Array]:
    if isinstance(modes, PreparedPeriodicPortModes):
        return (
            modes.incoming_electric_matrix,
            modes.incoming_magnetic_matrix,
            modes.outgoing_electric_matrix,
            modes.outgoing_magnetic_matrix,
        )
    sign = 1.0 if side == "left" else -1.0
    return (
        modes.electric_matrix,
        sign * modes.magnetic_matrix,
        modes.electric_matrix,
        -sign * modes.magnetic_matrix,
    )


def _relative_residual(matrix: Array, solution: Array, rhs: Array) -> Array:
    residual = contract("ij,jk->ik", matrix, solution) - rhs
    denominator = jnp.maximum(jnp.sqrt(jnp.sum(jnp.abs(rhs) ** 2)), 1.0)
    return jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2)) / denominator


def boundary_to_scattering(
    relation: BoundaryRelation,
    left_modes: PreparedFourierModalPortModes,
    right_modes: PreparedFourierModalPortModes,
    /,
) -> MaxwellPortScatteringOperator:
    lin_e, lin_h, lout_e, lout_h = _port_bases(left_modes, "left")
    rin_e, rin_h, rout_e, rout_h = _port_bases(right_modes, "right")
    size = relation.tangential_size
    if lin_e.shape != (size, size) or rin_e.shape != (size, size):
        raise ValueError("Port modes and boundary relation have incompatible sizes.")

    def mm(left: Array, right: Array) -> Array:
        return contract("ij,jk->ik", left, right)

    left_matrix = jnp.block(
        [
            [rout_e - mm(relation.b, rout_h), -mm(relation.a, lout_e)],
            [-mm(relation.d, rout_h), lout_h - mm(relation.c, lout_e)],
        ]
    )
    right_matrix = jnp.block(
        [
            [mm(relation.a, lin_e), -rin_e + mm(relation.b, rin_h)],
            [mm(relation.c, lin_e) - lin_h, mm(relation.d, rin_h)],
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


def _reference_phases(
    modes: PreparedFourierModalPortModes,
    distance: ArrayLike,
    side: str,
    /,
) -> tuple[Array, Array]:
    value = jnp.asarray(distance)
    if isinstance(modes, PreparedPeriodicPortModes):
        sign = -1.0 if side == "left" else 1.0
        return (
            jnp.exp(sign * modes.incoming_exponents * value),
            jnp.exp(sign * modes.outgoing_exponents * value),
        )
    wavevector = jnp.repeat(modes.longitudinal_wavevector, 2)
    phase = jnp.exp(1j * wavevector * value)
    return phase, phase


def shift_scattering_reference_planes(
    scattering: MaxwellPortScatteringOperator,
    left_distance: ArrayLike,
    right_distance: ArrayLike,
    /,
) -> MaxwellPortScatteringOperator:
    left_in_phase, left_out_phase = _reference_phases(
        scattering.left_modes, left_distance, "left"
    )
    right_in_phase, right_out_phase = _reference_phases(
        scattering.right_modes, right_distance, "right"
    )

    def phase_block(output_phase: Array, matrix: Array, input_phase: Array) -> Array:
        return output_phase[:, None] * matrix * input_phase[None, :]

    s11 = phase_block(right_out_phase, scattering.s11.matrix, left_in_phase)
    s12 = phase_block(right_out_phase, scattering.s12.matrix, right_in_phase)
    s21 = phase_block(left_out_phase, scattering.s21.matrix, left_in_phase)
    s22 = phase_block(left_out_phase, scattering.s22.matrix, right_in_phase)
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
    "PreparedFourierModalPortModes",
    "PreparedPeriodicPortModes",
    "PortScatteringDiagnostics",
    "boundary_to_scattering",
    "prepare_homogeneous_port_modes",
    "prepare_fourier_modal_port_modes",
    "prepare_periodic_port_modes",
    "redheffer_star_product",
    "shift_scattering_reference_planes",
]
