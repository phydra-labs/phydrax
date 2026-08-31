#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ...._strict import StrictModule
from ....discretization.spectral import LatticeHarmonicDiscretization
from ._factorization import _dense_solve, PreparedFourierMaterial


class LayerOperatorDiagnostics(StrictModule):
    constitutive_residual: Array
    reciprocity_residual: Array
    minimum_loss_diagonal: Array
    finite: Array


class PreparedLayerOperator(StrictModule):
    """First-order tangential Maxwell operator and longitudinal recovery maps."""

    matrix: Array
    electric_longitudinal_from_tangential: Array
    magnetic_longitudinal_from_tangential: Array
    permittivity: Array
    permeability: Array
    diagnostics: LayerOperatorDiagnostics

    @property
    def harmonic_count(self) -> int:
        return int(self.matrix.shape[-1] // 4)


def _relative_residual(matrix: Array, solution: Array, right_hand_side: Array) -> Array:
    residual = matrix @ solution - right_hand_side
    denominator = jnp.maximum(jnp.sqrt(jnp.sum(jnp.abs(right_hand_side) ** 2)), 1.0)
    return jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2)) / denominator


def prepare_layer_operator(
    material: PreparedFourierMaterial,
    lattice: LatticeHarmonicDiscretization,
    angular_frequency: Array,
    bloch_wavevector: Array,
    /,
) -> PreparedLayerOperator:
    """Assemble dψ/dz = Mψ for ψ = [Ex, Ey, Hx, Hy]."""
    omega = jnp.asarray(angular_frequency, dtype=material.permittivity.dtype)
    omega = eqx.error_if(
        omega,
        (~jnp.isfinite(omega)) | (jnp.abs(omega) <= jnp.finfo(omega.real.dtype).eps),
        "angular_frequency must be finite and nonzero.",
    )
    wavevectors = lattice.in_plane_wavevectors(bloch_wavevector).astype(
        material.permittivity.dtype
    )
    count = lattice.harmonic_count
    kx = jnp.diag(wavevectors[..., 0])
    ky = jnp.diag(wavevectors[..., 1])
    epsilon = material.permittivity
    mu = material.permeability

    epsilon_zz = epsilon[2, 2]
    mu_zz = mu[2, 2]
    electric_rhs = jnp.concatenate(
        (
            -epsilon[2, 0],
            -epsilon[2, 1],
            ky / omega,
            -kx / omega,
        ),
        axis=1,
    )
    magnetic_rhs = jnp.concatenate(
        (
            -ky / omega,
            kx / omega,
            -mu[2, 0],
            -mu[2, 1],
        ),
        axis=1,
    )
    electric_longitudinal = _dense_solve(epsilon_zz, electric_rhs)
    magnetic_longitudinal = _dense_solve(mu_zz, magnetic_rhs)
    ez = tuple(
        electric_longitudinal[:, index * count : (index + 1) * count]
        for index in range(4)
    )
    hz = tuple(
        magnetic_longitudinal[:, index * count : (index + 1) * count]
        for index in range(4)
    )

    rows: list[list[Array]] = [[], [], [], []]
    for column in range(4):
        mu_y = (
            mu[1, column - 2]
            if column >= 2
            else jnp.zeros((count, count), dtype=mu.dtype)
        )
        mu_x = (
            mu[0, column - 2]
            if column >= 2
            else jnp.zeros((count, count), dtype=mu.dtype)
        )
        epsilon_y = (
            epsilon[1, column]
            if column < 2
            else jnp.zeros((count, count), dtype=epsilon.dtype)
        )
        epsilon_x = (
            epsilon[0, column]
            if column < 2
            else jnp.zeros((count, count), dtype=epsilon.dtype)
        )
        rows[0].append(1j * kx @ ez[column] + 1j * omega * (mu_y + mu[1, 2] @ hz[column]))
        rows[1].append(1j * ky @ ez[column] - 1j * omega * (mu_x + mu[0, 2] @ hz[column]))
        rows[2].append(
            1j * kx @ hz[column] - 1j * omega * (epsilon_y + epsilon[1, 2] @ ez[column])
        )
        rows[3].append(
            1j * ky @ hz[column] + 1j * omega * (epsilon_x + epsilon[0, 2] @ ez[column])
        )
    matrix = jnp.block(rows)
    matrix = eqx.error_if(
        matrix,
        jnp.any(~jnp.isfinite(matrix)),
        "The Fourier-modal layer operator contains nonfinite values.",
    )
    constitutive_residual = jnp.maximum(
        _relative_residual(epsilon_zz, electric_longitudinal, electric_rhs),
        _relative_residual(mu_zz, magnetic_longitudinal, magnetic_rhs),
    )
    reciprocity_scale = jnp.maximum(
        jnp.sqrt(jnp.sum(jnp.abs(epsilon) ** 2) + jnp.sum(jnp.abs(mu) ** 2)),
        1.0,
    )
    reciprocity_residual = (
        jnp.sqrt(
            jnp.sum(jnp.abs(epsilon - jnp.swapaxes(epsilon, 0, 1)) ** 2)
            + jnp.sum(jnp.abs(mu - jnp.swapaxes(mu, 0, 1)) ** 2)
        )
        / reciprocity_scale
    )
    loss_diagonal = jnp.concatenate(
        tuple(jnp.imag(jnp.diag(epsilon[index, index])) for index in range(3))
        + tuple(jnp.imag(jnp.diag(mu[index, index])) for index in range(3))
    )
    diagnostics = LayerOperatorDiagnostics(
        constitutive_residual,
        reciprocity_residual,
        jnp.min(loss_diagonal),
        jnp.all(jnp.isfinite(matrix)),
    )
    return PreparedLayerOperator(
        matrix,
        electric_longitudinal,
        magnetic_longitudinal,
        epsilon,
        mu,
        diagnostics,
    )


def recover_longitudinal_fields(
    layer: PreparedLayerOperator,
    tangential_fields: Array,
    /,
) -> tuple[Array, Array]:
    """Recover Ez and Hz from tangential harmonic fields."""
    value = jnp.asarray(tangential_fields, dtype=layer.matrix.dtype)
    if value.shape[:1] != (4 * layer.harmonic_count,):
        raise ValueError("tangential_fields have an incompatible event dimension.")
    electric = layer.electric_longitudinal_from_tangential @ value
    magnetic = layer.magnetic_longitudinal_from_tangential @ value
    return electric, magnetic


__all__ = [
    "LayerOperatorDiagnostics",
    "PreparedLayerOperator",
    "prepare_layer_operator",
    "recover_longitudinal_fields",
]
