#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from opt_einsum import contract

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
    magnetoelectric_xi: Array
    magnetoelectric_zeta: Array
    diagnostics: LayerOperatorDiagnostics

    @property
    def harmonic_count(self) -> int:
        return int(self.matrix.shape[-1] // 4)


def _relative_residual(matrix: Array, solution: Array, right_hand_side: Array) -> Array:
    residual = contract("ij,jk->ik", matrix, solution) - right_hand_side
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
    xi = material.magnetoelectric_xi
    zeta = material.magnetoelectric_zeta
    zero = jnp.zeros((count, count), dtype=epsilon.dtype)

    longitudinal_matrix = jnp.block([[epsilon[2, 2], xi[2, 2]], [zeta[2, 2], mu[2, 2]]])
    longitudinal_rhs = jnp.block(
        [
            [
                -epsilon[2, 0],
                -epsilon[2, 1],
                ky / omega - xi[2, 0],
                -kx / omega - xi[2, 1],
            ],
            [
                -ky / omega - zeta[2, 0],
                kx / omega - zeta[2, 1],
                -mu[2, 0],
                -mu[2, 1],
            ],
        ]
    )
    longitudinal = _dense_solve(longitudinal_matrix, longitudinal_rhs)
    electric_longitudinal = longitudinal[:count]
    magnetic_longitudinal = longitudinal[count:]
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
        electric_column = column if column < 2 else None
        magnetic_column = column - 2 if column >= 2 else None
        b_y = (
            (zeta[1, electric_column] if electric_column is not None else zero)
            + (mu[1, magnetic_column] if magnetic_column is not None else zero)
            + contract("ij,jk->ik", zeta[1, 2], ez[column])
            + contract("ij,jk->ik", mu[1, 2], hz[column])
        )
        b_x = (
            (zeta[0, electric_column] if electric_column is not None else zero)
            + (mu[0, magnetic_column] if magnetic_column is not None else zero)
            + contract("ij,jk->ik", zeta[0, 2], ez[column])
            + contract("ij,jk->ik", mu[0, 2], hz[column])
        )
        d_y = (
            (epsilon[1, electric_column] if electric_column is not None else zero)
            + (xi[1, magnetic_column] if magnetic_column is not None else zero)
            + contract("ij,jk->ik", epsilon[1, 2], ez[column])
            + contract("ij,jk->ik", xi[1, 2], hz[column])
        )
        d_x = (
            (epsilon[0, electric_column] if electric_column is not None else zero)
            + (xi[0, magnetic_column] if magnetic_column is not None else zero)
            + contract("ij,jk->ik", epsilon[0, 2], ez[column])
            + contract("ij,jk->ik", xi[0, 2], hz[column])
        )
        rows[0].append(1j * contract("ij,jk->ik", kx, ez[column]) + 1j * omega * b_y)
        rows[1].append(1j * contract("ij,jk->ik", ky, ez[column]) - 1j * omega * b_x)
        rows[2].append(1j * contract("ij,jk->ik", kx, hz[column]) - 1j * omega * d_y)
        rows[3].append(1j * contract("ij,jk->ik", ky, hz[column]) + 1j * omega * d_x)
    matrix = jnp.block(rows)
    matrix = eqx.error_if(
        matrix,
        jnp.any(~jnp.isfinite(matrix)),
        "The Fourier-modal layer operator contains nonfinite values.",
    )
    constitutive_residual = _relative_residual(
        longitudinal_matrix, longitudinal, longitudinal_rhs
    )
    reciprocity_scale = jnp.maximum(
        jnp.sqrt(
            jnp.sum(jnp.abs(epsilon) ** 2)
            + jnp.sum(jnp.abs(mu) ** 2)
            + jnp.sum(jnp.abs(xi) ** 2)
            + jnp.sum(jnp.abs(zeta) ** 2)
        ),
        1.0,
    )
    reciprocity_residual = (
        jnp.sqrt(
            jnp.sum(jnp.abs(epsilon - jnp.swapaxes(epsilon, 0, 1)) ** 2)
            + jnp.sum(jnp.abs(mu - jnp.swapaxes(mu, 0, 1)) ** 2)
            + jnp.sum(jnp.abs(xi + jnp.swapaxes(zeta, 0, 1)) ** 2)
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
        xi,
        zeta,
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
    electric = contract(
        "ij,j...->i...", layer.electric_longitudinal_from_tangential, value
    )
    magnetic = contract(
        "ij,j...->i...", layer.magnetic_longitudinal_from_tangential, value
    )
    return electric, magnetic


__all__ = [
    "LayerOperatorDiagnostics",
    "PreparedLayerOperator",
    "prepare_layer_operator",
    "recover_longitudinal_fields",
]
