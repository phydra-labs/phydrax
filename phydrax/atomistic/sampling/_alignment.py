#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-correspondence optimal superposition through the quaternion spectrum.

The weighted mean-square deviation after the optimal proper rotation is

    MSD = (sum w |x_c|^2 + sum w |y_c|^2 - 2 lambda_max(K)) / sum w,

with ``K`` the symmetric 4x4 quaternion key matrix of the weighted
correlation ``R = sum w x_c y_c^T`` (Horn 1987; Coutsias et al. 2004).  The
largest eigenpair is computed by the native Hermitian spectrum owner.  Its
derivatives are defined only for a simple top eigenvalue: the JVP uses
``d lambda = q^T dK q`` and ``dq = (lambda I - K + q q^T)^{-1}(I - q q^T) dK q``,
which differentiates again for higher response and depends on the top gap
alone.  ``rmsd = sqrt(MSD)`` is not differentiable at zero; MSD is.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...ein import contract
from ...linalg import HermitianSpectrum, SmallLinearSolvePlan, solve_small_linear
from ._cv_geometry import CollectiveVariableStatus, status_flag


_SOLVE = SmallLinearSolvePlan(4, singular_tolerance=1e-14, maximum_condition=1e14)


def _key_matrix(correlation: Array, /) -> Array:
    r = correlation
    xx, xy, xz = r[0, 0], r[0, 1], r[0, 2]
    yx, yy, yz = r[1, 0], r[1, 1], r[1, 2]
    zx, zy, zz = r[2, 0], r[2, 1], r[2, 2]
    return jnp.stack(
        (
            jnp.stack((xx + yy + zz, yz - zy, zx - xz, xy - yx)),
            jnp.stack((yz - zy, xx - yy - zz, xy + yx, zx + xz)),
            jnp.stack((zx - xz, xy + yx, -xx + yy - zz, yz + zy)),
            jnp.stack((xy - yx, zx + xz, yz + zy, -xx - yy + zz)),
        )
    )


@jax.custom_jvp
def _top_eigenpair(matrix: Array, /) -> tuple[Array, Array]:
    spectrum = HermitianSpectrum(matrix)
    return spectrum.eigenvalues[-1], spectrum.eigenvectors[:, -1]


@_top_eigenpair.defjvp
def _top_eigenpair_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    (matrix,), (perturbation,) = primals, tangents
    value, vector = _top_eigenpair(matrix)
    symmetric = 0.5 * (perturbation + perturbation.T)
    action = symmetric @ vector
    dvalue = vector @ action
    shifted = value * jnp.eye(4, dtype=matrix.dtype) - matrix + jnp.outer(vector, vector)
    dvector = solve_small_linear(_SOLVE, shifted, action - dvalue * vector).value
    return (value, vector), (dvalue, dvector)


@jax.custom_jvp
def _top_eigenvalue(matrix: Array, /) -> Array:
    return _top_eigenpair(matrix)[0]


@_top_eigenvalue.defjvp
def _top_eigenvalue_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
    # First order needs only d lambda = q^T dK q (no eigenvector solve), so a
    # degenerate key (e.g. an empty history slot masked out of the energy)
    # cannot inject a singular solve into a gradient; higher order
    # differentiates ``q`` through the gap-dependent ``_top_eigenpair`` rule.
    (matrix,), (perturbation,) = primals, tangents
    value, vector = _top_eigenpair(matrix)
    symmetric = 0.5 * (perturbation + perturbation.T)
    return value, vector @ (symmetric @ vector)


def _quaternion_rotation(q: Array, /) -> Array:
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.stack(
        (
            jnp.stack(
                (w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y))
            ),
            jnp.stack(
                (2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x))
            ),
            jnp.stack(
                (2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z)
            ),
        )
    )


class AlignmentEvaluation(StrictModule):
    """Aligned MSD/RMSD with spectral evidence.

    ``eigengap`` is ``lambda_max - lambda_2`` of the key matrix and
    ``relative_gap`` its ratio to the spectral scale.  ``msd_regular`` admits
    MSD derivatives (simple top eigenvalue); ``rmsd_regular`` additionally
    requires a strictly positive MSD.  ``rotation`` (stopped) maps the
    centered coordinates onto the centered reference: ``x_c @ rotation.T``.
    """

    msd: Array
    rmsd: Array
    eigengap: Array
    relative_gap: Array
    rotation: Array
    successful: Array
    msd_regular: Array
    rmsd_regular: Array
    status: Array


def aligned_deviation(
    coordinates: ArrayLike,
    reference: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
    gap_tolerance: float = 1.0e-10,
    zero_tolerance: float = 1.0e-24,
) -> AlignmentEvaluation:
    """Optimal proper-rotation superposition of fixed corresponding points."""

    x = jnp.asarray(coordinates)
    y = jnp.asarray(reference, dtype=x.dtype)
    if x.ndim != 2 or x.shape[-1] != 3 or y.shape != x.shape or x.shape[0] < 3:
        raise ValueError("Alignment requires matching (n >= 3, 3) coordinates.")
    weight = (
        jnp.ones((x.shape[0],), dtype=x.dtype)
        if weights is None
        else jnp.asarray(weights, dtype=x.dtype)
    )
    if weight.shape != (x.shape[0],):
        raise ValueError("Alignment weights must align with the points.")
    total = jnp.sum(weight)
    x_c = x - jnp.sum(weight[:, None] * x, axis=0) / total
    y_c = y - jnp.sum(weight[:, None] * y, axis=0) / total
    correlation = contract("n,ni,nj->ij", weight, x_c, y_c)
    key = _key_matrix(correlation)
    largest = _top_eigenvalue(key)
    quaternion = _top_eigenpair(jax.lax.stop_gradient(key))[1]
    evidence = HermitianSpectrum(jax.lax.stop_gradient(key))
    values = evidence.eigenvalues
    gap = values[-1] - values[-2]
    scale = jnp.maximum(jnp.max(jnp.abs(values)), jnp.finfo(x.dtype).tiny)
    relative_gap = gap / scale
    inner = jnp.sum(weight * jnp.sum(x_c * x_c, axis=-1)) + jnp.sum(
        weight * jnp.sum(y_c * y_c, axis=-1)
    )
    raw = (inner - 2.0 * largest) / total
    msd = jnp.maximum(raw, 0.0)
    gap_ok = relative_gap > gap_tolerance
    positive = jax.lax.stop_gradient(msd) > zero_tolerance * jnp.maximum(
        jax.lax.stop_gradient(inner) / total, 1.0
    )
    rmsd = jnp.where(positive, jnp.sqrt(jnp.where(positive, msd, 1.0)), 0.0)
    finite = jnp.isfinite(msd) & evidence.valid & (total > 0.0)
    status = (
        status_flag(~gap_ok, CollectiveVariableStatus.ALIGNMENT_GAP)
        | status_flag(~positive, CollectiveVariableStatus.ZERO_DISTANCE)
        | status_flag(~finite, CollectiveVariableStatus.NONFINITE)
    )
    return AlignmentEvaluation(
        msd=msd,
        rmsd=rmsd,
        eigengap=gap,
        relative_gap=relative_gap,
        rotation=jax.lax.stop_gradient(_quaternion_rotation(quaternion)),
        successful=finite,
        msd_regular=finite & gap_ok,
        rmsd_regular=finite & gap_ok & positive,
        status=status,
    )


__all__ = ["AlignmentEvaluation", "aligned_deviation"]
