#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ....discretization import PeriodicCell
from ....geometry import MeshRegion
from ._periodic_core3d import (
    _ewald_green_host,
    _integer_cube,
    _prepare_periodic_scalar_dp0_3d,
    _reduced_bloch_wavevector,
    _require_periodic_cell_3d,
    periodic_reciprocal_vectors_3d,
    PeriodicEwaldPolicy3D,
    PeriodicScalarDP0Operator3D,
    PeriodicScalarResourceError,
)


class PeriodicHelmholtzWoodAnomalyError(ValueError):
    """Typed fail-closed rejection of a Wood mode or unguarded spectral tail."""

    def __init__(
        self,
        message: str,
        /,
        *,
        closest_mode_index: tuple[int, int, int],
        minimum_denominator: float,
        denominator_tolerance: float,
        unsearched_mode_lower_wavenumber: float,
    ):
        super().__init__(message)
        self.closest_mode_index = closest_mode_index
        self.minimum_denominator = float(minimum_denominator)
        self.denominator_tolerance = float(denominator_tolerance)
        self.unsearched_mode_lower_wavenumber = float(unsearched_mode_lower_wavenumber)


def _validated_wavenumber(wavenumber: float) -> float:
    value = float(wavenumber)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Helmholtz wavenumber must be finite and positive.")
    return value


def _guard_nonwood_modes(
    cell: PeriodicCell,
    wavenumber: float,
    bloch_wavevector: np.ndarray,
    policy: PeriodicEwaldPolicy3D,
) -> None:
    mode_count = (2 * policy.reciprocal_cutoff + 1) ** 3
    if mode_count * 128 > policy.max_preparation_workspace_bytes:
        raise PeriodicScalarResourceError(
            "Helmholtz Wood-mode guard exceeds max_preparation_workspace_bytes."
        )
    indices = _integer_cube(policy.reciprocal_cutoff)
    stored_reciprocal = np.asarray(periodic_reciprocal_vectors_3d(cell))
    reciprocal = stored_reciprocal.astype(float)
    modes = indices @ reciprocal + bloch_wavevector[None, :]
    mode_norm_squared = np.sum(modes * modes, axis=1)
    denominators = mode_norm_squared - wavenumber * wavenumber
    closest = int(np.argmin(np.abs(denominators)))
    minimum = float(abs(denominators[closest]))
    scale = max(
        1.0,
        wavenumber * wavenumber,
        float(mode_norm_squared[closest]),
    )
    relative_tolerance = max(
        policy.wood_tolerance,
        128.0 * np.finfo(stored_reciprocal.dtype).eps,
    )
    tolerance = relative_tolerance * scale
    singular_value = float(np.linalg.svd(reciprocal, compute_uv=False)[-1])
    outside_lower = max(
        0.0,
        singular_value * (policy.reciprocal_cutoff + 1)
        - float(np.linalg.norm(bloch_wavevector)),
    )
    mode_index = tuple(int(value) for value in indices[closest])
    if minimum <= tolerance:
        raise PeriodicHelmholtzWoodAnomalyError(
            "The quasi-periodic Helmholtz resolvent is at a retained Wood mode.",
            closest_mode_index=mode_index,
            minimum_denominator=minimum,
            denominator_tolerance=tolerance,
            unsearched_mode_lower_wavenumber=outside_lower,
        )
    if outside_lower <= wavenumber + relative_tolerance * max(1.0, wavenumber):
        raise PeriodicHelmholtzWoodAnomalyError(
            "The reciprocal cutoff cannot exclude a Wood mode in the "
            "unsearched spectral tail.",
            closest_mode_index=mode_index,
            minimum_denominator=minimum,
            denominator_tolerance=tolerance,
            unsearched_mode_lower_wavenumber=outside_lower,
        )


def periodic_helmholtz_green_3d(
    displacement: ArrayLike,
    cell: PeriodicCell,
    /,
    *,
    wavenumber: float,
    bloch_wavevector: ArrayLike | None = None,
    policy: PeriodicEwaldPolicy3D | None = None,
) -> Array:
    r"""Evaluate the guarded limiting-absorption 3D Helmholtz Ewald sum.

    The convention is ``G(r + A n) = exp(i alpha·A n) G(r)`` and the free-space
    central term is ``exp(i*k*r)/(4*pi*r)``. Preparation inspects every retained
    reciprocal denominator ``|B m + alpha|**2-k**2`` and proves the unsearched
    tail lies above ``k``; a pole or insufficient search envelope raises
    :class:`PeriodicHelmholtzWoodAnomalyError`. Finite Ewald shell error is not
    certified and no vector-wave claim is made.
    """

    _require_periodic_cell_3d(cell)
    selected = PeriodicEwaldPolicy3D() if policy is None else policy
    if not isinstance(selected, PeriodicEwaldPolicy3D):
        raise TypeError("policy must be PeriodicEwaldPolicy3D or None.")
    wavenumber_ = _validated_wavenumber(wavenumber)
    wavevector = _reduced_bloch_wavevector(cell, bloch_wavevector)
    _guard_nonwood_modes(cell, wavenumber_, wavevector, selected)
    evaluated = _ewald_green_host(
        displacement,
        cell,
        wavevector,
        selected,
        complex(0.0, -wavenumber_),
        subtract_central_laplace=False,
        remove_zero_mode=False,
    )
    return selected.precision.output(jnp.asarray(evaluated.value))


def prepare_periodic_helmholtz_single_layer_dp0_3d(
    region: MeshRegion,
    cell: PeriodicCell,
    /,
    *,
    wavenumber: float,
    certified_fractional_clearance: float,
    bloch_wavevector: ArrayLike | None = None,
    policy: PeriodicEwaldPolicy3D | None = None,
    numeric_version: str = "0",
) -> PeriodicScalarDP0Operator3D:
    r"""Prepare a non-Wood scalar 3D DP0 quasi-periodic Helmholtz layer.

    Envelope: outward watertight polyhedral inclusions strictly inside one
    affine cell, scalar DP0 density, and ``(-Delta-k**2)u=0`` off the repeated
    boundary under the limiting-absorption resolvent. The central singular
    Laplace DP0 Galerkin term is combined with the smooth outgoing central
    remainder, exact declared near images, and deterministic complex Ewald
    complement. Retained and unsearched reciprocal modes are guarded before
    allocation; anomalies fail with a typed error. The returned report states
    SciPy/NumPy host and fixed-shape JAX providers, realized precision,
    allocations, quadrature errors, and non-certified shell indicators. It
    makes no Wood-mode, continuum-certification, open-surface, or vector-wave
    claim.
    """

    _require_periodic_cell_3d(cell)
    selected = PeriodicEwaldPolicy3D() if policy is None else policy
    if not isinstance(selected, PeriodicEwaldPolicy3D):
        raise TypeError("policy must be PeriodicEwaldPolicy3D or None.")
    wavenumber_ = _validated_wavenumber(wavenumber)
    wavevector = _reduced_bloch_wavevector(cell, bloch_wavevector)
    _guard_nonwood_modes(cell, wavenumber_, wavevector, selected)
    return _prepare_periodic_scalar_dp0_3d(
        region,
        cell,
        family="helmholtz",
        screening=complex(0.0, -wavenumber_),
        bloch_wavevector=wavevector,
        policy=selected,
        certified_fractional_clearance=certified_fractional_clearance,
        pde=f"(-Delta - ({wavenumber_!r})^2) u = 0 off the repeated boundary",
        formulation=(
            "central singular Laplace DP0 Galerkin plus smooth outgoing central "
            "remainder, exact declared near images, and deterministic complex "
            "real/reciprocal Ewald complement"
        ),
        gauge="non-Wood limiting-absorption quasi-periodic scalar resolvent",
        non_goals=(
            "no continuum error certification",
            "no Wood-mode or resonant resolvent",
            "no open or cell-touching surfaces",
            "no vector Helmholtz, elasticity, or Maxwell claim",
            "no adaptive or unbounded image allocation",
        ),
        numeric_version=numeric_version,
    )


__all__ = [
    "PeriodicHelmholtzWoodAnomalyError",
    "periodic_helmholtz_green_3d",
    "prepare_periodic_helmholtz_single_layer_dp0_3d",
]
