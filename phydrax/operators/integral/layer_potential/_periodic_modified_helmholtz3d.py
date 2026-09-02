#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ....discretization import PeriodicCell
from ....geometry import MeshRegion
from ._periodic_core3d import (
    _direct_screened_image_sum_host,
    _ewald_green_host,
    _prepare_periodic_scalar_dp0_3d,
    _reduced_bloch_wavevector,
    _require_periodic_cell_3d,
    PeriodicEwaldPolicy3D,
    PeriodicScalarDP0Operator3D,
)


def _validated_screening(screening: float) -> float:
    value = float(screening)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Modified Helmholtz screening must be finite and positive.")
    return value


def periodic_modified_helmholtz_green_3d(
    displacement: ArrayLike,
    cell: PeriodicCell,
    /,
    *,
    screening: float,
    bloch_wavevector: ArrayLike | None = None,
    policy: PeriodicEwaldPolicy3D | None = None,
) -> Array:
    r"""Evaluate the bounded 3D quasi-periodic Yukawa Ewald sum.

    The convention is ``G(r + A n) = exp(i alpha·A n) G(r)``. Near images
    declared by the policy are evaluated with ``exp(-screening*r)/(4*pi*r)``;
    the rest is a deterministic real/reciprocal Ewald complement. The finite
    cutoffs are convergence-controlled inputs, not a continuum certificate.
    """

    _require_periodic_cell_3d(cell)
    selected = PeriodicEwaldPolicy3D() if policy is None else policy
    if not isinstance(selected, PeriodicEwaldPolicy3D):
        raise TypeError("policy must be PeriodicEwaldPolicy3D or None.")
    screening_ = _validated_screening(screening)
    wavevector = _reduced_bloch_wavevector(cell, bloch_wavevector)
    evaluated = _ewald_green_host(
        displacement,
        cell,
        wavevector,
        selected,
        complex(screening_),
        subtract_central_laplace=False,
        remove_zero_mode=False,
    )
    return selected.precision.output(jnp.asarray(evaluated.value))


def direct_periodic_modified_helmholtz_image_sum_3d(
    displacement: ArrayLike,
    cell: PeriodicCell,
    /,
    *,
    screening: float,
    image_cutoff: int,
    bloch_wavevector: ArrayLike | None = None,
    max_image_count: int = 2_000_000,
) -> Array:
    """Deterministic resource-bounded cube of the convergent Yukawa image sum."""

    _require_periodic_cell_3d(cell)
    screening_ = _validated_screening(screening)
    wavevector = _reduced_bloch_wavevector(cell, bloch_wavevector)
    return jnp.asarray(
        _direct_screened_image_sum_host(
            displacement,
            cell,
            screening_,
            wavevector,
            image_cutoff,
            max_image_count,
        )
    )


def prepare_periodic_modified_helmholtz_single_layer_dp0_3d(
    region: MeshRegion,
    cell: PeriodicCell,
    /,
    *,
    screening: float,
    certified_fractional_clearance: float,
    bloch_wavevector: ArrayLike | None = None,
    policy: PeriodicEwaldPolicy3D | None = None,
    numeric_version: str = "0",
) -> PeriodicScalarDP0Operator3D:
    r"""Prepare a scalar 3D DP0 quasi-periodic modified-Helmholtz layer.

    Envelope: outward watertight polyhedral inclusions strictly inside one
    affine cell, scalar DP0 density, and ``(-Delta + screening**2)u = 0`` off
    the repeated boundary. The formulation is central singular Laplace
    Galerkin plus a Yukawa regular remainder, exact declared near images, and a
    smooth Ewald complement. Host preparation is SciPy/NumPy complex128 and
    numeric actions are fixed-shape JAX. The returned report records realized
    precision, allocations, central quadrature errors, and Ewald shell
    indicators. It does not certify continuum error and makes no Laplace,
    oscillatory Helmholtz, open-surface, or vector-PDE claim.
    """

    screening_ = _validated_screening(screening)
    return _prepare_periodic_scalar_dp0_3d(
        region,
        cell,
        family="modified-helmholtz",
        screening=complex(screening_),
        bloch_wavevector=bloch_wavevector,
        policy=policy,
        certified_fractional_clearance=certified_fractional_clearance,
        pde=f"(-Delta + ({screening_!r})^2) u = 0 off the repeated boundary",
        formulation=(
            "central singular Laplace DP0 Galerkin plus smooth Yukawa central "
            "remainder, exact declared near images, and deterministic "
            "real/reciprocal Ewald complement"
        ),
        gauge="unique massive quasi-periodic resolvent; no gauge freedom",
        non_goals=(
            "no continuum error certification",
            "no open or cell-touching surfaces",
            "no Laplace or oscillatory Helmholtz claim",
            "no vector or Maxwell claim",
            "no adaptive or unbounded image allocation",
        ),
        numeric_version=numeric_version,
    )


__all__ = [
    "direct_periodic_modified_helmholtz_image_sum_3d",
    "periodic_modified_helmholtz_green_3d",
    "prepare_periodic_modified_helmholtz_single_layer_dp0_3d",
]
