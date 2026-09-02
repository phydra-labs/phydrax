#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ....discretization import PeriodicCell
from ....geometry import MeshRegion
from ._periodic_core3d import (
    _ewald_green_host,
    _laplace_zero_bloch,
    _prepare_periodic_scalar_dp0_3d,
    _reduced_bloch_wavevector,
    _require_periodic_cell_3d,
    PeriodicEwaldPolicy3D,
    PeriodicScalarCompatibilityError,
    PeriodicScalarDP0Operator3D,
)


PeriodicLaplaceNeutralityError = PeriodicScalarCompatibilityError


def periodic_laplace_green_3d(
    displacement: ArrayLike,
    cell: PeriodicCell,
    /,
    *,
    bloch_wavevector: ArrayLike | None = None,
    policy: PeriodicEwaldPolicy3D | None = None,
) -> Array:
    r"""Evaluate the 3D quasi-periodic Laplace Ewald Green function.

    At zero Bloch vector the reciprocal zero mode is removed and the constant
    is fixed by the zero-cell-mean periodic Green-function gauge. This scalar
    kernel is physically applicable only to neutral source combinations. At a
    nonzero Bloch vector the resolvent has no constant mode. Finite Ewald
    cutoffs remain reported convergence evidence rather than certified error.
    """

    _require_periodic_cell_3d(cell)
    selected = PeriodicEwaldPolicy3D() if policy is None else policy
    if not isinstance(selected, PeriodicEwaldPolicy3D):
        raise TypeError("policy must be PeriodicEwaldPolicy3D or None.")
    wavevector = _reduced_bloch_wavevector(cell, bloch_wavevector)
    evaluated = _ewald_green_host(
        displacement,
        cell,
        wavevector,
        selected,
        0.0j,
        subtract_central_laplace=False,
        remove_zero_mode=_laplace_zero_bloch(cell, wavevector),
    )
    return selected.precision.output(jnp.asarray(evaluated.value))


def prepare_periodic_laplace_single_layer_dp0_3d(
    region: MeshRegion,
    cell: PeriodicCell,
    /,
    *,
    certified_fractional_clearance: float,
    bloch_wavevector: ArrayLike | None = None,
    policy: PeriodicEwaldPolicy3D | None = None,
    numeric_version: str = "0",
) -> PeriodicScalarDP0Operator3D:
    r"""Prepare a scalar 3D DP0 quasi-periodic Laplace single layer.

    Envelope: outward watertight polyhedral inclusions strictly inside one
    affine cell and ``-Delta u = 0`` off the repeated boundary. The central
    singular interaction uses existing DP0 Galerkin; exact declared near images
    and a smooth deterministic Ewald complement provide periodicity. For zero
    Bloch phase, ``mv`` rejects non-neutral DP0 densities and uses the
    zero-reciprocal-mode/zero-cell-mean Green-function gauge. At nonzero Bloch
    phase no neutrality condition is needed. SciPy/NumPy host preparation and
    fixed-shape JAX actions are reported with precision, resources, central
    quadrature errors, and shell indicators. There is no continuum certificate,
    open-surface support, oscillatory/massive claim, or vector-PDE claim.
    """

    _require_periodic_cell_3d(cell)
    wavevector = _reduced_bloch_wavevector(cell, bloch_wavevector)
    zero_bloch = _laplace_zero_bloch(cell, wavevector)
    return _prepare_periodic_scalar_dp0_3d(
        region,
        cell,
        family="laplace",
        screening=0.0j,
        bloch_wavevector=wavevector,
        policy=policy,
        certified_fractional_clearance=certified_fractional_clearance,
        pde="-Delta u = 0 off the repeated boundary",
        formulation=(
            "central singular Laplace DP0 Galerkin, exact declared near images, "
            "and deterministic real/reciprocal Ewald complement"
        ),
        gauge=(
            "zero reciprocal mode removed; zero-cell-mean periodic Green "
            "function; neutral DP0 sources required"
            if zero_bloch
            else "nonzero-Bloch scalar resolvent; no constant-mode gauge"
        ),
        non_goals=(
            "no continuum error certification",
            "no non-neutral zero-Bloch source",
            "no open or cell-touching surfaces",
            "no modified or oscillatory Helmholtz claim",
            "no vector or Maxwell claim",
            "no adaptive or unbounded image allocation",
        ),
        numeric_version=numeric_version,
    )


__all__ = [
    "PeriodicLaplaceNeutralityError",
    "periodic_laplace_green_3d",
    "prepare_periodic_laplace_single_layer_dp0_3d",
]
