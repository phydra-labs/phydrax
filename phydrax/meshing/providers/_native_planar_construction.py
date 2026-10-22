#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Uniform-median planar construction admission and construction bounds.

A planar request admits the lattice construction exactly when every hard size
control is one uniform region control whose only target statistic is the median,
without a hard minimum, and whose tolerance interval contains ``[h/2, h]``. The
construction then targets the authored upper interval ``U`` (the median upper
``h + tol(h)``, tightened by every hard maximum itself; the maximum's
compliance tolerance is acceptance slack, never construction allowance) and
seeds the carved region natively with the equilateral lattice of spacing
``LATTICE_FRACTION * U``, whose edges stay strictly below ``U``. Requested
targets, statistics and the final compliance checks never change; graded fields
and other statistics keep the generic refinement.
"""

from __future__ import annotations

import numpy as np

from phydrax.meshing._contracts import SurfaceMeshingSpec

from .._sizing import SizeControlStrength, UniformSizeControl


# Equilateral lattice spacing as a fraction of the admitted construction upper:
# Lattice edges leave ten percent for roundoff and the carved boundary layer.
# Native hard edge/angle refinement and final compliance still decide admission.
LATTICE_FRACTION = 0.9


def uniform_median_strategy(specification: SurfaceMeshingSpec, size: float, /) -> bool:
    """Capability of the lattice construction, not an acceptance policy.

    The construction is designed for a single median with an interval containing
    [h/2, h]. Tight/additional statistics or a hard lower edge bound retain the
    generic refinement; their contracts cannot be inferred from a median.
    """
    hard = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
        and control.strength is SizeControlStrength.HARD
    )
    return (
        bool(hard)
        and specification.size_compliance.target_statistics == ("p50",)
        and all(
            control.scope.entity_dimension == 2
            and control.target_size == size
            and control.minimum_size is None
            and specification.size_compliance.tolerance(size) >= 0.5 * size
            for control in hard
        )
    )


def uniform_construction_size(specification: SurfaceMeshingSpec, size: float, /) -> float:
    """Use the authored upper interval when constructing a uniform median mesh.

    The target and all acceptance checks remain unchanged. Refining to the
    interval center instead can spend the entity/work allowance on a stronger
    constraint that the caller did not request. A hard per-edge maximum bounds
    construction by its declared value; its tolerance only absorbs acceptance
    measurement, so constructed edges never rely on it.
    """
    if not uniform_median_strategy(specification, size):
        return size
    policy = specification.size_compliance
    upper = size + policy.absolute_tolerance + policy.relative_tolerance * size
    for control in specification.size_controls:
        if (
            isinstance(control, UniformSizeControl)
            and control.strength is SizeControlStrength.HARD
            and control.maximum_size is not None
        ):
            upper = min(upper, control.maximum_size)
    return upper


def lattice_spacing(specification: SurfaceMeshingSpec, size: float, /) -> float:
    """Native lattice spacing of an admitted uniform median; zero disables it."""
    if not uniform_median_strategy(specification, size):
        return 0.0
    return LATTICE_FRACTION * uniform_construction_size(specification, size)


def lattice_site_upper(lower: np.ndarray, upper: np.ndarray, spacing: float, /) -> int:
    """Lattice sites the native seeding can visit in one bounding box."""
    if spacing == 0.0:
        return 0
    columns = int(np.floor(float(upper[0] - lower[0]) / spacing)) + 1
    rows = int(np.floor(float(upper[1] - lower[1]) / (spacing * 0.5 * np.sqrt(3.0)))) + 1
    return columns * rows + rows


__all__ = [
    "LATTICE_FRACTION",
    "lattice_site_upper",
    "lattice_spacing",
    "uniform_construction_size",
    "uniform_median_strategy",
]
