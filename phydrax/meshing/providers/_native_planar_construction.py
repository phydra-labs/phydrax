#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical construction targets for planar size controls."""

from __future__ import annotations

from phydrax.meshing._contracts import SurfaceMeshingSpec
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl


def _size_tolerance(specification: SurfaceMeshingSpec, size: float, /) -> float:
    policy = specification.size_compliance
    return policy.absolute_tolerance + policy.relative_tolerance * size


def uniform_median_strategy(
    specification: SurfaceMeshingSpec, size: float, /
) -> bool:
    """Whether the physical upper tolerance permits constructing for a hard median."""
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
            and _size_tolerance(specification, size) >= 0.5 * size
            for control in hard
        )
    )


def uniform_construction_size(
    specification: SurfaceMeshingSpec, size: float, /
) -> float:
    """Use only the hard request's physical upper tolerance for construction."""
    if not uniform_median_strategy(specification, size):
        return size
    upper = size + _size_tolerance(specification, size)
    for control in specification.size_controls:
        if (
            isinstance(control, UniformSizeControl)
            and control.strength is SizeControlStrength.HARD
            and control.maximum_size is not None
        ):
            upper = min(upper, control.maximum_size)
    return upper




__all__ = ["uniform_construction_size", "uniform_median_strategy"]
