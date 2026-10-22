"""Declared geometry scopes and affine-sweep requests for semantic-control tests."""

from collections.abc import Sequence
from typing import TypedDict, Unpack

import numpy as np

from phydrax.meshing import (
    BoundaryLayerCollisionPolicy,
    BoundaryLayerControl,
    BoundaryLayerRoute,
    LayerSchedule,
    MeshingEntityKind,
    MeshingScope,
)


class SweepFixtureOptions(TypedDict, total=False):
    feature_angle: float
    collision: BoundaryLayerCollisionPolicy
    growth_rate_bounds: tuple[float, float]
    minimum_thickness_fraction: float
    maximum_corner_stretch: float


def geometry_scope(
    dimension: int = 2,
    ids: Sequence[int] = (10, 20),
    *,
    revision: str = "revision-1",
) -> MeshingScope:
    """Bind authored entity IDs to the semantic fixture's geometry revision."""
    return MeshingScope(
        "shape",
        revision,
        MeshingEntityKind.GEOMETRY,
        dimension,
        f"entities-{dimension}",
        np.asarray(ids, dtype=np.int64),
    )


def exact_sweep_request(
    wall: MeshingScope,
    cap: MeshingScope,
    volume_scope: MeshingScope,
    schedule: LayerSchedule,
    **options: Unpack[SweepFixtureOptions],
) -> BoundaryLayerControl:
    """Construct the wall/cap/volume-bound sweep request with explicit overrides."""
    return BoundaryLayerControl(
        wall,
        schedule,
        route=BoundaryLayerRoute.EXACT_SWEEP,
        volume_scope=volume_scope,
        cap_scope=cap,
        **options,
    )
