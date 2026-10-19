# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Host surface triangulation arrays and generation evidence."""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np

from .._strict import StrictModule
from .._trainable import NonTrainableState


@final
class SurfaceConstruction(StrictModule, NonTrainableState):
    """Host arrays and evidence of one generated surface triangulation.

    ``triangles`` are counterclockwise about the oriented source normal;
    ``triangle_patches`` and ``triangle_charts`` (chart centroid) locate each
    triangle on its patch and ``triangle_deviations`` is its sampled deviation
    from the source. ``curve_edges`` are the published edges on feature curves
    with their curve, mid-parameter point on the curve and ``curve_directions``
    (the curve tangent sense from the first to the second vertex).
    ``chart_restriction_*`` retains rational affine source roots whenever the
    correctly rounded chart execution coordinate is not itself the root identity.
    """

    vertices: np.ndarray
    vertex_source_dimensions: np.ndarray
    vertex_source_indices: np.ndarray
    vertex_parameters: np.ndarray
    chart_restriction_vertices: np.ndarray
    chart_restriction_edges: np.ndarray
    chart_restriction_endpoint_parameters: np.ndarray
    chart_restriction_parameters: np.ndarray
    triangles: np.ndarray
    triangle_patches: np.ndarray
    triangle_charts: np.ndarray
    triangle_parameters: np.ndarray
    triangle_deviations: np.ndarray
    triangle_deviation_bounds: np.ndarray
    triangle_normal_bounds: np.ndarray
    curve_edges: np.ndarray
    curve_edge_curves: np.ndarray
    curve_edge_deviations: np.ndarray
    curve_edge_deviation_bounds: np.ndarray
    unresolved: np.ndarray
    chart_triangulations: tuple[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        ...,
    ]
    curve_parameters: tuple[tuple[int, np.ndarray], ...]
    minimum_angle: float = eqx.field(static=True)
    rounds: int = eqx.field(static=True)
    inserted: int = eqx.field(static=True)
    refused: int = eqx.field(static=True)
    flips: int = eqx.field(static=True)
    geometry_queries: int = eqx.field(static=True)
    work_units: int = eqx.field(static=True)
    incomplete_refinement: bool = eqx.field(static=True)


__all__ = ["SurfaceConstruction"]
