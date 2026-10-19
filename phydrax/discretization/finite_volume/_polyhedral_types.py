#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared planar-face polyhedral finite-volume geometry records."""

import equinox as eqx
from jax import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PreparedPolyhedralFiniteVolumeGeometry(StrictModule, NonTrainableState):
    """Certified fixed-capacity planar-face polyhedral FV geometry."""

    cell_volumes: Array
    cell_centers: Array
    face_centers: Array
    face_area_vectors: Array
    face_measures: Array
    face_quadrature_points: Array
    face_quadrature_weights: Array
    face_quadrature_valid: Array
    cell_quadrature_points: Array
    cell_quadrature_weights: Array
    cell_quadrature_valid: Array
    owner_cells: Array
    neighbor_cells: Array
    closure_residual: Array
    mesh_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    face_triangulation_id: str = eqx.field(static=True)
