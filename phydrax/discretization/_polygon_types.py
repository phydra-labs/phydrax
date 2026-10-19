#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepared polygonal face geometry shared by finite-volume source owners."""

import equinox as eqx
import numpy as np

from .._meshcore import NativeExecutionMemoryCounters
from .._strict import StrictModule
from .._trainable import NonTrainableState


class PolyhedralFaceTriangulation(StrictModule, NonTrainableState):
    """Shared native boundary-preserving triangles of planar polyhedral faces."""

    triangle_offsets: np.ndarray
    triangle_vertices: np.ndarray
    face_area_vectors: np.ndarray
    face_measures: np.ndarray
    face_centroids: np.ndarray
    native_work_evidence: np.ndarray
    native_memory_counters: tuple[NativeExecutionMemoryCounters, ...]
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    triangulation_id: str = eqx.field(static=True)
