#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntFlag


class ImplicitProjectionStatus(IntFlag):
    """Runtime failures of fixed-anchor implicit projection."""

    SUCCESS = 0
    INVALID_GEOMETRY = 1
    NONFINITE = 2
    ROOT_RESIDUAL = 4
    LOST_REGULARITY = 8
    TRUST_REGION_EXCEEDED = 16


class ImplicitSurfaceStatus(IntFlag):
    """Runtime failures of a fixed-topology implicit surface realization."""

    SUCCESS = 0
    INVALID_GEOMETRY = 1
    SIGN_PATTERN_CHANGED = 2
    PROJECTION_FAILED = 4
    QEF_FAILED = 8
    QEF_OUT_OF_CELL = 16
    DEGENERATE_FACE = 32
    ORIENTATION_CHANGED = 64
    SELF_INTERSECTION = 128


@dataclass(frozen=True, slots=True)
class ImplicitProjectionPolicy:
    """Static numerical policy for normal-gauge anchor projection."""

    maximum_steps: int = 12
    root_tolerance: float = 1.0e-8
    minimum_gradient_norm: float = 1.0e-8
    trust_fraction: float = 0.35

    def __post_init__(self):
        if self.maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive.")
        for name, value in (
            ("root_tolerance", self.root_tolerance),
            ("minimum_gradient_norm", self.minimum_gradient_norm),
            ("trust_fraction", self.trust_fraction),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.trust_fraction > 1.0:
            raise ValueError("trust_fraction must not exceed one.")


@dataclass(frozen=True, slots=True)
class ImplicitSurfacePolicy:
    """Static resource and numerical policy for dense dual contouring."""

    projection: ImplicitProjectionPolicy = field(default_factory=ImplicitProjectionPolicy)
    lattice_zero_tolerance: float = 1.0e-10
    qef_regularization: float = 5.0e-2
    minimum_face_area: float = 1.0e-12
    maximum_lattice_points: int = 2_000_000
    maximum_crossings: int = 1_000_000
    maximum_vertices: int = 1_000_000
    maximum_faces: int = 2_000_000
    maximum_intersection_pairs: int = 2_000_000
    allow_approximate_zero_set: bool = False
    allow_nonsmooth_field: bool = False

    def __post_init__(self):
        if not isinstance(self.projection, ImplicitProjectionPolicy):
            raise TypeError("projection must be an ImplicitProjectionPolicy.")
        for name, value in (
            ("lattice_zero_tolerance", self.lattice_zero_tolerance),
            ("qef_regularization", self.qef_regularization),
            ("minimum_face_area", self.minimum_face_area),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name, value in (
            ("maximum_lattice_points", self.maximum_lattice_points),
            ("maximum_crossings", self.maximum_crossings),
            ("maximum_vertices", self.maximum_vertices),
            ("maximum_faces", self.maximum_faces),
            ("maximum_intersection_pairs", self.maximum_intersection_pairs),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")


__all__ = [
    "ImplicitProjectionPolicy",
    "ImplicitProjectionStatus",
    "ImplicitSurfacePolicy",
    "ImplicitSurfaceStatus",
]
