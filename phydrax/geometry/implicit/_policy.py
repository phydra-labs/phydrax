#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntFlag
from typing import Literal, TypeAlias

from ...typing import parse


ImplicitDiscoveryEnclosure: TypeAlias = Literal["interval", "lipschitz", "sampled"]
"""Bound source of adaptive discovery.

``interval`` evaluates the source field program in outward-rounded interval
arithmetic with interval gradients; ``lipschitz`` uses the field certificate's
declared Lipschitz bound and evaluation error for values only; ``sampled`` uses
box samples and makes no enclosure claim.
"""

ImplicitDiscoveryAccuracy: TypeAlias = Literal["certified", "enclosed", "sampled"]
"""Achieved claim of adaptive discovery.

``certified``: rigorous value and gradient enclosures certify the cover, the
cell decomposition, and an embedded closed extraction homeomorphic to the zero
set. ``enclosed``: rigorous value enclosures certify that every zero-set point
lies in a reported surface or unresolved box; topology is not certified.
``sampled``: no enclosure claim.
"""


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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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


class AdaptiveImplicitSurfaceStatus(IntFlag):
    """Outcome flags of adaptive implicit surface discovery."""

    SUCCESS = 0
    UNRESOLVED_BOXES = 1
    BOX_BUDGET_EXHAUSTED = 2
    EVALUATION_BUDGET_EXHAUSTED = 4
    MAXIMUM_LEVEL_REACHED = 8
    NO_SURFACE = 16
    DOMAIN_BOUNDARY_CROSSING = 32
    DEGENERATE_FACE = 64
    SELF_INTERSECTION = 128
    NONMANIFOLD_EXTRACTION = 256


class AdaptiveImplicitBoxIssue(IntFlag):
    """Why one final octree leaf is unresolved."""

    NONE = 0
    SINGULAR = 1
    """The value and gradient enclosures both contain zero (near-zero gradient)."""
    EDGE = 2
    """An incident minimal edge has no certified root count."""
    FACE = 4
    """An incident minimal face has no certified arc structure."""
    SHEETS = 8
    """The leaf boundary carries more than one zero-set cycle."""
    DOMAIN_BOUNDARY = 16
    """The zero set is not excluded from the domain boundary."""
    FLATNESS = 32
    """The flatness bound exceeds ``flatness_tolerance``."""


@dataclass(frozen=True, slots=True)
class AdaptiveImplicitSurfacePolicy:
    """Resource, refinement, and extraction policy for adaptive discovery.

    Levels address the balanced octree over the discovery domain: level ``l``
    cells have ``2**-l`` of the domain extent per axis. ``maximum_boxes`` bounds
    the leaf count and ``maximum_evaluations`` the total box and point field
    evaluations; exhausting either stops refinement and reports the remaining
    failures as unresolved boxes. ``edge_isolation_depth`` bounds the interval
    bisection that certifies the root count of an edge the enclosure of its
    leaves cannot certify. ``flatness_tolerance`` bounds, per surface
    leaf, the leaf diagonal times the gradient-direction spread of its gradient
    enclosure; ``None`` disables that error control. ``feature_angle`` is the
    anchor-normal angle above which a vertex is reported as a sharp feature; on
    smooth patches the anchor-normal spread scales with leaf size times
    curvature, so coarse smooth leaves can report features unless
    ``flatness_tolerance`` refines them. Sharp creases cannot meet a flatness
    tolerance and are then reported as unresolved ``FLATNESS`` leaves.
    """

    enclosure: ImplicitDiscoveryEnclosure = "interval"
    initial_level: int = 2
    minimum_surface_level: int = 3
    maximum_level: int = 9
    maximum_boxes: int = 400_000
    maximum_evaluations: int = 8_000_000
    maximum_faces: int = 2_000_000
    maximum_intersection_pairs: int = 4_000_000
    flatness_tolerance: float | None = None
    edge_isolation_depth: int = 8
    root_tolerance: float = 1.0e-10
    qef_regularization: float = 5.0e-2
    minimum_face_area: float = 1.0e-14
    feature_angle: float = math.pi / 4.0
    allow_approximate_zero_set: bool = False

    def __post_init__(self) -> None:
        parse(self.enclosure, ImplicitDiscoveryEnclosure, "enclosure")
        for name, value in (
            ("initial_level", self.initial_level),
            ("minimum_surface_level", self.minimum_surface_level),
            ("maximum_level", self.maximum_level),
            ("maximum_boxes", self.maximum_boxes),
            ("maximum_evaluations", self.maximum_evaluations),
            ("maximum_faces", self.maximum_faces),
            ("maximum_intersection_pairs", self.maximum_intersection_pairs),
            ("edge_isolation_depth", self.edge_isolation_depth),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int.")
        if self.edge_isolation_depth < 0:
            raise ValueError("edge_isolation_depth must be nonnegative.")
        if not (
            1
            <= self.initial_level
            <= self.minimum_surface_level
            <= self.maximum_level
            <= _MAXIMUM_ADAPTIVE_LEVEL
        ):
            raise ValueError(
                "Adaptive levels require 1 <= initial_level <= minimum_surface_level "
                f"<= maximum_level <= {_MAXIMUM_ADAPTIVE_LEVEL}."
            )
        for name, value in (
            ("maximum_boxes", self.maximum_boxes),
            ("maximum_evaluations", self.maximum_evaluations),
            ("maximum_faces", self.maximum_faces),
            ("maximum_intersection_pairs", self.maximum_intersection_pairs),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        for name, value in (
            ("root_tolerance", self.root_tolerance),
            ("qef_regularization", self.qef_regularization),
            ("minimum_face_area", self.minimum_face_area),
            ("feature_angle", self.feature_angle),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.feature_angle >= math.pi:
            raise ValueError("feature_angle must be below pi.")
        if self.flatness_tolerance is not None and (
            not math.isfinite(self.flatness_tolerance) or self.flatness_tolerance <= 0.0
        ):
            raise ValueError("flatness_tolerance must be finite and positive or None.")
        if not isinstance(self.allow_approximate_zero_set, bool):
            raise TypeError("allow_approximate_zero_set must be a bool.")


# Integer entity keys pack (2**level + 1)**3 vertex keys with an axis and a size
# exponent into int64, which bounds the finest adaptive level.
_MAXIMUM_ADAPTIVE_LEVEL = 16


__all__ = [
    "AdaptiveImplicitBoxIssue",
    "AdaptiveImplicitSurfacePolicy",
    "AdaptiveImplicitSurfaceStatus",
    "ImplicitDiscoveryAccuracy",
    "ImplicitDiscoveryEnclosure",
    "ImplicitProjectionPolicy",
    "ImplicitProjectionStatus",
    "ImplicitSurfacePolicy",
    "ImplicitSurfaceStatus",
]
