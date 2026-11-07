#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntFlag
from typing import Literal, TypeAlias

from ...typing import parse


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
    """Static numerical policy for fixed-topology dual contouring."""

    projection: ImplicitProjectionPolicy = field(default_factory=ImplicitProjectionPolicy)
    lattice_zero_tolerance: float = 1.0e-10
    qef_regularization: float = 5.0e-2
    minimum_face_area: float = 1.0e-12
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


ImplicitDiscoveryEnclosure: TypeAlias = Literal["interval", "lipschitz", "sampled"]
"""Bound source of adaptive discovery.

``interval`` evaluates the scalar program with outward-rounded value and
directional-gradient enclosures. ``lipschitz`` encloses values from the source
certificate, but does not establish regularity. ``sampled`` is exploratory:
samples are never promoted to an enclosure.
"""

ImplicitDiscoveryAccuracy: TypeAlias = Literal["certified", "enclosed", "sampled"]
"""Achieved claim of adaptive discovery.

``certified`` requires a rigorous whole-domain cover, regular zero-set charts,
and an embedded closed extraction. ``enclosed`` means every possible zero is
retained in a reported surface or unresolved region; it does not certify
topology. ``sampled`` makes no whole-domain claim.
"""


class AdaptiveImplicitSurfaceStatus(IntFlag):
    """Outcome flags of exhaustive adaptive implicit discovery."""

    SUCCESS = 0
    UNRESOLVED_BOXES = 1
    PHYSICAL_RESOLUTION_REACHED = 2
    REPRESENTABILITY_LIMIT_REACHED = 4
    NO_SURFACE = 8
    DOMAIN_BOUNDARY_CROSSING = 16
    DEGENERATE_EXTRACTION = 32
    SELF_INTERSECTION = 64
    NONMANIFOLD_EXTRACTION = 128


class AdaptiveImplicitBoxIssue(IntFlag):
    """Why a final possible-zero box could not be resolved.

    ``FACE`` marks a contour triangle whose nondegenerate orientation is not
    certified by the error bounds of its roots; no absolute area floor applies.
    """

    NONE = 0
    SINGULAR = 1
    EDGE = 2
    FACE = 4
    MULTIPLE_SHEETS = 8
    DOMAIN_BOUNDARY = 16
    PHYSICAL_RESOLUTION = 32
    REPRESENTABILITY = 64


@dataclass(frozen=True, slots=True)
class AdaptiveImplicitSurfacePolicy:
    """Numerical and physical-resolution policy for adaptive discovery.

    ``relative_cell_diameter`` is a geometric stopping criterion, scaled by
    the diagonal of the authored discovery domain. It is not a depth, leaf,
    query, or allocation quota: all possible-zero boxes are traversed until
    their topology is resolved or this physical resolution (or binary64
    representability) prevents further subdivision.
    """

    enclosure: ImplicitDiscoveryEnclosure = "interval"
    relative_cell_diameter: float = 1.0e-3
    root_tolerance: float = 1.0e-10
    allow_approximate_zero_set: bool = False
    allow_nonsmooth_field: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "enclosure",
            parse(self.enclosure, ImplicitDiscoveryEnclosure, "enclosure"),
        )
        for name, value in (
            ("relative_cell_diameter", self.relative_cell_diameter),
            ("root_tolerance", self.root_tolerance),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar.")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not isinstance(self.allow_approximate_zero_set, bool):
            raise TypeError("allow_approximate_zero_set must be a bool.")
        if not isinstance(self.allow_nonsmooth_field, bool):
            raise TypeError("allow_nonsmooth_field must be a bool.")


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
