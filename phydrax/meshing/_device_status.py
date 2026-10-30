#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host interpretation of scientific device adaptation status.

Device adaptation calls accumulate scientific failure and predicate-resolution
flags in one prepared epoch. Capacity requests are transient execution signals
handled by the adaptation host wrapper before this module sees an update. This
module maps real scientific failures and resolves uncertain orientation with
exact host predicates before a commit is accepted.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from .._geometry_predicates import (
    orient2d,
    orient3d,
    PredicateMode,
    PredicateSign,
    resolve_host_predicate_mode,
)
from ..discretization._adaptive_simplex import AdaptiveSimplexStatus
from ._contracts import MeshingFailure, MeshingFailureCategory


class DeviceEpoch(Enum):
    """Device adaptation family: its failure wording and stage."""

    BISECTION = ("Device bisection", "device-bisection")
    METRIC = ("Device metric adaptation", "device-metric")

    @property
    def operation(self) -> str:
        return self.value[0]

    @property
    def stage(self) -> str:
        return self.value[1]


def require_applied_status(status: int, epoch: DeviceEpoch, /) -> None:
    """Raise the first real scientific failure recorded in ``status``."""

    flags = AdaptiveSimplexStatus(status)
    operation, stage = epoch.operation, epoch.stage
    if flags & AdaptiveSimplexStatus.PROTECTED_CONFLICT:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            f"{operation}: the union of bisection closures split a protected edge.",
            stage=stage,
        )
    if flags & AdaptiveSimplexStatus.INVALID_GEOMETRY:
        raise MeshingFailure(
            MeshingFailureCategory.QUALITY_REJECTED,
            f"{operation} certified an inverted or degenerate cell.",
            stage=stage,
        )


def _require_positive_orientation(
    coordinates: np.ndarray, cells: np.ndarray, epoch: DeviceEpoch, /
) -> None:
    """Certify every committed cell POSITIVE with exact host predicates."""

    dimension = cells.shape[1] - 1
    if coordinates.shape[1] != dimension:
        # Surface simplices carry no uncertain device orientation predicate.
        raise ValueError("Orientation resolution requires volumetric simplices.")
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    corners = coordinates[cells]
    predicate = orient2d if dimension == 2 else orient3d
    result = predicate(*(corners[:, index] for index in range(dimension + 1)), mode=mode)
    certain = np.asarray(result.certain)
    positive = certain & (np.asarray(result.signs) == int(PredicateSign.POSITIVE))
    if np.all(positive):
        return
    uncertain = np.count_nonzero(~certain)
    invalid = np.count_nonzero(certain & ~positive)
    raise MeshingFailure(
        MeshingFailureCategory.QUALITY_REJECTED,
        f"{epoch.operation} left unresolved or invalid cell orientation: "
        f"{invalid} zero or negative and {uncertain} unresolved {mode.value} "
        "orientation signs.",
        stage="device-commit",
    )


def require_committed_status(
    status: int, coordinates: np.ndarray, cells: np.ndarray, epoch: DeviceEpoch, /
) -> None:
    """Accept one epoch commit only for applied flags with exactly valid cells.

    ``coordinates`` are the epoch's vertex slots and ``cells`` its active cell
    rows. Terminal flags raise their mapped failure; ``NEEDS_HOST_RESOLUTION``
    requires every cell to be certified POSITIVE by exact host predicates
    (unresolved signs when meshcore is unavailable reject the commit).
    """

    require_applied_status(status, epoch)
    if status & AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION:
        _require_positive_orientation(coordinates, cells, epoch)
