#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Declared robustness policies and evidence of point-cloud reconstruction.

Each policy states what the imperfect input means and reports what it did:

- statistical outlier removal (Rusu et al. 2008) drops samples whose mean
  distance to their ``k`` nearest neighbors exceeds the cloud mean by
  ``outlier_std_ratio`` standard deviations;
- sampling coverage measures the extracted surface farther than
  ``coverage_distance`` from every sample (Poisson closes such holes by
  interpolation; trimming removes them, as in Kazhdan's density trimming);
- thin features are samples with a second sheet stacked along their normal
  line closer than the Poisson resolution, where the indicator cannot separate
  the two sheets;
- component fit measures, per connected sample component, the spread of the
  sampled indicator around the iso-value relative to its unit jump; a
  component whose samples are not fit by the extracted level set is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import equinox as eqx
import numpy as np

from ..._strict import StrictModule
from ..._validation import positive_finite_float, positive_integer
from ...typing import parse
from ._normals import _merge_labels, nearest_neighbors
from ._poisson import nearest_sample_distance


IncompleteSamplingPolicy: TypeAlias = Literal["close", "refuse"]
"""``close`` keeps the watertight Poisson closure of unsampled regions and
reports them; ``refuse`` rejects a solid with any unsupported surface."""

_THIN_PROBE_FACTOR = 4
_THIN_SUPPORT = 3


class ReconstructionRobustness(StrictModule):
    """Declared imperfect-input policies of point-cloud surface reconstruction.

    ``outlier_std_ratio`` enables statistical outlier removal (``None`` keeps
    every sample). ``coverage_distance`` is the largest sample distance at which
    the extracted surface counts as supported (default: twice the finest
    Poisson width). ``incomplete_sampling`` decides whether unsupported surface
    is reported or refused, ``thin_feature_distance`` is the sheet separation
    below which a thin feature is reported (default: twice the finest Poisson
    width), and ``component_fit_tolerance`` is the largest sampled-indicator
    spread, in units of the unit indicator jump, of an accepted component.
    """

    outlier_std_ratio: float | None = eqx.field(static=True)
    coverage_distance: float | None = eqx.field(static=True)
    incomplete_sampling: IncompleteSamplingPolicy = eqx.field(static=True)
    thin_feature_distance: float | None = eqx.field(static=True)
    component_fit_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        outlier_std_ratio: float | None = None,
        coverage_distance: float | None = None,
        incomplete_sampling: IncompleteSamplingPolicy = "close",
        thin_feature_distance: float | None = None,
        component_fit_tolerance: float = 0.25,
    ) -> None:
        self.outlier_std_ratio = (
            None
            if outlier_std_ratio is None
            else positive_finite_float(outlier_std_ratio, "outlier_std_ratio")
        )
        self.coverage_distance = (
            None
            if coverage_distance is None
            else positive_finite_float(coverage_distance, "coverage_distance")
        )
        self.incomplete_sampling = parse(
            incomplete_sampling, IncompleteSamplingPolicy, "incomplete_sampling"
        )
        self.thin_feature_distance = (
            None
            if thin_feature_distance is None
            else positive_finite_float(thin_feature_distance, "thin_feature_distance")
        )
        self.component_fit_tolerance = positive_finite_float(
            component_fit_tolerance, "component_fit_tolerance"
        )


@dataclass(frozen=True, slots=True)
class OutlierRemovalEvidence:
    """Statistical outlier removal over mean ``k``-nearest-neighbor distances.

    ``threshold = mean_distance + std_ratio * std_distance``; samples above it
    are removed and listed by input index in ``removed_indices``.
    """

    neighborhood_size: int
    std_ratio: float
    mean_distance: float
    std_distance: float
    threshold: float
    removed_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SamplingCoverageEvidence:
    """Extracted surface unsupported by samples within ``coverage_distance``.

    A triangle is unsupported when its centroid lies farther than
    ``coverage_distance`` from every sample; ``unsupported_patches`` counts the
    vertex-connected components of unsupported triangles, and
    ``maximum_sample_distance`` is the largest centroid-to-sample distance.
    """

    coverage_distance: float
    surface_area: float
    unsupported_area: float
    unsupported_triangles: int
    unsupported_patches: int
    maximum_sample_distance: float


@dataclass(frozen=True, slots=True)
class ThinFeatureEvidence:
    """Samples with a stacked sheet closer than ``thin_feature_distance``.

    ``minimum_separation`` is the smallest supported normal-line distance to a
    stacked sheet among the probed ``probe_size`` nearest samples (``None``
    when no stacked sheet was found).
    """

    thin_feature_distance: float
    probe_size: int
    thin_samples: int
    minimum_separation: float | None


@dataclass(frozen=True, slots=True)
class ComponentFitEvidence:
    """Sampled-indicator fit of every connected sample component.

    Components are ordered by their minimum sample index. ``mean_offsets`` and
    ``spreads`` are the area-weighted mean and root-mean-square of the sampled
    indicator minus the iso-value; ``unresolved_components`` lists the
    component positions whose spread exceeds ``tolerance``.
    """

    tolerance: float
    sample_counts: tuple[int, ...]
    represented_areas: tuple[float, ...]
    mean_offsets: tuple[float, ...]
    spreads: tuple[float, ...]
    unresolved_components: tuple[int, ...]


def remove_statistical_outliers(
    points: np.ndarray, neighborhood_size: int, std_ratio: float, /
) -> tuple[np.ndarray, OutlierRemovalEvidence]:
    """Retained-sample mask and evidence of statistical outlier removal."""

    size = positive_integer(neighborhood_size, "neighborhood_size")
    if size >= points.shape[0]:
        raise ValueError("neighborhood_size must be smaller than the number of points.")
    _, distances = nearest_neighbors(points, size)
    mean = np.mean(distances, axis=1)
    center = float(np.mean(mean))
    deviation = float(np.std(mean))
    threshold = center + std_ratio * deviation
    retained = mean <= threshold
    return retained, OutlierRemovalEvidence(
        neighborhood_size=size,
        std_ratio=std_ratio,
        mean_distance=center,
        std_distance=deviation,
        threshold=threshold,
        removed_indices=tuple(int(index) for index in np.flatnonzero(~retained)),
    )


def thin_features(
    points: np.ndarray,
    normals: np.ndarray,
    neighborhood_size: int,
    thin_feature_distance: float,
    /,
) -> ThinFeatureEvidence:
    """Probe the nearest samples for a second sheet stacked along each normal.

    A probed sample lies on a stacked sheet when its normal line is parallel
    (``|n_i . n_j| > 1/2``) and its offset lies within 45 degrees of the normal
    line; the separation of sample ``i`` is the ``_THIN_SUPPORT``-th smallest
    such normal offset, so a sheet counts only when several samples support it.
    Stacking is tested on normal lines, not signs, so a thin feature is found
    whether its sheets face apart or were oriented alike by propagation.
    """

    probe = min(_THIN_PROBE_FACTOR * neighborhood_size, points.shape[0] - 1)
    neighbors, _ = nearest_neighbors(points, probe)
    offsets = points[neighbors] - points[:, None, :]
    along = np.sum(offsets * normals[:, None, :], axis=-1)
    across = np.linalg.norm(offsets - along[..., None] * normals[:, None, :], axis=-1)
    parallel = np.abs(np.sum(normals[neighbors] * normals[:, None, :], axis=-1)) > 0.5
    stacked = parallel & (np.abs(along) >= across) & (along != 0.0)
    offsets_ = np.sort(np.where(stacked, np.abs(along), np.inf), axis=1)
    separation = offsets_[:, min(_THIN_SUPPORT, probe) - 1]
    finite = np.isfinite(separation)
    return ThinFeatureEvidence(
        thin_feature_distance=thin_feature_distance,
        probe_size=probe,
        thin_samples=int(np.count_nonzero(separation < thin_feature_distance)),
        minimum_separation=float(np.min(separation[finite])) if np.any(finite) else None,
    )


def sampling_coverage(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    coverage_distance: float,
    /,
) -> tuple[np.ndarray, SamplingCoverageEvidence]:
    """Unsupported-triangle mask and coverage evidence of an extracted surface."""

    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    distance = nearest_sample_distance(points, np.mean(triangles, axis=1))
    unsupported = distance > coverage_distance
    patches = 0
    if np.any(unsupported):
        corners = faces[unsupported].astype(np.int64)
        labels = _merge_labels(
            np.arange(vertices.shape[0], dtype=np.int64),
            np.concatenate((corners[:, 0], corners[:, 1])),
            np.concatenate((corners[:, 1], corners[:, 2])),
        )
        patches = np.unique(labels[corners[:, 0]]).size
    return unsupported, SamplingCoverageEvidence(
        coverage_distance=coverage_distance,
        surface_area=float(np.sum(areas)),
        unsupported_area=float(np.sum(areas[unsupported])),
        unsupported_triangles=int(np.count_nonzero(unsupported)),
        unsupported_patches=patches,
        maximum_sample_distance=float(np.max(distance)),
    )


def component_fit(
    sample_values: np.ndarray,
    areas: np.ndarray,
    components: np.ndarray,
    tolerance: float,
    /,
) -> ComponentFitEvidence:
    """Per-component area-weighted fit of the sampled indicator minus its iso-value."""

    labels, inverse = np.unique(components, return_inverse=True)
    counts = np.bincount(inverse, minlength=labels.size)
    represented = np.bincount(inverse, weights=areas, minlength=labels.size)
    mean = np.bincount(inverse, weights=areas * sample_values) / represented
    spread = np.sqrt(np.bincount(inverse, weights=areas * sample_values**2) / represented)
    return ComponentFitEvidence(
        tolerance=tolerance,
        sample_counts=tuple(int(value) for value in counts),
        represented_areas=tuple(float(value) for value in represented),
        mean_offsets=tuple(float(value) for value in mean),
        spreads=tuple(float(value) for value in spread),
        unresolved_components=tuple(
            int(index) for index in np.flatnonzero(spread > tolerance)
        ),
    )
