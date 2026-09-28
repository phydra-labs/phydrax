#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-side geometric measurements of periodic label fields for qualification.

These are independent NumPy references over committed label arrays: pairwise
contact counts (grain neighbours), triple-junction angles, and the von
Neumann–Mullins rate fit. They never drive the dynamics.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from jax.typing import ArrayLike

from .._validation import finite_real_scalar, positive_integer
from ..typing import AnyShape, as_host_array, HostFloat64, HostInteger


def label_contacts(labels: ArrayLike, label_count: int, /) -> np.ndarray:
    """Symmetric ``L x L`` counts of face-adjacent site pairs on a periodic grid."""
    count = positive_integer(label_count, "label_count")
    field = as_host_array(labels, HostInteger[AnyShape], "labels")
    if field.ndim < 1 or np.any(field < 0) or np.any(field >= count):
        raise ValueError("labels must index the declared labels on a grid.")
    contacts = np.zeros((count, count), dtype=np.int64)
    for axis in range(field.ndim):
        shifted = np.roll(field, -1, axis=axis)
        different = field != shifted
        np.add.at(contacts, (field[different], shifted[different]), 1)
    return contacts + contacts.T


def label_neighbor_counts(
    labels: ArrayLike, label_count: int, /, *, minimum_contact: int = 1
) -> np.ndarray:
    """Number of distinct face-adjacent neighbour labels of every label.

    Pairs sharing fewer than ``minimum_contact`` adjacent site pairs are ignored,
    which suppresses pixel-scale corner contacts in grain statistics.
    """
    threshold = positive_integer(minimum_contact, "minimum_contact")
    return np.sum(label_contacts(labels, label_count) >= threshold, axis=1)


def _periodic_offsets(
    points: np.ndarray, origin: np.ndarray, periods: np.ndarray, /
) -> np.ndarray:
    displacement = points - origin
    return displacement - periods * np.round(displacement / periods)


def triple_junction_angles(
    labels: ArrayLike,
    lengths: Sequence[float],
    triple: tuple[int, int, int],
    /,
    *,
    center: Sequence[float],
    inner_radius: float,
    outer_radius: float,
) -> np.ndarray:
    """Opening angles (degrees) of three phases at the 2D junction nearest ``center``.

    Interfaces are the midpoints of face-adjacent site pairs of each label pair;
    each interface direction is the mean unit vector from the junction to its
    midpoints inside the annulus ``[inner_radius, outer_radius]``. The returned
    angles follow ``triple`` and sum to 360 degrees.
    """
    field = as_host_array(labels, HostInteger[AnyShape], "labels")
    if field.ndim != 2:
        raise ValueError("Triple-junction angles are measured on 2D label fields.")
    periods = as_host_array(lengths, HostFloat64[AnyShape], "lengths")
    hint = as_host_array(center, HostFloat64[AnyShape], "center")
    if periods.shape != (2,) or hint.shape != (2,):
        raise ValueError("lengths and center need two components.")
    inner = finite_real_scalar(inner_radius, "inner_radius")
    outer = finite_real_scalar(outer_radius, "outer_radius")
    if not 0.0 <= inner < outer:
        raise ValueError("Need 0 <= inner_radius < outer_radius.")
    first, second, third = triple
    if len({first, second, third}) != 3:
        raise ValueError("triple must name three distinct labels.")
    spacing = periods / np.asarray(field.shape, dtype=np.float64)
    index = np.indices(field.shape).transpose(1, 2, 0).astype(np.float64)
    block = np.stack(
        (
            field,
            np.roll(field, -1, axis=0),
            np.roll(field, -1, axis=1),
            np.roll(np.roll(field, -1, axis=0), -1, axis=1),
        )
    )
    meets = np.all(np.stack([np.any(block == label, axis=0) for label in triple]), axis=0)
    if not np.any(meets):
        raise ValueError("The three labels never meet on this grid.")
    corners = (index[meets] + 1.0) * spacing
    offsets = _periodic_offsets(corners, hint, periods)
    junction = corners[np.argmin(np.sum(offsets**2, axis=1))]
    directions = {}
    for pair in ((first, second), (first, third), (second, third)):
        midpoints = []
        for axis in range(2):
            shifted = np.roll(field, -1, axis=axis)
            mask = ((field == pair[0]) & (shifted == pair[1])) | (
                (field == pair[1]) & (shifted == pair[0])
            )
            step = np.zeros((2,), dtype=np.float64)
            step[axis] = 0.5
            midpoints.append((index[mask] + 0.5 + step) * spacing)
        points = _periodic_offsets(np.concatenate(midpoints), junction, periods)
        radius = np.linalg.norm(points, axis=1)
        ring = (radius >= inner) & (radius <= outer)
        if not np.any(ring):
            raise ValueError(f"No interface points of labels {pair} in the annulus.")
        mean = np.mean(points[ring] / radius[ring, None], axis=0)
        directions[pair] = np.arctan2(mean[1], mean[0])
    order = sorted(directions, key=lambda pair: directions[pair])
    angles = {}
    for position, pair in enumerate(order):
        following = order[(position + 1) % 3]
        sector = np.degrees(directions[following] - directions[pair]) % 360.0
        (shared,) = set(pair) & set(following)
        angles[shared] = sector
    return np.asarray([angles[label] for label in triple], dtype=np.float64)


def von_neumann_mullins_fit(
    times: ArrayLike, areas: ArrayLike, neighbor_counts: ArrayLike, /
) -> tuple[float, float]:
    """Least-squares ``dA/dt = k (n - n0)`` over labels present at every time.

    ``areas`` has shape ``(T, L)``; each label's rate is the slope of a linear fit
    of its area history. Returns ``(k, n0)``; the 2D isotropic law predicts
    ``k = pi * mu * sigma / 3`` and ``n0 = 6``.
    """
    t = as_host_array(times, HostFloat64[AnyShape], "times")
    history = as_host_array(areas, HostFloat64[AnyShape], "areas")
    sides = as_host_array(neighbor_counts, HostFloat64[AnyShape], "neighbor_counts")
    if history.ndim != 2 or history.shape[0] != t.shape[0] or t.shape[0] < 2:
        raise ValueError("areas must have one row per time (at least two).")
    if sides.shape != (history.shape[1],):
        raise ValueError("neighbor_counts must have one entry per label.")
    present = np.all(history > 0.0, axis=0)
    if np.count_nonzero(present) < 2:
        raise ValueError("At least two labels must survive the whole window.")
    rates = np.polyfit(t, history[:, present], 1)[0]
    slope, intercept = np.polyfit(sides[present], rates, 1)
    return float(slope), float(-intercept / slope)


__all__ = [
    "label_contacts",
    "label_neighbor_counts",
    "triple_junction_angles",
    "von_neumann_mullins_fit",
]
