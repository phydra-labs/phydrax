#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Boxwise implicit-field enclosures bound to their source, state and domain.

A cover is a finite set of closed axis-aligned boxes with interval enclosures of
the field value and of every gradient component. Covers record where their
bounds come from: ``user_supplied`` bounds are trusted input, while
``lipschitz_enclosure`` and ``interval_arithmetic`` bounds were established by
Phydrax routines from the owning geometry. Completeness (the boxes tile the
declared domain) is decided exactly on the binary64 box corners.

Topology premises are checked per box, never asserted by name:

- ``regular_value``: every box meeting the zero set has a gradient enclosure
  excluding zero, so the zero set inside the domain is a regular hypersurface
  (implicit function theorem).
- ``small_normal_variation``: additionally the gradient enclosure ``G(B)`` of
  every box meeting the zero set satisfies ``0 not in <G(B), G(B)>`` (the
  Plantinga-Vegter premise under which their subdivision extraction is
  isotopic to the zero set).
- ``directional_monotone``: every box meeting the zero set has a directional
  derivative enclosure along one supplied direction excluding zero, so the
  field is strictly monotone along it (Lebourg mean value theorem) and the zero
  set in the box is a Lipschitz graph over the orthogonal complement (Clarke
  implicit function theorem); this admits nonsmooth creases.
"""

from __future__ import annotations

from fractions import Fraction
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..discretization._topology import CellComplexTopology
from ..typing import Dim, HostBool, HostFloat64, parse, Scope
from ._contracts import CompiledGeometry


ImplicitBoundOrigin: TypeAlias = Literal[
    "user_supplied", "lipschitz_enclosure", "interval_arithmetic"
]
ImplicitTopologyPremise: TypeAlias = Literal[
    "regular_value", "small_normal_variation", "directional_monotone"
]


class _CoverBoxDim(Dim, minimum=1):
    """Boxes of one implicit cover."""


class _CoverAmbientDim(Dim, minimum=1):
    """Ambient dimension of one implicit cover."""


class _CoverDirectionDim(Dim):
    """Directions with directional-derivative enclosures (possibly none)."""


def implicit_state_id(geometry: CompiledGeometry, /) -> str:
    """Canonical identity of a compiled geometry's kernel type and design state."""

    if not isinstance(geometry, CompiledGeometry):
        raise TypeError("geometry must be CompiledGeometry.")
    return canonical_fingerprint(
        {
            "kind": "implicit-geometry-state",
            "kernel": type(geometry.kernel).__qualname__,
            "state": array_tree_fingerprint(
                [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(geometry.state)]
            ),
        }
    )


def _exact_volume(lower: np.ndarray, upper: np.ndarray, /) -> Fraction:
    total = Fraction(0)
    for low, high in zip(lower.tolist(), upper.tolist(), strict=True):
        volume = Fraction(1)
        for start, stop in zip(low, high, strict=True):
            volume *= Fraction(stop) - Fraction(start)
        total += volume
    return total


def _cover_completeness(
    boxes: np.ndarray, domain: np.ndarray, /
) -> tuple[bool, int, float]:
    """Exact tiling test: containment, interior disjointness and volume sum.

    Returns ``(complete, overlapping_pairs, uncovered_volume)``. Contained boxes
    with pairwise disjoint interiors whose exact volumes sum to the domain
    volume cover the closed domain.
    """

    lower = boxes[:, 0]
    upper = boxes[:, 1]
    contained = bool(
        np.all(lower >= domain[0][None]) and np.all(upper <= domain[1][None])
    )
    bvh = prepare_bvh(lower, upper, dtype=np.float64)
    overlaps = 0
    for first, second in bvh_overlap_pair_blocks(bvh, bvh):
        overlaps += int(np.count_nonzero(first < second))
    uncovered = _exact_volume(domain[:1], domain[1:]) - _exact_volume(lower, upper)
    complete = contained and overlaps == 0 and uncovered == 0
    return complete, overlaps, float(uncovered)


def _gradient_norm_lower(lower: np.ndarray, upper: np.ndarray, /) -> np.ndarray:
    """Norm lower bound from componentwise gradient mignitudes (rounded down)."""

    mignitude = np.where(
        (lower > 0.0) | (upper < 0.0), np.minimum(np.abs(lower), np.abs(upper)), 0.0
    )
    squared = np.sum(mignitude * mignitude, axis=-1)
    return np.nextafter(np.sqrt(squared), 0.0) * (squared > 0.0)


def _directional_bounds(
    box_count: int,
    dimension: int,
    directions: ArrayLike | None,
    lower: ArrayLike | None,
    upper: ArrayLike | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validated directional-derivative enclosures (empty when not supplied)."""

    supplied = (directions is not None, lower is not None, upper is not None)
    if not any(supplied):
        empty = np.empty((box_count, 0), dtype=np.float64)
        return np.empty((0, dimension), dtype=np.float64), empty, empty.copy()
    if not all(supplied):
        raise ValueError(
            "directions, directional_lower and directional_upper go together."
        )
    scope = Scope()
    values = parse(
        np.asarray(directions, dtype=np.float64),
        HostFloat64[_CoverDirectionDim, _CoverAmbientDim],
        "directions",
        scope=scope,
    )
    low = parse(
        np.asarray(lower, dtype=np.float64),
        HostFloat64[_CoverBoxDim, _CoverDirectionDim],
        "directional_lower",
        scope=scope,
    )
    high = parse(
        np.asarray(upper, dtype=np.float64),
        HostFloat64[_CoverBoxDim, _CoverDirectionDim],
        "directional_upper",
        scope=scope,
    )
    if values.shape[1] != dimension or low.shape[0] != box_count:
        raise ValueError("Directional bounds must follow the cover boxes and dimension.")
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(low))):
        raise ValueError("Directional bounds must be finite.")
    if not np.all(np.isfinite(high)) or not np.all(low <= high):
        raise ValueError("Directional bounds must be finite ordered intervals.")
    if np.any(np.all(values == 0.0, axis=1)):
        raise ValueError("Directions must be nonzero.")
    return values, low, high


@final
class CertifiedImplicitCover(StrictModule, NonTrainableState):
    """Boxwise sign and regular-value evidence bound to one source state and domain.

    ``value_lower``/``value_upper`` enclose the field and ``gradient_lower``/
    ``gradient_upper`` every gradient component over each closed box. ``complete``
    holds when the boxes exactly tile ``domain`` (lower/upper corners);
    ``certified`` additionally requires every box meeting the zero set to be
    regular. ``bound_origin`` distinguishes trusted user bounds from bounds
    established by Phydrax routines; public construction always records
    ``user_supplied`` (``_bound_origin`` is reserved for the establishing
    routines of the geometry package).
    """

    __strict_contract__ = True

    boxes: HostFloat64[_CoverBoxDim, Literal[2], _CoverAmbientDim]
    value_lower: HostFloat64[_CoverBoxDim]
    value_upper: HostFloat64[_CoverBoxDim]
    gradient_lower: HostFloat64[_CoverBoxDim, _CoverAmbientDim]
    gradient_upper: HostFloat64[_CoverBoxDim, _CoverAmbientDim]
    gradient_norm_lower: HostFloat64[_CoverBoxDim]
    intersects: HostBool[_CoverBoxDim]
    excluded: HostBool[_CoverBoxDim]
    regular: HostBool[_CoverBoxDim]
    domain: HostFloat64[Literal[2], _CoverAmbientDim]
    directions: HostFloat64[_CoverDirectionDim, _CoverAmbientDim]
    directional_lower: HostFloat64[_CoverBoxDim, _CoverDirectionDim]
    directional_upper: HostFloat64[_CoverBoxDim, _CoverDirectionDim]
    source_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)
    bound_origin: ImplicitBoundOrigin = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    overlap_count: int = eqx.field(static=True)
    uncovered_volume: float = eqx.field(static=True)
    certified: bool = eqx.field(static=True)
    cover_id: str = eqx.field(static=True)

    def __init__(
        self,
        boxes: ArrayLike,
        value_lower: ArrayLike,
        value_upper: ArrayLike,
        gradient_lower: ArrayLike,
        gradient_upper: ArrayLike,
        /,
        *,
        domain: ArrayLike,
        source_id: str,
        state_id: str,
        directions: ArrayLike | None = None,
        directional_lower: ArrayLike | None = None,
        directional_upper: ArrayLike | None = None,
        _bound_origin: ImplicitBoundOrigin = "user_supplied",
    ) -> None:
        scope = Scope()
        boxes_ = parse(
            np.asarray(boxes, dtype=np.float64),
            HostFloat64[_CoverBoxDim, Literal[2], _CoverAmbientDim],
            "boxes",
            scope=scope,
        )
        lower = parse(
            np.asarray(value_lower, dtype=np.float64),
            HostFloat64[_CoverBoxDim],
            "value_lower",
            scope=scope,
        )
        upper = parse(
            np.asarray(value_upper, dtype=np.float64),
            HostFloat64[_CoverBoxDim],
            "value_upper",
            scope=scope,
        )
        gradient_low = parse(
            np.asarray(gradient_lower, dtype=np.float64),
            HostFloat64[_CoverBoxDim, _CoverAmbientDim],
            "gradient_lower",
            scope=scope,
        )
        gradient_high = parse(
            np.asarray(gradient_upper, dtype=np.float64),
            HostFloat64[_CoverBoxDim, _CoverAmbientDim],
            "gradient_upper",
            scope=scope,
        )
        domain_ = parse(
            np.asarray(domain, dtype=np.float64),
            HostFloat64[Literal[2], _CoverAmbientDim],
            "domain",
            scope=scope,
        )
        origin = parse(_bound_origin, ImplicitBoundOrigin, "bound_origin")
        source = canonical_identifier(source_id, "source_id")
        state = canonical_identifier(state_id, "state_id")
        values = (boxes_, lower, upper, gradient_low, gradient_high, domain_)
        if not all(np.all(np.isfinite(value)) for value in values):
            raise ValueError("Implicit cover bounds must be finite.")
        if not np.all(boxes_[:, 0] < boxes_[:, 1]) or not np.all(domain_[0] < domain_[1]):
            raise ValueError("Implicit cover boxes require strict lower/upper bounds.")
        if not np.all(lower <= upper) or not np.all(gradient_low <= gradient_high):
            raise ValueError("Implicit value/gradient bounds are inconsistent.")
        complete, overlaps, uncovered = _cover_completeness(boxes_, domain_)
        directions_, directional_low, directional_high = _directional_bounds(
            boxes_.shape[0],
            boxes_.shape[2],
            directions,
            directional_lower,
            directional_upper,
        )
        norm = _gradient_norm_lower(gradient_low, gradient_high)
        intersects = (lower <= 0.0) & (upper >= 0.0)
        regular = ~intersects | (norm > 0.0)
        self.boxes = boxes_
        self.value_lower = lower
        self.value_upper = upper
        self.gradient_lower = gradient_low
        self.gradient_upper = gradient_high
        self.gradient_norm_lower = norm
        self.intersects = intersects
        self.excluded = ~intersects
        self.regular = regular
        self.domain = domain_
        self.directions = directions_
        self.directional_lower = directional_low
        self.directional_upper = directional_high
        self.source_id = source
        self.state_id = state
        self.bound_origin = origin
        self.complete = complete
        self.overlap_count = overlaps
        self.uncovered_volume = uncovered
        self.certified = complete and bool(np.all(regular))
        self.cover_id = canonical_fingerprint(
            {
                "kind": "certified-implicit-cover",
                "boxes": array_tree_fingerprint(boxes_),
                "value": array_tree_fingerprint((lower, upper)),
                "gradient": array_tree_fingerprint((gradient_low, gradient_high)),
                "domain": array_tree_fingerprint(domain_),
                "directional": array_tree_fingerprint(
                    (directions_, directional_low, directional_high)
                ),
                "source": source,
                "state": state,
                "bound_origin": origin,
            }
        )

    @property
    def established(self) -> bool:
        return self.bound_origin != "user_supplied"

    def require_bound(self, source_id: str, state_id: str, /) -> None:
        """Refuse use of this cover for another source or design state."""

        if source_id != self.source_id or state_id != self.state_id:
            raise ValueError("Implicit cover is stale for this source or state.")


def _established_implicit_cover(
    boxes: ArrayLike,
    value_lower: ArrayLike,
    value_upper: ArrayLike,
    gradient_lower: ArrayLike,
    gradient_upper: ArrayLike,
    /,
    *,
    domain: ArrayLike,
    source_id: str,
    state_id: str,
    bound_origin: ImplicitBoundOrigin,
    directions: ArrayLike | None = None,
    directional_lower: ArrayLike | None = None,
    directional_upper: ArrayLike | None = None,
) -> CertifiedImplicitCover:
    """Construct a cover whose bounds a Phydrax routine established itself."""

    return CertifiedImplicitCover(
        boxes,
        value_lower,
        value_upper,
        gradient_lower,
        gradient_upper,
        domain=domain,
        source_id=source_id,
        state_id=state_id,
        directions=directions,
        directional_lower=directional_lower,
        directional_upper=directional_upper,
        _bound_origin=bound_origin,
    )


def establish_implicit_cover(
    geometry: CompiledGeometry,
    boxes: ArrayLike,
    /,
    *,
    domain: ArrayLike,
    source_id: str,
) -> CertifiedImplicitCover:
    """Establish value and gradient enclosures from the owner's Lipschitz bound.

    Requires a reliable field certificate with owner-established Lipschitz and
    evaluation-error bounds ``L`` and ``e``. Each box gets
    ``phi(center) +- (L r + e)`` for half-diagonal ``r`` and the gradient
    enclosure ``[-L, L]`` per component, which is valid but never regular:
    regular-value premises need sharper gradient enclosures (for example from
    interval arithmetic).
    """

    if not isinstance(geometry, CompiledGeometry):
        raise TypeError("geometry must be CompiledGeometry.")
    certificate = geometry.field_certificate
    if not certificate.bounds_established or certificate.validity_region != "all_space":
        raise ValueError(
            "Lipschitz enclosures need owner-established global Lipschitz and "
            "evaluation-error bounds."
        )
    boxes_ = np.asarray(boxes, dtype=np.float64)
    if boxes_.ndim != 3 or boxes_.shape[1] != 2:
        raise ValueError("Implicit cover boxes require shape (box, 2, dimension).")
    lipschitz = float(certificate.lipschitz_upper_bound or 0.0)
    error = float(certificate.evaluation_error or 0.0)
    center = 0.5 * (boxes_[:, 0] + boxes_[:, 1])
    radius = 0.5 * np.linalg.norm(boxes_[:, 1] - boxes_[:, 0], axis=-1)
    # The center is rounded by at most one ulp per axis; the relative slack
    # covers that displacement and the rounding of the enclosure itself.
    width = (lipschitz * radius + error) * (1.0 + 64.0 * np.finfo(np.float64).eps)
    value = np.asarray(
        geometry.boundary_field(jnp.asarray(center, dtype=jnp.float64)),
        dtype=np.float64,
    ).reshape(-1)
    gradient = np.full(center.shape, lipschitz)
    return _established_implicit_cover(
        boxes_,
        value - width,
        value + width,
        -gradient,
        gradient,
        domain=domain,
        source_id=source_id,
        state_id=implicit_state_id(geometry),
        bound_origin="lipschitz_enclosure",
    )


def _premise_boxes(
    cover: CertifiedImplicitCover, premise: ImplicitTopologyPremise, /
) -> np.ndarray:
    match premise:
        case "regular_value":
            return cover.regular
        case "small_normal_variation":
            low = cover.gradient_lower
            high = cover.gradient_upper
            # Lower bound of <G, G> for independent interval factors.
            products = np.minimum.reduce((low * low, low * high, high * low, high * high))
            inner = np.sum(products, axis=-1)
            return ~cover.intersects | (inner > 0.0)
        case "directional_monotone":
            monotone = (cover.directional_lower > 0.0) | (cover.directional_upper < 0.0)
            return ~cover.intersects | np.any(monotone, axis=1)
        case _:
            raise ValueError(f"Unknown implicit topology premise {premise!r}.")


@final
class CertifiedImplicitTopology(StrictModule, NonTrainableState):
    """Finite topology extracted from one cover with its checked premise.

    ``premise_boxes`` marks boxes satisfying the premise. ``certified`` requires
    a complete cover and the premise on every box; ``established`` states that
    the cover's bounds were established by Phydrax rather than supplied. The
    extraction routine that produced ``topology`` owns the claim that it used
    this cover under this premise.
    """

    __strict_contract__ = True

    cover: CertifiedImplicitCover
    topology: CellComplexTopology
    premise_boxes: HostBool[_CoverBoxDim]
    premise: ImplicitTopologyPremise = eqx.field(static=True)
    unresolved_box_count: int = eqx.field(static=True)
    certified: bool = eqx.field(static=True)
    established: bool = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        cover: CertifiedImplicitCover,
        topology: CellComplexTopology,
        /,
        *,
        premise: ImplicitTopologyPremise,
    ) -> None:
        if not isinstance(cover, CertifiedImplicitCover):
            raise TypeError("cover must be CertifiedImplicitCover.")
        if not isinstance(topology, CellComplexTopology):
            raise TypeError("topology must be CellComplexTopology.")
        premise_ = parse(premise, ImplicitTopologyPremise, "premise")
        satisfied = np.asarray(_premise_boxes(cover, premise_), dtype=np.bool_)
        self.cover = cover
        self.topology = topology
        self.premise_boxes = satisfied
        self.premise = premise_
        self.unresolved_box_count = int(np.count_nonzero(~satisfied))
        self.certified = cover.complete and bool(np.all(satisfied))
        self.established = cover.established
        self.result_id = canonical_fingerprint(
            {
                "kind": "certified-implicit-topology",
                "cover": cover.cover_id,
                "topology": topology.topology_id,
                "premise": premise_,
                "certified": self.certified,
            }
        )


__all__ = [
    "CertifiedImplicitCover",
    "CertifiedImplicitTopology",
    "ImplicitBoundOrigin",
    "ImplicitTopologyPremise",
    "establish_implicit_cover",
    "implicit_state_id",
]
